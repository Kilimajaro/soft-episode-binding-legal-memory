#!/usr/bin/env python3
"""Query channels for LegalMem-MT v3.

- u1_exact / uk_followup / u_last: taken from real CAIL user turns
- u_para: LLM paraphrase with overlap constraints (optional offline cache)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def lcs_ratio(a: str, b: str) -> float:
    """Normalized LCS length (chars)."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # banded DP for long strings: use char bigram Jaccard as cheap proxy + prefix LCS cap
    n, m = len(a), len(b)
    if n * m > 2_000_000:
        sa, sb = set(a[i:i + 2] for i in range(max(0, n - 1))), set(b[i:i + 2] for i in range(max(0, m - 1)))
        inter = len(sa & sb)
        return inter / max(1, len(sa | sb))
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur.append(prev[j - 1])
            else:
                cur.append(1 + min(prev[j], cur[-1], prev[j - 1]))
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(n, m)


def trigram_overlap(a: str, b: str) -> float:
    def grams(s):
        s = re.sub(r"\s+", "", s)
        return set(s[i:i + 3] for i in range(max(0, len(s) - 2)))
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def build_queries_for_dialog(session: dict, channels: List[str], rng: random.Random) -> List[dict]:
    """Emit query records for one GOLD dialog."""
    assert session["role"] == "gold"
    users = session["user_turns"]
    assts = session["assistant_turns"]
    sid = session["session_id"]
    out = []

    def pack(channel: str, qtext: str, user_idx: int, evidence_mode: str = "all_assistant"):
        if evidence_mode == "all_assistant":
            # evidence resolved later via tid maps; store textual gold answers
            evidence_answers = assts
        elif evidence_mode == "after_query":
            evidence_answers = assts[user_idx:] if user_idx < len(assts) else assts[-1:]
        else:
            evidence_answers = assts
        out.append({
            "session_id": sid,
            "channel": channel,
            "query": qtext,
            "user_idx": user_idx,
            "evidence_mode": evidence_mode,
            "n_gold_answers": len(evidence_answers),
        })

    if "u1_exact" in channels and users:
        pack("u1_exact", users[0], 0, "all_assistant")

    if "uk_followup" in channels and len(users) >= 2:
        # pick a mid/late real follow-up user turn (not first)
        idx = rng.randint(1, len(users) - 1)
        pack("uk_followup", users[idx], idx, "after_query")

    if "u_last" in channels and users:
        pack("u_last", users[-1], len(users) - 1, "after_query")

    if "u_para" in channels and users:
        # placeholder: actual paraphrase filled by paraphrase_cache or online LLM
        pack("u_para", users[0], 0, "all_assistant")

    return out


def build_query_set(manifest: dict, channels: List[str], seed: int) -> List[dict]:
    rng = random.Random(seed)
    gold = [s for s in manifest["sessions"] if s["role"] == "gold"]
    queries = []
    for s in gold:
        queries.extend(build_queries_for_dialog(s, channels, rng))
    return queries


def load_legalep_queries(
    queries_path: Path,
    channels: List[str],
    para_cache: Optional[Path] = None,
    max_lcs: float = 0.55,
    max_tri: float = 0.45,
) -> List[dict]:
    """Load prebuilt LegalEp queries.json (exact / advice_recall / u_para).

    Channel aliases accepted: u1_exact→exact, u_para→u_para.
    """
    alias = {
        "u1_exact": "exact",
        "exact": "exact",
        "advice_recall": "advice_recall",
        "advice-recall": "advice_recall",
        "u_para": "u_para",
        "paraphrase": "u_para",
    }
    want = {alias.get(c, c) for c in channels}
    rows = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    cache: Dict[str, str] = {}
    if para_cache and Path(para_cache).exists():
        cache = json.loads(Path(para_cache).read_text(encoding="utf-8"))

    # exact text per session for overlap checks
    exact_by_sid = {
        r["session_id"]: r["query"]
        for r in rows
        if r.get("channel") == "exact"
    }
    out: List[dict] = []
    skipped_para = 0
    for r in rows:
        ch = r.get("channel")
        if ch not in want:
            continue
        qtext = r.get("query") or ""
        if ch == "u_para":
            sid = r["session_id"]
            if sid in cache:
                qtext = cache[sid]
            src = exact_by_sid.get(sid, "")
            if not qtext or qtext == src:
                skipped_para += 1
                continue
            if src and (lcs_ratio(qtext, src) > max_lcs or trigram_overlap(qtext, src) > max_tri):
                skipped_para += 1
                continue
        out.append({
            "session_id": r["session_id"],
            "channel": ch,
            "query": qtext,
            "user_idx": 0,
            "evidence_mode": "all_assistant",
            "template_family": r.get("template_family"),
        })
    if skipped_para:
        print(f"[legalep-queries] skipped {skipped_para} u_para rows (placeholder/overlap)", flush=True)
    return out


def apply_paraphrase_cache(queries: List[dict], cache_path: Path, max_lcs: float = 0.55, max_tri: float = 0.45) -> List[dict]:
    """Replace u_para queries using an offline JSON cache {session_id: paraphrased}."""
    if not cache_path.is_file():
        return queries
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    out = []
    for q in queries:
        if q["channel"] != "u_para":
            out.append(q)
            continue
        para = cache.get(q["session_id"])
        if not para:
            out.append(q)
            continue
        # original first user turn is not stored here; skip strict check if missing
        out.append({**q, "query": para, "paraphrase_source": str(cache_path)})
    return out


def paraphrase_with_ollama(
    text: str,
    model: str = "qwen3:14b",
    host: str = "http://localhost:11434",
) -> str:
    """LLM paraphrase; caller should disable thinking for qwen3."""
    import urllib.request
    prompt = (
        "请将下面的法律咨询用户问句改写成语义等价、语气自然的另一句中文提问。"
        "不要复制原句，不要补充新事实，不要回答问题。只输出改写后的问句。\n\n"
        f"原句：{text}"
    )
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip().splitlines()[0].strip()


def build_paraphrase_cache(
    gold_sessions: List[dict],
    out_path: Path,
    model: str,
    max_lcs: float = 0.55,
    max_tri: float = 0.45,
    limit: Optional[int] = None,
) -> dict:
    cache = {}
    if out_path.is_file():
        cache = json.loads(out_path.read_text(encoding="utf-8"))
    items = gold_sessions[: limit or len(gold_sessions)]
    for i, s in enumerate(items):
        sid = s["session_id"]
        if sid in cache:
            continue
        src = s["user_turns"][0]
        ok = None
        for _try in range(3):
            cand = paraphrase_with_ollama(src, model=model)
            if not cand or cand == src:
                continue
            if lcs_ratio(src, cand) <= max_lcs and trigram_overlap(src, cand) <= max_tri:
                ok = cand
                break
        cache[sid] = ok or ("请问" + src + "，这种情况法律上怎么处理？")
        if (i + 1) % 20 == 0:
            out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  paraphrase cache {i+1}/{len(items)}", flush=True)
    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", nargs="+", default=["u1_exact", "uk_followup", "u_last", "u_para"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--para_cache", default="")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qs = build_query_set(man, args.channels, args.seed)
    if args.para_cache:
        qs = apply_paraphrase_cache(qs, Path(args.para_cache))
    Path(args.out).write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(qs)} queries → {args.out}")
