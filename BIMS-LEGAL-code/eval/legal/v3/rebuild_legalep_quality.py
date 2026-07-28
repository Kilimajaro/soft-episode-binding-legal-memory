#!/usr/bin/env python3
"""Rebuild high-quality LegalEp-DISC / LegalEp-Lawyer episode corpora.

Honest definition: each episode is a natural consultation pair
  (user question, assistant answer) with a session_id.
This is NOT fake multi-turn; multi-turn authenticity is reserved for CAIL.

Outputs under data/legal/legalep_v4/{disc,lawyer}/:
  episodes.jsonl, needles.json, distractors.json, queries.json,
  QUALITY_REPORT.json, audit_sample.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval" / "legal"))
from prepare_legal_datasets import ensure_downloaded, _clean, LAWYER_LLAMA_CONSULT_SOURCES  # noqa: E402

OUT_ROOT = REPO / "data" / "legal" / "legalep_v4"

BAD_Q = ("下列说法", "选择题", "判断题", "填空", "以下哪项", "正确答案", "单项选择", "多项选择")
LEGAL_A = (
    "法", "条", "款", "责任", "合同", "诉讼", "仲裁", "赔偿", "权利", "义务",
    "民法典", "刑法", "劳动", "婚姻", "法院", "起诉", "证据", "违约", "侵权",
    "律师", "当事人", "管辖", "程序", "行政处罚",
)


def _norm_key(q: str) -> str:
    s = re.sub(r"\s+", "", q.strip().lower())
    return s[:160]


def _char_trigrams(s: str):
    s = re.sub(r"\s+", "", s)
    return set(s[i:i + 3] for i in range(max(0, len(s) - 2)))


def near_dup(a: str, b: str, thr: float = 0.85) -> bool:
    ga, gb = _char_trigrams(a), _char_trigrams(b)
    if not ga or not gb:
        return False
    return len(ga & gb) / len(ga | gb) >= thr


def quality_gate(q: str, a: str) -> Tuple[bool, str]:
    q, a = _clean(q), _clean(a)
    if not q or not a:
        return False, "empty"
    if not (12 <= len(q) <= 240):
        return False, "q_len"
    if not (80 <= len(a) <= 4000):
        return False, "a_len"
    if any(x in q for x in BAD_Q):
        return False, "exam_like"
    if not any(x in a for x in LEGAL_A):
        return False, "no_legal_signal"
    if re.search(r"<think>|链式思考|思考过程", a, re.I):
        return False, "cot"
    # reject Q almost equal to A prefix (degenerate)
    if q[:20] and q[:20] in a[:40]:
        return False, "qa_leak"
    return True, "ok"


def load_raw_disc(max_n: int) -> List[Tuple[str, str, dict]]:
    path = ensure_downloaded("disc_law")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("input") or obj.get("question") or obj.get("query") or ""
            a = obj.get("output") or obj.get("answer") or obj.get("response") or ""
            # DISC pair format variants
            if not q and "conversations" in obj:
                continue
            out.append((q, a, {"src": "disc_law"}))
            if len(out) >= max_n * 3:  # oversample before filter
                break
    return out


def load_raw_lawyer(max_n: int) -> List[Tuple[str, str, dict]]:
    path = ensure_downloaded("lawyer_llama")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        it = dict(it)
                        it["_bucket"] = k
                        items.append(it)
        data = items
    for obj in data:
        src = str(obj.get("source") or obj.get("_bucket") or "")
        # Strict: consultation sources only (exclude judicial exam / unlabeled)
        if src not in LAWYER_LLAMA_CONSULT_SOURCES:
            continue
        q = (obj.get("input") or obj.get("question") or "").strip()
        a = (obj.get("output") or obj.get("answer") or obj.get("response") or "").strip()
        # Consult files store the user question in `instruction` (input is empty).
        if not q:
            q = (obj.get("instruction") or "").strip()
        if isinstance(obj.get("conversations"), list):
            users = [t.get("value") or t.get("content", "") for t in obj["conversations"] if t.get("from") in ("human", "user")]
            assts = [t.get("value") or t.get("content", "") for t in obj["conversations"] if t.get("from") in ("gpt", "assistant")]
            if users and assts:
                q, a = users[0], assts[-1]
        if not q or not a:
            continue
        out.append((q, a, {"src": "lawyer_llama", "source_file": src}))
    return out


def rebuild(name: str, raw: List[Tuple[str, str, dict]], seed: int, n_needle: int, n_distractor: int, n_audit: int):
    rng = random.Random(seed)
    funnel = Counter()
    episodes = []
    seen_exact = set()
    # first pass quality
    candidates = []
    for q, a, meta in raw:
        funnel["raw"] += 1
        ok, reason = quality_gate(q, a)
        if not ok:
            funnel[f"drop_{reason}"] += 1
            continue
        q, a = _clean(q), _clean(a)
        key = _norm_key(q)
        if key in seen_exact:
            funnel["drop_exact_dup"] += 1
            continue
        seen_exact.add(key)
        candidates.append({"question": q, "answer": a, **meta})
        funnel["pass_gate"] += 1

    rng.shuffle(candidates)

    # greedy near-dup suppression within question text
    kept = []
    for c in candidates:
        if any(near_dup(c["question"], k["question"]) for k in kept[-200:]):  # windowed
            funnel["drop_near_dup"] += 1
            continue
        kept.append(c)
        funnel["pass_near_dup"] += 1
        if len(kept) >= n_needle + n_distractor + 500:
            break

    if len(kept) < n_needle + 100:
        raise RuntimeError(f"{name}: insufficient episodes after filter: {len(kept)}")

    needles = kept[:n_needle]
    distractors = kept[n_needle: n_needle + n_distractor]

    def to_episode(i: str, c: dict, role: str):
        return {
            "session_id": i,
            "role": role,
            "dataset": name,
            "turns": [
                {"role": "user", "content": c["question"]},
                {"role": "assistant", "content": c["answer"]},
            ],
            "question": c["question"],
            "answer": c["answer"],
            "n_turns": 2,
            "meta": {k: c[k] for k in c if k not in ("question", "answer")},
        }

    needle_eps = [to_episode(f"{name}_needle_{i:05d}", c, "needle") for i, c in enumerate(needles)]
    dist_eps = [to_episode(f"{name}_dist_{i:05d}", c, "distractor") for i, c in enumerate(distractors)]
    all_eps = needle_eps + dist_eps

    # queries: exact + advice_recall template + paraphrase placeholder
    templates = [
        "上次咨询里，律师对「{core}」最终是怎么建议的？",
        "关于此前那个问题（涉及：{core}），结论与依据是什么？",
        "请根据档案中该次咨询，说明律师给出的处理意见。",
    ]
    queries = []
    for ep in needle_eps:
        core = ep["question"][:28]
        queries.append({
            "session_id": ep["session_id"],
            "channel": "exact",
            "query": ep["question"],
        })
        queries.append({
            "session_id": ep["session_id"],
            "channel": "advice_recall",
            "query": rng.choice(templates).format(core=core),
            "template_family": "advice_recall_v4",
        })
        queries.append({
            "session_id": ep["session_id"],
            "channel": "u_para",
            "query": ep["question"],  # filled later by LLM cache
            "needs_paraphrase": True,
        })

    audit = rng.sample(needle_eps, min(n_audit, len(needle_eps)))
    audit_rows = [{
        "session_id": e["session_id"],
        "question": e["question"],
        "answer_head": e["answer"][:200],
        "audit_axes": {
            "is_legal_consultation": None,
            "answer_substantive": None,
            "safe_for_research_use": None,
        },
    } for e in audit]

    length_bins = Counter()
    for e in needle_eps:
        L = len(e["answer"])
        if L < 150:
            length_bins["ans_short"] += 1
        elif L < 500:
            length_bins["ans_mid"] += 1
        else:
            length_bins["ans_long"] += 1

    report = {
        "dataset": name,
        "protocol": "LegalEp-v4",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "definition": (
            "Each episode is a natural consultation Q+A pair (2 turns). "
            "Not claimed as multi-turn dialogue; use CAIL for multi-turn."
        ),
        "funnel": dict(funnel),
        "n_needles": len(needle_eps),
        "n_distractors": len(dist_eps),
        "n_episodes_total": len(all_eps),
        "n_turns_total": len(all_eps) * 2,
        "needle_answer_length_bins": dict(length_bins),
        "query_channels": ["exact", "advice_recall", "u_para"],
        "audit_sample_n": len(audit_rows),
        "audit_instructions": (
            "Human raters label audit_axes as yes/no; report agreement (kappa) in paper."
        ),
    }
    return all_eps, needle_eps, dist_eps, queries, audit_rows, report


def write_bundle(name: str, all_eps, needles, dists, queries, audit, report, out_root: Path):
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in all_eps:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (d / "needles.json").write_text(json.dumps(needles, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "distractors.json").write_text(json.dumps(dists, ensure_ascii=False), encoding="utf-8")
    (d / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "audit_sample.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "QUALITY_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # also a shared-corpus manifest at M scale = needles + distractors
    manifest = {
        "tier": "M",
        "dataset": name,
        "protocol": "LegalEp-v4",
        "n_sessions": len(all_eps),
        "n_gold": len(needles),
        "n_distractor": len(dists),
        "n_turns": len(all_eps) * 2,
        "gold_session_ids": [e["session_id"] for e in needles],
        "sessions": [
            {
                "session_id": e["session_id"],
                "role": "gold" if e["role"] == "needle" else "distractor",
                "source": name,
                "turns": e["turns"],
                "user_turns": [e["question"]],
                "assistant_turns": [e["answer"]],
            }
            for e in all_eps
        ],
    }
    (d / "corpus_manifest_M.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_needle", type=int, default=500)
    ap.add_argument("--n_distractor", type=int, default=2500)
    ap.add_argument("--n_audit", type=int, default=50)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()
    out = Path(args.out)

    disc_raw = load_raw_disc(args.n_needle + args.n_distractor)
    lawyer_raw = load_raw_lawyer(args.n_needle + args.n_distractor)

    for name, raw in [("legalep_disc", disc_raw), ("legalep_lawyer", lawyer_raw)]:
        bundle = rebuild(name, raw, args.seed, args.n_needle, args.n_distractor, args.n_audit)
        write_bundle(name, *bundle, out_root=out)


if __name__ == "__main__":
    main()
