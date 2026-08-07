#!/usr/bin/env python3
"""LegalMem-MT v3 retrieval runner.

Indexes a corpus_manifest_{tier}.json into VectorMemoryManager, evaluates
turn-level configs on gold queries, writes per-query hits for significance tests.

Supports LegalEp prebuilt queries.json, BM25 turn/joint, dense RRF, beta sweeps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_DEVICE", "0"))

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval" / "legal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OLLAMA_BASE_URL, CUDA_DEVICE, _DATA_ROOT  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from query_channels import (  # noqa: E402
    build_query_set,
    apply_paraphrase_cache,
    load_legalep_queries,
)
from stats_sig import bootstrap_ci, paired_report  # noqa: E402
from run_revision_protocol import (  # noqa: E402
    parent_hydrate,
    session_max_expand,
    metrics_at_k as _metrics_at_k_base,
)
from legal_metrics import failure_taxonomy  # noqa: E402


def metrics_at_k(retrieved, gold_sess_tids, gold_ans_tids, k, gold_q_tids=None):
    """Prefer revision AH/EC path; attach failure taxonomy when query tids exist."""
    out = _metrics_at_k_base(retrieved, gold_sess_tids, gold_ans_tids, k)
    if gold_q_tids is not None:
        out = dict(out)
        out["failure_mode"] = failure_taxonomy(
            retrieved[:k], gold_sess_tids, gold_ans_tids, gold_q_tids, k
        )
    return out

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore


DENSE_CONFIGS = ["dense_flat", "dense_o2", "parent_hydrate", "session_max", "shuffled_o2", "joint_qa", "dense_ce"]
SPARSE_CONFIGS = ["bm25_turn", "bm25_joint", "dense_rrf"]
CONFIGS = DENSE_CONFIGS + SPARSE_CONFIGS

_CE_MODEL = None


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
    """Lazy-load multilingual CE reranker (Chinese-capable)."""
    global _CE_MODEL
    if _CE_MODEL is not None:
        return _CE_MODEL
    from sentence_transformers import CrossEncoder
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ce] loading {model_name} on {device}", flush=True)
    _CE_MODEL = CrossEncoder(model_name, device=device, trust_remote_code=True)
    return _CE_MODEL


def ce_rerank(mgr, query: str, candidates: List[dict], top_k: int, model_name: str) -> List[dict]:
    """Rerank dense candidates with a cross-encoder; keep top_k."""
    if not candidates:
        return []
    ce = get_cross_encoder(model_name)
    pairs = []
    for r in candidates:
        tid = r.get("tid")
        text = r.get("text") or r.get("full_text") or ""
        if not text and tid is not None and hasattr(mgr, "para_tree"):
            node = mgr.para_tree.get(tid)
            text = getattr(node, "text", "") if node is not None else ""
        pairs.append([query, text[:2000]])
    scores = ce.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: -float(x[1]),
    )[:top_k]
    out = []
    for r, sc in ranked:
        item = dict(r)
        item["score"] = float(sc)
        item["final_score"] = float(sc)
        out.append(item)
    return out


def apply_cfg(mgr: VectorMemoryManager, name: str, beta: float = 0.98):
    # Keep embedding LRU across configs (same queries), but drop cached search
    # rankings — otherwise Soft O2 / shuffled_o2 / β reuse FlatIP hit lists.
    if hasattr(mgr, "clear_search_cache"):
        mgr.clear_search_cache()
    else:
        mgr.lru_cache.clear()
    mgr._retrieval_augment = None
    mgr._query_projection = None
    mgr._exact_match_boost = 0.0
    mgr._session_first_rerank = False
    if name == "dense_flat":
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    elif name in ("dense_o2", "shuffled_o2"):
        mgr.vector_store.use_pq = False
        mgr._session_expand = True
        mgr._session_coherence = float(beta)
        mgr._session_first_rerank = True
    elif name in ("parent_hydrate", "session_max", "joint_qa", "bm25_turn", "bm25_joint", "dense_rrf", "dense_ce"):
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    else:
        raise ValueError(name)


def index_turn_store(sessions: List[dict], *, skip_finalize: bool = False) -> tuple[VectorMemoryManager, dict]:
    """Index every turn; return mgr and maps session_id -> {all_tids, ans_tids}."""
    mgr = VectorMemoryManager()
    mgr.reset(use_pq=False)
    mgr.vector_store.use_pq = False
    mgr._bulk_load = True
    n_turns = sum(len(s["turns"]) for s in sessions)
    try:
        # Keep paragraph + sentence embeddings in LRU during bulk index.
        mgr.lru_cache.capacity = max(getattr(mgr.lru_cache, "capacity", 500), n_turns * 20 + 2000)
    except Exception:
        pass
    texts = [t["content"] for s in sessions for t in s["turns"]]
    # Also warm sentence fragments so add_dialog hits disk/LRU instead of Ollama.
    sent_texts: List[str] = []
    _tmp = VectorMemoryManager()
    for text in texts:
        try:
            sent_texts.extend(_tmp._split_sentences(text))
        except Exception:
            pass
    del _tmp
    try:
        n_warm = mgr.warm_embed_cache(texts + sent_texts)
        print(
            f"[index] warm_embed_cache paras={len(texts)} sents={len(sent_texts)} newly_loaded={n_warm}",
            flush=True,
        )
    except Exception as e:
        print(f"[index] warm_embed_cache skipped: {e}", flush=True)
        try:
            mgr.warm_embed_cache(texts)
        except Exception:
            pass
    meta = {}
    for i, s in enumerate(sessions):
        sid = s["session_id"]
        all_tids, ans_tids, user_tids = [], [], []
        for t in s["turns"]:
            tid = mgr.add_dialog(t["role"], t["content"], session_id=sid)
            all_tids.append(tid)
            if t["role"] == "assistant":
                ans_tids.append(tid)
            elif t["role"] == "user":
                user_tids.append(tid)
        meta[sid] = {
            "all_tids": all_tids,
            "ans_tids": ans_tids or all_tids[-1:],
            "user_tids": user_tids or all_tids[:1],
            "role": s["role"],
        }
        if (i + 1) % 500 == 0:
            print(f"[index] sessions {i+1}/{len(sessions)}", flush=True)
    if not skip_finalize:
        mgr.finalize_bulk_load()
    mgr._bulk_load = False
    print(f"[index] turn store done sessions={len(sessions)}", flush=True)
    return mgr, meta


def index_joint_store(sessions: List[dict]) -> tuple[VectorMemoryManager, dict]:
    mgr = VectorMemoryManager()
    mgr.reset(use_pq=False)
    mgr.vector_store.use_pq = False
    mgr._bulk_load = True
    texts = []
    for s in sessions:
        blob = "\n".join(f"{t['role']}: {t['content']}" for t in s["turns"])
        texts.append(blob)
    try:
        n_warm = mgr.warm_embed_cache(texts)
        print(f"[index] joint warm_embed_cache={n_warm}", flush=True)
        mgr.lru_cache.capacity = max(getattr(mgr.lru_cache, "capacity", 500), len(sessions) + 500)
    except Exception:
        pass
    meta = {}
    for s, blob in zip(sessions, texts):
        sid = s["session_id"]
        tid = mgr.add_dialog("assistant", blob, session_id=sid)
        meta[sid] = {"all_tids": [tid], "ans_tids": [tid], "role": s["role"]}
    mgr.finalize_bulk_load()
    mgr._bulk_load = False
    return mgr, meta


def shuffle_sids(mgr: VectorMemoryManager, seed: int = 7):
    rng = np.random.default_rng(seed)
    tids = list(mgr._tid_to_session.keys())
    sids = [mgr._tid_to_session[t] for t in tids]
    perm = rng.permutation(len(sids))
    new_map = {tids[i]: sids[int(perm[i])] for i in range(len(tids))}
    mgr._tid_to_session = new_map
    members: Dict[str, List[str]] = {}
    for tid, sid in new_map.items():
        members.setdefault(sid, []).append(tid)
    mgr._session_members = members


def uniquify_session_ids(sessions: List[dict]) -> int:
    """Ensure each session row has a unique session_id.

    CAIL prelim manifests can reuse the same session_id for two different
    dialogues. Collisions overwrite ``meta[sid]`` and make Soft~O2 incorrectly
    bind turns across dialogues; BM25 then hits ``len(turns) != len(tids)``.
    Returns the number of renamed (non-first) occurrences.
    """
    seen: Dict[str, int] = {}
    n_fix = 0
    for s in sessions:
        sid = str(s["session_id"])
        if sid not in seen:
            seen[sid] = 0
            continue
        seen[sid] += 1
        s["session_id"] = f"{sid}__occ{seen[sid]}"
        n_fix += 1
    return n_fix


def results_complete(path: Path, channels: Sequence[str], configs: Sequence[str]) -> bool:
    """True iff results.json exists and contains every channel×config cell."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ch_map = payload.get("channels") or {}
    for ch in channels:
        # LegalEp may map u1_exact→exact; callers pass the stored channel names.
        block = ch_map.get(ch) or {}
        cfgs = (block.get("configs") or {}) if isinstance(block, dict) else {}
        for cfg in configs:
            if cfg not in cfgs:
                return False
    return True


def build_bm25_from_sessions(sessions: List[dict], meta: dict):
    """Build BM25 over turns and joint session blobs using tid maps from index_turn_store."""
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 is required for sparse configs")
    turn_docs, turn_meta = [], []
    joint_docs, joint_meta = [], []
    for s in sessions:
        sid = s["session_id"]
        if sid not in meta:
            raise KeyError(f"BM25: session_id {sid!r} missing from turn-store meta")
        gold = meta[sid]
        turns = s["turns"]
        tids = gold["all_tids"]
        if len(turns) != len(tids):
            raise RuntimeError(
                f"BM25: sid={sid!r} turns={len(turns)} tids={len(tids)}; "
                "likely duplicate session_id collision — call uniquify_session_ids first"
            )
        for t, tid in zip(turns, tids):
            turn_docs.append(_tok(t["content"]))
            turn_meta.append({"tid": tid, "sid": sid})
        blob = " ".join(t["content"] for t in turns)
        joint_docs.append(_tok(blob))
        joint_meta.append({"sid": sid, "tids": list(tids)})
    return BM25Okapi(turn_docs), turn_meta, BM25Okapi(joint_docs), joint_meta


def _tok(text: str) -> List[str]:
    text = (text or "").strip().lower()
    chars = [c for c in text if not c.isspace()]
    return chars if chars else ["_"]


def bm25_turn_search(bm25, meta, query: str, top_k: int) -> List[dict]:
    scores = bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in order:
        m = meta[int(idx)]
        sc = float(scores[int(idx)])
        out.append({"tid": m["tid"], "score": sc, "final_score": sc})
    return out


def bm25_joint_search(bm25, meta, query: str, top_k: int) -> List[dict]:
    scores = bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1]
    out, seen = [], set()
    for idx in order:
        for tid in meta[int(idx)]["tids"]:
            if tid in seen:
                continue
            seen.add(tid)
            sc = float(scores[int(idx)])
            out.append({"tid": tid, "score": sc, "final_score": sc})
            if len(out) >= top_k:
                return out
    return out


def rrf_fuse(lists: List[List[dict]], top_k: int, k: int = 60) -> List[dict]:
    scores: Dict[str, float] = {}
    payload: Dict[str, dict] = {}
    for lst in lists:
        for rank, r in enumerate(lst):
            tid = str(r.get("tid", ""))
            if not tid:
                continue
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + rank + 1)
            payload[tid] = r
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    out = []
    for tid, sc in ranked:
        r = dict(payload[tid])
        r["final_score"] = sc
        r["score"] = sc
        out.append(r)
    return out


def eval_config(
    mgr,
    meta,
    queries,
    config,
    top_k,
    beta=0.98,
    bm25_pack=None,
    ce_model: str = "BAAI/bge-reranker-v2-m3",
):
    base_cfg = "dense_flat" if config == "dense_ce" else ("dense_o2" if config == "shuffled_o2" else config)
    apply_cfg(mgr, base_cfg, beta=beta)
    orig = None
    if config == "shuffled_o2":
        orig = (dict(mgr._tid_to_session), {k: list(v) for k, v in mgr._session_members.items()})
        shuffle_sids(mgr)

    metric_keys = ["session_hit@k", "answer_hit@k", "episode_completeness@k", "ndcg@k", "mrr@k"]
    acc = {k: [] for k in metric_keys}
    failure_modes = []
    t0 = time.time()
    for n, q in enumerate(queries):
        sid = q["session_id"]
        gold = meta[sid]
        if q.get("evidence_mode") == "after_query" and len(gold["ans_tids"]) > q.get("user_idx", 0):
            ans = gold["ans_tids"][q["user_idx"] :]
        else:
            ans = gold["ans_tids"]
        q_tids = gold.get("user_tids") or gold["all_tids"][:1]

        if config == "bm25_turn":
            assert bm25_pack is not None
            retrieved = bm25_turn_search(bm25_pack[0], bm25_pack[1], q["query"], top_k)
        elif config == "bm25_joint":
            assert bm25_pack is not None
            retrieved = bm25_joint_search(bm25_pack[2], bm25_pack[3], q["query"], top_k)
        elif config == "dense_rrf":
            assert bm25_pack is not None
            dense = mgr.search(q["query"], top_k=max(top_k * 3, 30), is_temporal_task=False)[:top_k]
            sparse = bm25_turn_search(bm25_pack[0], bm25_pack[1], q["query"], top_k)
            retrieved = rrf_fuse([dense, sparse], top_k)
        elif config == "dense_ce":
            cand = mgr.search(q["query"], top_k=max(top_k * 3, 30), is_temporal_task=False)
            retrieved = ce_rerank(mgr, q["query"], cand, top_k, ce_model)
        else:
            retrieved = mgr.search(q["query"], top_k=max(top_k * 3, 30), is_temporal_task=False)
            if config == "parent_hydrate":
                retrieved = parent_hydrate(mgr, retrieved, top_k)
            elif config == "session_max":
                retrieved = session_max_expand(mgr, retrieved, top_k)
            else:
                retrieved = retrieved[:top_k]

        m = metrics_at_k(retrieved, gold["all_tids"], ans, top_k, gold_q_tids=q_tids)
        for k in metric_keys:
            acc[k].append(m[k])
        if m.get("failure_mode"):
            failure_modes.append(m["failure_mode"])
        if (n + 1) % 50 == 0:
            print(f"  [{config}] {n+1}/{len(queries)} AH={np.mean(acc['answer_hit@k']):.3f}", flush=True)

    if orig is not None:
        mgr._tid_to_session, mgr._session_members = orig

    hits = [int(x) for x in acc["answer_hit@k"]]
    from legal_metrics import aggregate_failure_counts

    out = {
        "config": config,
        "beta": beta,
        "n": len(queries),
        "session_hit@k": float(np.mean(acc["session_hit@k"])) if acc["session_hit@k"] else 0.0,
        "answer_hit@k": float(np.mean(acc["answer_hit@k"])) if acc["answer_hit@k"] else 0.0,
        "episode_completeness@k": float(np.mean(acc["episode_completeness@k"])) if acc["episode_completeness@k"] else 0.0,
        "ndcg@k": float(np.mean(acc["ndcg@k"])) if acc["ndcg@k"] else 0.0,
        "mrr@k": float(np.mean(acc["mrr@k"])) if acc["mrr@k"] else 0.0,
        "ah_ci": bootstrap_ci(hits),
        "per_query_ah": hits,
        "elapsed_seconds": round(time.time() - t0, 1),
        "ce_model": ce_model if config == "dense_ce" else None,
    }
    if failure_modes:
        out["failure_taxonomy"] = aggregate_failure_counts(failure_modes)
        out["per_query_failure"] = failure_modes
    return out


def resolve_queries(man: dict, args) -> List[dict]:
    """Prefer LegalEp queries.json when present and channels look LegalEp-native."""
    man_path = Path(args.manifest)
    qpath = man_path.parent / "queries.json"
    legalep_channels = {"exact", "advice_recall", "u_para", "advice-recall", "paraphrase"}
    requested = set(args.channels)
    use_legalep = qpath.exists() and (
        bool(requested & legalep_channels)
        or args.force_queries_json
        or "legalep" in str(man_path).lower()
    )
    # If only CAIL-style channels requested on LegalEp, map u1_exact→exact via queries.json when forced
    if use_legalep and qpath.exists():
        mapped = []
        for c in args.channels:
            if c == "u1_exact":
                mapped.append("exact")
            elif c in ("uk_followup", "u_last"):
                # LegalEp has no multi-turn follow-ups; skip silently
                continue
            else:
                mapped.append(c)
        if not mapped:
            mapped = ["exact"]
        qs = load_legalep_queries(
            qpath,
            mapped,
            para_cache=Path(args.para_cache) if args.para_cache else None,
        )
        if qs:
            return qs
    queries = build_query_set(man, args.channels, args.seed)
    if args.para_cache:
        queries = apply_paraphrase_cache(queries, Path(args.para_cache))
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tier", default="")
    ap.add_argument("--channels", nargs="+", default=["u1_exact", "uk_followup", "u_last"])
    ap.add_argument("--configs", nargs="+", default=["dense_flat", "dense_o2", "parent_hydrate", "joint_qa"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.98)
    ap.add_argument("--betas", nargs="*", type=float, default=None, help="If set, sweep Soft O2 beta values")
    ap.add_argument(
        "--beta_channel",
        default="",
        help="Channel name for Soft O2 beta sweep (default: first channel in query set)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--para_cache", default="")
    ap.add_argument("--force_queries_json", action="store_true")
    ap.add_argument("--ce_model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out_dir", default=str(REPO / "results" / "legalmem_mt_v3"))
    ap.add_argument("--max_queries", type=int, default=0, help="0=all gold queries")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tier = args.tier or man.get("tier", "X")
    sessions = man["sessions"]
    n_sid_fix = uniquify_session_ids(sessions)
    if n_sid_fix:
        print(f"[fix] uniquified {n_sid_fix} duplicate session_id occurrences", flush=True)
    print(
        f"[env] CUDA_DEVICE={CUDA_DEVICE} OLLAMA={OLLAMA_BASE_URL} DATA={_DATA_ROOT} "
        f"sessions={len(sessions)} gold={man.get('n_gold')}",
        flush=True,
    )
    queries = resolve_queries(man, args)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    out_dir = Path(args.out_dir) / f"tier_{tier}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[index] turn store sessions={len(sessions)} turns={sum(len(s['turns']) for s in sessions)}", flush=True)
    mgr_turn, meta_turn = index_turn_store(sessions)
    need_joint = "joint_qa" in args.configs
    if need_joint:
        print("[index] joint store", flush=True)
        mgr_joint, meta_joint = index_joint_store(sessions)
    else:
        mgr_joint = meta_joint = None

    need_sparse = any(c in SPARSE_CONFIGS for c in args.configs)
    bm25_pack = None
    if need_sparse:
        print("[index] BM25", flush=True)
        bm25_pack = build_bm25_from_sessions(sessions, meta_turn)

    by_ch: Dict[str, List[dict]] = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    payload = {
        "protocol": "LegalMem-MT-v3",
        "tier": tier,
        "n_sessions": man["n_sessions"],
        "n_gold": man["n_gold"],
        "n_distractor": man["n_distractor"],
        "n_turns": man["n_turns"],
        "channels": {},
        "comparisons": {},
        "beta_sweep": {},
    }

    for ch, qs in by_ch.items():
        print(f"=== channel {ch} n={len(qs)} ===", flush=True)
        payload["channels"][ch] = {"configs": {}}
        for cfg in args.configs:
            print(f"[{ch}/{cfg}] start", flush=True)
            if cfg == "joint_qa":
                res = eval_config(mgr_joint, meta_joint, qs, cfg, args.top_k, beta=args.beta, ce_model=args.ce_model)
            else:
                res = eval_config(
                    mgr_turn, meta_turn, qs, cfg, args.top_k, beta=args.beta,
                    bm25_pack=bm25_pack, ce_model=args.ce_model,
                )
            payload["channels"][ch]["configs"][cfg] = res
            path = out_dir / "results.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{ch}/{cfg}] AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f} "
                f"nDCG={res['ndcg@k']:.3f} CI=[{res['ah_ci']['ci_low']:.3f},{res['ah_ci']['ci_high']:.3f}]",
                flush=True,
            )

        cfgs = payload["channels"][ch]["configs"]
        comps = {}
        if "dense_o2" in cfgs and "dense_flat" in cfgs:
            comps["o2_vs_flat"] = paired_report(
                "dense_o2", cfgs["dense_o2"]["per_query_ah"],
                "dense_flat", cfgs["dense_flat"]["per_query_ah"],
            )
        if "dense_o2" in cfgs and "parent_hydrate" in cfgs:
            comps["o2_vs_hard"] = paired_report(
                "dense_o2", cfgs["dense_o2"]["per_query_ah"],
                "parent_hydrate", cfgs["parent_hydrate"]["per_query_ah"],
            )
        if "dense_o2" in cfgs and "dense_ce" in cfgs:
            comps["o2_vs_ce"] = paired_report(
                "dense_o2", cfgs["dense_o2"]["per_query_ah"],
                "dense_ce", cfgs["dense_ce"]["per_query_ah"],
            )
        if "dense_o2" in cfgs and "joint_qa" in cfgs:
            comps["o2_vs_joint"] = paired_report(
                "dense_o2", cfgs["dense_o2"]["per_query_ah"],
                "joint_qa", cfgs["joint_qa"]["per_query_ah"],
            )
        payload["comparisons"][ch] = comps

    # Optional Soft O2 beta sweep (default: first channel; override with --beta_channel)
    if args.betas:
        primary_ch = args.beta_channel or next(iter(by_ch))
        if primary_ch not in by_ch:
            raise SystemExit(
                f"--beta_channel={primary_ch!r} not in loaded channels {list(by_ch)}"
            )
        qs = by_ch[primary_ch]
        print(f"=== beta sweep on {primary_ch} n={len(qs)} ===", flush=True)
        for b in args.betas:
            print(f"[beta={b}] start", flush=True)
            res = eval_config(mgr_turn, meta_turn, qs, "dense_o2", args.top_k, beta=b)
            payload["beta_sweep"][str(b)] = {
                "channel": primary_ch,
                "answer_hit@k": res["answer_hit@k"],
                "episode_completeness@k": res["episode_completeness@k"],
                "ndcg@k": res["ndcg@k"],
                "ah_ci": res["ah_ci"],
            }
            (out_dir / "results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[beta={b}] AH={res['answer_hit@k']:.3f}", flush=True)

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
