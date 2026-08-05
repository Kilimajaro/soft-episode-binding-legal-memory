#!/usr/bin/env python3
"""LegalMem-MT semantic-path ablation for the IPM manuscript.

This runner isolates the BIRCH semantic-cluster pathway from the episodic turn
index. To avoid conflating slow-path effects, summary retrieval is disabled for
all configs in this script; only BIRCH knowledge-cluster retrieval is toggled.

Configs:
  - ep_flat:     episodic-only FlatIP (no BIRCH, no O2)
  - ep_o2:       episodic-only + Soft O2
  - birch1_flat: episodic + primary BIRCH-cluster retrieval (no assoc expand)
  - birch1_o2:   birch1 + Soft O2
  - birch2_flat: episodic + full BIRCH associative retrieval
  - birch2_o2:   birch2 + Soft O2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval" / "legal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_manager import VectorMemoryManager  # noqa: E402
from stats_sig import bootstrap_ci, paired_report  # noqa: E402
from run_legalmem_mt import (  # noqa: E402
    index_turn_store,
    metrics_at_k,
    resolve_queries,
    uniquify_session_ids,
)

CONFIGS = [
    "ep_flat",
    "ep_o2",
    "birch1_flat",
    "birch1_o2",
    "birch2_flat",
    "birch2_o2",
]


def apply_semantic_cfg(
    mgr: VectorMemoryManager,
    name: str,
    *,
    beta: float,
    full_kg: dict,
) -> dict:
    if hasattr(mgr, "clear_search_cache"):
        mgr.clear_search_cache()
    else:
        mgr.lru_cache.clear()

    mgr._retrieval_augment = None
    mgr._query_projection = None
    mgr._exact_match_boost = 0.0
    mgr._session_expand = False
    mgr._session_first_rerank = False
    mgr._session_coherence = float(beta)

    # Hold summary retrieval off throughout this ablation so that "slow path"
    # means the BIRCH semantic-cluster route only.
    mgr.summary_nodes = {}

    if name.startswith("ep_"):
        mgr.knowledge_graph = {}
        mgr._ablation_no_assoc = True
    elif name.startswith("birch1_"):
        mgr.knowledge_graph = full_kg
        mgr._ablation_no_assoc = True
    elif name.startswith("birch2_"):
        mgr.knowledge_graph = full_kg
        mgr._ablation_no_assoc = False
    else:
        raise ValueError(name)

    if name.endswith("_o2"):
        mgr._session_expand = True
        mgr._session_first_rerank = True

    return {
        "summary_enabled": False,
        "knowledge_enabled": bool(mgr.knowledge_graph),
        "assoc_expand": not bool(getattr(mgr, "_ablation_no_assoc", False)),
        "session_expand": bool(mgr._session_expand),
        "session_first_rerank": bool(mgr._session_first_rerank),
        "knowledge_nodes": len(mgr.knowledge_graph),
    }


def eval_config(mgr, meta, queries, config, top_k, *, beta, full_kg):
    cfg_meta = apply_semantic_cfg(mgr, config, beta=beta, full_kg=full_kg)
    metric_keys = ["session_hit@k", "answer_hit@k", "episode_completeness@k", "ndcg@k", "mrr@k"]
    acc = {k: [] for k in metric_keys}
    t0 = time.time()
    for n, q in enumerate(queries):
        sid = q["session_id"]
        gold = meta[sid]
        if q.get("evidence_mode") == "after_query" and len(gold["ans_tids"]) > q.get("user_idx", 0):
            ans = gold["ans_tids"][q["user_idx"] :]
        else:
            ans = gold["ans_tids"]
        retrieved = mgr.search(q["query"], top_k=max(top_k * 3, 30), is_temporal_task=False)[:top_k]
        m = metrics_at_k(retrieved, gold["all_tids"], ans, top_k)
        for k in metric_keys:
            acc[k].append(m[k])
        if (n + 1) % 50 == 0:
            print(f"  [{config}] {n+1}/{len(queries)} AH={sum(acc['answer_hit@k'])/len(acc['answer_hit@k']):.3f}", flush=True)

    hits = [int(x) for x in acc["answer_hit@k"]]
    return {
        "config": config,
        "beta": beta,
        "n": len(queries),
        "session_hit@k": float(sum(acc["session_hit@k"]) / len(acc["session_hit@k"])) if acc["session_hit@k"] else 0.0,
        "answer_hit@k": float(sum(acc["answer_hit@k"]) / len(acc["answer_hit@k"])) if acc["answer_hit@k"] else 0.0,
        "episode_completeness@k": float(sum(acc["episode_completeness@k"]) / len(acc["episode_completeness@k"])) if acc["episode_completeness@k"] else 0.0,
        "ndcg@k": float(sum(acc["ndcg@k"]) / len(acc["ndcg@k"])) if acc["ndcg@k"] else 0.0,
        "mrr@k": float(sum(acc["mrr@k"]) / len(acc["mrr@k"])) if acc["mrr@k"] else 0.0,
        "ah_ci": bootstrap_ci(hits),
        "per_query_ah": hits,
        "elapsed_seconds": round(time.time() - t0, 1),
        "semantic_path": cfg_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tier", default="")
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--para_cache", default="")
    ap.add_argument("--force_queries_json", action="store_true")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--out_dir", default=str(REPO / "results" / "bims_legal_semantic_ablation"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tier = args.tier or man.get("tier", "X")
    sessions = man["sessions"]
    n_sid_fix = uniquify_session_ids(sessions)
    if n_sid_fix:
        print(f"[fix] uniquified {n_sid_fix} duplicate session_id occurrences", flush=True)

    queries = resolve_queries(man, args)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    by_ch: Dict[str, List[dict]] = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    print(f"[index] turn store sessions={len(sessions)} turns={sum(len(s['turns']) for s in sessions)}", flush=True)
    mgr_turn, meta_turn = index_turn_store(sessions)
    full_kg = dict(mgr_turn.knowledge_graph)
    out_dir = Path(args.out_dir) / Path(args.manifest).parent.name / f"tier_{tier}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": "LegalMem-MT-v3-semantic-ablation",
        "tier": tier,
        "manifest": str(Path(args.manifest)),
        "n_sessions": man["n_sessions"],
        "n_gold": man["n_gold"],
        "n_distractor": man["n_distractor"],
        "n_turns": man["n_turns"],
        "semantic_ablation_note": "Summary retrieval disabled in all configs; BIRCH cluster retrieval toggled on/off and with/without associative expansion.",
        "knowledge_nodes_indexed": len(full_kg),
        "channels": {},
        "comparisons": {},
    }

    for ch, qs in by_ch.items():
        print(f"=== channel {ch} n={len(qs)} ===", flush=True)
        payload["channels"][ch] = {"configs": {}}
        for cfg in args.configs:
            print(f"[{ch}/{cfg}] start", flush=True)
            res = eval_config(mgr_turn, meta_turn, qs, cfg, args.top_k, beta=args.beta, full_kg=full_kg)
            payload["channels"][ch]["configs"][cfg] = res
            (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{ch}/{cfg}] AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f} "
                f"nDCG={res['ndcg@k']:.3f}",
                flush=True,
            )

        cfgs = payload["channels"][ch]["configs"]
        comps = {}
        for a, b, key in [
            ("ep_o2", "ep_flat", "ep_o2_vs_ep_flat"),
            ("birch1_flat", "ep_flat", "birch1_flat_vs_ep_flat"),
            ("birch1_o2", "ep_o2", "birch1_o2_vs_ep_o2"),
            ("birch2_flat", "birch1_flat", "birch2_flat_vs_birch1_flat"),
            ("birch2_o2", "birch1_o2", "birch2_o2_vs_birch1_o2"),
            ("birch2_o2", "ep_o2", "birch2_o2_vs_ep_o2"),
        ]:
            if a in cfgs and b in cfgs:
                comps[key] = paired_report(
                    a, cfgs[a]["per_query_ah"],
                    b, cfgs[b]["per_query_ah"],
                )
        payload["comparisons"][ch] = comps

    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {(out_dir / 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
