#!/usr/bin/env python3
"""规模/效率曲线（修改意见补充五）：M∈{100,400,1600} × FlatIP/IVFPQ/O2。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "eval"), os.path.join(REPO_ROOT, "eval", "legal")):
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE  # noqa: E402
from prepare_legal_datasets import load_pairs  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402
from run_revision_protocol import metrics_at_k  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))


def run_one(pairs, M, Q, top_k, mode, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    sel = [pairs[int(i)] for i in idx[:M]]
    q_idx = list(range(min(M, Q)))

    t_build0 = time.time()
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = (mode == "ivfpq")
    sessions = build_corpus(mgr, sel, M)
    build_s = time.time() - t_build0

    if mode == "o2":
        mgr._session_expand = True
        mgr._session_coherence = 0.98
        mgr._session_first_rerank = True
    else:
        mgr._session_expand = False

    lat = []
    mets = []
    for qi in q_idx:
        s = sessions[qi]
        t0 = time.time()
        retrieved = mgr.search(s["question"], top_k=top_k, is_temporal_task=False)
        lat.append(time.time() - t0)
        mets.append(metrics_at_k(
            retrieved, s["evidence_session_tids"], s["evidence_answer_tids"], top_k,
        ))
    lat = np.asarray(lat, dtype=np.float64)
    return {
        "M": M, "Q": len(q_idx), "mode": mode,
        "build_seconds": round(build_s, 2),
        "build_sec_per_turn": round(build_s / max(2 * M, 1), 4),
        "latency_p50": float(np.percentile(lat, 50)),
        "latency_p95": float(np.percentile(lat, 95)),
        "latency_mean": float(np.mean(lat)),
        "answer_hit@k": float(np.mean([m["answer_hit@k"] for m in mets])),
        "episode_completeness@k": float(np.mean([m["episode_completeness@k"] for m in mets])),
        "ndcg@k": float(np.mean([m["ndcg@k"] for m in mets])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="disc_law")
    ap.add_argument("--Ms", nargs="+", type=int, default=[100, 400, 1600, 6400])
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", nargs="+", default=["ivfpq", "flat", "o2"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    pairs = load_pairs(args.dataset)
    rows = []
    for M in args.Ms:
        if 2 * M > len(pairs) * 2 and M > len(pairs):
            print(f"skip M={M}: only {len(pairs)} pairs", flush=True)
            continue
        M_eff = min(M, len(pairs))
        for mode in args.modes:
            for r in range(args.repeats):
                print(f"=== M={M_eff} mode={mode} rep={r+1} ===", flush=True)
                row = run_one(pairs, M_eff, args.n_queries, args.top_k, mode, args.seed + r)
                row["repeat"] = r
                rows.append(row)
                print(json.dumps(row), flush=True)

    out = args.output or os.path.join(
        REPO_ROOT, "results", "legal_revision", f"scale_curve_{args.dataset}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {"generated_at": datetime.now().isoformat(), "config": vars(args), "rows": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
