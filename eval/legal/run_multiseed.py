#!/usr/bin/env python3
"""多随机种子 + 配对显著性（修改意见补充四）。

对 paraphrase 主协议，在 seeds∈{42,43,44,45,46} 上比较 dense_flat vs dense_o2；
报告均值±std、95% CI，并对 Answer Hit 做 McNemar、对 nDCG 做 paired bootstrap。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "eval"), os.path.join(REPO_ROOT, "eval", "legal")):
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE  # noqa: E402
from prepare_legal_datasets import load_pairs, DATASET_FILES  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402
from run_revision_protocol import build_queries, evaluate  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b01: base hit / o2 miss, b10: base miss / o2 hit)."""
    n = b01 + b10
    if n == 0:
        return 1.0
    # exact binomial two-sided
    from math import comb
    p = 0.0
    for k in range(0, n + 1):
        pk = comb(n, k) * (0.5 ** n)
        if pk <= comb(n, min(b10, b01)) * (0.5 ** n) + 1e-18:
            p += pk
    # simpler: sum of tails for observed min
    obs = min(b01, b10)
    p = sum(comb(n, k) for k in range(0, obs + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = []
    n = len(diff)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(float(np.mean(diff[idx])))
    means = np.asarray(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.mean(diff))


def per_query_flags(mgr, sessions, queries, config, top_k):
    """Return list of (answer_hit, ndcg) per query."""
    from run_revision_protocol import apply_config, metrics_at_k
    apply_config(mgr, config)
    hits, ndcgs = [], []
    for qi, qtext in queries:
        s = sessions[qi]
        retrieved = mgr.search(qtext, top_k=top_k, is_temporal_task=False)[:top_k]
        m = metrics_at_k(
            retrieved, s["evidence_session_tids"], s["evidence_answer_tids"], top_k,
        )
        hits.append(int(m["answer_hit@k"]))
        ndcgs.append(float(m["ndcg@k"]))
    return np.asarray(hits), np.asarray(ndcgs)


def run_one_seed(dataset_key, n_corpus, n_queries, top_k, seed):
    rng = np.random.default_rng(seed)
    pairs = load_pairs(dataset_key)
    rng.shuffle(pairs)
    n_corpus = min(n_corpus, len(pairs))
    q_idx = list(range(min(n_corpus, n_queries)))
    print(f"[{dataset_key}] seed={seed} building FlatIP M={n_corpus}", flush=True)
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    sessions = build_corpus(mgr, pairs, n_corpus)
    queries = build_queries(sessions, q_idx, "paraphrase", seed)

    base_hit, base_ndcg = per_query_flags(mgr, sessions, queries, "dense_flat", top_k)
    o2_hit, o2_ndcg = per_query_flags(mgr, sessions, queries, "dense_o2", top_k)

    # McNemar counts
    b01 = int(np.sum((base_hit == 1) & (o2_hit == 0)))
    b10 = int(np.sum((base_hit == 0) & (o2_hit == 1)))
    p_mc = mcnemar_exact(b01, b10)
    lo, hi, mean_d = paired_bootstrap_ci(o2_ndcg - base_ndcg, seed=seed)

    return {
        "seed": seed,
        "n": len(queries),
        "dense_flat": {
            "answer_hit@k": float(np.mean(base_hit)),
            "ndcg@k": float(np.mean(base_ndcg)),
        },
        "dense_o2": {
            "answer_hit@k": float(np.mean(o2_hit)),
            "ndcg@k": float(np.mean(o2_ndcg)),
        },
        "delta_answer_hit": float(np.mean(o2_hit) - np.mean(base_hit)),
        "delta_ndcg": float(mean_d),
        "mcnemar": {"b01": b01, "b10": b10, "p_two_sided": p_mc},
        "ndcg_delta_ci95": [lo, hi],
    }


def summarize(seed_rows):
    ah_b = [r["dense_flat"]["answer_hit@k"] for r in seed_rows]
    ah_o = [r["dense_o2"]["answer_hit@k"] for r in seed_rows]
    nd_b = [r["dense_flat"]["ndcg@k"] for r in seed_rows]
    nd_o = [r["dense_o2"]["ndcg@k"] for r in seed_rows]

    def ms(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "ci95": [
                float(np.mean(a) - 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float(np.mean(a)),
                float(np.mean(a) + 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float(np.mean(a)),
            ],
        }

    return {
        "dense_flat_answer_hit": ms(ah_b),
        "dense_o2_answer_hit": ms(ah_o),
        "dense_flat_ndcg": ms(nd_b),
        "dense_o2_ndcg": ms(nd_o),
        "mean_delta_answer_hit": float(np.mean(ah_o) - np.mean(ah_b)),
        "mean_delta_ndcg": float(np.mean(nd_o) - np.mean(nd_b)),
        "seeds": seed_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"])
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument(
        "--output_root",
        default=os.path.join(REPO_ROOT, "results", "legal_revision"),
    )
    args = ap.parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    all_out = {"generated_at": datetime.now().isoformat(), "config": vars(args), "datasets": {}}
    for ds in args.datasets:
        if ds not in DATASET_FILES:
            continue
        rows = []
        path = os.path.join(args.output_root, ds, "multiseed.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prev = {}
        if os.path.isfile(path):
            try:
                prev = json.load(open(path, encoding="utf-8"))
            except Exception:
                prev = {}
        done = {r["seed"]: r for r in prev.get("seeds", [])}
        for seed in args.seeds:
            if seed in done:
                print(f"[{ds}] seed={seed} skip", flush=True)
                rows.append(done[seed])
                continue
            row = run_one_seed(ds, args.n_corpus, args.n_queries, args.top_k, seed)
            rows.append(row)
            summary = summarize(rows)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(
                f"[{ds}] seed={seed} "
                f"flat AH={row['dense_flat']['answer_hit@k']:.3f} "
                f"o2 AH={row['dense_o2']['answer_hit@k']:.3f} "
                f"McNemar p={row['mcnemar']['p_two_sided']:.4g}",
                flush=True,
            )
        all_out["datasets"][ds] = summarize(rows)
    with open(os.path.join(args.output_root, "multiseed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, ensure_ascii=False, indent=2)
    print("DONE multiseed", args.output_root)


if __name__ == "__main__":
    main()
