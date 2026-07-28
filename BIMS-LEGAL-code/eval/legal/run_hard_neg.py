#!/usr/bin/env python3
"""困难负样本 / 鲁棒性分层（修改意见补充六）。

在固定 M=400 FlatIP 语料上，按干扰强度分层 paraphrase 查询：
  - near_dup：近重复问题（同语料高字面重叠 distractors 存在时）
  - same_prefix：问题前缀相近（同主题簇）
  - cross_topic：与目标主题字面重叠低
  - wrong_sibling：O2 负对照——随机 sibling 映射
并报告 Answer Hit / EC / nDCG。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "eval"), os.path.join(REPO_ROOT, "eval", "legal")):
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE  # noqa: E402
from prepare_legal_datasets import load_pairs, DATASET_FILES  # noqa: E402
from prepare_legal_hard import paraphrase_query  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402
from run_revision_protocol import apply_config, metrics_at_k, shuffle_session_ids  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def stratify_queries(sessions, seed: int):
    """Assign each query index a hardness bucket based on nearest distractor overlap."""
    rng = np.random.default_rng(seed)
    qs = [s["question"] for s in sessions]
    buckets = {"near_dup": [], "same_prefix": [], "cross_topic": []}
    for i, q in enumerate(qs):
        others = [j for j in range(len(qs)) if j != i]
        sims = [(char_jaccard(q, qs[j]), j) for j in others]
        sims.sort(reverse=True)
        best = sims[0][0] if sims else 0.0
        pref = q[:8]
        same_pref = any(qs[j].startswith(pref) for j in others)
        if best >= 0.55:
            buckets["near_dup"].append(i)
        elif same_pref or best >= 0.35:
            buckets["same_prefix"].append(i)
        else:
            buckets["cross_topic"].append(i)
    # ensure non-empty by fallback fill
    for name in list(buckets):
        if len(buckets[name]) < 20:
            need = 20 - len(buckets[name])
            pool = [i for i in range(len(qs)) if i not in buckets[name]]
            rng.shuffle(pool)
            buckets[name].extend(pool[:need])
    return buckets


def eval_bucket(mgr, sessions, indices, config, top_k, seed, paraphrase=True):
    import random
    rng = random.Random(seed)
    apply_config(mgr, config)
    acc = defaultdict(list)
    for qi in indices:
        q = sessions[qi]["question"]
        qtext = paraphrase_query(q, rng) if paraphrase else q
        if paraphrase and qtext.strip() == q.strip():
            qtext = "请问" + q + "，法律上如何处理？"
        retrieved = mgr.search(qtext, top_k=top_k, is_temporal_task=False)[:top_k]
        m = metrics_at_k(
            retrieved,
            sessions[qi]["evidence_session_tids"],
            sessions[qi]["evidence_answer_tids"],
            top_k,
        )
        for k, v in m.items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()} | {"n": len(indices)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"])
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output_root",
        default=os.path.join(REPO_ROOT, "results", "legal_revision"),
    )
    args = ap.parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    summary = {"generated_at": datetime.now().isoformat(), "config": vars(args), "datasets": {}}

    for ds in args.datasets:
        if ds not in DATASET_FILES:
            continue
        rng = np.random.default_rng(args.seed)
        pairs = load_pairs(ds)
        rng.shuffle(pairs)
        n = min(args.n_corpus, len(pairs))
        print(f"[{ds}] building FlatIP M={n}", flush=True)
        mgr = VectorMemoryManager()
        mgr.vector_store.use_pq = False
        sessions = build_corpus(mgr, pairs, n)
        buckets = stratify_queries(sessions, args.seed)
        out = {"buckets": {k: len(v) for k, v in buckets.items()}, "results": {}}
        for bucket, idxs in buckets.items():
            print(f"[{ds}] bucket={bucket} n={len(idxs)}", flush=True)
            out["results"][bucket] = {
                "dense_flat": eval_bucket(mgr, sessions, idxs, "dense_flat", args.top_k, args.seed),
                "dense_o2": eval_bucket(mgr, sessions, idxs, "dense_o2", args.top_k, args.seed),
            }
        # wrong sibling negative control
        orig_map = dict(mgr._tid_to_session)
        orig_mem = {k: list(v) for k, v in mgr._session_members.items()}
        shuffle_session_ids(mgr, seed=99)
        all_idx = list(range(n))
        out["results"]["wrong_sibling_o2"] = {
            "shuffled_o2": eval_bucket(mgr, sessions, all_idx, "dense_o2", args.top_k, args.seed),
        }
        mgr._tid_to_session = orig_map
        mgr._session_members = orig_mem

        path = os.path.join(args.output_root, ds, "hard_neg.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        summary["datasets"][ds] = out
        print(f"[{ds}] wrote {path}", flush=True)

    with open(os.path.join(args.output_root, "hard_neg_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE hard-neg")


if __name__ == "__main__":
    main()
