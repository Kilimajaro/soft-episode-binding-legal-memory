#!/usr/bin/env python3
"""BM25 / dense+RRF 标准基线（修改意见补充二）。

在与 revision_protocol 相同的 M/S/协议上评估：
  - bm25_turn：按 turn 文本 BM25
  - bm25_joint：按 session Q+A 拼接文档 BM25，再展开为 turns
  - dense_rrf：FlatIP dense 与 BM25 的 RRF 融合
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "eval"), os.path.join(REPO_ROOT, "eval", "legal")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rank_bm25 import BM25Okapi  # noqa: E402

from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE  # noqa: E402
from prepare_legal_datasets import load_pairs, DATASET_FILES  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402
from run_revision_protocol import (  # noqa: E402
    build_queries,
    metrics_at_k,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))


def _tok(text: str) -> List[str]:
    # 中文按字 + 空白分词的轻量 tokenizer（无需额外分词模型）
    text = (text or "").strip().lower()
    chars = [c for c in text if not c.isspace()]
    return chars if chars else ["_"]


def build_bm25_indexes(sessions: List[dict]):
    turn_docs, turn_meta = [], []
    joint_docs, joint_meta = [], []
    for i, s in enumerate(sessions):
        q, a = s["question"], s["answer"]
        tid_u, tid_a = s["tid_user"], s["tid_assistant"]
        turn_docs.append(_tok(q))
        turn_meta.append({"tid": tid_u, "sid": i, "kind": "q"})
        turn_docs.append(_tok(a))
        turn_meta.append({"tid": tid_a, "sid": i, "kind": "a"})
        joint_docs.append(_tok(f"{q} {a}"))
        joint_meta.append({"sid": i, "tids": [tid_u, tid_a]})
    return (
        BM25Okapi(turn_docs), turn_meta,
        BM25Okapi(joint_docs), joint_meta,
    )


def bm25_search(bm25, meta, query: str, top_k: int) -> List[dict]:
    scores = bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in order:
        m = meta[int(idx)]
        out.append({
            "tid": m["tid"],
            "score": float(scores[int(idx)]),
            "final_score": float(scores[int(idx)]),
        })
    return out


def bm25_joint_search(bm25, meta, sessions, query: str, top_k: int) -> List[dict]:
    scores = bm25.get_scores(_tok(query))
    order = np.argsort(scores)[::-1]
    out, seen = [], set()
    for idx in order:
        for tid in meta[int(idx)]["tids"]:
            if tid in seen:
                continue
            seen.add(tid)
            out.append({
                "tid": tid,
                "score": float(scores[int(idx)]),
                "final_score": float(scores[int(idx)]),
            })
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


def evaluate_sparse(
    sessions: List[dict],
    queries: Sequence[Tuple[int, str]],
    *,
    mode: str,
    top_k: int,
    turn_bm25=None,
    turn_meta=None,
    joint_bm25=None,
    joint_meta=None,
    mgr: VectorMemoryManager | None = None,
) -> dict:
    acc = {k: [] for k in (
        "session_hit@k", "answer_hit@k", "episode_completeness@k", "ndcg@k", "mrr@k"
    )}
    t0 = time.time()
    for n, (qi, qtext) in enumerate(queries):
        s = sessions[qi]
        if mode == "bm25_turn":
            retrieved = bm25_search(turn_bm25, turn_meta, qtext, top_k)
        elif mode == "bm25_joint":
            retrieved = bm25_joint_search(joint_bm25, joint_meta, sessions, qtext, top_k)
        elif mode == "dense_rrf":
            dense = mgr.search(qtext, top_k=top_k * 3, is_temporal_task=False)[: top_k * 2]
            sparse = bm25_search(turn_bm25, turn_meta, qtext, top_k * 2)
            retrieved = rrf_fuse([dense, sparse], top_k)
        else:
            raise ValueError(mode)
        m = metrics_at_k(
            retrieved, s["evidence_session_tids"], s["evidence_answer_tids"], top_k,
        )
        for k, v in m.items():
            acc[k].append(v)
        if (n + 1) % 50 == 0:
            print(
                f"  [{mode}] {n+1}/{len(queries)} "
                f"AH={np.mean(acc['answer_hit@k']):.3f}",
                flush=True,
            )
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    out["n"] = len(queries)
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    out["config"] = mode
    return out


def run_dataset(dataset_key, n_corpus, n_queries, top_k, seed, protocols, modes, output_root):
    rng = np.random.default_rng(seed)
    pairs = load_pairs(dataset_key)
    rng.shuffle(pairs)
    n_corpus = min(n_corpus, len(pairs))
    q_idx = list(range(min(n_corpus, n_queries)))
    out_dir = os.path.join(output_root, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "bm25_rrf.json")
    payload = {
        "dataset_key": dataset_key,
        "n_corpus": n_corpus,
        "n_queries": len(q_idx),
        "top_k": top_k,
        "seed": seed,
        "protocols": {},
        "generated_at": datetime.now().isoformat(),
    }
    if os.path.isfile(result_path):
        try:
            prev = json.load(open(result_path, encoding="utf-8"))
            if prev.get("seed") == seed and prev.get("n_corpus") == n_corpus:
                payload = prev
        except Exception:
            pass

    need_dense = "dense_rrf" in modes
    mgr, sessions = None, None
    if need_dense:
        print(f"[{dataset_key}] building FlatIP for RRF", flush=True)
        mgr = VectorMemoryManager()
        mgr.vector_store.use_pq = False
        sessions = build_corpus(mgr, pairs, n_corpus)
    else:
        # lightweight sessions without full FAISS for pure BM25
        print(f"[{dataset_key}] building text sessions (no embed)", flush=True)
        sessions = []
        for k, (q, a) in enumerate(pairs[:n_corpus]):
            sessions.append({
                "question": q, "answer": a,
                "tid_user": f"u{k}", "tid_assistant": f"a{k}",
                "evidence_session_tids": [f"u{k}", f"a{k}"],
                "evidence_answer_tids": [f"a{k}"],
            })

    turn_bm25, turn_meta, joint_bm25, joint_meta = build_bm25_indexes(sessions)

    for protocol in protocols:
        payload["protocols"].setdefault(protocol, {"configs": {}})
        queries = build_queries(sessions, q_idx, protocol, seed)
        for mode in modes:
            if mode in payload["protocols"][protocol]["configs"]:
                print(f"[{dataset_key}/{protocol}/{mode}] skip", flush=True)
                continue
            print(f"[{dataset_key}/{protocol}/{mode}] start", flush=True)
            res = evaluate_sparse(
                sessions, queries, mode=mode, top_k=top_k,
                turn_bm25=turn_bm25, turn_meta=turn_meta,
                joint_bm25=joint_bm25, joint_meta=joint_meta, mgr=mgr,
            )
            payload["protocols"][protocol]["configs"][mode] = res
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(
                f"[{dataset_key}/{protocol}/{mode}] "
                f"AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f}",
                flush=True,
            )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"])
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocols", nargs="+", default=["exact", "paraphrase", "followup"])
    ap.add_argument("--modes", nargs="+", default=["bm25_turn", "bm25_joint", "dense_rrf"])
    ap.add_argument(
        "--output_root",
        default=os.path.join(REPO_ROOT, "results", "legal_revision"),
    )
    args = ap.parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    for ds in args.datasets:
        if ds not in DATASET_FILES:
            continue
        run_dataset(
            ds, args.n_corpus, args.n_queries, args.top_k, args.seed,
            args.protocols, args.modes, args.output_root,
        )
    print("DONE bm25/rrf", args.output_root)


if __name__ == "__main__":
    main()
