#!/usr/bin/env python3
"""投稿前修订实验：独立查询协议 + 标准基线对照 + 拆分指标。

协议（修改意见补充一/二）：
  - exact：诊断对照（原文 question，不应作为主结论）
  - paraphrase：规则改写 query（主协议）
  - followup：指代式追问（主协议子集）

系统配置：
  - baseline_pq：IVFPQ dense
  - dense_flat：O1 FlatIP
  - dense_o2：O1+O2 session expand（论文主方法）
  - parent_hydrate：命中 child 后无条件返回同 session 全部 turns（对照）
  - joint_qa：把 Q+A 拼成单文档索引（对照）
  - session_max：按 session 聚合 max score 再展开
  - shuffled_o2：随机打乱 session ID 后仍开 O2（负对照）

指标（修改意见问题三）：
  SessionHit@k / AnswerHit@k / EpisodeCompleteness@k / nDCG@k / MRR@k
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
LEGAL_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, EVAL_DIR, LEGAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE  # noqa: E402
from prepare_legal_datasets import load_pairs, DATASET_FILES  # noqa: E402
from prepare_legal_hard import paraphrase_query  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))

CONFIGS = [
    "baseline_pq",
    "dense_flat",
    "dense_o2",
    "parent_hydrate",
    "joint_qa",
    "session_max",
    "shuffled_o2",
]


def make_followup(q: str, rng: random.Random) -> str:
    templates = [
        "刚才那个问题，律师最后建议怎么处理？",
        "针对上述情形，法律上的结论是什么？",
        "这个问题里，对方应当承担什么责任？",
        "前面咨询的案件，有哪些关键法条依据？",
        "关于「{core}」这一点，请补充律师意见。",
    ]
    core = q.strip()[:24] if len(q) > 24 else q.strip()
    t = rng.choice(templates)
    return t.format(core=core) if "{core}" in t else t


def build_queries(
    sessions: List[dict],
    q_idx: Sequence[int],
    protocol: str,
    seed: int,
) -> List[Tuple[int, str]]:
    rng = random.Random(seed + hash(protocol) % 997)
    out = []
    for qi in q_idx:
        q = sessions[qi]["question"]
        if protocol == "exact":
            qq = q
        elif protocol == "paraphrase":
            qq = paraphrase_query(q, rng)
            # 避免改写后仍与原文完全相同
            if qq.strip() == q.strip():
                qq = "请问" + q + "，法律上如何处理？"
        elif protocol == "followup":
            qq = make_followup(q, rng)
        else:
            raise ValueError(protocol)
        out.append((qi, qq))
    return out


def _ids(retrieved) -> List[str]:
    return [str(r.get("tid", "")) for r in retrieved if r.get("tid")]


def metrics_at_k(retrieved, gold_sess_tids, gold_ans_tids, k: int) -> dict:
    top = retrieved[:k]
    tids = set(_ids(top))
    g_s = set(gold_sess_tids)
    g_a = set(gold_ans_tids)
    sess_hit = 1.0 if (tids & g_s) else 0.0
    ans_hit = 1.0 if (tids & g_a) else 0.0
    ep_comp = len(tids & g_s) / len(g_s) if g_s else 0.0
    # nDCG over answer relevance (binary on gold answer tids)
    rel = [1.0 if t in g_a else (0.5 if t in g_s else 0.0) for t in _ids(top)]
    if rel:
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
        ideal = sorted(rel, reverse=True)
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal)) or 1.0
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    mrr = 0.0
    for i, t in enumerate(_ids(top)):
        if t in g_a or t in g_s:
            mrr = 1.0 / (i + 1)
            break
    return {
        "session_hit@k": sess_hit,
        "answer_hit@k": ans_hit,
        "episode_completeness@k": ep_comp,
        "ndcg@k": float(ndcg),
        "mrr@k": float(mrr),
    }


def parent_hydrate(mgr, retrieved, top_k: int):
    """Hard parent hydration: child hit → insert all siblings with the trigger's score (hard copy)."""
    out, seen = [], set()
    for r in retrieved:
        tid = r.get("tid")
        sid = mgr._tid_to_session.get(tid) if hasattr(mgr, "_tid_to_session") else None
        members = mgr._session_members.get(sid, [tid]) if sid else [tid]
        for m in members:
            if m in seen:
                continue
            seen.add(m)
            node = mgr.para_tree.get(m)
            if node is None:
                continue
            out.append({
                "tid": m, "text": node.text, "full_text": node.text,
                "type": "paragraph",
                "score": float(r.get("score", 0) or 0),
                "final_score": float(r.get("final_score", r.get("score", 0)) or 0),
            })
            if len(out) >= top_k:
                return out
    return out[:top_k]


def session_max_expand(mgr, retrieved, top_k: int):
    """Aggregate by session max score, then expand members in score order."""
    best = {}
    for r in retrieved:
        tid = r.get("tid")
        sid = mgr._tid_to_session.get(tid, tid)
        sc = float(r.get("final_score", r.get("score", 0)) or 0)
        if sid not in best or sc > best[sid][0]:
            best[sid] = (sc, tid)
    ranked = sorted(best.items(), key=lambda x: -x[1][0])
    out, seen = [], set()
    for sid, (sc, _) in ranked:
        for m in mgr._session_members.get(sid, []):
            if m in seen:
                continue
            seen.add(m)
            node = mgr.para_tree.get(m)
            if node is None:
                continue
            out.append({
                "tid": m, "text": node.text, "full_text": node.text,
                "type": "paragraph", "score": sc, "final_score": sc,
            })
            if len(out) >= top_k:
                return out
    return out[:top_k]


def build_joint_qa_store(pairs, n_corpus: int) -> Tuple[VectorMemoryManager, List[dict]]:
    """Index joint Q+A documents; evidence tids point to joint docs."""
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    mgr.reset(use_pq=False)
    mgr._bulk_load = True
    try:
        mgr.lru_cache.capacity = max(getattr(mgr.lru_cache, "capacity", 500), n_corpus * 4 + 500)
    except Exception:
        pass
    texts = [f"咨询问题：{q}\n律师解答：{a}" for q, a in pairs[:n_corpus]]
    try:
        mgr.warm_embed_cache(texts)
    except Exception:
        pass
    sessions = []
    for k, (q, a) in enumerate(pairs[:n_corpus]):
        sid = f"joint_{k}"
        text = texts[k]
        tid = mgr.add_dialog("assistant", text, session_id=sid)
        sessions.append({
            "question": q, "answer": a,
            "tid_user": tid, "tid_assistant": tid,
            "evidence_session_tids": [tid],
            "evidence_answer_tids": [tid],
        })
    mgr.finalize_bulk_load()
    mgr._bulk_load = False
    return mgr, sessions


def shuffle_session_ids(mgr, seed: int = 7):
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


def apply_config(mgr, name: str):
    mgr.lru_cache.clear()
    mgr._retrieval_augment = None
    mgr._query_projection = None
    mgr._exact_match_boost = 0.0
    mgr._session_first_rerank = False
    if name == "baseline_pq":
        mgr.vector_store.use_pq = True
        mgr._session_expand = False
    elif name == "dense_flat":
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    elif name == "dense_o2":
        mgr.vector_store.use_pq = False
        mgr._session_expand = True
        mgr._session_coherence = 0.98
        mgr._session_first_rerank = True
    elif name in ("parent_hydrate", "session_max"):
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    elif name == "shuffled_o2":
        mgr.vector_store.use_pq = False
        mgr._session_expand = True
        mgr._session_coherence = 0.98
        mgr._session_first_rerank = True
    elif name == "joint_qa":
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    else:
        raise ValueError(name)


def evaluate(
    mgr,
    sessions: List[dict],
    queries: List[Tuple[int, str]],
    config: str,
    top_k: int,
) -> dict:
    apply_config(mgr, config if config != "shuffled_o2" else "dense_o2")
    if config == "shuffled_o2":
        # copy map then shuffle
        shuffle_session_ids(mgr, seed=7)

    acc = {k: [] for k in (
        "session_hit@k", "answer_hit@k", "episode_completeness@k", "ndcg@k", "mrr@k"
    )}
    t0 = time.time()
    for n, (qi, qtext) in enumerate(queries):
        s = sessions[qi]
        retrieved = mgr.search(qtext, top_k=max(top_k * 3, 30), is_temporal_task=False)
        if config == "parent_hydrate":
            retrieved = parent_hydrate(mgr, retrieved, top_k)
        elif config == "session_max":
            retrieved = session_max_expand(mgr, retrieved, top_k)
        else:
            retrieved = retrieved[:top_k]
        m = metrics_at_k(
            retrieved, s["evidence_session_tids"], s["evidence_answer_tids"], top_k,
        )
        for k, v in m.items():
            acc[k].append(v)
        if (n + 1) % 50 == 0:
            print(
                f"  [{config}] {n+1}/{len(queries)} "
                f"AH={np.mean(acc['answer_hit@k']):.3f} "
                f"SH={np.mean(acc['session_hit@k']):.3f} "
                f"EC={np.mean(acc['episode_completeness@k']):.3f}",
                flush=True,
            )
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    out["n"] = len(queries)
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    out["config"] = config
    return out


def run_dataset(
    dataset_key: str,
    *,
    n_corpus: int,
    n_queries: int,
    top_k: int,
    seed: int,
    protocols: List[str],
    configs: List[str],
    output_root: str,
):
    rng = np.random.default_rng(seed)
    pairs = load_pairs(dataset_key)
    rng.shuffle(pairs)
    n_corpus = min(n_corpus, len(pairs))
    q_idx = list(range(min(n_corpus, n_queries)))

    out_dir = os.path.join(output_root, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "revision_protocol.json")
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
                print(f"[{dataset_key}] resume from {result_path}", flush=True)
        except Exception:
            pass

    # shared turn-level corpus
    print(f"[{dataset_key}] building FlatIP corpus M={n_corpus}", flush=True)
    t0 = time.time()
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    sessions = build_corpus(mgr, pairs, n_corpus)
    print(f"[{dataset_key}] FlatIP build {time.time()-t0:.0f}s", flush=True)

    # PQ baseline corpus (separate)
    print(f"[{dataset_key}] building PQ corpus M={n_corpus}", flush=True)
    t0 = time.time()
    mgr_pq = VectorMemoryManager()
    mgr_pq.vector_store.use_pq = True
    sessions_pq = build_corpus(mgr_pq, pairs, n_corpus)
    print(f"[{dataset_key}] PQ build {time.time()-t0:.0f}s", flush=True)

    # joint QA corpus
    need_joint = "joint_qa" in configs
    mgr_joint, sessions_joint = (None, None)
    if need_joint:
        print(f"[{dataset_key}] building joint Q+A corpus", flush=True)
        mgr_joint, sessions_joint = build_joint_qa_store(pairs, n_corpus)

    for protocol in protocols:
        payload["protocols"].setdefault(protocol, {"configs": {}})
        queries = build_queries(sessions, q_idx, protocol, seed)
        # save a few sample queries for audit
        payload["protocols"][protocol]["sample_queries"] = [
            {"qi": qi, "query": qq[:120], "orig": sessions[qi]["question"][:120]}
            for qi, qq in queries[:5]
        ]
        for cfg in configs:
            if cfg in payload["protocols"][protocol]["configs"]:
                print(f"[{dataset_key}/{protocol}/{cfg}] skip (done)", flush=True)
                continue
            print(f"[{dataset_key}/{protocol}/{cfg}] start", flush=True)
            if cfg == "baseline_pq":
                res = evaluate(mgr_pq, sessions_pq, queries, cfg, top_k)
            elif cfg == "joint_qa":
                # rebuild query list indices against joint sessions (same order)
                jq = [(qi, qq) for qi, qq in queries]
                res = evaluate(mgr_joint, sessions_joint, jq, cfg, top_k)
            elif cfg == "shuffled_o2":
                # work on a shallow copy of expand flags; shuffle mutates session map
                # rebuild tid map from original after run by reloading members from para_tree
                # Safer: clone mapping
                orig_map = dict(mgr._tid_to_session)
                orig_mem = {k: list(v) for k, v in mgr._session_members.items()}
                res = evaluate(mgr, sessions, queries, cfg, top_k)
                mgr._tid_to_session = orig_map
                mgr._session_members = orig_mem
            else:
                res = evaluate(mgr, sessions, queries, cfg, top_k)
            payload["protocols"][protocol]["configs"][cfg] = res
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(
                f"[{dataset_key}/{protocol}/{cfg}] "
                f"AH={res['answer_hit@k']:.3f} SH={res['session_hit@k']:.3f} "
                f"EC={res['episode_completeness@k']:.3f} nDCG={res['ndcg@k']:.3f}",
                flush=True,
            )
    return payload


def main():
    ap = argparse.ArgumentParser(description="投稿前修订：查询协议 + 标准基线")
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"])
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocols", nargs="+", default=["exact", "paraphrase", "followup"])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument(
        "--output_root",
        default=os.path.join(REPO_ROOT, "results", "legal_revision"),
    )
    args = ap.parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    all_p = []
    for ds in args.datasets:
        if ds not in DATASET_FILES:
            print(f"skip unknown dataset {ds}")
            continue
        all_p.append(run_dataset(
            ds,
            n_corpus=args.n_corpus,
            n_queries=args.n_queries,
            top_k=args.top_k,
            seed=args.seed,
            protocols=args.protocols,
            configs=args.configs,
            output_root=args.output_root,
        ))
    summary = {
        "generated_at": datetime.now().isoformat(),
        "config": vars(args),
        "datasets": [
            {k: v for k, v in p.items() if k != "protocols"}
            | {"protocols": {
                prot: {
                    "configs": {
                        c: {kk: vv for kk, vv in cres.items()}
                        for c, cres in pdata.get("configs", {}).items()
                    }
                }
                for prot, pdata in p.get("protocols", {}).items()
            }}
            for p in all_p
        ],
    }
    with open(os.path.join(args.output_root, "revision_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE", args.output_root)


if __name__ == "__main__":
    main()
