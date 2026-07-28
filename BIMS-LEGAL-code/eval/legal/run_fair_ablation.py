#!/usr/bin/env python3
"""Fair Soft-O2 re-run + dense Joint Q+A + β sweep + optional CAIL multi-turn.

Uses soft β inheritance only (no hard completeness). Writes under
results/legal_revision_fair/ so prior JSON is not resumed incorrectly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval" / "legal"))

from memory_manager import VectorMemoryManager  # noqa: E402
from prepare_legal_datasets import load_pairs  # noqa: E402
from prepare_legal_hard import paraphrase_query  # noqa: E402
from run_legal_scaled import build_corpus  # noqa: E402
from run_revision_protocol import (  # noqa: E402
    build_joint_qa_store,
    build_queries,
    metrics_at_k,
    make_followup,
    parent_hydrate,
    session_max_expand,
    shuffle_session_ids,
)

BETA_GRID = [0.5, 0.7, 0.9, 0.95, 0.98, 1.0]
CORE_CONFIGS = [
    "dense_flat", "dense_o2", "parent_hydrate", "session_max",
    "shuffled_o2", "joint_qa",
]


def apply_config(mgr, name: str, beta: float = 0.98):
    mgr.lru_cache.clear()
    mgr._retrieval_augment = None
    mgr._query_projection = None
    mgr._exact_match_boost = 0.0
    mgr._session_first_rerank = False
    if name == "dense_flat":
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    elif name in ("dense_o2", "shuffled_o2", "beta_o2"):
        mgr.vector_store.use_pq = False
        mgr._session_expand = True
        mgr._session_coherence = float(beta)
        mgr._session_first_rerank = True
    elif name in ("parent_hydrate", "session_max", "joint_qa"):
        mgr.vector_store.use_pq = False
        mgr._session_expand = False
    else:
        raise ValueError(name)


def evaluate_cfg(mgr, sessions, queries, config, top_k, beta=0.98):
    apply_config(mgr, "dense_o2" if config == "shuffled_o2" else config, beta=beta)
    if config == "shuffled_o2":
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
                f"  [{config} β={beta}] {n+1}/{len(queries)} "
                f"AH={np.mean(acc['answer_hit@k']):.3f}",
                flush=True,
            )
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    out.update({
        "n": len(queries), "elapsed_seconds": round(time.time() - t0, 1),
        "config": config, "beta": beta,
    })
    return out


def build_cail_corpus(mgr, dialogs: List[dict]) -> List[dict]:
    """Index multi-turn CAIL dialogs; evidence = all assistant turns."""
    mgr.reset(use_pq=False)
    mgr.vector_store.use_pq = False
    mgr._bulk_load = True
    try:
        mgr.lru_cache.capacity = max(getattr(mgr.lru_cache, "capacity", 500), len(dialogs) * 20 + 500)
    except Exception:
        pass
    texts = []
    for d in dialogs:
        for t in d["turns"]:
            texts.append(t["content"])
    try:
        mgr.warm_embed_cache(texts)
    except Exception:
        pass
    sessions = []
    for i, d in enumerate(dialogs):
        sid = d.get("id", f"cail_{i}")
        tids_all, tids_ans = [], []
        for t in d["turns"]:
            tid = mgr.add_dialog(t["role"], t["content"], session_id=sid)
            tids_all.append(tid)
            if t["role"] == "assistant":
                tids_ans.append(tid)
        sessions.append({
            "question": d["question"],
            "answer": d["answer"],
            "query_user": d.get("query_user", d["question"]),
            "tid_user": tids_all[0],
            "tid_assistant": tids_ans[-1] if tids_ans else tids_all[-1],
            "evidence_session_tids": tids_all,
            "evidence_answer_tids": tids_ans or [tids_all[-1]],
        })
    mgr.finalize_bulk_load()
    mgr._bulk_load = False
    return sessions


def build_cail_queries(sessions, protocol: str, seed: int):
    """exact=first user; followup=last user; paraphrase=rewrite first user."""
    import random
    rng = random.Random(seed + hash(protocol) % 997)
    out = []
    for qi, s in enumerate(sessions):
        if protocol == "exact":
            q = s["question"]
        elif protocol == "followup":
            q = s.get("query_user") or make_followup(s["question"], rng)
        else:
            q = paraphrase_query(s["question"], rng)
            if q.strip() == s["question"].strip():
                q = "请问" + s["question"] + "，法律上如何处理？"
        out.append((qi, q))
    return out


def run_standard_dataset(dataset_key, n_corpus, n_queries, top_k, seed, protocols, out_root):
    rng = np.random.default_rng(seed)
    pairs = load_pairs(dataset_key)
    rng.shuffle(pairs)
    n_corpus = min(n_corpus, len(pairs))
    q_idx = list(range(min(n_corpus, n_queries)))
    out_dir = Path(out_root) / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "fair_protocol.json"
    payload = {
        "dataset_key": dataset_key, "n_corpus": n_corpus, "n_queries": len(q_idx),
        "top_k": top_k, "seed": seed, "protocols": {},
        "soft_o2_definition": "expand+rerank+completeness all use β·s (no hard score copy)",
        "generated_at": datetime.now().isoformat(),
    }
    if result_path.is_file():
        try:
            prev = json.loads(result_path.read_text(encoding="utf-8"))
            if prev.get("seed") == seed and prev.get("n_corpus") == n_corpus:
                payload = prev
                print(f"[{dataset_key}] resume {result_path}", flush=True)
        except Exception:
            pass

    print(f"[{dataset_key}] build FlatIP M={n_corpus}", flush=True)
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    sessions = build_corpus(mgr, pairs, n_corpus)

    print(f"[{dataset_key}] build joint Q+A", flush=True)
    mgr_joint, sessions_joint = build_joint_qa_store(pairs, n_corpus)

    for protocol in protocols:
        payload["protocols"].setdefault(protocol, {"configs": {}})
        queries = build_queries(sessions, q_idx, protocol, seed)
        for cfg in CORE_CONFIGS:
            if cfg in payload["protocols"][protocol]["configs"]:
                print(f"[{dataset_key}/{protocol}/{cfg}] skip", flush=True)
                continue
            print(f"[{dataset_key}/{protocol}/{cfg}] start", flush=True)
            if cfg == "joint_qa":
                res = evaluate_cfg(mgr_joint, sessions_joint, queries, cfg, top_k)
            elif cfg == "shuffled_o2":
                orig_map = dict(mgr._tid_to_session)
                orig_mem = {k: list(v) for k, v in mgr._session_members.items()}
                res = evaluate_cfg(mgr, sessions, queries, cfg, top_k)
                mgr._tid_to_session = orig_map
                mgr._session_members = orig_mem
            else:
                res = evaluate_cfg(mgr, sessions, queries, cfg, top_k)
            payload["protocols"][protocol]["configs"][cfg] = res
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{dataset_key}/{protocol}/{cfg}] "
                f"AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f}",
                flush=True,
            )
    return payload, mgr, sessions, q_idx


def run_beta_sweep(mgr, sessions, q_idx, seed, top_k, out_root, dataset_key):
    out_path = Path(out_root) / dataset_key / "beta_sweep.json"
    payload = {"dataset_key": dataset_key, "protocol": "paraphrase", "betas": {},
               "generated_at": datetime.now().isoformat()}
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    queries = build_queries(sessions, q_idx, "paraphrase", seed)
    for beta in BETA_GRID:
        key = f"{beta:.2f}"
        if key in payload["betas"]:
            print(f"[{dataset_key}/beta={key}] skip", flush=True)
            continue
        print(f"[{dataset_key}/beta={key}] start", flush=True)
        orig_map = dict(mgr._tid_to_session)
        orig_mem = {k: list(v) for k, v in mgr._session_members.items()}
        res = evaluate_cfg(mgr, sessions, queries, "dense_o2", top_k, beta=beta)
        mgr._tid_to_session = orig_map
        mgr._session_members = orig_mem
        payload["betas"][key] = res
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dataset_key}/beta={key}] AH={res['answer_hit@k']:.3f}", flush=True)
    return payload


def run_cail(split: str, prepared_path: Path, top_k: int, seed: int, protocols, out_root):
    dialogs = json.loads(prepared_path.read_text(encoding="utf-8"))
    out_dir = Path(out_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "fair_protocol.json"
    payload = {
        "dataset_key": split, "n_corpus": len(dialogs), "n_queries": len(dialogs),
        "top_k": top_k, "seed": seed, "protocols": {},
        "generated_at": datetime.now().isoformat(),
    }
    if result_path.is_file():
        try:
            prev = json.loads(result_path.read_text(encoding="utf-8"))
            if prev.get("n_corpus") == len(dialogs):
                payload = prev
        except Exception:
            pass

    print(f"[{split}] build multi-turn corpus n={len(dialogs)}", flush=True)
    mgr = VectorMemoryManager()
    sessions = build_cail_corpus(mgr, dialogs)

    # joint: concatenate full dialogue text
    pairs = [("\n".join(t["content"] for t in d["turns"] if t["role"] == "user"),
              "\n".join(t["content"] for t in d["turns"] if t["role"] == "assistant"))
             for d in dialogs]
    mgr_joint, sessions_joint = build_joint_qa_store(pairs, len(pairs))

    for protocol in protocols:
        payload["protocols"].setdefault(protocol, {"configs": {}})
        queries = build_cail_queries(sessions, protocol, seed)
        for cfg in CORE_CONFIGS:
            if cfg in payload["protocols"][protocol]["configs"]:
                print(f"[{split}/{protocol}/{cfg}] skip", flush=True)
                continue
            print(f"[{split}/{protocol}/{cfg}] start", flush=True)
            if cfg == "joint_qa":
                res = evaluate_cfg(mgr_joint, sessions_joint, queries, cfg, top_k)
            elif cfg == "shuffled_o2":
                orig_map = dict(mgr._tid_to_session)
                orig_mem = {k: list(v) for k, v in mgr._session_members.items()}
                res = evaluate_cfg(mgr, sessions, queries, cfg, top_k)
                mgr._tid_to_session = orig_map
                mgr._session_members = orig_mem
            else:
                res = evaluate_cfg(mgr, sessions, queries, cfg, top_k)
            payload["protocols"][protocol]["configs"][cfg] = res
            result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{split}/{protocol}/{cfg}] "
                f"AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f}",
                flush=True,
            )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "cail", "all"], default="all")
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"])
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocols", nargs="+", default=["exact", "paraphrase", "followup"])
    ap.add_argument("--output_root", default=str(REPO_ROOT / "results" / "legal_revision_fair"))
    ap.add_argument("--skip_beta", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    if args.mode in ("standard", "all"):
        for ds in args.datasets:
            payload, mgr, sessions, q_idx = run_standard_dataset(
                ds, args.n_corpus, args.n_queries, args.top_k, args.seed,
                args.protocols, args.output_root,
            )
            if not args.skip_beta:
                run_beta_sweep(mgr, sessions, q_idx, args.seed, args.top_k, args.output_root, ds)

    if args.mode in ("cail", "all"):
        # prepare if needed
        prep = REPO_ROOT / "eval" / "legal" / "prepare_cail2024.py"
        os.system(f"{sys.executable} {prep}")
        cail_dir = REPO_ROOT / "data" / "legal" / "cail2024"
        for split, fname in [("cail_prelim", "cail_prelim.json"), ("cail_final", "cail_final.json")]:
            run_cail(
                split, cail_dir / fname, args.top_k, args.seed,
                args.protocols, args.output_root,
            )


if __name__ == "__main__":
    main()
