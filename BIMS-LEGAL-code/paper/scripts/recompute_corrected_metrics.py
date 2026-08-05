#!/usr/bin/env python3
"""Recompute corrected metrics (fixed-gold IDCG) for appendix grids.

Sets BIMS_DATA_ROOT BEFORE importing BIMS-LEGAL-code.
Supports --reuse_index when a previous run saved vectors + eval_meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "BIMS-LEGAL-code"


def _parse_early():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--embed_cache", type=Path, default=None)
    args, _ = ap.parse_known_args()
    return args


def _bootstrap_env():
    early = _parse_early()
    if early.workdir:
        early.workdir.mkdir(parents=True, exist_ok=True)
        (early.workdir / "vectors").mkdir(exist_ok=True)
        (early.workdir / "knowledge").mkdir(exist_ok=True)
        os.environ["BIMS_DATA_ROOT"] = str(early.workdir.resolve())
    if early.embed_cache:
        os.environ["EMBED_DISK_CACHE_DIR"] = str(early.embed_cache.resolve())
        os.environ["USE_EMBED_DISK_CACHE"] = "1"
    os.environ.setdefault("USE_EMBED_DISK_CACHE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


_bootstrap_env()

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "eval" / "legal"))
sys.path.insert(0, str(CODE / "eval" / "legal" / "v3"))

from config import _DATA_ROOT  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from run_legalmem_mt import (  # noqa: E402
    build_bm25_from_sessions,
    eval_config,
    index_joint_store,
    index_turn_store,
    resolve_queries,
    uniquify_session_ids,
)


METRIC_KEYS = (
    "session_hit@k",
    "answer_hit@k",
    "episode_completeness@k",
    "ndcg@k",
    "mrr@k",
    "failure_taxonomy",
    "ah_ci",
    "n",
)


def save_eval_state(workdir: Path, mgr: VectorMemoryManager, meta: dict) -> None:
    mgr._save_vector_db()
    state = {
        "meta": meta,
        "tid_to_session": dict(mgr._tid_to_session),
        "session_members": {k: list(v) for k, v in mgr._session_members.items()},
    }
    (workdir / "eval_meta.json").write_text(json.dumps(state), encoding="utf-8")
    print(
        f"[save] index ntotal={mgr.vector_store.ntotal()} meta_sessions={len(meta)}",
        flush=True,
    )


def load_eval_state(workdir: Path):
    state_path = workdir / "eval_meta.json"
    if not state_path.exists():
        return None, None
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    ntotal = mgr.vector_store.ntotal()
    if ntotal <= 0:
        print(f"[reuse] refuse empty index ntotal={ntotal}", flush=True)
        return None, None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mgr._tid_to_session = {str(k): v for k, v in state["tid_to_session"].items()}
    mgr._session_members = {k: list(v) for k, v in state["session_members"].items()}
    print(f"[reuse] ntotal={ntotal} sessions={len(state['meta'])}", flush=True)
    return mgr, state["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--embed_cache", type=Path, default=None)
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["dense_flat", "dense_o2", "parent_hydrate", "shuffled_o2"],
    )
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--skip_finalize", action="store_true")
    ap.add_argument("--reuse_index", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", action="store_true", help="Merge into existing --out JSON")
    args = ap.parse_args()

    print(f"[env] BIMS_DATA_ROOT={os.environ.get('BIMS_DATA_ROOT')}", flush=True)
    print(f"[env] config._DATA_ROOT={_DATA_ROOT}", flush=True)
    if Path(_DATA_ROOT).resolve() != args.workdir.resolve():
        raise SystemExit(f"FATAL data root mismatch {_DATA_ROOT} vs {args.workdir}")

    man = load_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sessions = man["sessions"]
    uniquify_session_ids(sessions)

    mgr = meta = None
    if args.reuse_index:
        mgr, meta = load_eval_state(args.workdir)

    if mgr is None:
        talk = args.workdir / "talk.txt"
        if talk.exists():
            talk.unlink()
        for sub in ("vectors", "knowledge"):
            p = args.workdir / sub
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
        if (args.workdir / "eval_meta.json").exists():
            (args.workdir / "eval_meta.json").unlink()
        print("[index] building turn store...", flush=True)
        mgr, meta = index_turn_store(sessions, skip_finalize=args.skip_finalize)
        print(f"[index] ntotal={mgr.vector_store.ntotal()} sessions={len(meta)}", flush=True)
        save_eval_state(args.workdir, mgr, meta)

    class Args:
        manifest = str(args.manifest)
        channels = args.channels
        force_queries_json = True
        para_cache = ""
        seed = 42

    queries = resolve_queries(man, Args())
    by_ch = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    out = {"manifest": str(args.manifest), "workdir": str(args.workdir), "channels": {}}
    if args.merge and args.out.exists():
        out = json.loads(args.out.read_text(encoding="utf-8"))
        out.setdefault("channels", {})

    bm25_pack = None
    if any(c.startswith("bm25") for c in args.configs):
        bm25_pack = build_bm25_from_sessions(sessions, meta)

    for ch in args.channels:
        qs = by_ch.get(ch) or by_ch.get(ch.replace("-", "_"))
        if not qs:
            # LegalEp maps
            alt = {"u1_exact": "exact", "advice-recall": "advice_recall", "paraphrase": "u_para"}.get(ch)
            qs = by_ch.get(alt) if alt else None
        if not qs:
            print(f"[skip] no queries for channel {ch}", flush=True)
            continue
        out["channels"].setdefault(ch, {})
        for cfg in args.configs:
            if cfg in out["channels"][ch] and args.merge:
                print(f"[skip-existing] {ch}/{cfg}", flush=True)
                continue
            if cfg == "joint_qa":
                jmgr, jmeta = index_joint_store(sessions)
                res = eval_config(jmgr, jmeta, qs, "joint_qa", args.top_k)
            else:
                res = eval_config(mgr, meta, qs, cfg, args.top_k, bm25_pack=bm25_pack)
            out["channels"][ch][cfg] = {k: res[k] for k in METRIC_KEYS if k in res}
            print(
                f"[{ch}/{cfg}] AH={res['answer_hit@k']:.3f} EC={res['episode_completeness@k']:.3f} "
                f"nDCG={res['ndcg@k']:.3f} SH={res.get('session_hit@k', float('nan')):.3f}",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
