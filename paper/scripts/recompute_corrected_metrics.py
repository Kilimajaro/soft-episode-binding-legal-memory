#!/usr/bin/env python3
"""Recompute corrected nDCG + failure taxonomy on primary BIMS-LEGAL grids.

IMPORTANT: BIMS_DATA_ROOT / embed-cache env must be set BEFORE importing
BIMS-LEGAL-code (config.py freezes paths at import time).

Example:
  export CUDA_VISIBLE_DEVICES=1 USE_EMBED_DISK_CACHE=1
  export EMBED_DISK_CACHE_DIR=/path/to/embed_cache_shared/vectors.sqlite
  python paper/scripts/recompute_corrected_metrics.py \\
    --manifest BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json \\
    --workdir /tmp/bims_recompute_disc \\
    --channels u_para advice_recall \\
    --configs dense_flat dense_o2 \\
    --skip_finalize \\
    --out paper/ipm/figures/corrected_metrics_disc.json
"""
from __future__ import annotations

import argparse
import json
import os
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
from run_legalmem_mt import (  # noqa: E402
    build_bm25_from_sessions,
    eval_config,
    index_joint_store,
    index_turn_store,
    resolve_queries,
    uniquify_session_ids,
)


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True, help="Fresh BIMS_DATA_ROOT (will be used for index)")
    ap.add_argument("--embed_cache", type=Path, default=None)
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=["dense_flat", "dense_o2"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--skip_finalize", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "paper/ipm/figures/corrected_metrics.json")
    ap.add_argument("--expect_flat_ah", type=float, default=None, help="Abort if FlatIP AH differs by >0.05")
    args = ap.parse_args()

    print(f"[env] BIMS_DATA_ROOT={os.environ.get('BIMS_DATA_ROOT')}", flush=True)
    print(f"[env] config._DATA_ROOT={_DATA_ROOT}", flush=True)
    print(f"[env] EMBED_DISK_CACHE_DIR={os.environ.get('EMBED_DISK_CACHE_DIR')}", flush=True)
    if Path(_DATA_ROOT).resolve() != args.workdir.resolve():
        raise SystemExit(
            f"FATAL: config._DATA_ROOT={_DATA_ROOT} != --workdir={args.workdir}. "
            "Env must be set before importing config."
        )

    # Fresh store each run
    talk = args.workdir / "talk.txt"
    if talk.exists():
        talk.unlink()
    for sub in ("vectors", "knowledge"):
        p = args.workdir / sub
        if p.exists():
            import shutil
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    man = load_manifest(args.manifest)
    sessions = man["sessions"]
    uniquify_session_ids(sessions)

    class Args:
        manifest = str(args.manifest)
        channels = args.channels
        force_queries_json = "legalep" in str(args.manifest).lower()
        para_cache = ""
        seed = 42

    print("[index] building turn store...", flush=True)
    mgr, meta = index_turn_store(sessions, skip_finalize=args.skip_finalize)
    print(f"[index] ntotal={mgr.vector_store.ntotal()} sessions={len(meta)}", flush=True)

    queries = resolve_queries(man, Args())
    by_ch = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    out = {
        "manifest": str(args.manifest),
        "workdir": str(args.workdir),
        "data_root": str(_DATA_ROOT),
        "channels": {},
    }
    for ch in args.channels:
        qs = by_ch.get(ch) or by_ch.get(ch.replace("-", "_"))
        if not qs:
            print(f"[skip] no queries for channel {ch}", flush=True)
            continue
        out["channels"][ch] = {}
        bm25_pack = None
        if any(c.startswith("bm25") for c in args.configs):
            bm25_pack = build_bm25_from_sessions(sessions, meta)
        for cfg in args.configs:
            if cfg == "joint_qa":
                jmgr, jmeta = index_joint_store(sessions)
                res = eval_config(jmgr, jmeta, qs, "joint_qa", args.top_k)
            else:
                res = eval_config(mgr, meta, qs, cfg, args.top_k, bm25_pack=bm25_pack)
            out["channels"][ch][cfg] = {
                k: res[k]
                for k in (
                    "answer_hit@k",
                    "episode_completeness@k",
                    "ndcg@k",
                    "failure_taxonomy",
                    "n",
                )
                if k in res
            }
            print(
                f"[{ch}/{cfg}] AH={res['answer_hit@k']:.3f} nDCG={res['ndcg@k']:.3f} "
                f"fail={res.get('failure_taxonomy')}",
                flush=True,
            )
            if (
                cfg == "dense_flat"
                and args.expect_flat_ah is not None
                and abs(res["answer_hit@k"] - args.expect_flat_ah) > 0.05
            ):
                raise SystemExit(
                    f"FlatIP AH {res['answer_hit@k']:.3f} != expected {args.expect_flat_ah:.3f} (±0.05). Abort."
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
