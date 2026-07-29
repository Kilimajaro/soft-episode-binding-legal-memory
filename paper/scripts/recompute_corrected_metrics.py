#!/usr/bin/env python3
"""Recompute corrected nDCG + failure taxonomy on primary BIMS-LEGAL grids.

Uses fixed-gold IDCG from eval/legal/legal_metrics.py. Requires an indexed
workdir (BIMS_DATA_ROOT) or will index from manifest (slow).

Example:
  python paper/scripts/recompute_corrected_metrics.py \\
    --manifest BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json \\
    --workdir /path/to/workdir_disc_para0 \\
    --channels u_para advice_recall \\
    --configs dense_flat dense_o2 bm25_joint joint_qa
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "BIMS-LEGAL-code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "eval" / "legal"))
sys.path.insert(0, str(CODE / "eval" / "legal" / "v3"))

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
    ap.add_argument("--workdir", type=Path, default=None, help="Pre-built BIMS_DATA_ROOT")
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=["dense_flat", "dense_o2", "parent_hydrate", "bm25_joint", "joint_qa"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--skip_finalize", action="store_true", help="Skip BIRCH finalize for faster dense-only metrics")
    ap.add_argument("--out", type=Path, default=REPO / "paper/ipm/figures/corrected_metrics.json")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    sessions = man["sessions"]
    uniquify_session_ids(sessions)
    class Args:
        manifest = str(args.manifest)
        channels = args.channels
        force_queries_json = "legalep" in str(args.manifest).lower()
        para_cache = ""
        seed = 42

    if args.workdir:
        os.environ["BIMS_DATA_ROOT"] = str(args.workdir)
        os.environ.setdefault("USE_EMBED_DISK_CACHE", "1")
    print("[index] building turn store from manifest (embed cache via BIMS_DATA_ROOT)...", flush=True)
    mgr, meta = index_turn_store(sessions, skip_finalize=args.skip_finalize)

    queries = resolve_queries(man, Args())
    by_ch = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    out = {"manifest": str(args.manifest), "workdir": str(args.workdir) if args.workdir else None, "channels": {}}
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
    out.pop("_bm25", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
