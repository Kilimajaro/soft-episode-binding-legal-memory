#!/usr/bin/env python3
"""Recompute corrected nDCG by LOADING an existing V4 workdir (no re-index).

Preserves the published store so Answer Hit should match primary_results.
BIMS_DATA_ROOT must be set via --workdir before config import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "BIMS-LEGAL-code"


def _bootstrap():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workdir", type=Path, required=False)
    ap.add_argument("--embed_cache", type=Path, default=None)
    args, _ = ap.parse_known_args()
    if not args.workdir:
        return
    os.environ["BIMS_DATA_ROOT"] = str(args.workdir.resolve())
    if args.embed_cache:
        os.environ["EMBED_DISK_CACHE_DIR"] = str(args.embed_cache.resolve())
        os.environ["USE_EMBED_DISK_CACHE"] = "1"
    os.environ.setdefault("USE_EMBED_DISK_CACHE", "1")


_bootstrap()
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "eval" / "legal"))
sys.path.insert(0, str(CODE / "eval" / "legal" / "v3"))

from config import _DATA_ROOT  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from run_legalmem_mt import eval_config, resolve_queries, uniquify_session_ids  # noqa: E402


def _parse_talk(talk_path: Path) -> list[dict]:
    rows = []
    for line in talk_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        head, text = line.split("|", 1)
        try:
            meta = json.loads(head)
        except Exception:
            continue
        rows.append(
            {
                "tid": str(meta.get("tid")),
                "role": meta.get("role"),
                "text": text,
                "timestamp": meta.get("timestamp"),
            }
        )
    return rows


def build_meta_from_manifest(mgr: VectorMemoryManager, sessions: list[dict], talk_path: Path) -> dict:
    """Map manifest sessions to tids by talk.txt order (same construction order as V4)."""
    talk = _parse_talk(talk_path)
    expected = sum(len(s["turns"]) for s in sessions)
    print(f"[meta] talk_rows={len(talk)} expected_turns={expected}", flush=True)
    if len(talk) != expected:
        # Fall back to text matching if lengths diverge.
        print("[meta] WARN talk/manifest length mismatch; using text match fallback", flush=True)
        return _build_meta_by_text(mgr, sessions)

    mgr._tid_to_session = {}
    mgr._session_members = {}
    out = {}
    i = 0
    for s in sessions:
        sid = s["session_id"]
        all_tids, ans_tids, user_tids = [], [], []
        for t in s["turns"]:
            row = talk[i]
            i += 1
            tid = row["tid"]
            all_tids.append(tid)
            role = t.get("role") or row.get("role")
            if role == "assistant":
                ans_tids.append(tid)
            elif role == "user":
                user_tids.append(tid)
            mgr._tid_to_session[tid] = sid
            mgr._session_members.setdefault(sid, []).append(tid)
        out[sid] = {
            "all_tids": all_tids,
            "ans_tids": ans_tids or all_tids[-1:],
            "user_tids": user_tids or all_tids[:1],
            "role": s.get("role"),
        }
    print(f"[meta] sessions_mapped={len(out)} via talk-order", flush=True)
    return out


def _build_meta_by_text(mgr: VectorMemoryManager, sessions: list[dict]) -> dict:
    text_to_tids: dict[str, list[str]] = {}
    for meta in mgr.vector_store.metadata:
        if meta.get("type") != "paragraph":
            continue
        text = (meta.get("text") or "").strip()
        tid = str(meta.get("id"))
        text_to_tids.setdefault(text, []).append(tid)
    mgr._tid_to_session = {}
    mgr._session_members = {}
    out = {}
    missing = 0
    for s in sessions:
        sid = s["session_id"]
        all_tids, ans_tids, user_tids = [], [], []
        for t in s["turns"]:
            text = (t.get("content") or "").strip()
            cands = text_to_tids.get(text) or []
            if not cands:
                missing += 1
                continue
            tid = cands.pop(0)
            all_tids.append(tid)
            if t.get("role") == "assistant":
                ans_tids.append(tid)
            elif t.get("role") == "user":
                user_tids.append(tid)
            mgr._tid_to_session[tid] = sid
            mgr._session_members.setdefault(sid, []).append(tid)
        if not all_tids:
            continue
        out[sid] = {
            "all_tids": all_tids,
            "ans_tids": ans_tids or all_tids[-1:],
            "user_tids": user_tids or all_tids[:1],
            "role": s.get("role"),
        }
    print(f"[meta] sessions_mapped={len(out)} missing_turns={missing}", flush=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--embed_cache", type=Path, default=None)
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=["dense_flat", "dense_o2"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect_flat_ah", type=float, default=None)
    args = ap.parse_args()

    print(f"[env] BIMS_DATA_ROOT={os.environ.get('BIMS_DATA_ROOT')}", flush=True)
    print(f"[env] config._DATA_ROOT={_DATA_ROOT}", flush=True)
    if Path(_DATA_ROOT).resolve() != args.workdir.resolve():
        raise SystemExit(f"FATAL data root mismatch {_DATA_ROOT} vs {args.workdir}")

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    sessions = man["sessions"]
    uniquify_session_ids(sessions)

    print("[load] VectorMemoryManager from existing workdir (no reset)...", flush=True)
    mgr = VectorMemoryManager()
    mgr.vector_store.use_pq = False
    print(
        f"[load] ntotal={mgr.vector_store.ntotal()} meta={len(mgr.vector_store.metadata)}",
        flush=True,
    )
    meta = build_meta_from_manifest(mgr, sessions, args.workdir / "talk.txt")

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

    out = {
        "manifest": str(args.manifest),
        "workdir": str(args.workdir),
        "mode": "load_existing",
        "channels": {},
    }
    for ch in args.channels:
        qs = by_ch.get(ch) or by_ch.get(ch.replace("-", "_"))
        if not qs:
            print(f"[skip] {ch}", flush=True)
            continue
        qs = [q for q in qs if q["session_id"] in meta]
        print(f"[channel] {ch} n_queries={len(qs)}", flush=True)
        out["channels"][ch] = {}
        for cfg in args.configs:
            res = eval_config(mgr, meta, qs, cfg, args.top_k)
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
                and abs(res["answer_hit@k"] - args.expect_flat_ah) > 0.03
            ):
                raise SystemExit(
                    f"FlatIP AH {res['answer_hit@k']:.3f} != expected {args.expect_flat_ah:.3f}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
