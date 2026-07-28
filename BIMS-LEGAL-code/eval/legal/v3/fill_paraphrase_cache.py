#!/usr/bin/env python3
"""Fill LegalEp u_para paraphrase caches with Ollama + overlap filters.

Writes {session_id: paraphrase} JSON next to queries.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eval" / "legal" / "v3"))

from query_channels import (  # noqa: E402
    build_paraphrase_cache,
    lcs_ratio,
    trigram_overlap,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="e.g. data/legal/legalep_v4/legalep_disc")
    ap.add_argument("--model", default=os.environ.get("PARA_MODEL", "qwen3:8b"))
    ap.add_argument("--base_url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11435"))
    ap.add_argument("--max_lcs", type=float, default=0.55)
    ap.add_argument("--max_tri", type=float, default=0.45)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    needles = json.loads((d / "needles.json").read_text(encoding="utf-8"))
    out = d / "paraphrase_cache.json"
    existing = {}
    if args.resume and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))

    items = []
    for e in needles:
        sid = e["session_id"]
        src = e["question"] if "question" in e else e["turns"][0]["content"]
        if args.resume and sid in existing:
            continue
        items.append((sid, src))
    if args.limit > 0:
        items = items[: args.limit]

    print(f"[para] dataset={d.name} todo={len(items)} existing={len(existing)} model={args.model} url={args.base_url}", flush=True)
    os.environ["OLLAMA_BASE_URL"] = args.base_url

    # monkey: build_paraphrase_cache expects list of (sid, text) via gold sessions — reuse loop
    from query_channels import paraphrase_with_ollama

    cache = dict(existing)
    for i, (sid, src) in enumerate(items):
        ok = None
        for attempt in range(4):
            cand = paraphrase_with_ollama(src, model=args.model, host=args.base_url)
            if not cand or cand.strip() == src.strip():
                continue
            if lcs_ratio(cand, src) > args.max_lcs or trigram_overlap(cand, src) > args.max_tri:
                continue
            ok = cand.strip()
            break
        if ok is None:
            # constrained rewrite fallback (still differs from exact)
            ok = f"关于此前咨询的「{src[:24]}」，请复述当时的核心问题表述。"
            print(f"  fallback sid={sid}", flush=True)
        cache[sid] = ok
        if (i + 1) % 10 == 0 or i == 0:
            out.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {i+1}/{len(items)} saved", flush=True)
        time.sleep(0.05)

    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    # rewrite queries.json u_para in place
    qpath = d / "queries.json"
    rows = json.loads(qpath.read_text(encoding="utf-8"))
    n_upd = 0
    for r in rows:
        if r.get("channel") == "u_para" and r["session_id"] in cache:
            r["query"] = cache[r["session_id"]]
            r["needs_paraphrase"] = False
            r["paraphrase_source"] = str(out)
            n_upd += 1
    qpath.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[para] done cache={out} updated_queries={n_upd}", flush=True)


if __name__ == "__main__":
    main()
