#!/usr/bin/env python3
"""Merge per_query_ah/query_ids from a PQ recompute into corrected_metrics_*.json.

Requires AH@k to match within --atol so table point estimates stay locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "ipm" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=["cail", "disc", "lawyer"])
    ap.add_argument("--src", type=Path, required=True, help="PQ recompute JSON")
    ap.add_argument("--atol", type=float, default=5e-3)
    ap.add_argument("--force", action="store_true", help="Allow AH drift and overwrite aggregates")
    args = ap.parse_args()
    dst = FIG / f"corrected_metrics_{args.name}.json"
    old = json.loads(dst.read_text(encoding="utf-8"))
    new = json.loads(args.src.read_text(encoding="utf-8"))
    for ch, cfgs in old["channels"].items():
        if ch not in new["channels"]:
            raise SystemExit(f"missing channel {ch} in {args.src}")
        for cfg, cell in cfgs.items():
            if cfg not in new["channels"][ch]:
                raise SystemExit(f"missing {ch}/{cfg}")
            ncell = new["channels"][ch][cfg]
            if "per_query_ah" not in ncell:
                raise SystemExit(f"no per_query_ah for {ch}/{cfg}")
            ah_old = float(cell["answer_hit@k"])
            ah_new = float(ncell["answer_hit@k"])
            if abs(ah_old - ah_new) > args.atol and not args.force:
                raise SystemExit(
                    f"AH drift {ch}/{cfg}: paper={ah_old:.6f} src={ah_new:.6f} "
                    f"(use --force to adopt src aggregates)"
                )
            if args.force:
                # adopt full metric cell from src but keep old provenance notes
                keep = {k: cell[k] for k in cell.keys() - ncell.keys()}
                cell.clear()
                cell.update(ncell)
                cell.update(keep)
            else:
                cell["per_query_ah"] = ncell["per_query_ah"]
                if "query_ids" in ncell:
                    cell["query_ids"] = ncell["query_ids"]
            print(f"ok {ch}/{cfg} AH={cell['answer_hit@k']:.3f} n={len(cell['per_query_ah'])}")
    old.setdefault("provenance", {})
    old["provenance"]["per_query_source"] = str(args.src)
    old["provenance"]["per_query_note"] = (
        "per_query_ah aligned to unified rebuild for McNemar/Holm"
    )
    dst.write_text(json.dumps(old, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", dst)


if __name__ == "__main__":
    main()
