#!/usr/bin/env python3
"""Patch appendix nDCG cells from corrected_metrics_*.json artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/ipm/ipm-article.tex"
FIG = ROOT / "paper/ipm/figures"


def fmt(x: float) -> str:
    return f"{float(x):.3f}"


def load_metrics() -> dict:
    out = {}
    for name in ("corrected_metrics_disc.json", "corrected_metrics_cail.json", "corrected_metrics_lawyer.json"):
        p = FIG / name
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def patch_table_ndcg(tex: str, label: str, channel: str, cfg: str, ndcg: float) -> str:
    # Best-effort: find row with cfg name near label table - manual mapping preferred.
    return tex


def main() -> None:
    blobs = load_metrics()
    if not blobs:
        print("[skip] no corrected metrics json files")
        return
    tex = TEX.read_text(encoding="utf-8")
    mapping = {
        ("disc", "u_para", "dense_flat"): ("tab:disc_main", "FlatIP"),
        ("disc", "u_para", "dense_o2"): ("tab:disc_main", "Soft O2"),
        ("disc", "advice_recall", "dense_flat"): ("tab:disc_advice", "FlatIP"),
        ("disc", "advice_recall", "dense_o2"): ("tab:disc_advice", "Soft O2"),
    }
    for blob in blobs.values():
        for ch, cfgs in blob.get("channels", {}).items():
            for cfg, m in cfgs.items():
                key = (blob.get("corpus", "disc"), ch, cfg)
                if key not in mapping:
                    continue
                label, sysname = mapping[key]
                ndcg = m.get("ndcg@k")
                if ndcg is None:
                    continue
                # replace nDCG in row containing sysname within table after label
                mlabel = re.search(rf"\\label\{{{label}\}}.*?\\midrule\n(.*?)\\bottomrule", tex, re.S)
                if not mlabel:
                    continue
                body = mlabel.group(1)
                new_body = re.sub(
                    rf"(& {re.escape(sysname)} & [^&]+ & [^&]+ & )[0-9.]+",
                    rf"\g<1>{fmt(ndcg)}",
                    body,
                    count=1,
                )
                tex = tex.replace(body, new_body)
    TEX.write_text(tex, encoding="utf-8")
    print(f"[wrote] {TEX} from {list(blobs)}")


if __name__ == "__main__":
    main()
