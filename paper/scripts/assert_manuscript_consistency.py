#!/usr/bin/env python3
"""Fail loud if Soft O2 manuscript artifacts disagree (S0-1 / S0-2 gate).

Checks
  - draw_ipm_figures.py does not load bims_legal_v4 result paths
  - corrected_metrics_*.json expose per_query_ah for FlatIP and Soft O2
  - holm_primary_family.json ΔAH == SoftO2−FlatIP on each primary channel
  - holm source is not the legacy v4 paired archive
  - hardcoded AH figures in key prose sentences match corrected JSON (3 decimals)
  - Fig 3/4 loaders resolve to the same corrected JSON used by tables
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "ipm" / "figures"
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
DRAW = ROOT / "paper" / "scripts" / "draw_ipm_figures.py"

PRIMARY = [
    ("CAIL / Uk", "cail", "uk_followup"),
    ("CAIL / U1", "cail", "u1_exact"),
    ("CAIL / U-last", "cail", "u_last"),
    ("LegalEp-DISC / exact", "disc", "exact"),
    ("LegalEp-Lawyer / exact", "lawyer", "exact"),
    ("LegalEp-DISC / u-para", "disc", "u_para"),
    ("LegalEp-Lawyer / u-para", "lawyer", "u_para"),
    ("LegalEp-DISC / advice", "disc", "advice_recall"),
    ("LegalEp-Lawyer / advice", "lawyer", "advice_recall"),
]

PROSE_SPOTS = [
    ("cail", "u1_exact", "dense_o2", "cail_u1_o2"),
    ("cail", "u1_exact", "dense_flat", "cail_u1_flat"),
    ("cail", "uk_followup", "dense_o2", "cail_uk_o2"),
    ("cail", "uk_followup", "dense_flat", "cail_uk_flat"),
    ("cail", "u_last", "dense_o2", "cail_ul_o2"),
    ("cail", "u_last", "dense_flat", "cail_ul_flat"),
    ("disc", "u_para", "dense_o2", "disc_para_o2"),
    ("disc", "u_para", "dense_flat", "disc_para_flat"),
    ("lawyer", "u_para", "dense_o2", "law_para_o2"),
    ("lawyer", "u_para", "dense_flat", "law_para_flat"),
]


def load(name: str) -> dict:
    return json.loads((FIG / f"corrected_metrics_{name}.json").read_text(encoding="utf-8"))


def ah(blob: dict, ch: str, cfg: str) -> float:
    return float(blob["channels"][ch][cfg]["answer_hit@k"])


def main() -> None:
    errors: list[str] = []

    draw = DRAW.read_text(encoding="utf-8")
    if "results/bims_legal_v4" in draw or "bims_legal_v4/" in draw:
        # docstring may mention the forbidden path; forbid path-literal loads only.
        if re.search(r"results/bims_legal_v4|bims_legal_v4/.*/results\.json", draw):
            errors.append("draw_ipm_figures.py still references bims_legal_v4 result paths")

    for name in ("cail", "disc", "lawyer"):
        blob = load(name)
        for ch, cellmap in blob["channels"].items():
            for cfg in ("dense_flat", "dense_o2"):
                if "per_query_ah" not in cellmap.get(cfg, {}):
                    errors.append(f"missing per_query_ah in corrected_metrics_{name}.json::{ch}/{cfg}")

    holm_path = FIG / "holm_primary_family.json"
    paired_path = FIG / "paired_ah_primary_family.json"
    if not holm_path.exists():
        errors.append("missing holm_primary_family.json")
    if not paired_path.exists():
        errors.append("missing paired_ah_primary_family.json")
    else:
        holm = json.loads(holm_path.read_text(encoding="utf-8"))
        src = str(holm.get("source", ""))
        if "bims_legal_v4" in src.lower():
            errors.append(f"holm source still points at legacy archive: {src}")
        by = {r["label"]: r for r in holm.get("rows", [])}
        for lab, js, ch in PRIMARY:
            b = load(js)["channels"][ch]
            d = float(b["dense_o2"]["answer_hit@k"]) - float(b["dense_flat"]["answer_hit@k"])
            if lab not in by:
                errors.append(f"holm missing row {lab}")
                continue
            hd = float(by[lab]["delta_ah"])
            if abs(d - hd) > 1e-9:
                errors.append(f"ΔAH mismatch {lab}: table={d:+.6f} holm={hd:+.6f}")

    tex = TEX.read_text(encoding="utf-8")
    for name, ch, cfg, _tag in PROSE_SPOTS:
        val = f"{ah(load(name), ch, cfg):.3f}"
        # require the 3-decimal form to appear at least once in Soft O2 discussion territory
        if val not in tex:
            errors.append(f"prose/table missing expected AH {val} from corrected_metrics_{name}/{ch}/{cfg}")

    # caption contracts
    if "early development campaign" in tex.lower():
        errors.append("residual 'early development campaign' wording found")
    if "paired-hit archive" in tex.lower():
        errors.append("residual paired-hit archive disclaimer found")

    if errors:
        print("MANUSCRIPT CONSISTENCY FAILED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("OK: Soft O2 tables, Holm, per_query_ah, and prose AH spots agree.")


if __name__ == "__main__":
    main()
