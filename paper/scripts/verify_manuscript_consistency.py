#!/usr/bin/env python3
"""Guard Soft O2 manuscript consistency before release (S0-1 / S0-2 / S0-3).

Exit 0 only when:
  - Fig/table Soft O2 point estimates come from corrected_metrics_*.json
  - holm_primary_family.json ΔAH == SoftO2-FlatIP on those JSON grids
  - draw_ipm_figures.py does not load legacy bims_legal_v4 result paths
  - beta narrative is theory + grid search (not early-campaign / primary-eval wording)
  - corrected_metrics cells include per_query_ah for the primary Holm family
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

BANNED = [
    "early development campaign",
    "separate early development",
    "paired-hit archive",
    "primary shared-corpus evaluation",
]

REQUIRED = [
    "grid search",
    "gap-ratio",
]


def load_corrected(name: str) -> dict:
    return json.loads((FIG / f"corrected_metrics_{name}.json").read_text(encoding="utf-8"))


def main() -> int:
    issues: list[str] = []
    tex = TEX.read_text(encoding="utf-8")
    draw = DRAW.read_text(encoding="utf-8")
    holm = json.loads((FIG / "holm_primary_family.json").read_text(encoding="utf-8"))
    by = {r["label"]: r for r in holm.get("rows", [])}

    if re.search(r"results/bims_legal_v4", draw):
        issues.append("draw_ipm_figures.py still references results/bims_legal_v4 data paths")

    src = str(holm.get("source", ""))
    if "bims_legal_v4" in src.lower():
        issues.append(f"holm_primary_family.json source is legacy: {src}")

    blobs = {n: load_corrected(n) for n in ("cail", "disc", "lawyer")}
    for lab, js, ch in PRIMARY:
        block = blobs[js]["channels"][ch]
        for cfg in ("dense_flat", "dense_o2"):
            if "per_query_ah" not in block.get(cfg, {}):
                issues.append(f"missing per_query_ah for {lab}/{cfg}")
        soft = float(block["dense_o2"]["answer_hit@k"])
        flat = float(block["dense_flat"]["answer_hit@k"])
        delta = soft - flat
        if lab not in by:
            issues.append(f"holm row missing: {lab}")
            continue
        hd = float(by[lab]["delta_ah"])
        if abs(delta - hd) > 1e-6:
            issues.append(f"Holm ΔAH != Soft−Flat for {lab}: table={delta:+.6f} holm={hd:+.6f}")

        # Table A.2 body digit check (3 d.p.)
        needle = f"{hd:+.3f}"
        if needle not in tex and f"{hd:.3f}" not in tex:
            # sign-forced form used in tabular
            if f"{hd:+.3f}" not in tex:
                issues.append(f"tab:sig may not contain Holmed ΔAH {needle} for {lab}")

    for bad in BANNED:
        if bad.lower() in tex.lower():
            issues.append(f"banned phrase in manuscript: {bad!r}")

    for need in REQUIRED:
        if need not in tex:
            issues.append(f"required narrative missing: {need!r}")

    # Caption overclaim without matching numbers is checked via Holm mismatches above.
    if issues:
        print("FAIL: manuscript consistency")
        for i in issues:
            print(" -", i)
        return 1
    print("OK: Soft O2 tables, Holm ΔAH, fig script, and beta narrative are aligned")
    print("holm source:", src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
