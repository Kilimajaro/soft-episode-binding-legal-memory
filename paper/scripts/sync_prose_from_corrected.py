#!/usr/bin/env python3
"""Sync key Soft O2 prose numbers in ipm-article.tex to corrected_metrics_*.json.

Tables are owned by regenerate_unified_tables.py; this patch updates narrative
sentences that hard-code AH values so they cannot drift from the unified rebuild.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
FIG = ROOT / "paper" / "ipm" / "figures"


def load(name: str) -> dict:
    return json.loads((FIG / f"corrected_metrics_{name}.json").read_text(encoding="utf-8"))


def ah(blob: dict, ch: str, cfg: str) -> str:
    return f"{float(blob['channels'][ch][cfg]['answer_hit@k']):.3f}"


def main() -> None:
    cail, disc, lawyer = load("cail"), load("disc"), load("lawyer")
    vals = {
        "cail_u1_flat": ah(cail, "u1_exact", "dense_flat"),
        "cail_u1_o2": ah(cail, "u1_exact", "dense_o2"),
        "cail_uk_flat": ah(cail, "uk_followup", "dense_flat"),
        "cail_uk_o2": ah(cail, "uk_followup", "dense_o2"),
        "cail_uk_hard": ah(cail, "uk_followup", "parent_hydrate"),
        "cail_ul_flat": ah(cail, "u_last", "dense_flat"),
        "cail_ul_o2": ah(cail, "u_last", "dense_o2"),
        "disc_exact_flat": ah(disc, "exact", "dense_flat"),
        "disc_exact_o2": ah(disc, "exact", "dense_o2"),
        "disc_para_flat": ah(disc, "u_para", "dense_flat"),
        "disc_para_o2": ah(disc, "u_para", "dense_o2"),
        "disc_adv_flat": ah(disc, "advice_recall", "dense_flat"),
        "disc_adv_o2": ah(disc, "advice_recall", "dense_o2"),
        "law_exact_flat": ah(lawyer, "exact", "dense_flat"),
        "law_exact_o2": ah(lawyer, "exact", "dense_o2"),
        "law_para_flat": ah(lawyer, "u_para", "dense_flat"),
        "law_para_o2": ah(lawyer, "u_para", "dense_o2"),
        "law_adv_flat": ah(lawyer, "advice_recall", "dense_flat"),
        "law_adv_o2": ah(lawyer, "advice_recall", "dense_o2"),
    }
    tex = TEX.read_text(encoding="utf-8")

    replacements = [
        (
            r"Soft~O2 raises Answer Hit over FlatIP on every channel \(U1 \$[0-9.]+\$ vs \$[0-9.]+\$; Uk \$[0-9.]+\$ vs \$[0-9.]+\$; U-last \$[0-9.]+\$ vs \$[0-9.]+\$\)\.",
            f"Soft~O2 raises Answer Hit over FlatIP on every channel "
            f"(U1 ${vals['cail_u1_o2']}$ vs ${vals['cail_u1_flat']}$; "
            f"Uk ${vals['cail_uk_o2']}$ vs ${vals['cail_uk_flat']}$; "
            f"U-last ${vals['cail_ul_o2']}$ vs ${vals['cail_ul_flat']}$).",
        ),
        (
            r"Hard hydration can raise AH further by copying sibling scores without attenuation \(e\.g\., Uk \$[0-9.]+\$ vs Soft~O2 \$[0-9.]+\$\)\.",
            f"Hard hydration can raise AH further by copying sibling scores without attenuation "
            f"(e.g., Uk ${vals['cail_uk_hard']}$ vs Soft~O2 ${vals['cail_uk_o2']}$).",
        ),
        (
            r"Soft~O2 also raises exact replay \(DISC \$[0-9.]+\$ vs FlatIP \$[0-9.]+\$; Lawyer \$[0-9.]+\$ vs \$[0-9.]+\$\)",
            f"Soft~O2 also raises exact replay "
            f"(DISC ${vals['disc_exact_o2']}$ vs FlatIP ${vals['disc_exact_flat']}$; "
            f"Lawyer ${vals['law_exact_o2']}$ vs ${vals['law_exact_flat']}$)",
        ),
        (
            r"Paraphrase gains are large: \$[0-9.]+\$ vs \$[0-9.]+\$ on DISC and \$[0-9.]+\$ vs \$[0-9.]+\$ on Lawyer\.",
            f"Paraphrase gains are large: "
            f"${vals['disc_para_o2']}$ vs ${vals['disc_para_flat']}$ on DISC and "
            f"${vals['law_para_o2']}$ vs ${vals['law_para_flat']}$ on Lawyer.",
        ),
        (
            r"Advice-recall rises to \$[0-9.]+\$ from FlatIP \$[0-9.]+\$ on DISC and to \$[0-9.]+\$ from \$[0-9.]+\$ on Lawyer\.",
            f"Advice-recall rises to ${vals['disc_adv_o2']}$ from FlatIP ${vals['disc_adv_flat']}$ on DISC "
            f"and to ${vals['law_adv_o2']}$ from ${vals['law_adv_flat']}$ on Lawyer.",
        ),
    ]

    n = 0
    for pat, repl in replacements:
        tex2, k = re.subn(pat, repl, tex, count=1)
        if k != 1:
            raise SystemExit(f"prose pattern not uniquely matched ({k}): {pat[:80]}")
        tex = tex2
        n += 1
    TEX.write_text(tex, encoding="utf-8")
    print(f"updated {n} prose sentences in {TEX}")
    for k, v in vals.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
