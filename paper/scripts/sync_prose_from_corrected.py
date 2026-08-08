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
        "law_exact_flat": ah(lawyer, "exact", "dense_flat"),
        "law_exact_o2": ah(lawyer, "exact", "dense_o2"),
        "law_para_flat": ah(lawyer, "u_para", "dense_flat"),
        "law_para_o2": ah(lawyer, "u_para", "dense_o2"),
    }
    tex = TEX.read_text(encoding="utf-8")

    replacements = [
        (
            r"Soft~O2 lifts Answer Hit over FlatIP on every channel under the unified rebuild \(U1 \$[0-9.]+\$ vs \$[0-9.]+\$; Uk \$[0-9.]+\$ vs \$[0-9.]+\$; U-last \$[0-9.]+\$ vs \$[0-9.]+\$\)\.",
            f"Soft~O2 lifts Answer Hit over FlatIP on every channel under the unified rebuild "
            f"(U1 ${vals['cail_u1_o2']}$ vs ${vals['cail_u1_flat']}$; "
            f"Uk ${vals['cail_uk_o2']}$ vs ${vals['cail_uk_flat']}$; "
            f"U-last ${vals['cail_ul_o2']}$ vs ${vals['cail_ul_flat']}$).",
        ),
        (
            r"Hard expansion can raise AH further by unrestricted score copy \(e\.g\., Uk \$[0-9.]+\$ vs Soft~O2 \$[0-9.]+\$\)\.",
            f"Hard expansion can raise AH further by unrestricted score copy "
            f"(e.g., Uk ${vals['cail_uk_hard']}$ vs Soft~O2 ${vals['cail_uk_o2']}$).",
        ),
        (
            r"Under the unified rebuild Soft~O2 also lifts exact replay \(DISC \$[0-9.]+\$ vs FlatIP \$[0-9.]+\$; Lawyer \$[0-9.]+\$ vs \$[0-9.]+\$\)",
            f"Under the unified rebuild Soft~O2 also lifts exact replay "
            f"(DISC ${vals['disc_exact_o2']}$ vs FlatIP ${vals['disc_exact_flat']}$; "
            f"Lawyer ${vals['law_exact_o2']}$ vs ${vals['law_exact_flat']}$)",
        ),
        (
            r"Paraphrase yields large Soft~O2 gains over FlatIP: \$[0-9.]+\$ vs \$[0-9.]+\$ on DISC and \$[0-9.]+\$ vs \$[0-9.]+\$ on Lawyer\.",
            f"Paraphrase yields large Soft~O2 gains over FlatIP: "
            f"${vals['disc_para_o2']}$ vs ${vals['disc_para_flat']}$ on DISC and "
            f"${vals['law_para_o2']}$ vs ${vals['law_para_flat']}$ on Lawyer.",
        ),
        (
            r"On CAIL under the unified rebuild, Soft~O2 raises Answer Hit by large margins over FlatIP \(Uk \$[0-9.]+\$ vs \$[0-9.]+\$; U-last \$[0-9.]+\$ vs \$[0-9.]+\$\)",
            f"On CAIL under the unified rebuild, Soft~O2 raises Answer Hit by large margins over FlatIP "
            f"(Uk ${vals['cail_uk_o2']}$ vs ${vals['cail_uk_flat']}$; "
            f"U-last ${vals['cail_ul_o2']}$ vs ${vals['cail_ul_flat']}$)",
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
