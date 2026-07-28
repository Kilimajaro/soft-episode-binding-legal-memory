#!/usr/bin/env python3
"""将 results/legal_revision/*.json 汇总为 Markdown/LaTeX 片段，便于填入 IPM 表。"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "legal_revision"
OUT = REV / "tables_for_paper.md"


def row(ds, prot, cfg, r):
    return (
        f"| {ds} | {prot} | {cfg} | "
        f"{r.get('answer_hit@k', 0):.3f} | "
        f"{r.get('session_hit@k', 0):.3f} | "
        f"{r.get('episode_completeness@k', 0):.3f} | "
        f"{r.get('ndcg@k', 0):.3f} | "
        f"{r.get('mrr@k', 0):.3f} |"
    )


def load_all():
    lines = [
        "# Revision protocol tables",
        "",
        "| Dataset | Protocol | Config | AH@10 | SH@10 | EC@10 | nDCG@10 | MRR@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for ds in ("disc_law", "lawyer_llama"):
        for fname in ("revision_protocol.json", "bm25_rrf.json"):
            path = REV / ds / fname
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for prot, pdata in data.get("protocols", {}).items():
                for cfg, r in pdata.get("configs", {}).items():
                    lines.append(row(ds, prot, cfg, r))
        for extra in ("multiseed.json", "hard_neg.json"):
            path = REV / ds / extra
            if path.is_file():
                lines.append(f"\n## {ds}/{extra}\n")
                lines.append("```json\n" + path.read_text(encoding="utf-8")[:8000] + "\n```")
    scale = list(REV.glob("scale_curve_*.json"))
    if scale:
        lines.append("\n## Scale curves\n")
        for p in scale:
            lines.append(f"- `{p.name}`")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    load_all()
