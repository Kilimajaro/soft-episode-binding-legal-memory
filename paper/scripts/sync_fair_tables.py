#!/usr/bin/env python3
"""Sync fair-ablation JSON into ipm-article.tex (beta, CAIL, dense paraphrase tables).

Bold rule: within each metric column (and within each dataset block), bold the
maximum numeric cell — never blanket-bold the ``Ours'' / Soft-O2 row.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results" / "legal_revision_fair"
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"

# Display order for dense comparison tables (no BM25 / IVFPQ here).
DENSE_ORDER = [
    ("dense_flat", "Dense FlatIP"),
    ("joint_qa", "Dense joint Q+A"),
    ("parent_hydrate", "Parent hydration"),
    ("session_max", "Session-max"),
    ("shuffled_o2", "Shuffled O2"),
    ("dense_o2", "Dense + soft O2"),
]

METRICS = [
    ("answer_hit@k", "AH"),
    ("session_hit@k", "SH"),
    ("episode_completeness@k", "EC"),
    ("ndcg@k", "nDCG"),
]


def load_fair(dataset: str):
    p = FAIR / dataset / "fair_protocol.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def cfg_metrics(payload, protocol, cfg):
    try:
        return payload["protocols"][protocol]["configs"][cfg]
    except Exception:
        return None


def fmt(x):
    return "--" if x is None else f"{x:.3f}"


def bold_max_columns(rows):
    """rows: list[list[float|None]]; return list[list[str]] with per-column max bolded."""
    if not rows:
        return []
    ncols = len(rows[0])
    best = [None] * ncols
    for c in range(ncols):
        vals = [r[c] for r in rows if isinstance(r[c], float)]
        if vals:
            best[c] = max(vals)
    out = []
    for r in rows:
        cells = []
        for c, v in enumerate(r):
            if isinstance(v, float) and best[c] is not None and abs(v - best[c]) < 1e-12:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(fmt(v))
        out.append(cells)
    return out


def bold_row(vals):
    return bold_max_columns([vals])[0]


def dense_block(payload, protocol="paraphrase"):
    raw = []
    for cfg, _label in DENSE_ORDER:
        m = cfg_metrics(payload, protocol, cfg)
        if m is None:
            raw.append([None] * 4)
        else:
            raw.append([float(m[k]) for k, _ in METRICS])
    return bold_max_columns(raw)


def replace_between(tex, start_pat, end_pat, body):
    """Replace text after start_pat up to end_pat (exclusive of end)."""
    m = re.search(start_pat, tex, flags=re.S)
    if not m:
        raise RuntimeError(f"start not found: {start_pat}")
    start = m.end()
    m2 = re.search(end_pat, tex[start:], flags=re.S)
    if not m2:
        raise RuntimeError(f"end not found: {end_pat}")
    end = start + m2.start()
    return tex[:start] + body + tex[end:]


def main():
    tex = TEX.read_text(encoding="utf-8")

    # ---- Beta table ----
    beta_rows = []
    for ds, label in [("disc_law", "DISC-Law"), ("lawyer_llama", "Lawyer-LLaMA")]:
        p = FAIR / ds / "beta_sweep.json"
        if not p.is_file():
            beta_rows.append(f"{label} & -- & -- & -- & -- & -- & -- \\\\")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        betas = ["0.50", "0.70", "0.90", "0.95", "0.98", "1.00"]
        vals = []
        for b in betas:
            r = data.get("betas", {}).get(b) or data.get("betas", {}).get(f"{float(b):.2f}")
            vals.append(None if r is None else float(r["answer_hit@k"]))
        cells = bold_row(vals)
        beta_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex = replace_between(
        tex,
        r"\\label\{tab:beta\}.*?\\midrule\n",
        r"\\bottomrule",
        "\n".join(beta_rows) + "\n",
    )

    # ---- CAIL table ----
    cail_rows = []
    for split, label in [("cail_prelim", "CAIL-Prelim"), ("cail_final", "CAIL-Final")]:
        data = load_fair(split)
        for proto, pname in [("paraphrase", "paraphrase"), ("followup", "follow-up")]:
            vals = [
                None if cfg_metrics(data, proto, c) is None
                else float(cfg_metrics(data, proto, c)["answer_hit@k"])
                for c in ("dense_flat", "dense_o2", "parent_hydrate", "joint_qa")
            ]
            cells = bold_row(vals)
            cail_rows.append(f"{label} & {pname} & " + " & ".join(cells) + " \\\\")
    tex = replace_between(
        tex,
        r"\\label\{tab:cail\}.*?\\midrule\n",
        r"\\bottomrule",
        "\n".join(cail_rows) + "\n",
    )

    # ---- Dense paraphrase comparison (tab:revision_dense) ----
    dense_lines = []
    for ds, label, n_rows in [
        ("disc_law", "DISC-Law", 6),
        ("lawyer_llama", "Lawyer-LLaMA", 6),
    ]:
        payload = load_fair(ds)
        cells_rows = dense_block(payload, "paraphrase")
        for i, ((cfg, sys_label), cells) in enumerate(zip(DENSE_ORDER, cells_rows)):
            prefix = f"\\multirow{{{n_rows}}}{{*}}{{{label}}}\n" if i == 0 else ""
            dense_lines.append(
                f"{prefix} & {sys_label} & " + " & ".join(cells) + " \\\\"
            )
        if ds == "disc_law":
            dense_lines.append("\\midrule")
    tex = replace_between(
        tex,
        r"\\label\{tab:revision_dense\}.*?\\midrule\n",
        r"\\bottomrule",
        "\n".join(dense_lines) + "\n",
    )

    TEX.write_text(tex, encoding="utf-8")
    print("updated", TEX)
    print("--- beta ---")
    print("\n".join(beta_rows))
    print("--- cail ---")
    print("\n".join(cail_rows))
    print("--- revision_dense ---")
    print("\n".join(dense_lines))


if __name__ == "__main__":
    main()
