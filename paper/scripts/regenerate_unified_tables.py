#!/usr/bin/env python3
"""Regenerate paper tables from a SINGLE metric source (S0-1/S0-5/S0-6).

Default source: paper/ipm/figures/corrected_metrics_{cail,disc,lawyer}.json
(same FlatIP rebuild for AH, EC, nDCG, failure_taxonomy, and per_query_ah).

Holm primary family is computed from unified-rebuild per_query_ah when present.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "legal" / "v3"))
sys.path.insert(0, str(ROOT / "eval" / "legal"))

from stats_sig import paired_report  # noqa: E402

TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
FIG = ROOT / "paper" / "ipm" / "figures"

CFG = {
    "FlatIP": "dense_flat",
    "Soft O2": "dense_o2",
    "Hard hydr.": "parent_hydrate",
    "Shuffled O2": "shuffled_o2",
}

PRIMARY_FAMILY = [
    # (tex label, corrected json stem, channel key)
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


def holm_adjust(pvals: list[float]) -> list[float]:
    """Standard Holm step-down; monotone non-decreasing in sorted order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvals[idx] * (m - rank)
        running = max(running, raw)
        adj[idx] = min(1.0, running)
    return adj


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def load_corrected(name: str) -> dict:
    return json.loads((FIG / f"corrected_metrics_{name}.json").read_text(encoding="utf-8"))


def replace_table_body(tex: str, label: str, new_body: str) -> str:
    """Replace tabular body between \\midrule and \\bottomrule for a labeled table."""
    lab = f"\\label{{{label}}}"
    i = tex.find(lab)
    if i < 0:
        raise SystemExit(f"missing {label}")
    # find begin tabular after label (or before if caption order)
    # tables have label then tabular OR caption/label then tabular
    t0 = tex.find("\\begin{tabular}", i)
    if t0 < 0 or t0 - i > 800:
        # label may be after tabular start in some layouts — search back
        t0 = tex.rfind("\\begin{tabular}", 0, i)
    mid = tex.find("\\midrule", t0)
    bot = tex.find("\\bottomrule", mid)
    if mid < 0 or bot < 0:
        raise SystemExit(f"mid/bottom missing for {label}")
    return tex[: mid + len("\\midrule\n")] + new_body.rstrip() + "\n" + tex[bot:]


def rewrite_appendix_grids(tex: str) -> str:
    packs = {
        "tab:cail_main": (
            "cail",
            [
                ("U1", "u1_exact"),
                ("Uk-followup", "uk_followup"),
                ("U-last", "u_last"),
            ],
            False,
        ),
        "tab:disc_main": (
            "disc",
            [("exact", "exact"), ("advice-recall", "advice_recall"), ("u-para", "u_para")],
            False,
        ),
        "tab:lawyer_main": (
            "lawyer",
            [("exact", "exact"), ("advice-recall", "advice_recall"), ("u-para", "u_para")],
            False,
        ),
    }
    for label, (js, channels, with_ci) in packs.items():
        data = load_corrected(js)
        rows = []
        for ch_tex, ch_key in channels:
            rows.append(f"\\multirow{{4}}{{*}}{{{ch_tex}}}")
            block = data["channels"][ch_key]
            # bold best AH and best EC among four systems
            ahs = {name: float(block[cfg]["answer_hit@k"]) for name, cfg in CFG.items()}
            ecs = {name: float(block[cfg]["episode_completeness@k"]) for name, cfg in CFG.items()}
            best_ah = max(ahs.values())
            best_ec = max(ecs.values())
            for name, cfg in CFG.items():
                m = block[cfg]
                ah = float(m["answer_hit@k"])
                ec = float(m["episode_completeness@k"])
                nd = float(m["ndcg@k"])
                ah_s = f"\\textbf{{{fmt3(ah)}}}" if abs(ah - best_ah) < 1e-12 else fmt3(ah)
                ec_s = f"\\textbf{{{fmt3(ec)}}}" if abs(ec - best_ec) < 1e-12 else fmt3(ec)
                if with_ci:
                    # binomial-style CI not in corrected JSON; leave --- or omit
                    rows.append(f" & {name} & {ah_s} & {ec_s} & {fmt3(nd)} & --- \\\\")
                else:
                    rows.append(f" & {name} & {ah_s} & {ec_s} & {fmt3(nd)} \\\\")
            rows.append("\\midrule")
        # drop trailing midrule
        if rows and rows[-1] == "\\midrule":
            rows.pop()
        body = "\n".join(rows) + "\n"
        tex = replace_table_body(tex, label, body)
        print("rewrote", label)
    return tex


def rewrite_soft_o2_controls(tex: str) -> str:
    """Main-text Soft O2 controls table from corrected metrics."""
    # Map rows in tab:soft-o2-controls
    specs = [
        ("CAIL / U1", "cail", "u1_exact"),
        ("CAIL / Uk", "cail", "uk_followup"),
        ("CAIL / U-last", "cail", "u_last"),
        ("LegalEp-DISC / u-para", "disc", "u_para"),
        ("LegalEp-DISC / advice", "disc", "advice_recall"),
        ("LegalEp-Lawyer / u-para", "lawyer", "u_para"),
        ("LegalEp-Lawyer / advice", "lawyer", "advice_recall"),
    ]
    rows = []
    for lab, js, ch in specs:
        block = load_corrected(js)["channels"][ch]
        vals = {n: float(block[c]["answer_hit@k"]) for n, c in CFG.items()}
        best = max(vals.values())
        cells = []
        for n in ["FlatIP", "Soft O2", "Hard hydr.", "Shuffled O2"]:
            # table headers use Hard / Shuffled shortened
            key = n
            v = vals[key]
            cells.append(f"\\textbf{{{fmt3(v)}}}" if abs(v - best) < 1e-12 else fmt3(v))
        # tex columns: FlatIP Soft O2 Hard Shuffled
        rows.append(
            f"{lab} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
        )
    body = "\n".join(rows) + "\n"
    # soft-o2-controls uses Hard not Hard hydr. in header — values still map
    # Fix: CFG uses Hard hydr. but table may say Hard — check tex header
    tex = replace_table_body(tex, "tab:soft-o2-controls", body.replace("Hard hydr.", "Hard"))
    # Actually cells already filled; Soft O2 name in CFG is fine. Header is Hard/Shuffled.
    # Rebuild without wrong replace:
    rows = []
    for lab, js, ch in specs:
        block = load_corrected(js)["channels"][ch]
        order = [("FlatIP", "dense_flat"), ("Soft O2", "dense_o2"), ("Hard", "parent_hydrate"), ("Shuffled", "shuffled_o2")]
        vals = [float(block[c]["answer_hit@k"]) for _, c in order]
        best = max(vals)
        cells = [f"\\textbf{{{fmt3(v)}}}" if abs(v - best) < 1e-12 else fmt3(v) for v in vals]
        rows.append(f"{lab} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    body = "\n".join(rows) + "\n"
    tex = replace_table_body(tex, "tab:soft-o2-controls", body)
    print("rewrote tab:soft-o2-controls")
    return tex


def rewrite_ndcg_graded(tex: str) -> str:
    specs = [
        ("CAIL", "Uk", "cail", "uk_followup"),
        ("CAIL", "U-last", "cail", "u_last"),
        ("LegalEp-DISC", "u-para", "disc", "u_para"),
        ("LegalEp-DISC", "advice", "disc", "advice_recall"),
        ("LegalEp-Lawyer", "u-para", "lawyer", "u_para"),
        ("LegalEp-Lawyer", "advice", "lawyer", "advice_recall"),
    ]
    rows = []
    for corpus, ch, js, key in specs:
        block = load_corrected(js)["channels"][key]
        for name, cfg in [("FlatIP", "dense_flat"), ("Soft O2", "dense_o2"), ("Shuffled O2", "shuffled_o2")]:
            m = block[cfg]
            nd = float(m["ndcg@k"])
            inc = float(m["failure_taxonomy"]["incomplete"])
            # bold Soft O2 nDCG if > FlatIP
            nd_s = fmt3(nd)
            if name == "Soft O2" and nd > float(block["dense_flat"]["ndcg@k"]):
                nd_s = f"\\textbf{{{nd_s}}}"
            rows.append(f"{corpus} & {ch} & {name} & {nd_s} & {pct(inc)} \\\\")
    tex = replace_table_body(tex, "tab:ndcg-graded", "\n".join(rows) + "\n")
    print("rewrote tab:ndcg-graded")
    return tex


def rewrite_failure_taxonomy(tex: str) -> str:
    """Four-way taxonomy from corrected_metrics failure_taxonomy (compact Table 4 layout)."""
    specs = [
        ("DISC / u-para", "disc", "u_para"),
        ("DISC / advice", "disc", "advice_recall"),
        ("Lawyer / u-para", "lawyer", "u_para"),
        ("CAIL / Uk", "cail", "uk_followup"),
    ]
    systems = [
        ("FlatIP", "dense_flat"),
        ("Soft O2", "dense_o2"),
        ("Hard", "parent_hydrate"),
        ("Shuf.\\ O2", "shuffled_o2"),
    ]
    rows = []
    for lab, js, ch in specs:
        block = load_corrected(js)["channels"][ch]
        focus = ["dense_flat", "dense_o2", "shuffled_o2"]
        completes = {c: float(block[c]["failure_taxonomy"]["complete"]) for c in focus}
        incompletes = {c: float(block[c]["failure_taxonomy"]["incomplete"]) for c in focus}
        best_c = max(completes.values())
        best_i = min(incompletes.values())
        for name, cfg in systems:
            ft = block[cfg]["failure_taxonomy"]
            c, inc, ao, miss = (
                float(ft["complete"]),
                float(ft["incomplete"]),
                float(ft["answer_only"]),
                float(ft["session_miss"]),
            )
            s = c + inc + ao + miss
            if abs(s - 1.0) > 0.002:
                raise SystemExit(f"taxonomy sum {s} for {lab}/{name}")
            # compact numeric cells (caption states values are %)
            def cell(x: float) -> str:
                return f"{100.0 * x:.1f}"

            c_s, i_s = cell(c), cell(inc)
            if cfg in focus:
                if abs(c - best_c) < 1e-12:
                    c_s = f"\\textbf{{{c_s}}}"
                if abs(inc - best_i) < 1e-12:
                    i_s = f"\\textbf{{{i_s}}}"
            rows.append(
                f"{lab} & {name} & {c_s} & {i_s} & {cell(ao)} & {cell(miss)} \\\\"
            )
        rows.append("\\midrule")
    if rows[-1] == "\\midrule":
        rows.pop()
    body = "\n".join(rows) + "\n"
    tex = replace_table_body(tex, "tab:rq1-failure", body)
    print("rewrote tab:rq1-failure")
    return tex


def rewrite_holm(tex: str) -> str:
    """Recompute Soft O2 vs FlatIP primary family from unified-rebuild per_query_ah."""
    rows_out = []
    pvals = []
    metas = []
    paired_export = {"family": "SoftO2_vs_FlatIP_primary9", "source": "corrected_metrics_*.json per_query_ah", "channels": {}}
    for lab, js, ch in PRIMARY_FAMILY:
        data = load_corrected(js)
        block = data["channels"][ch]
        soft = block["dense_o2"]
        flat = block["dense_flat"]
        if "per_query_ah" not in soft or "per_query_ah" not in flat:
            raise SystemExit(
                f"missing per_query_ah for {lab} in corrected_metrics_{js}.json; "
                "re-run paper/scripts/recompute_corrected_metrics.py with --reuse_index"
            )
        a = soft["per_query_ah"]
        b = flat["per_query_ah"]
        if len(a) != len(b):
            raise SystemExit(f"paired length mismatch for {lab}: {len(a)} vs {len(b)}")
        # Delta AH must match table point estimates (rounding aside).
        delta_table = float(soft["answer_hit@k"]) - float(flat["answer_hit@k"])
        delta_vec = float(sum(a) / len(a) - sum(b) / len(b))
        if abs(delta_table - delta_vec) > 1e-9:
            raise SystemExit(f"internal AH inconsistency for {lab}: {delta_table} vs {delta_vec}")
        rep = paired_report("Soft O2", a, "FlatIP", b)
        mid_p = float(rep["mcnemar"]["mid_p"])
        delta = float(rep["mcnemar"]["delta_mean"])
        if abs(delta - delta_table) > 5e-4:
            raise SystemExit(f"McNemar delta {delta} != table delta {delta_table} for {lab}")
        pvals.append(mid_p)
        metas.append((lab, delta_table, mid_p))
        paired_export["channels"][lab] = {
            "json": f"corrected_metrics_{js}.json",
            "channel": ch,
            "n": len(a),
            "query_ids": soft.get("query_ids") or flat.get("query_ids"),
            "soft_o2_ah": a,
            "flatip_ah": b,
            "delta_ah": delta_table,
            "mid_p": mid_p,
        }
    holm = holm_adjust(pvals)
    for (lab, delta, mid_p), h in zip(metas, holm):
        rows_out.append(
            f"{lab} & O2 vs FlatIP & {delta:+.3f} & {mid_p:.2e} & {h:.2e} \\\\"
            if mid_p < 1e-3
            else f"{lab} & O2 vs FlatIP & {delta:+.3f} & {mid_p:.4g} & {h:.4g} \\\\"
        )
        paired_export["channels"][lab]["holm_p"] = h
    body = "\n".join(rows_out) + "\n"
    tex = replace_table_body(tex, "tab:sig", body)
    print("rewrote tab:sig Holm family")
    art = {
        "family": "SoftO2_vs_FlatIP_primary9",
        "source": "paper/ipm/figures/corrected_metrics_*.json per_query_ah (unified FlatIP rebuild)",
        "rows": [
            {"label": lab, "delta_ah": d, "mid_p": p, "holm_p": h}
            for (lab, d, p), h in zip(metas, holm)
        ],
    }
    out = FIG / "holm_primary_family.json"
    out.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print("wrote", out)
    paired_path = FIG / "paired_ah_primary_family.json"
    paired_path.write_text(json.dumps(paired_export, indent=2), encoding="utf-8")
    print("wrote", paired_path)
    return tex


def main():
    tex = TEX.read_text(encoding="utf-8")
    tex = rewrite_appendix_grids(tex)
    tex = rewrite_soft_o2_controls(tex)
    tex = rewrite_ndcg_graded(tex)
    tex = rewrite_failure_taxonomy(tex)
    tex = rewrite_holm(tex)
    TEX.write_text(tex, encoding="utf-8")
    print("updated", TEX)


if __name__ == "__main__":
    main()
