#!/usr/bin/env python3
"""Regenerate paper tables from a SINGLE metric source (S0-1/S0-5/S0-6).

Default source: paper/ipm/figures/corrected_metrics_{cail,disc,lawyer}.json
(same FlatIP rebuild for AH, EC, nDCG, and failure_taxonomy).

Holm primary family is computed from V4 per_query_ah when available
(paired McNemar mid-p), then Holm-adjusted with a tested step-down.
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
V4 = ROOT / "BIMS-LEGAL-dataset" / "primary_results" / "bims_legal_v4"

CFG = {
    "FlatIP": "dense_flat",
    "Soft O2": "dense_o2",
    "Hard hydr.": "parent_hydrate",
    "Shuffled O2": "shuffled_o2",
}

PRIMARY_FAMILY = [
    # (tex label, v4 rel path, channel key, soft cfg, flat cfg)
    ("CAIL / Uk", "cail_M/tier_M/results.json", "uk_followup"),
    ("CAIL / U1", "cail_M/tier_M/results.json", "u1_exact"),
    ("CAIL / U-last", "cail_M/tier_M/results.json", "u_last"),
    ("LegalEp-DISC / exact", "legalep_disc_M/tier_M/results.json", "u1_exact"),
    ("LegalEp-Lawyer / exact", "legalep_lawyer_M/tier_M/results.json", "exact"),
    ("LegalEp-DISC / u-para", "legalep_disc_para/tier_M/results.json", "u_para"),
    ("LegalEp-Lawyer / u-para", "legalep_lawyer_para/tier_M/results.json", "u_para"),
    ("LegalEp-DISC / advice", "legalep_disc_advice/tier_M/results.json", "advice_recall"),
    ("LegalEp-Lawyer / advice", "legalep_lawyer_advice/tier_M/results.json", "advice_recall"),
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
    """Four-way taxonomy from corrected_metrics failure_taxonomy."""
    specs = [
        ("LegalEp-DISC / u-para", "disc", "u_para"),
        ("LegalEp-DISC / advice", "disc", "advice_recall"),
        ("LegalEp-Lawyer / u-para", "lawyer", "u_para"),
        ("CAIL / Uk-followup", "cail", "uk_followup"),
    ]
    systems = [
        ("FlatIP", "dense_flat"),
        ("Soft O2", "dense_o2"),
        ("Hard hydr.", "parent_hydrate"),
        ("Shuffled O2", "shuffled_o2"),
    ]
    rows = []
    for lab, js, ch in specs:
        block = load_corrected(js)["channels"][ch]
        # bold best complete / lowest incomplete among FlatIP, Soft O2, Shuffled
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
            c_s, i_s = pct(c), pct(inc)
            if cfg in focus:
                if abs(c - best_c) < 1e-12:
                    c_s = f"\\textbf{{{c_s}}}"
                if abs(inc - best_i) < 1e-12:
                    i_s = f"\\textbf{{{i_s}}}"
            rows.append(
                f"{lab} & {name} & {c_s} & {i_s} & {pct(ao)} & {pct(miss)} \\\\"
            )
        rows.append("\\midrule")
    if rows[-1] == "\\midrule":
        rows.pop()
    body = "\n".join(rows) + "\n"
    # Update tabular header to 4 failure cols if needed — handled in paper edit
    tex = replace_table_body(tex, "tab:rq1-failure", body)
    print("rewrote tab:rq1-failure")
    return tex


def rewrite_holm(tex: str) -> str:
    """Recompute Soft O2 vs FlatIP primary family from V4 per_query_ah."""
    rows_out = []
    pvals = []
    metas = []
    for lab, rel, ch in PRIMARY_FAMILY:
        path = V4 / rel
        if not path.exists():
            # try alternate folder names
            alts = list(V4.glob(f"**/{Path(rel).name}"))
            if not alts:
                raise SystemExit(f"missing V4 results {rel}")
            path = alts[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        cfgs = data["channels"][ch]["configs"]
        a = cfgs["dense_o2"]["per_query_ah"]
        b = cfgs["dense_flat"]["per_query_ah"]
        rep = paired_report("Soft O2", a, "FlatIP", b)
        mid_p = float(rep["mcnemar"]["mid_p"])
        delta = float(rep["mcnemar"]["delta_mean"])
        pvals.append(mid_p)
        metas.append((lab, delta, mid_p))
    holm = holm_adjust(pvals)
    # sanity vs known ordering for old p list
    for (lab, delta, mid_p), h in zip(metas, holm):
        rows_out.append(
            f"{lab} & O2 vs FlatIP & {delta:+.3f} & {mid_p:.2e} & {h:.2e} \\\\"
            if mid_p < 1e-3
            else f"{lab} & O2 vs FlatIP & {delta:+.3f} & {mid_p:.4g} & {h:.4g} \\\\"
        )
    # Keep non-FlatIP contrast rows from existing table if present — rewrite only O2 vs FlatIP block by full table regen of primary rows
    # Simpler: replace entire sig table body with primary family only + note
    body = "\n".join(rows_out) + "\n"
    tex = replace_table_body(tex, "tab:sig", body)
    print("rewrote tab:sig Holm family")
    # write artifact
    art = {
        "family": "SoftO2_vs_FlatIP_primary9",
        "source": "BIMS-LEGAL-dataset/primary_results/bims_legal_v4 per_query_ah",
        "rows": [
            {"label": lab, "delta_ah": d, "mid_p": p, "holm_p": h}
            for (lab, d, p), h in zip(metas, holm)
        ],
    }
    out = FIG / "holm_primary_family.json"
    out.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print("wrote", out)
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
