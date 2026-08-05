#!/usr/bin/env python3
"""Fill tab:csce from post-gate Mix results into root paper/ipm/ipm-article.tex.

Default results root:
  BIMS-LEGAL-dataset/primary_results/bims_legal_csce_mix
Override with --results.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "paper" / "ipm" / "ipm-article.tex"
DEFAULT_ROOT = REPO / "BIMS-LEGAL-dataset" / "primary_results" / "bims_legal_csce_mix"

CORPORA = [
    ("legalep_disc", "advice_recall", "LegalEp-DISC Mix"),
    ("legalep_lawyer", "advice_recall", "LegalEp-Lawyer Mix"),
    ("legalmem_mt", "uk_followup", "LegalMem-MT Mix"),
]


def fmt(x, nd=3):
    if x is None:
        return "---"
    return f"{float(x):.{nd}f}"


def stars(mid_p: float | None) -> str:
    if mid_p is None:
        return ""
    if mid_p < 0.01:
        return "${}^{**}$"
    if mid_p < 0.05:
        return "${}^{*}$"
    return ""


def load(root: Path, corpus: str, channel: str):
    path = root / corpus / "tier_M" / "results.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    ch = ((blob.get("channels") or {}).get(channel) or {})
    cfgs = ch.get("configs") or {}
    comps = (blob.get("comparisons") or {}).get(channel) or ch.get("comparisons") or {}
    return blob, cfgs, comps


def scope(cfg, key):
    return ((cfg.get("ah_by_scope") or {}).get(key) or {}).get("answer_hit@k")


def build_table(root: Path, exploratory: bool = False) -> str:
    dagger = "$^{\\dagger}$" if exploratory else ""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Mix evaluation of Soft~O2-C and gated Hybrid (same store, unrestricted dense; AH@10). "
        "Gold is $70$\\% cross-session Sa/Sb + $30$\\% intact same-session. "
        "Boldface: best overall AH and best cross-session AH within each corpus. "
        "Significance vs Soft~O2: ${}^{*}\\,p{<}0.05$, ${}^{**}\\,p{<}0.01$"
        + (" (Hybrid rows marked exploratory)." if exploratory else ".")
        + "}",
        "\\label{tab:csce}",
        "\\compacttab\\footnotesize",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "\\textbf{Corpus} & \\textbf{System} & \\textbf{AH} & \\textbf{AH$_{\\mathrm{cross}}$} & "
        "\\textbf{AH$_{\\mathrm{same}}$} & \\textbf{mid-$p$ vs Soft~O2} \\\\",
        "\\midrule",
    ]
    for i, (corpus, channel, label) in enumerate(CORPORA):
        _, cfgs, comps = load(root, corpus, channel)
        flat, sess, clus, hyb = cfgs["ep_flat"], cfgs["sess_o2"], cfgs["cluster_o2"], cfgs["hybrid_xsess"]
        cmp_c = (comps.get("cluster_o2_vs_sess_o2") or {}).get("mcnemar") or {}
        cmp_h = (comps.get("hybrid_xsess_vs_sess_o2") or {}).get("mcnemar") or {}

        # Bold: best overall AH among four; also bold best AH_cross among four.
        ahs = {
            "FlatIP": float(flat["answer_hit@k"]),
            "Soft O2": float(sess["answer_hit@k"]),
            "Soft O2-C": float(clus["answer_hit@k"]),
            "Hybrid": float(hyb["answer_hit@k"]),
        }
        cross = {
            "FlatIP": float(scope(flat, "cross_session") or 0),
            "Soft O2": float(scope(sess, "cross_session") or 0),
            "Soft O2-C": float(scope(clus, "cross_session") or 0),
            "Hybrid": float(scope(hyb, "cross_session") or 0),
        }
        best_ah = max(ahs, key=ahs.get)
        best_cross = max(cross, key=cross.get)

        def cell(name, val, which):
            winners = best_ah if which == "ah" else best_cross
            s = fmt(val)
            return f"\\textbf{{{s}}}" if name == winners else s

        def pcell(cmpd):
            mid = cmpd.get("mid_p")
            if mid is None:
                return "---"
            return f"{fmt(mid, 4)}{stars(mid)}"

        rows = [
            ("FlatIP", flat, "---", False),
            ("Soft O2", sess, "---", False),
            ("Soft O2-C", clus, pcell(cmp_c), False),
            (f"Hybrid{dagger}", hyb, pcell(cmp_h), True),
        ]
        lines.append(f"\\multirow{{4}}{{*}}{{{label}}}")
        for name, cfg, p, is_hyb in rows:
            key = "Hybrid" if is_hyb else name
            lines.append(
                f" & {name} & {cell(key, cfg['answer_hit@k'], 'ah')} & "
                f"{cell(key, scope(cfg,'cross_session'), 'cross')} & "
                f"{fmt(scope(cfg,'same_session'))} & {p} \\\\"
            )
        if i < len(CORPORA) - 1:
            lines.append("\\midrule")
    foot = (
        "{\\footnotesize Soft~O2-C $\\mathrm{AH}_{\\mathrm{cross}}$ is the mechanism contrast. "
        + (
            "$^{\\dagger}$Hybrid rows are exploratory measurements from a pre-fix gate implementation and are not used for primary claims. "
            if exploratory
            else "Hybrid uses the post-gate direct-dense trigger (direct hits recorded before Soft~O2 session expansion). "
        )
        + "Cluster co-membership of Sa/Sb pairs follows Section~\\ref{sec:csce-protocol}.}"
    )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", foot]
    return "\n".join(lines)


def replace_csce_block(tex: str, new_block: str) -> str:
    """Replace only the labeled Mix table and its immediate footnote."""
    # Anchor on \label{tab:csce} so we never eat surrounding sections.
    lab = "\\label{tab:csce}"
    i = tex.find(lab)
    if i < 0:
        raise SystemExit("tab:csce label not found")
    start = tex.rfind("\\begin{table}", 0, i)
    end_table = tex.find("\\end{table}", i)
    if start < 0 or end_table < 0:
        raise SystemExit("tab:csce table bounds not found")
    end = end_table + len("\\end{table}")
    # Optional single footnote paragraph immediately after the table.
    rest = tex[end:]
    m = re.match(r"\n\{\\footnotesize Soft~O2-C.*?\\}\n?", rest, flags=re.S)
    if not m:
        m = re.match(r"\n\{\\footnotesize Hybrid uses the post-gate.*?\\}\n?", rest, flags=re.S)
    if m:
        end += m.end()
    return tex[:start] + new_block + tex[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    table = build_table(args.results, exploratory=args.exploratory)
    if args.dry_run:
        print(table)
        return
    tex = TEX.read_text(encoding="utf-8")
    TEX.write_text(replace_csce_block(tex, table), encoding="utf-8")
    print(f"updated {TEX} from {args.results}")


if __name__ == "__main__":
    main()
