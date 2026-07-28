#!/usr/bin/env python3
"""Fill tab:csce Mix fair results into ipm-article.tex."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "paper" / "ipm" / "ipm-article.tex"
ROOT = REPO / "results" / "bims_legal_csce_mix"

CORPORA = [
    ("legalep_disc", "advice_recall", "LegalEp-DISC Mix"),
    ("legalep_lawyer", "advice_recall", "LegalEp-Lawyer Mix"),
    ("legalmem_mt", "uk_followup", "LegalMem-MT Mix"),
]


def fmt(x, nd=3):
    if x is None:
        return "---"
    return f"{float(x):.{nd}f}"


def load(corpus, channel):
    path = ROOT / corpus / "tier_M" / "results.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    cfgs = ((blob.get("channels") or {}).get(channel) or {}).get("configs") or {}
    comps = ((blob.get("comparisons") or {}).get(channel) or {})
    return blob, cfgs, comps


def build_table():
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Fair Mix protocol (same store, unrestricted dense; AH@10). "
        "Gold is a $70\\%$/$30\\%$ mix of cross-session Sa/Sb splits and intact same-session episodes. "
        "Soft~O2-C denotes gated cluster binding ($\\texttt{cluster\\_o2}$); Hybrid is Soft~O2+Soft~O2-C. "
        "Boldface: best AH within each corpus.}",
        "\\label{tab:csce}",
        "\\compacttab\\footnotesize",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "\\textbf{Corpus} & \\textbf{System} & \\textbf{AH} & \\textbf{AH$_{\\mathrm{cross}}$} & "
        "\\textbf{AH$_{\\mathrm{same}}$} & \\textbf{mid-$p$ vs Soft~O2} \\\\",
        "\\midrule",
    ]
    for i, (corpus, channel, label) in enumerate(CORPORA):
        _, cfgs, comps = load(corpus, channel)
        # Prefer hybrid_xsess if it beats sess_o2; else cluster_o2
        flat = cfgs["ep_flat"]
        sess = cfgs["sess_o2"]
        clus = cfgs["cluster_o2"]
        hyb = cfgs.get("hybrid_xsess") or clus
        cmp_c = (comps.get("cluster_o2_vs_sess_o2") or {}).get("mcnemar") or {}
        cmp_h = (comps.get("hybrid_xsess_vs_sess_o2") or {}).get("mcnemar") or {}
        # Choose winner vs Soft O2 for bold + p
        use_hyb = float(hyb["answer_hit@k"]) >= float(clus["answer_hit@k"])
        best_name = "Hybrid" if use_hyb else "Soft O2-C"
        best = hyb if use_hyb else clus
        mid = (cmp_h if use_hyb else cmp_c).get("mid_p")
        mid_s = f"{mid:.2e}" if mid is not None and mid < 1e-3 else (fmt(mid, 4) if mid is not None else "---")

        def scope(cfg, key):
            return ((cfg.get("ah_by_scope") or {}).get(key) or {}).get("answer_hit@k")

        rows = [
            ("FlatIP", flat, "---"),
            ("Soft O2", sess, "---"),
            ("Soft O2-C", clus, fmt(cmp_c.get("mid_p"), 4) if cmp_c.get("mid_p") is not None else "---"),
            ("Hybrid", hyb, mid_s if use_hyb else (fmt(cmp_h.get("mid_p"), 4) if cmp_h.get("mid_p") is not None else "---")),
        ]
        lines.append(f"\\multirow{{4}}{{*}}{{{label}}}")
        for j, (name, cfg, p) in enumerate(rows):
            ah = cfg["answer_hit@k"]
            bold = name == best_name and float(ah) >= float(sess["answer_hit@k"])
            ah_s = f"\\textbf{{{fmt(ah)}}}" if bold else fmt(ah)
            lines.append(
                f" & {name} & {ah_s} & {fmt(scope(cfg,'cross_session'))} & "
                f"{fmt(scope(cfg,'same_session'))} & {p} \\\\"
            )
        if i < len(CORPORA) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    table = build_table()
    tex = TEX.read_text(encoding="utf-8")
    pat = re.compile(
        r"\\begin\{table\}\[htbp\]\s*\\centering\s*"
        r"\\caption\{(?:Fair Mix|Split-Episode CSCE).*?\\end\{table\}",
        re.S,
    )
    if not pat.search(tex):
        raise SystemExit("tab:csce block not found")
    TEX.write_text(pat.sub(lambda _m: table, tex, count=1), encoding="utf-8")
    print(f"updated {TEX}")
    print(table)


if __name__ == "__main__":
    main()
