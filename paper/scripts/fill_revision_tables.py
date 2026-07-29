#!/usr/bin/env python3
"""Fill RQ1 failure taxonomy and Holm-corrected p-values in ipm-article.tex."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
V4 = ROOT / "BIMS-LEGAL-dataset" / "primary_results" / "bims_legal_v4"

FAILURE_SOURCES = [
    ("LegalEp-DISC / u-para", "legalep_disc_para/tier_M/results.json", "u_para"),
    ("LegalEp-DISC / advice", "legalep_disc_advice/tier_M/results.json", "advice_recall"),
    ("LegalEp-Lawyer / u-para", "legalep_lawyer_para/tier_M/results.json", "u_para"),
    ("CAIL / Uk-followup", "cail_M/tier_M/results.json", "uk_followup"),
]

SIG_ROWS = [
    ("CAIL / Uk", "O2 vs FlatIP", 8.99e-48),
    ("CAIL / U1", "O2 vs FlatIP", 5.19e-18),
    ("CAIL / U-last", "O2 vs FlatIP", 5.51e-65),
    ("LegalEp-DISC / exact", "O2 vs FlatIP", 0.934),
    ("LegalEp-Lawyer / exact", "O2 vs FlatIP", 0.0239),
    ("LegalEp-DISC / u-para", "O2 vs FlatIP", 7.95e-05),
    ("LegalEp-Lawyer / u-para", "O2 vs FlatIP", 2.10e-15),
    ("LegalEp-DISC / advice", "O2 vs FlatIP", 1.25e-04),
    ("LegalEp-Lawyer / advice", "O2 vs FlatIP", 1.18e-10),
]


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def approx_taxonomy(m: dict) -> tuple[str, str, str]:
    ah = float(m["answer_hit@k"])
    sh = float(m["session_hit@k"])
    complete = ah
    incomplete = max(0.0, sh - ah)
    miss = max(0.0, 1.0 - sh)
    return pct(complete), pct(incomplete), pct(miss)


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvals[idx] * (m - rank)
        running = max(running, raw)
        adj[idx] = min(1.0, running)
    return adj


def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4g}"


def fill_failure_table(tex: str) -> str:
    rows = []
    for label, rel, ch in FAILURE_SOURCES:
        data = json.loads((V4 / rel).read_text(encoding="utf-8"))
        m = data["channels"][ch]["configs"]["dense_flat"]
        c, inc, miss = approx_taxonomy(m)
        rows.append(f"{label} & FlatIP & {c} & {inc} & {miss} \\\\")
    body = "\n".join(rows)
    return re.sub(
        r"(\\midrule\n)(?:LegalEp-DISC / u-para.*?CAIL / Uk-followup & FlatIP & --- & --- & --- \\\\)",
        r"\1" + body,
        tex,
        flags=re.S,
    )


def fill_sig_table(tex: str) -> str:
    pvals = [p for _, _, p in SIG_ROWS]
    holm = holm_adjust(pvals)
    lines = []
    holm_map = {(a, b): h for (a, b, _), h in zip(SIG_ROWS, holm)}
    for corpus, contrast, p in SIG_ROWS:
        holm_p = holm_map.get((corpus, contrast))
        holm_cell = fmt_p(holm_p) if holm_p is not None and contrast == "O2 vs FlatIP" else "---"
        # keep existing delta from tex when possible
        m = re.search(
            rf"{re.escape(corpus)} & {re.escape(contrast)} & ([^&]+) & ([^\\]+)",
            tex,
        )
        delta = m.group(1).strip() if m else "---"
        raw = fmt_p(p) if contrast == "O2 vs FlatIP" else (m.group(2).strip() if m else "---")
        if contrast != "O2 vs FlatIP" and m:
            lines.append(f"{corpus} & {contrast} & {delta} & {m.group(2).strip()} & --- \\\\")
            continue
        if contrast == "O2 vs FlatIP":
            lines.append(f"{corpus} & {contrast} & {delta} & {raw} & {holm_cell} \\\\")
    # Rebuild only O2 vs FlatIP primary family with holm; keep other rows from tex
    new_body = []
    for line in tex.splitlines():
        if re.match(r"^CAIL / Uk & O2 vs FlatIP", line):
            break
        if "\\midrule" in line and "tab:sig" in tex[max(0, tex.find(line) - 400) : tex.find(line)]:
            new_body.append(line)
            break
        new_body.append(line)
    # simpler: replace header and all sig rows
    header = (
        "\\caption{McNemar mid-$p$ on paired Answer Hit (Soft~O2 vs FlatIP / hard hydration / FlatIP+CE). "
        "Holm-adjusted $p$ reported for the nine primary Soft~O2 vs FlatIP contrasts.}\n"
        "\\label{tab:sig}\n"
        "\\compacttab\\footnotesize\n"
        "\\begin{tabular}{@{}llccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Corpus / channel} & \\textbf{Contrast} & \\textbf{$\\Delta$AH} & \\textbf{mid-$p$} & \\textbf{Holm $p$} \\\\"
    )
    tex = re.sub(
        r"\\caption\{McNemar mid-\$p\$ on paired Answer Hit \(Soft~O2 vs FlatIP / hard hydration / FlatIP\+CE\)\.\}.*?\\midrule",
        header + "\n\\midrule",
        tex,
        flags=re.S,
        count=1,
    )
    # extract existing body between midrule and bottomrule
    m = re.search(r"\\label\{tab:sig\}.*?\\midrule\n(.*?)\\bottomrule", tex, re.S)
    if not m:
        return tex
    old_lines = m.group(1).strip().splitlines()
    out_lines = []
    holm_idx = 0
    for line in old_lines:
        if "O2 vs FlatIP" in line and holm_idx < len(holm):
            parts = [p.strip() for p in line.split("&")]
            if len(parts) >= 4:
                parts = parts[:4] + [fmt_p(holm[holm_idx])]
                holm_idx += 1
                out_lines.append(" & ".join(parts) + " \\\\")
                continue
        if len(line.split("&")) == 4:
            out_lines.append(line.rstrip("\\") + " & --- \\\\")
        else:
            out_lines.append(line)
    return re.sub(
        r"(\\label\{tab:sig\}.*?\\midrule\n)(.*?)(\\bottomrule)",
        r"\1" + "\n".join(out_lines) + r"\n\3",
        tex,
        flags=re.S,
        count=1,
    )


def apply_corrected_ndcg(tex: str) -> str:
    corrected = ROOT / "paper/ipm/figures/corrected_metrics_disc.json"
    if not corrected.exists():
        print("[skip] no corrected_metrics_disc.json yet")
        return tex
    data = json.loads(corrected.read_text(encoding="utf-8"))
    # placeholder for future table overlay
    print("[info] corrected metrics available:", list(data.get("channels", {}).keys()))
    return tex


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    tex = fill_failure_table(tex)
    tex = fill_sig_table(tex)
    tex = apply_corrected_ndcg(tex)
    TEX.write_text(tex, encoding="utf-8")
    print(f"[wrote] {TEX}")


if __name__ == "__main__":
    main()
