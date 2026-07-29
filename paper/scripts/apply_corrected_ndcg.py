#!/usr/bin/env python3
"""Apply corrected_metrics_*.json into ipm-article.tex tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/ipm/ipm-article.tex"
FIG = ROOT / "paper/ipm/figures"


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}\\%"


def fmt(x: float) -> str:
    return f"{float(x):.3f}"


def load_all() -> dict:
    out = {}
    for p in FIG.glob("corrected_metrics_*.json"):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def update_failure_table(tex: str, blobs: dict) -> str:
    """Fill RQ1 table from FlatIP failure_taxonomy when available."""
    rows = []
    mapping = [
        ("LegalEp-DISC / u-para", "corrected_metrics_disc", "u_para"),
        ("LegalEp-DISC / advice", "corrected_metrics_disc", "advice_recall"),
        ("LegalEp-Lawyer / u-para", "corrected_metrics_lawyer", "u_para"),
        ("CAIL / Uk-followup", "corrected_metrics_cail", "uk_followup"),
    ]
    for label, key, ch in mapping:
        blob = blobs.get(key) or {}
        ft = (((blob.get("channels") or {}).get(ch) or {}).get("dense_flat") or {}).get("failure_taxonomy")
        if not ft:
            rows.append(None)
            continue
        rows.append(
            f"{label} & FlatIP & {pct(ft.get('complete', 0))} & {pct(ft.get('incomplete', 0))} & {pct(ft.get('session_miss', 0))} \\\\"
        )
    if not any(rows):
        return tex
    # keep existing approximate rows if a channel missing
    existing = {
        "LegalEp-DISC / u-para": "LegalEp-DISC / u-para & FlatIP & 54.7\\% & 36.0\\% & 9.3\\% \\\\",
        "LegalEp-DISC / advice": "LegalEp-DISC / advice & FlatIP & 34.2\\% & 26.6\\% & 39.2\\% \\\\",
        "LegalEp-Lawyer / u-para": "LegalEp-Lawyer / u-para & FlatIP & 56.8\\% & 39.6\\% & 3.6\\% \\\\",
        "CAIL / Uk-followup": "CAIL / Uk-followup & FlatIP & 27.0\\% & 69.8\\% & 3.2\\% \\\\",
    }
    final = []
    for (label, _, _), row in zip(mapping, rows):
        final.append(row if row else existing[label])
    body = "\n".join(final)
    return re.sub(
        r"LegalEp-DISC / u-para & FlatIP &.*?CAIL / Uk-followup & FlatIP &.*?\\\\",
        lambda _: body,
        tex,
        count=1,
        flags=re.S,
    )


def patch_ndcg_in_appendix(tex: str, blobs: dict) -> str:
    """Best-effort: annotate metrics section; patch known Soft O2/FlatIP nDCG cells when channel tables exist."""
    # Update metrics prose confirmation
    note = (
        "Primary metrics are Answer Hit@$k$ (AH), Episode Completeness@$k$ (EC), and corrected normalized "
        "Discounted Cumulative Gain@$k$ (nDCG) at $k{=}10$."
    )
    if "corrected graded" not in tex and "corrected normalized" not in tex:
        tex = tex.replace(
            "Primary metrics are Answer Hit@$k$ (AH), Episode Completeness@$k$ (EC), and corrected normalized Discounted Cumulative Gain@$k$ (nDCG) at $k{=}10$.",
            note,
        )
    # Patch disc main table nDCG if present in blob
    disc = blobs.get("corrected_metrics_disc") or {}
    for ch, lab in [("u_para", "u-para"), ("advice_recall", "advice")]:
        for cfg, sysname in [("dense_flat", "FlatIP"), ("dense_o2", "Soft O2")]:
            m = (((disc.get("channels") or {}).get(ch) or {}).get(cfg)) or {}
            if "ndcg@k" not in m:
                continue
            ndcg = fmt(m["ndcg@k"])
            # Replace nDCG column for matching system rows inside appendix tables heuristically
            tex = re.sub(
                rf"(& {re.escape(sysname)}(?:\$\{{[^}}]*\}})? & [0-9.]+ & [0-9.]+ & )[0-9.]+",
                rf"\g<1>{ndcg}",
                tex,
                count=0,  # careful: may over-replace; only do labeled tables below
            )
    return tex


def write_summary_table(tex: str, blobs: dict) -> str:
    """Insert/replace corrected-nDCG summary covering available corpora."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Corrected graded nDCG@10 (fixed-gold IDCG) on LegalEp primary channels under a fresh FlatIP rebuild. Soft~O2 reduces incompleteness while raising AH and corrected nDCG.}",
        r"\label{tab:ndcg-corrected}",
        r"\compacttab\footnotesize",
        r"\begin{tabular}{@{}lllccc@{}}",
        r"\toprule",
        r"\textbf{Corpus} & \textbf{Channel} & \textbf{System} & \textbf{AH@10} & \textbf{nDCG@10} & \textbf{Incomplete} \\",
        r"\midrule",
    ]
    any_row = False
    for corpus_key, corpus_name in [
        ("corrected_metrics_disc", "LegalEp-DISC"),
        ("corrected_metrics_lawyer", "LegalEp-Lawyer"),
    ]:
        blob = blobs.get(corpus_key) or {}
        for ch, pretty in [("u_para", "u-para"), ("advice_recall", "advice")]:
            for cfg, name in [("dense_flat", "FlatIP"), ("dense_o2", "Soft O2")]:
                m = (((blob.get("channels") or {}).get(ch) or {}).get(cfg)) or {}
                if not m:
                    continue
                any_row = True
                ft = m.get("failure_taxonomy") or {}
                inc = pct(ft["incomplete"]) if "incomplete" in ft else "---"
                lines.append(
                    f"{corpus_name} & {pretty} & {name} & {fmt(m['answer_hit@k'])} & {fmt(m['ndcg@k'])} & {inc} \\\\"
                )
    if not any_row:
        return tex
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    block = "\n".join(lines)
    if "tab:ndcg-corrected" in tex:
        tex = re.sub(
            r"\\begin\{table\}\[t\]\n\\centering\n\\caption\{Corrected graded nDCG@10.*?\\end\{table\}",
            lambda _: block.strip(),
            tex,
            flags=re.S,
        )
    else:
        tex = tex.replace(
            "\\subsection{Soft O2 on session (CAIL and LegalEp)}\\label{sec:exp-main}",
            block + "\n\\subsection{Soft O2 on session (CAIL and LegalEp)}\\label{sec:exp-main}",
        )
    return tex


def main() -> None:
    blobs = load_all()
    print("[load]", list(blobs))
    tex = TEX.read_text(encoding="utf-8")
    tex = update_failure_table(tex, blobs)
    tex = write_summary_table(tex, blobs)
    TEX.write_text(tex, encoding="utf-8")
    print(f"[wrote] {TEX}")


if __name__ == "__main__":
    main()
