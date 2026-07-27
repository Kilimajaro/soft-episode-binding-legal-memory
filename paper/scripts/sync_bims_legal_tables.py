#!/usr/bin/env python3
"""Sync completed BIMS-LEGAL V4 JSON results into paper/ipm/ipm-article.tex tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
RES = ROOT / "results" / "bims_legal_v4"


def fmt(x, nd=3):
    if x is None:
        return "---"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def load_results(*candidates: Path):
    for p in candidates:
        if p and p.exists():
            return json.loads(p.read_text(encoding="utf-8")), p
    return None, None


# Main tables: turn-level systems only. joint_qa is an atomic-index bound (prose only).
SYS_ORDER = [
    ("dense_flat", "FlatIP"),
    ("dense_o2", "Soft O2"),
    ("parent_hydrate", "Hard hydr."),
    ("session_max", "Session-max"),
    ("shuffled_o2", "Shuffled O2"),
]


def cfg_metrics(ch_block, cfg):
    m = (ch_block or {}).get("configs", {}).get(cfg) or {}
    ci = m.get("ah_ci") or {}
    ci_s = "---"
    if "ci_low" in ci and "ci_high" in ci:
        ci_s = f"[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"
    return {
        "AH": m.get("answer_hit@k"),
        "EC": m.get("episode_completeness@k"),
        "nDCG": m.get("ndcg@k"),
        "CI": ci_s,
        "SH": m.get("session_hit@k"),
    }


def bold_best_rows(rows, key_idx):
    """rows: list of list[str]; bold max numeric in column key_idx within block."""
    vals = []
    for r in rows:
        try:
            vals.append(float(r[key_idx]))
        except Exception:
            vals.append(float("-inf"))
    if not vals or max(vals) == float("-inf"):
        return rows
    best = max(vals)
    out = []
    for r, v in zip(rows, vals):
        rr = list(r)
        if v == best and v != float("-inf"):
            rr[key_idx] = f"\\textbf{{{rr[key_idx]}}}"
        out.append(rr)
    return out


def render_channel_block(channel_label, ch_data, systems=SYS_ORDER, with_ci=True):
    rows = []
    for cfg, label in systems:
        m = cfg_metrics(ch_data, cfg)
        if m["AH"] is None:
            continue
        row = [label, fmt(m["AH"]), fmt(m["EC"]), fmt(m["nDCG"]) if m["nDCG"] is not None else "---"]
        if with_ci:
            row.append(m["CI"])
        rows.append(row)
    if not rows:
        return None
    rows = bold_best_rows(rows, 1)  # AH column
    lines = []
    for i, r in enumerate(rows):
        if i == 0:
            lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{channel_label}}}\n & " + " & ".join(r) + " \\\\")
        else:
            lines.append(" & " + " & ".join(r) + " \\\\")
    return "\n".join(lines)


def replace_table_body(tex: str, label: str, new_body: str) -> str:
    """Replace tabular body between \label{label} ... \begin{tabular}... and \end{tabular}."""
    # find label then next tabular
    m = re.search(rf"\\label\{{{re.escape(label)}\}}", tex)
    if not m:
        print(f"[warn] label {label} not found")
        return tex
    start = tex.find("\\begin{tabular}", m.end())
    if start < 0:
        print(f"[warn] tabular after {label} not found")
        return tex
    # skip header until \midrule after toprule... find first \midrule after begin
    mid = tex.find("\\midrule", start)
    end = tex.find("\\bottomrule", mid)
    if mid < 0 or end < 0:
        print(f"[warn] mid/bottom for {label} not found")
        return tex
    return tex[: mid + len("\\midrule")] + "\n" + new_body + "\n" + tex[end:]


def main():
    tex = TEX.read_text(encoding="utf-8")

    disc, disc_p = load_results(
        RES / "legalep_disc_M_multi" / "tier_M" / "results.json",
        RES / "legalep_disc_M" / "tier_M" / "results.json",
    )
    lawyer, lawyer_p = load_results(
        RES / "legalep_lawyer_M_multi" / "tier_M" / "results.json",
        RES / "legalep_lawyer_M" / "tier_M" / "results.json",
    )
    cail, cail_p = load_results(RES / "cail_M" / "tier_M" / "results.json")

    # LegalEp-DISC
    if disc:
        ch = disc["channels"]
        # prefer exact then u1_exact
        exact = ch.get("exact") or ch.get("u1_exact")
        advice = ch.get("advice_recall")
        blocks = []
        if exact:
            blocks.append(render_channel_block("exact", exact, with_ci=False))
        if advice:
            # only FlatIP / Soft O2 / Hard for advice block as in tex
            adv_sys = [s for s in SYS_ORDER if s[0] in ("dense_flat", "dense_o2", "parent_hydrate")]
            blocks.append(render_channel_block("advice-recall", advice, systems=adv_sys, with_ci=False))
        body = "\n\\midrule\n".join(b for b in blocks if b)
        if body:
            tex = replace_table_body(tex, "tab:disc_main", body)
            print(f"[ok] tab:disc_main from {disc_p}")

    if lawyer:
        ch = lawyer["channels"]
        exact = ch.get("exact") or ch.get("u1_exact")
        advice = ch.get("advice_recall")
        blocks = []
        if exact:
            blocks.append(render_channel_block("exact", exact, with_ci=False))
        if advice:
            adv_sys = [s for s in SYS_ORDER if s[0] in ("dense_flat", "dense_o2", "parent_hydrate")]
            blocks.append(render_channel_block("advice-recall", advice, systems=adv_sys, with_ci=False))
        body = "\n\\midrule\n".join(b for b in blocks if b)
        if body:
            tex = replace_table_body(tex, "tab:lawyer_main", body)
            print(f"[ok] tab:lawyer_main from {lawyer_p}")

    if cail:
        ch = cail["channels"]
        blocks = []
        for key, lab in [("u1_exact", "U1"), ("uk_followup", "Uk-followup"), ("u_last", "U-last")]:
            if key in ch and ch[key].get("configs"):
                # only include systems that have finished
                avail = [(c, n) for c, n in SYS_ORDER if c in ch[key]["configs"]]
                if avail:
                    blocks.append(render_channel_block(lab, ch[key], systems=avail, with_ci=True))
        body = "\n\\midrule\n".join(b for b in blocks if b)
        if body:
            tex = replace_table_body(tex, "tab:cail_main", body)
            print(f"[ok] tab:cail_main from {cail_p}")

        # significance from comparisons
        comps = cail.get("comparisons") or {}
        uk = comps.get("uk_followup") or comps.get("u1_exact") or {}
        sig_lines = []
        for corpus_lab, src in [
            ("CAIL / Uk", comps.get("uk_followup") or {}),
            ("CAIL / U1", comps.get("u1_exact") or {}),
        ]:
            for key, contrast in [
                ("o2_vs_flat", "O2 vs FlatIP"),
                ("o2_vs_hard", "O2 vs Hard hydr."),
            ]:
                if key in src:
                    m = src[key]["mcnemar"]
                    sig_lines.append(
                        f"{corpus_lab} & {contrast} & {m['delta_mean']:+.3f} & {m['mid_p']:.3g} \\\\"
                    )
        if disc and "comparisons" in disc:
            src = (disc["comparisons"].get("exact") or disc["comparisons"].get("u1_exact") or {})
            if "o2_vs_flat" in src:
                m = src["o2_vs_flat"]["mcnemar"]
                sig_lines.append(
                    f"LegalEp-DISC / exact & O2 vs FlatIP & {m['delta_mean']:+.3f} & {m['mid_p']:.3g} \\\\"
                )
        if lawyer and "comparisons" in lawyer:
            src = (lawyer["comparisons"].get("exact") or lawyer["comparisons"].get("u1_exact") or {})
            if "o2_vs_flat" in src:
                m = src["o2_vs_flat"]["mcnemar"]
                sig_lines.append(
                    f"LegalEp-Lawyer / exact & O2 vs FlatIP & {m['delta_mean']:+.3f} & {m['mid_p']:.3g} \\\\"
                )
        if sig_lines:
            tex = replace_table_body(tex, "tab:sig", "\n".join(sig_lines))
            print("[ok] tab:sig")

        # beta
        beta = cail.get("beta_sweep") or {}
        # also from multi runs
        for extra in (disc, lawyer):
            if extra and extra.get("beta_sweep"):
                pass
        # build beta table if any beta_sweep exists
        beta_sources = []
        for name, blob in [("CAIL", cail), ("LegalEp-DISC", disc), ("LegalEp-Lawyer", lawyer)]:
            if blob and blob.get("beta_sweep"):
                beta_sources.append((name, blob["beta_sweep"]))
        sparse, _ = load_results(RES / "cail_M_sparse" / "tier_M" / "results.json")
        if sparse and sparse.get("beta_sweep"):
            beta_sources = [("CAIL", sparse["beta_sweep"])] + [x for x in beta_sources if x[0] != "CAIL"]
        betas = ["0.5", "0.7", "0.9", "0.95", "0.98", "1.0"]
        if beta_sources:
            lines = []
            for name, sweep in beta_sources:
                vals = []
                for b in betas:
                    cell = sweep.get(b) or sweep.get(str(float(b)))
                    vals.append(fmt(cell.get("answer_hit@k")) if cell else "---")
                lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
            tex = replace_table_body(tex, "tab:beta", "\n".join(lines))
            print("[ok] tab:beta")

    # BM25 table
    bm25_lines = []
    for name, blob, chans in [
        ("CAIL", cail, [("u1_exact", "U1"), ("uk_followup", "Uk"), ("u_last", "U-last")]),
        ("LegalEp-DISC", disc, [("exact", "exact"), ("advice_recall", "advice-recall")]),
        ("LegalEp-Lawyer", lawyer, [("exact", "exact"), ("advice_recall", "advice-recall")]),
    ]:
        if not blob:
            continue
        turns, joints = [], []
        labels = []
        for key, lab in chans:
            cf = (blob.get("channels") or {}).get(key, {}).get("configs") or {}
            if "bm25_turn" in cf or "bm25_joint" in cf:
                labels.append(lab)
                turns.append(fmt(cf.get("bm25_turn", {}).get("answer_hit@k")) if "bm25_turn" in cf else "---")
                joints.append(fmt(cf.get("bm25_joint", {}).get("answer_hit@k")) if "bm25_joint" in cf else "---")
        if labels:
            bm25_lines.append(
                f"{name} & {' / '.join(labels)} & {' / '.join(turns)} & {' / '.join(joints)} \\\\"
            )
    sparse, _ = load_results(RES / "cail_M_sparse" / "tier_M" / "results.json")
    if sparse:
        # prefer sparse for CAIL BM25
        pass
    if bm25_lines:
        tex = replace_table_body(tex, "tab:bm25", "\n".join(bm25_lines))
        print("[ok] tab:bm25")

    # Remove pending language when we have disc numbers
    if disc:
        tex = tex.replace(
            "Empirical tables are reported as they become available under this protocol.",
            "Primary legal tables report completed V4 runs under this protocol.",
        )
        tex = tex.replace(
            "Result cells marked \\tbd{} are reserved for completed V4 runs.",
            "Tables below report completed V4 runs; remaining cells stay blank only when a channel is still running.",
        )

    TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
