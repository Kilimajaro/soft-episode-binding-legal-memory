#!/usr/bin/env python3
"""Fill IPM manuscript tables from completed BIMS-LEGAL V4 result JSONs.

Publication rules:
- Drop Session-max as a separate row (identical to hard hydration on all channels).
- Skip Soft O2 / shuffled cells whose elapsed_seconds≈0 and AH equals FlatIP
  (legacy cache collision); after a clean advice re-run those cells are kept.
- CAIL primary table: AH / EC / nDCG / 95% CI.
- Soft O2 column in the CE table always comes from the primary evaluation.
- β table uses cache-fixed primary sweeps; stale DISC-exact flat sweep is omitted.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "ipm" / "ipm-article.tex"
RES = ROOT / "BIMS-LEGAL-dataset" / "primary_results" / "bims_legal_v4"
if not RES.exists():
    RES = ROOT / "results" / "bims_legal_v4"
sys.path.insert(0, str(ROOT / "eval" / "legal" / "v3"))

# Main turn-level systems shown in CAIL / LegalEp grids (no Session-max duplicate).
SYS = [
    ("dense_flat", "FlatIP"),
    ("dense_o2", "Soft O2"),
    ("parent_hydrate", "Hard hydr."),
    ("shuffled_o2", "Shuffled O2"),
]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd=3):
    if x is None:
        return "---"
    return f"{float(x):.{nd}f}"


def cfg(ch, name):
    return (ch or {}).get("configs", {}).get(name) or {}


def is_poisoned(cfgs: dict, key: str = "dense_o2") -> bool:
    """True if Soft O2/shuffled looks like the old FlatIP search-cache collision."""
    m = cfgs.get(key) or {}
    flat = cfgs.get("dense_flat") or {}
    if not m or not flat:
        return False
    elapsed = float(m.get("elapsed_seconds") or 0.0)
    if elapsed > 0.5:
        return False
    return m.get("answer_hit@k") == flat.get("answer_hit@k")


def bold_max(rows, col):
    vals = []
    for r in rows:
        try:
            vals.append(float(re.sub(r"\\textbf\{([^}]+)\}", r"\1", r[col])))
        except Exception:
            vals.append(float("-inf"))
    best = max(vals) if vals else None
    out = []
    for r, v in zip(rows, vals):
        rr = list(r)
        if best is not None and v == best and v != float("-inf"):
            raw = re.sub(r"\\textbf\{([^}]+)\}", r"\1", rr[col])
            rr[col] = f"\\textbf{{{raw}}}"
        out.append(rr)
    return out


def block(label, ch_data, with_ci=False, systems=None, skip_poisoned=True):
    systems = systems or SYS
    cfgs = (ch_data or {}).get("configs") or {}
    rows = []
    for key, name in systems:
        if skip_poisoned and key in ("dense_o2", "shuffled_o2") and is_poisoned(cfgs, key if key != "shuffled_o2" else "dense_o2"):
            # If Soft O2 is poisoned, also drop shuffled (same collision class).
            if key == "dense_o2" or is_poisoned(cfgs, "dense_o2"):
                continue
        if skip_poisoned and key == "shuffled_o2" and is_poisoned(cfgs, "dense_o2"):
            continue
        m = cfgs.get(key) or {}
        if not m:
            continue
        row = [name, fmt(m.get("answer_hit@k")), fmt(m.get("episode_completeness@k")), fmt(m.get("ndcg@k"))]
        if with_ci:
            ci = m.get("ah_ci") or {}
            row.append(f"[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]" if "ci_low" in ci else "---")
        rows.append(row)
    if not rows:
        return ""
    rows = bold_max(rows, 1)
    lines = []
    for i, r in enumerate(rows):
        prefix = f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}\n & " if i == 0 else " & "
        lines.append(prefix + " & ".join(r) + " \\\\")
    return "\n".join(lines)


def replace_body(tex, label, body):
    m = re.search(rf"\\label\{{{re.escape(label)}\}}", tex)
    if not m:
        print(f"[warn] missing {label}")
        return tex
    start = tex.find("\\begin{tabular}", m.end())
    mid = tex.find("\\midrule", start)
    end = tex.find("\\bottomrule", mid)
    if mid < 0 or end < 0:
        print(f"[warn] mid/bottom {label}")
        return tex
    return tex[: mid + len("\\midrule")] + "\n" + body + "\n" + tex[end:]


def main():
    cail = load(RES / "cail_M" / "tier_M" / "results.json")
    disc_exact = load(RES / "legalep_disc_M" / "tier_M" / "results.json")
    disc_advice = load(RES / "legalep_disc_advice" / "tier_M" / "results.json")
    disc_para = load(RES / "legalep_disc_para" / "tier_M" / "results.json")
    lawyer_exact = load(RES / "legalep_lawyer_M" / "tier_M" / "results.json")
    lawyer_advice = load(RES / "legalep_lawyer_advice" / "tier_M" / "results.json")
    lawyer_para = load(RES / "legalep_lawyer_para" / "tier_M" / "results.json")
    disc_bm25 = load(RES / "legalep_disc_bm25" / "tier_M" / "results.json")
    scale = load(RES / "scale_curve.json")
    cail_bm25 = load(RES / "cail_bm25" / "tier_M" / "results.json")
    lawyer_bm25 = load(RES / "legalep_lawyer_bm25" / "tier_M" / "results.json")
    cail_beta = load(RES / "cail_beta" / "tier_M" / "results.json")
    lawyer_beta = load(RES / "legalep_lawyer_beta" / "tier_M" / "results.json")
    cail_ce = load(RES / "cail_ce" / "tier_M" / "results.json")
    disc_ce = load(RES / "legalep_disc_ce" / "tier_M" / "results.json")
    lawyer_ce = load(RES / "legalep_lawyer_ce" / "tier_M" / "results.json")

    tex = TEX.read_text(encoding="utf-8")

    # --- CAIL: AH EC nDCG CI (header/colspec maintained in the .tex scaffold) ---
    if cail:
        parts = []
        for key, lab in [("u1_exact", "U1"), ("uk_followup", "Uk-followup"), ("u_last", "U-last")]:
            if key in cail["channels"]:
                parts.append(block(lab, cail["channels"][key], with_ci=True))
        tex = replace_body(tex, "tab:cail_main", "\n\\midrule\n".join(p for p in parts if p))
        print("[ok] cail")

    disc_parts = []
    if disc_exact:
        ch = disc_exact["channels"].get("exact") or disc_exact["channels"].get("u1_exact")
        disc_parts.append(block("exact", ch))
    if disc_advice:
        disc_parts.append(block("advice-recall", disc_advice["channels"]["advice_recall"]))
    if disc_para:
        disc_parts.append(block("u-para", disc_para["channels"]["u_para"]))
    if disc_parts:
        tex = replace_body(tex, "tab:disc_main", "\n\\midrule\n".join(disc_parts))
        print("[ok] disc")

    law_parts = []
    if lawyer_exact:
        ch = lawyer_exact["channels"].get("exact") or lawyer_exact["channels"].get("u1_exact")
        law_parts.append(block("exact", ch))
    if lawyer_advice:
        law_parts.append(block("advice-recall", lawyer_advice["channels"]["advice_recall"]))
    if lawyer_para:
        law_parts.append(block("u-para", lawyer_para["channels"]["u_para"]))
    if law_parts:
        tex = replace_body(tex, "tab:lawyer_main", "\n\\midrule\n".join(law_parts))
        print("[ok] lawyer")

    bm25_lines = []
    if disc_bm25:
        ex = disc_bm25["channels"].get("exact", {}).get("configs", {})
        adv = disc_bm25["channels"].get("advice_recall", {}).get("configs", {})
        bm25_lines += [
            f"LegalEp-DISC & exact & {fmt(ex.get('bm25_turn', {}).get('answer_hit@k'))} & {fmt(ex.get('bm25_joint', {}).get('answer_hit@k'))} \\\\",
            f"LegalEp-DISC & advice-recall & {fmt(adv.get('bm25_turn', {}).get('answer_hit@k'))} & {fmt(adv.get('bm25_joint', {}).get('answer_hit@k'))} \\\\",
            f"LegalEp-DISC & exact (dense RRF) & {fmt(ex.get('dense_rrf', {}).get('answer_hit@k'))} & --- \\\\",
            f"LegalEp-DISC & advice (dense RRF) & {fmt(adv.get('dense_rrf', {}).get('answer_hit@k'))} & --- \\\\",
        ]
    if cail_bm25:
        parts_t, parts_j = [], []
        for key in ("u1_exact", "uk_followup", "u_last"):
            cf = cail_bm25["channels"].get(key, {}).get("configs", {})
            if cf:
                parts_t.append(fmt(cf.get("bm25_turn", {}).get("answer_hit@k")))
                parts_j.append(fmt(cf.get("bm25_joint", {}).get("answer_hit@k")))
        if parts_t:
            bm25_lines.append(
                f"CAIL & U1 / Uk / U-last & {' / '.join(parts_t)} & {' / '.join(parts_j)} \\\\"
            )
    if lawyer_bm25:
        ex = lawyer_bm25["channels"].get("exact", {}).get("configs", {})
        adv = lawyer_bm25["channels"].get("advice_recall", {}).get("configs", {})
        bm25_lines.append(
            f"LegalEp-Lawyer & exact / advice & "
            f"{fmt(ex.get('bm25_turn', {}).get('answer_hit@k'))} / {fmt(adv.get('bm25_turn', {}).get('answer_hit@k'))} & "
            f"{fmt(ex.get('bm25_joint', {}).get('answer_hit@k'))} / {fmt(adv.get('bm25_joint', {}).get('answer_hit@k'))} \\\\"
        )
    if bm25_lines:
        tex = replace_body(tex, "tab:bm25", "\n".join(bm25_lines))
        print("[ok] bm25")

    betas = ["0.5", "0.7", "0.9", "0.95", "0.98", "1.0"]

    def _beta_row(name, blob):
        if not blob:
            return None
        sweep = blob.get("beta_sweep") or {}
        vals = []
        any_ok = False
        for b in betas:
            cell = sweep.get(b)
            ah = (cell or {}).get("answer_hit@k") if isinstance(cell, dict) else None
            if ah is not None:
                any_ok = True
            vals.append(fmt(ah))
        if not any_ok:
            return None
        # Drop flat/uninformative sweeps (all equal) from a mismatched older store.
        numeric = [float(v) for v in vals if v != "---"]
        if len(numeric) >= 2 and len(set(numeric)) == 1:
            return None
        return f"{name} & " + " & ".join(vals) + " \\\\"

    beta_lines = []
    for name, blob in [
        ("CAIL / Uk", cail if cail and (cail.get("beta_sweep") or {}) else cail_beta),
        ("LegalEp-Lawyer / exact", lawyer_exact if lawyer_exact and (lawyer_exact.get("beta_sweep") or {}) else lawyer_beta),
        ("LegalEp-Lawyer / u-para", lawyer_para),
        ("LegalEp-DISC / u-para", disc_para),
    ]:
        row = _beta_row(name, blob)
        if row:
            beta_lines.append(row)
    if beta_lines:
        tex = replace_body(tex, "tab:beta", "\n".join(beta_lines))
        print("[ok] beta")

    def _ah(blob, ch, key):
        if not blob or ch not in (blob.get("channels") or {}):
            return None
        return ((blob["channels"][ch].get("configs") or {}).get(key) or {}).get("answer_hit@k")

    o2_primary = {
        "CAIL / U1": _ah(cail, "u1_exact", "dense_o2"),
        "CAIL / Uk": _ah(cail, "uk_followup", "dense_o2"),
        "CAIL / U-last": _ah(cail, "u_last", "dense_o2"),
        "LegalEp-DISC / exact": _ah(disc_exact, "exact", "dense_o2") or _ah(disc_exact, "u1_exact", "dense_o2"),
        "LegalEp-Lawyer / exact": _ah(lawyer_exact, "exact", "dense_o2"),
    }
    ce_lines = []
    for lab, blob, ch in [
        ("CAIL / U1", cail_ce, "u1_exact"),
        ("CAIL / Uk", cail_ce, "uk_followup"),
        ("CAIL / U-last", cail_ce, "u_last"),
        ("LegalEp-DISC / exact", disc_ce, "exact"),
        ("LegalEp-Lawyer / exact", lawyer_ce, "exact"),
    ]:
        flat_ah = _ah(blob, ch, "dense_flat")
        ce_ah = _ah(blob, ch, "dense_ce")
        o2_ah = o2_primary.get(lab)
        if o2_ah is None:
            o2_ah = _ah(blob, ch, "dense_o2")
        ce_lines.append(f"{lab} & {fmt(flat_ah)} & {fmt(ce_ah)} & {fmt(o2_ah)} \\\\")
    tex = replace_body(tex, "tab:ce", "\n".join(ce_lines))
    print("[ok] ce")

    def _sig_append(lines, lab, name, mcnemar):
        if not mcnemar:
            return
        p = mcnemar["mid_p"]
        p_s = f"{p:.3g}" if p >= 1e-3 else f"{p:.2e}"
        lines.append(f"{lab} & {name} & {mcnemar['delta_mean']:+.3f} & {p_s} \\\\")

    def _paired_from_cfgs(cfgs, a="dense_o2", b="dense_flat"):
        if a not in cfgs or b not in cfgs:
            return None
        if a in ("dense_o2", "shuffled_o2") and is_poisoned(cfgs, "dense_o2"):
            return None
        from stats_sig import paired_report

        return paired_report(a, cfgs[a]["per_query_ah"], b, cfgs[b]["per_query_ah"])["mcnemar"]

    sig_lines = []
    if cail:
        for ch, lab in [("uk_followup", "CAIL / Uk"), ("u1_exact", "CAIL / U1"), ("u_last", "CAIL / U-last")]:
            comps = (cail.get("comparisons") or {}).get(ch) or {}
            cfgs = (cail["channels"].get(ch) or {}).get("configs") or {}
            for key, name, a, b in [
                ("o2_vs_flat", "O2 vs FlatIP", "dense_o2", "dense_flat"),
                ("o2_vs_hard", "O2 vs Hard hydr.", "dense_o2", "parent_hydrate"),
            ]:
                m = (comps.get(key) or {}).get("mcnemar") or _paired_from_cfgs(cfgs, a, b)
                _sig_append(sig_lines, lab, name, m)
        if cail_ce:
            for ch, lab in [("uk_followup", "CAIL / Uk"), ("u1_exact", "CAIL / U1"), ("u_last", "CAIL / U-last")]:
                comps = (cail_ce.get("comparisons") or {}).get(ch) or {}
                m = (comps.get("o2_vs_ce") or {}).get("mcnemar")
                _sig_append(sig_lines, lab, "O2 vs FlatIP+CE", m)
    for blob, lab, prefer_ch in [
        (disc_exact, "LegalEp-DISC / exact", None),
        (lawyer_exact, "LegalEp-Lawyer / exact", None),
        (disc_para, "LegalEp-DISC / u-para", "u_para"),
        (lawyer_para, "LegalEp-Lawyer / u-para", "u_para"),
        (disc_advice, "LegalEp-DISC / advice", "advice_recall"),
        (lawyer_advice, "LegalEp-Lawyer / advice", "advice_recall"),
    ]:
        if not blob:
            continue
        comps = blob.get("comparisons") or {}
        ch = prefer_ch or next(iter(comps), None) or next(iter(blob.get("channels") or {}), None)
        if not ch:
            continue
        cfgs = (blob.get("channels") or {}).get(ch, {}).get("configs") or {}
        src = comps.get(ch) or {}
        if is_poisoned(cfgs, "dense_o2"):
            if "advice" in lab:
                m_hf = _paired_from_cfgs(cfgs, "parent_hydrate", "dense_flat")
                _sig_append(sig_lines, lab, "Hard hydr. vs FlatIP", m_hf)
            continue
        for key, name, a, b in [
            ("o2_vs_flat", "O2 vs FlatIP", "dense_o2", "dense_flat"),
            ("o2_vs_hard", "O2 vs Hard hydr.", "dense_o2", "parent_hydrate"),
        ]:
            m = (src.get(key) or {}).get("mcnemar") or _paired_from_cfgs(cfgs, a, b)
            _sig_append(sig_lines, lab, name, m)
    if sig_lines:
        tex = replace_body(tex, "tab:sig", "\n".join(sig_lines))
        print("[ok] sig")

    if scale:
        agg = defaultdict(list)
        for r in scale["rows"]:
            agg[(r["M"], r["mode"])].append(r)
        lines = []
        mode_name = {"ivfpq": "IVFPQ", "flat": "FlatIP", "o2": "Soft O2"}
        for (M, mode), rows in sorted(agg.items()):
            n = len(rows)
            ah = sum(r["answer_hit@k"] for r in rows) / n
            b = sum(r["build_seconds"] for r in rows) / n
            p50 = sum(r["latency_p50"] for r in rows) / n * 1000
            p95 = sum(r["latency_p95"] for r in rows) / n * 1000
            lines.append(
                f"{M} & {mode_name.get(mode, mode)} & {ah:.3f} & {b:.1f} & {p50:.0f} & {p95:.0f} \\\\"
            )
        tex = replace_body(tex, "tab:scale", "\n".join(lines))
        print("[ok] scale")

    if cail and scale:
        u1 = cail["channels"]["u1_exact"]["configs"]
        ivf = [r for r in scale["rows"] if r["M"] == 1600 and r["mode"] == "ivfpq"]
        ivf_ah = sum(r["answer_hit@k"] for r in ivf) / len(ivf) if ivf else None
        ablate = "\n".join([
            f"Soft O2 (CAIL U1) & {fmt(u1['dense_o2']['answer_hit@k'])} & {fmt(u1['dense_o2']['episode_completeness@k'])} & --- \\\\",
            f"FlatIP (CAIL U1) & {fmt(u1['dense_flat']['answer_hit@k'])} & {fmt(u1['dense_flat']['episode_completeness@k'])} & --- \\\\",
            f"Hard hydration (CAIL U1) & {fmt(u1['parent_hydrate']['answer_hit@k'])} & {fmt(u1['parent_hydrate']['episode_completeness@k'])} & --- \\\\",
            f"Shuffled $sid$ Soft O2 (CAIL U1) & {fmt(u1['shuffled_o2']['answer_hit@k'])} & {fmt(u1['shuffled_o2']['episode_completeness@k'])} & --- \\\\",
            f"IVFPQ (DISC $M{{=}}1600$) & {fmt(ivf_ah)} & --- & --- \\\\",
        ])
        tex = replace_body(tex, "tab:ablate", ablate)
        print("[ok] ablate")

    TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
