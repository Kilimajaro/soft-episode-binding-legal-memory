#!/usr/bin/env python3
"""Fill appendix nDCG (optionally AH/EC) from corrected_metrics_*.json into ipm-article.tex.

Default: nDCG-only — keeps published V4 AH/EC; fills corrected fixed-gold nDCG
from FlatIP rebuild. Use --rewrite-all to replace AH/EC/nDCG with rebuild numbers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "paper" / "ipm" / "ipm-article.tex"
FIG = REPO / "paper" / "ipm" / "figures"

CFG_MAP = {
    "FlatIP": "dense_flat",
    "Soft O2": "dense_o2",
    "Hard hydr.": "parent_hydrate",
    "Shuffled O2": "shuffled_o2",
}

TABLES = {
    "cail_main": {
        "json": "corrected_metrics_cail.json",
        "channels": {
            "U1": "u1_exact",
            "Uk-followup": "uk_followup",
            "Uk": "uk_followup",
            "U-last": "u_last",
        },
    },
    "disc_main": {
        "json": "corrected_metrics_disc.json",
        "channels": {
            "exact": "exact",
            "advice-recall": "advice_recall",
            "advice": "advice_recall",
            "u-para": "u_para",
        },
    },
    "lawyer_main": {
        "json": "corrected_metrics_lawyer.json",
        "channels": {
            "exact": "exact",
            "advice-recall": "advice_recall",
            "advice": "advice_recall",
            "u-para": "u_para",
        },
    },
}


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def load_json(name: str) -> dict:
    p = FIG / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def cell(data: dict, channel: str, cfg: str, key: str):
    ch = data.get("channels", {}).get(channel, {})
    row = ch.get(cfg)
    if not row or key not in row:
        return None
    return row[key]


def replace_num_field(s: str, new_val: str) -> str:
    """Replace a numeric field that may be wrapped in \\textbf{...}."""
    m = re.fullmatch(r"(\\textbf\{)([0-9.]+|---)(\})", s.strip())
    if m:
        return f"{m.group(1)}{new_val}{m.group(3)}"
    if re.fullmatch(r"[0-9.]+|---", s.strip()):
        return new_val
    return s


def replace_grid_rows(tex: str, label: str, data: dict, channel_map: dict, rewrite_all: bool) -> str:
    pat = re.compile(
        rf"(\\label\{{tab:{re.escape(label)}\}}.*?\\begin\{{tabular\}}.*?\n)(.*?)(\\end\{{tabular\}})",
        re.S,
    )
    m = pat.search(tex)
    if not m:
        raise SystemExit(f"could not find tabular for tab:{label}")

    body = m.group(2)
    out_lines = []
    current_channel_key = None
    n_filled = 0

    for line in body.splitlines(keepends=True):
        raw = line.rstrip("\n")
        mm_ch = re.match(r"^\\multirow\{\d+\}\{\*\}\{([^}]+)\}\s*$", raw.strip())
        if mm_ch:
            tex_ch = mm_ch.group(1)
            current_channel_key = channel_map.get(tex_ch)
            if current_channel_key is None:
                print(f"[warn] unknown channel label {tex_ch!r} in tab:{label}")
            out_lines.append(line)
            continue

        mm = re.match(
            r"^(\s*&\s*)(FlatIP|Soft O2|Hard hydr\.|Shuffled O2)(\s*&\s*)"
            r"(\\textbf\{[0-9.]+\}|[0-9.]+|---)(\s*&\s*)"
            r"(\\textbf\{[0-9.]+\}|[0-9.]+|---)(\s*&\s*)"
            r"(\\textbf\{[0-9.]+\}|[0-9.]+|---)(.*)$",
            raw,
        )
        if mm and current_channel_key:
            sys_name = mm.group(2)
            cfg = CFG_MAP[sys_name]
            ah_s, ec_s, nd_s = mm.group(4), mm.group(6), mm.group(8)
            if rewrite_all:
                v_ah = cell(data, current_channel_key, cfg, "answer_hit@k")
                v_ec = cell(data, current_channel_key, cfg, "episode_completeness@k")
                if v_ah is not None:
                    ah_s = replace_num_field(ah_s, fmt(v_ah))
                if v_ec is not None:
                    ec_s = replace_num_field(ec_s, fmt(v_ec))
            v_nd = cell(data, current_channel_key, cfg, "ndcg@k")
            if v_nd is not None:
                nd_s = replace_num_field(nd_s, fmt(v_nd))
                n_filled += 1
            raw = (
                f"{mm.group(1)}{sys_name}{mm.group(3)}{ah_s}"
                f"{mm.group(5)}{ec_s}{mm.group(7)}{nd_s}{mm.group(9)}"
            )
            out_lines.append(raw + "\n")
            continue

        out_lines.append(line)

    print(f"[fill] tab:{label} filled {n_filled} nDCG cells")
    new_body = "".join(out_lines)
    return tex[: m.start(2)] + new_body + tex[m.end(2) :]


def update_ndcg_corrected(tex: str) -> str:
    disc = load_json("corrected_metrics_disc.json")
    lawyer = load_json("corrected_metrics_lawyer.json")
    cail = load_json("corrected_metrics_cail.json")

    def row(corpus, channel_tex, data, ch, cfg_tex, cfg):
        ah = cell(data, ch, cfg, "answer_hit@k")
        nd = cell(data, ch, cfg, "ndcg@k")
        fail = (cell(data, ch, cfg, "failure_taxonomy") or {}).get("incomplete")
        if ah is None or nd is None or fail is None:
            return None
        inc = f"{100.0 * float(fail):.1f}\\%"
        if cfg == "dense_o2":
            return (
                f"{corpus} & {channel_tex} & {cfg_tex} & "
                f"\\textbf{{{fmt(ah)}}} & \\textbf{{{fmt(nd)}}} & {inc} \\\\"
            )
        return f"{corpus} & {channel_tex} & {cfg_tex} & {fmt(ah)} & {fmt(nd)} & {inc} \\\\"

    plan = [
        ("CAIL", "U1", cail, "u1_exact"),
        ("CAIL", "Uk", cail, "uk_followup"),
        ("CAIL", "U-last", cail, "u_last"),
        ("LegalEp-DISC", "exact", disc, "exact"),
        ("LegalEp-DISC", "u-para", disc, "u_para"),
        ("LegalEp-DISC", "advice", disc, "advice_recall"),
        ("LegalEp-Lawyer", "exact", lawyer, "exact"),
        ("LegalEp-Lawyer", "u-para", lawyer, "u_para"),
        ("LegalEp-Lawyer", "advice", lawyer, "advice_recall"),
    ]
    rows = []
    for corpus, ch_tex, data, ch in plan:
        for cfg_tex, cfg in (("FlatIP", "dense_flat"), ("Soft O2", "dense_o2")):
            r = row(corpus, ch_tex, data, ch, cfg_tex, cfg)
            if r:
                rows.append(r)
    if not rows:
        return tex

    pat = re.compile(
        r"(\\label\{tab:ndcg-corrected\}.*?\\begin\{tabular\}.*?\n)(.*?)(\\end\{tabular\})",
        re.S,
    )
    m = pat.search(tex)
    if not m:
        print("[warn] tab:ndcg-corrected not found; skip")
        return tex

    old = m.group(2)
    header_lines = []
    for line in old.splitlines(keepends=True):
        if re.match(r"^(CAIL|LegalEp)", line.strip()):
            break
        header_lines.append(line)
    new_body = "".join(header_lines) + "\n".join(rows) + "\n"
    print(f"[fill] tab:ndcg-corrected rows={len(rows)}")
    return tex[: m.start(2)] + new_body + tex[m.end(2) :]


def update_captions(tex: str, rewrite_all: bool) -> str:
    if rewrite_all:
        tex = tex.replace(
            "nDCG@10 marked --- pending fixed-gold IDCG recompute (Section~\\ref{sec:protocol}); AH/EC unchanged.",
            "AH/EC/nDCG from a fresh FlatIP rebuild with fixed-gold IDCG (Section~\\ref{sec:protocol}); absolute AH may exceed published V4 IVFPQ grids.",
        )
    else:
        tex = tex.replace(
            "nDCG@10 marked --- pending fixed-gold IDCG recompute (Section~\\ref{sec:protocol}); AH/EC unchanged.",
            "nDCG@10: corrected fixed-gold IDCG on FlatIP rebuild (Section~\\ref{sec:protocol}); published V4 AH/EC unchanged.",
        )
        tex = tex.replace(
            "Legacy retrieved-item IDCG values removed from this grid; Soft~O2 claims use AH/EC. Corrected nDCG script: \\texttt{recompute\\_corrected\\_metrics.py}.",
            "nDCG@10 filled from fixed-gold IDCG FlatIP rebuild (\\texttt{recompute\\_corrected\\_metrics.py}); published V4 AH/EC unchanged.",
        )
        # disc/lawyer captions: append nDCG note if missing
        for label, needle in (
            (
                "tab:disc_main",
                "only hard hydration is shown.}",
            ),
            (
                "tab:lawyer_main",
                "only hard hydration is shown.}",
            ),
        ):
            # find caption ending near label
            pass
        tex = re.sub(
            r"(\\caption\{LegalEp-DISC[^}]*)(only hard hydration is shown\.)(\})",
            r"\1\2 nDCG@10: corrected fixed-gold IDCG (FlatIP rebuild); V4 AH/EC unchanged.\3",
            tex,
            count=1,
        )
        tex = re.sub(
            r"(\\caption\{LegalEp-Lawyer[^}]*)(only hard hydration is shown\.)(\})",
            r"\1\2 nDCG@10: corrected fixed-gold IDCG (FlatIP rebuild); V4 AH/EC unchanged.\3",
            tex,
            count=1,
        )
    return tex


def completeness_report() -> list[str]:
    missing = []
    for label, spec in TABLES.items():
        data = load_json(spec["json"])
        seen = set(spec["channels"].values())
        for ch in sorted(seen):
            for cfg in CFG_MAP.values():
                if cell(data, ch, cfg, "ndcg@k") is None:
                    missing.append(f"{spec['json']}:{ch}/{cfg}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite-all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    missing = completeness_report()
    if missing:
        print("[missing]", len(missing))
        for x in missing:
            print(" ", x)
        if args.require_complete:
            raise SystemExit("incomplete artifacts")
    else:
        print("[ok] all appendix channel/config nDCG cells present")

    tex = TEX.read_text(encoding="utf-8")
    for label, spec in TABLES.items():
        data = load_json(spec["json"])
        tex = replace_grid_rows(tex, label, data, spec["channels"], args.rewrite_all)
    tex = update_ndcg_corrected(tex)
    tex = update_captions(tex, args.rewrite_all)

    if args.dry_run:
        print("[dry-run] not writing")
        for label in TABLES:
            m = re.search(rf"\\label\{{tab:{label}\}}.*?\\end\{{tabular\}}", tex, re.S)
            if m:
                # count --- only in nDCG column-ish: lines with systems
                n = 0
                for line in m.group(0).splitlines():
                    if re.search(r"FlatIP|Soft O2|Hard hydr|Shuffled O2", line) and "---" in line:
                        n += 1
                print(f"  tab:{label} system-rows still containing ---: {n}")
        return

    TEX.write_text(tex, encoding="utf-8")
    print(f"[wrote] {TEX}")


if __name__ == "__main__":
    main()
