#!/usr/bin/env python3
"""Generate anonymized IPM manuscript from ipm-article.tex."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper/ipm/ipm-article.tex"
OUT = ROOT / "paper/ipm/ipm-article-anonymous.tex"

# Use doubled backslashes so "\a" never becomes the bell character.
ANON_FRONT = (
    "%% Anonymous submission frontmatter\n"
    "\\author[]{Anonymous Author(s)}\n"
)


def main() -> None:
    tex = SRC.read_text(encoding="utf-8")
    tex2, n = re.subn(
        r"%% Author names and affiliations appear under the title.*?\\affiliation\[idl\]\{.*?country=\{China\}\}",
        lambda _: ANON_FRONT.rstrip(),
        tex,
        count=1,
        flags=re.S,
    )
    if n != 1:
        tex2, n = re.subn(
            r"\\author\[uw\]\{Linrui Xu\}.*?\\affiliation\[idl\]\{.*?country=\{China\}\}",
            lambda _: ANON_FRONT.rstrip(),
            tex,
            count=1,
            flags=re.S,
        )
    if n != 1:
        raise SystemExit(f"anonymize frontmatter failed (replacements={n})")
    tex = tex2

    front_end = tex.find("\\begin{abstract}")
    if front_end < 0:
        raise SystemExit("missing abstract")
    if re.search(
        r"\\ead\{|\\cortext\{|Linrui|uwinnipeg|163\.com|Xitucheng|Winnipeg|CUPL",
        tex[:front_end],
    ):
        raise SystemExit("anonymize left identifying frontmatter")

    tex = tex.replace(
        "Code, processed evaluation corpora, and primary result JSON files are available at\n"
        "\\url{https://github.com/Kilimajaro/soft-episode-binding-legal-memory}\n"
        "(archived release to accompany camera-ready).",
        "Code, processed evaluation corpora, and primary result JSON files will be made available upon acceptance.",
    )
    tex = tex.replace(
        "\\url{https://github.com/Kilimajaro/soft-episode-binding-legal-memory}",
        "URL withheld for anonymous review.",
    )
    tex = tex.replace(
        "The BIMS-LEGAL implementation (O1 FlatIP, Soft~O2, O3 bulk-load, Soft~O2-C), corpus builders, Mix constructors, and table/figure scripts ship in the same repository, with one JSON artifact per main table.",
        "The BIMS-LEGAL implementation, evaluation pipelines, and table/figure scripts will be released with the accepted publication.",
    )
    contrib = (
        "\\subsection*{Author contributions}\n"
        "Author contributions are withheld for anonymous review.\n\n"
        "\\FloatBarrier"
    )
    tex2, n = re.subn(
        r"\\subsection\*\{Author contributions\}.*?\\FloatBarrier",
        lambda _: contrib,
        tex,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit(f"anonymize author contributions failed (replacements={n})")
    tex = tex2

    if "github.com/Kilimajaro" in tex or "xu-l81@" in tex or "linrui_han@" in tex:
        raise SystemExit("anonymize left identifying URLs or emails")
    tex = tex.replace(
        "Computational resources were provided by the Data Law Laboratory, China University of Political Science and Law.",
        "Computational resources were provided by the authors' host institutions.",
    )
    if "University of Winnipeg" in tex or "CUPL Data Law Lab" in tex:
        raise SystemExit("anonymize left identifying affiliations in body")

    OUT.write_text(tex, encoding="utf-8")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
