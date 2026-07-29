#!/usr/bin/env python3
"""Generate anonymized IPM manuscript from ipm-article.tex."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper/ipm/ipm-article.tex"
OUT = ROOT / "paper/ipm/ipm-article-anonymous.tex"

ANON_FRONT = r"""\author[]{Anonymous Author(s)}
\ead{anonymous@anonymous.edu}
"""


def main() -> None:
    tex = SRC.read_text(encoding="utf-8")
    tex = re.sub(
        r"\\author\[.*?\]\{Linrui Xu\}.*?\\affiliation\[idl\]\{organization=\{The Institute for Data Law, China University of Political Science and Law\},\s*city=\{Beijing\},\s*country=\{China\}\}",
        lambda _: ANON_FRONT.strip(),
        tex,
        flags=re.S,
    )
    tex = tex.replace(
        "Code, processed evaluation corpora, and primary result files are publicly available at\n\\url{https://github.com/Kilimajaro/soft-episode-binding-legal-memory}.",
        "Code, processed evaluation corpora, and primary result files will be made available upon acceptance.",
    )
    tex = tex.replace(
        "The BIMS-LEGAL implementation (O1 FlatIP, Soft~O2, O3 bulk-load, Soft~O2-C), LegalMem/LegalEp pipelines, Mix builders, and table/figure scripts are released in the same repository with JSON artifacts for each main table.",
        "The BIMS-LEGAL implementation, evaluation pipelines, and table/figure scripts will be released with the accepted publication.",
    )
    OUT.write_text(tex, encoding="utf-8")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
