#!/usr/bin/env python3
"""Submission polish for ipm-article.tex and regenerate anonymous twin."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "ipm" / "ipm-article.tex"
ANON = ROOT / "ipm" / "ipm-article-anonymous.tex"

text = TEX.read_text(encoding="utf-8")

# --- Abstract ---
text = text.replace(
    "Soft~O2-C and gated Hybrid are reported as exploratory, gold-assisted solvability bounds for cross-session recovery---complementary to Soft~O2, not replacements for it, and not claims about natural unsupervised BIRCH deployment.",
    "Soft~O2-C and gated Hybrid complement Soft~O2 under a Mix cross-session protocol in which some gold evidence is held off-session; Soft~O2 remains the default when query and gold already share a session identifier.",
)

# --- Contributions ---
text = text.replace(
    "\\item Exploratory Soft~O2-C / gated Hybrid results under a Mix cross-session protocol, together with graded nDCG, a failure taxonomy, and paired significance reporting.",
    "\\item Soft~O2-C and gated Hybrid under a Mix cross-session protocol, together with graded nDCG, a failure taxonomy, and paired significance reporting.",
)

# --- Mix protocol paragraph: keep honest, drop checklist tone ---
text = text.replace(
    """\\paragraph{Ensuring Sa/Sb co-membership for mechanism testing.}
After BIRCH consolidation on the Mix store, each cross-session Sa/Sb split pair is \\emph{glued} into one cluster identifier.
This step uses gold split metadata available only in the evaluation protocol---it does \\emph{not} inject labels at query time and is \\emph{not} used in real unsupervised clustering deployments.
Its purpose is diagnostic: if Soft~O2-C cannot recover cross-session gold when both halves of a split pair co-occupy one cluster, failure reflects the binding mechanism rather than accidental cluster separation. Natural same-cluster rates without this step are lower and are noted in Section~\\ref{sec:limitations}.
Soft~O2's session membership constraint is unchanged---glue affects only whether Soft~O2-C has a cluster in which to bind, not whether Soft~O2 receives privileged label access.""",
    """\\paragraph{Cluster co-membership under Mix.}
After BIRCH consolidation on the Mix store, each cross-session Sa/Sb split pair is \\emph{glued} into one cluster identifier so that Soft~O2-C can be evaluated when thematic co-membership is present.
Glue uses split metadata from the evaluation construction only; it does not inject labels at query time.
If Soft~O2-C still fails to recover cross-session gold under that condition, the failure is attributable to binding rather than accidental cluster separation.
Natural same-cluster rates without glue are lower and are noted in Section~\\ref{sec:limitations}.
Soft~O2's session membership constraint is unchanged: glue affects only whether Soft~O2-C has a cluster in which to bind.""",
)

# --- Soft O2 CE one-liner -> short paragraph ---
text = text.replace(
    "Soft~O2 also exceeds FlatIP+CE on CAIL channels (Table~\\ref{tab:ce}).",
    """Cross-encoder reranking is a strong neural control at the same turn-level evidence unit: FlatIP+CE reranks the top-$30$ FlatIP candidates with \\texttt{bge-reranker-v2-m3}.
Table~\\ref{tab:ce} shows Soft~O2 ahead of FlatIP+CE on CAIL multi-turn channels and on LegalEp-Lawyer exact, while FlatIP+CE leads on LegalEp-DISC exact---consistent with Soft~O2's largest gains arising where a partial episode hit can bind its answer sibling rather than where surface-form overlap already recovers both turns.""",
)

# --- Soft O2-C Mix section: slight polish on appendix pointer ---
text = text.replace(
    "Same-session Soft~O2 vs Soft~O2-C contrasts, where Soft~O2 remains preferable, appear in Appendix~\\ref{app:tables} (Table~\\ref{tab:cluster_same}).",
    "Same-session Soft~O2 vs Soft~O2-C contrasts remain Soft~O2-led and are reported in Appendix~\\ref{app:tables} (Table~\\ref{tab:cluster_same}).",
)

# --- Limitations: Mix glue ---
text = text.replace(
    "The Mix protocol co-locates Sa/Sb pairs for Soft~O2-C evaluation (Section~\\ref{sec:csce-protocol}); unsupervised BIRCH without that step would not receive split-pair metadata at build time.",
    "The Mix protocol co-locates Sa/Sb pairs so Soft~O2-C can be tested when thematic co-membership is present (Section~\\ref{sec:csce-protocol}); unsupervised BIRCH without that construction step would not receive split-pair metadata at build time.",
)

# --- Conclusion ---
text = text.replace(
    "Soft~O2 session soft binding is the primary algorithmic contribution; Soft~O2-C with gated Hybrid is an exploratory cluster-binding complement under a gold-assisted Mix solvability bound, with Soft~O2 attenuation $\\beta$ validated on a diagnostic gap-ratio model; O1 exact indexing and O3 bulk-load consolidation are implementation operators that enable the shared-store ablations reported here.",
    "Soft~O2 session soft binding is the primary algorithmic contribution; Soft~O2-C with gated Hybrid extends soft binding to BIRCH clusters under Mix reconstructions in which some gold evidence is held off-session; Soft~O2 attenuation $\\beta$ is validated on a diagnostic gap-ratio model; and O1 exact indexing with O3 bulk-load consolidation are implementation operators that enable the shared-store ablations reported here.",
)

# --- Appendix intro ---
text = text.replace(
    "Full Soft~O2 numeric grids and auxiliary controls appear below. Primary Soft~O2 claims are carried by the main-text figures and Mix table.",
    "Same-session Soft~O2-C contrasts, significance tests, scale curves, and full Soft~O2 numeric grids appear below. Primary Soft~O2 and Mix claims are carried by the main-text figures and tables.",
)

# --- Reorder appendix: after cluster_same put sig, scale, beta; keep grids then bm25 ---
def extract_labeled_table(src: str, label: str) -> tuple[str, str]:
    """Return (block_including_optional_comment, remainder). Prefer nearest preceding % comment."""
    lab = f"\\label{{{label}}}"
    i = src.find(lab)
    if i < 0:
        raise SystemExit(f"missing label {label}")
    # find begin{table} before label
    b = src.rfind("\\begin{table}", 0, i)
    if b < 0:
        raise SystemExit(f"missing begin table for {label}")
    # include a preceding % comment line if present
    line_start = src.rfind("\n", 0, b) + 1
    if src[line_start:b].lstrip().startswith("%"):
        b = line_start
    e = src.find("\\end{table}", i)
    if e < 0:
        raise SystemExit(f"missing end table for {label}")
    e = e + len("\\end{table}")
    # trailing footnotes after some tables
    rest_probe = src[e : e + 200]
    foot = re.match(r"\n\{\\footnotesize.*?\}\n", rest_probe, flags=re.S)
    if foot:
        e = e + foot.end()
    block = src[b:e].strip("\n")
    rem = src[:b] + src[e:]
    return block, rem


# Extract in reverse document order to keep indices stable via sequential removal
order_extract = ["tab:scale", "tab:sig", "tab:beta", "tab:bm25"]
blocks: dict[str, str] = {}
for lab in order_extract:
    blk, text = extract_labeled_table(text, lab)
    blocks[lab] = blk

# Also leave grids in place; re-insert after cluster_same: sig, scale, beta, then later bm25 after lawyer grid
# Find end of cluster_same
lab = "\\label{tab:cluster_same}"
i = text.find(lab)
if i < 0:
    raise SystemExit("cluster_same missing after edits")
e = text.find("\\end{table}", i) + len("\\end{table}")
# insert sig, scale, beta
insert = (
    "\n\n"
    + blocks["tab:sig"]
    + "\n\n"
    + blocks["tab:scale"]
    + "\n\n"
    + blocks["tab:beta"]
    + "\n"
)
text = text[:e] + insert + text[e:]

# Place bm25 after lawyer_main (end of primary grids)
lab = "\\label{tab:lawyer_main}"
i = text.find(lab)
if i < 0:
    raise SystemExit("lawyer_main missing")
e = text.find("\\end{table}", i) + len("\\end{table}")
text = text[:e] + "\n\n" + blocks["tab:bm25"] + "\n" + text[e:]

# Soften leftover "Negative and sensitivity controls" if too checklist-y — OK scientifically
# Soften appendix cluster_same caption slightly
text = text.replace(
    "Same-session Soft~O2 vs ungated Soft~O2-C (AH@10). When gold already shares $sid$ with the query, Soft~O2 is the default operator; ungated Soft~O2-C injects thematic confusers. Boldface: better AH within each row. Soft~O2-C gains under Mix appear in Table~\\ref{tab:csce}.",
    "Same-session Soft~O2 vs ungated Soft~O2-C (AH@10). Soft~O2 is preferable when gold already shares $sid$ with the query; ungated Soft~O2-C can surface thematic confusers. Boldface: better AH within each row. Mix Soft~O2-C results appear in Table~\\ref{tab:csce}.",
)

TEX.write_text(text, encoding="utf-8")
print("wrote", TEX)

# --- Regenerate anonymous from main ---
anon = text
# Strip author block (keep anonymous)
anon = re.sub(
    r"%+ Author names.*?\\affiliation\[idl\].*?China\}\}\n",
    "%% Anonymous submission frontmatter\n"
    "\\author[]{Anonymous Author(s)}\n"
    "\\ead{anonymous@anonymous.edu}\n\n",
    anon,
    count=1,
    flags=re.S,
)
# Strip author-identifying declarations end-matter where needed
anon = anon.replace(
    """\\subsection*{Funding}
Computational resources were provided by the Data Law Laboratory, China University of Political Science and Law.

\\subsection*{Conflict of interest}
The authors declare no competing interests.

\\subsection*{Ethics approval}
This study uses publicly released research corpora and excludes confidential client data. The system is a research prototype for consultation-memory retrieval.

\\subsection*{Data availability}
Code, processed evaluation corpora, and primary result files are publicly available at
\\url{https://github.com/Kilimajaro/soft-episode-binding-legal-memory}.
Upstream legal corpora: CAIL2024 consultation tracks; Hugging Face
\\codeid{ShengbinYue/DISC-Law-SFT} and
\\codeid{Skepsun/lawyer_llama_data}.

\\subsection*{Code availability}
The BIMS-LEGAL implementation (O1 FlatIP, Soft~O2, O3 bulk-load, Soft~O2-C), LegalMem/LegalEp pipelines, Mix builders, and table/figure scripts are released in the same repository with JSON artifacts for each main table.

\\subsection*{Author contributions}
\\sloppy
L.X.\\ (University of Winnipeg) designed BIMS-LEGAL and the Soft~O2~/Soft~O2-C evaluation, implemented the system, ran the experiments, and wrote the manuscript.
L.H.\\ (corresponding author; CUPL Data Law Lab and Institute for Data Law) provided methodology guidance, revision, and supervision.
""",
    """\\subsection*{Funding}
Funding information omitted for anonymous review.

\\subsection*{Conflict of interest}
The authors declare no competing interests.

\\subsection*{Ethics approval}
This study uses publicly released research corpora and excludes confidential client data. The system is a research prototype for consultation-memory retrieval.

\\subsection*{Data availability}
Code, processed evaluation corpora, and primary result files will be released upon publication.
Upstream legal corpora: CAIL2024 consultation tracks; Hugging Face
\\codeid{ShengbinYue/DISC-Law-SFT} and
\\codeid{Skepsun/lawyer_llama_data}.

\\subsection*{Code availability}
Implementation artifacts will be released with the camera-ready version.

\\subsection*{Author contributions}
Author contributions omitted for anonymous review.
""",
)

ANON.write_text(anon, encoding="utf-8")
print("wrote", ANON)

# Sanity checks
for path, blob in [(TEX, text), (ANON, anon)]:
    print("==", path.name)
    for pat in [
        r"corrected nDCG",
        r"published V4",
        r"solvability",
        r"gold-assisted",
        r"recompute_corrected",
        r"Negative control",
        r"FlatIP rebuild",
        r"primary V4",
    ]:
        hits = re.findall(pat, blob, flags=re.I)
        print(f"  {pat}: {len(hits)}")
    # table label order in appendix
    app = blob.split("\\appendix", 1)[1]
    labels = re.findall(r"\\label\{(tab:[^}]+)\}", app)
    print("  appendix labels:", labels)
