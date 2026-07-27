#!/usr/bin/env python3
"""Convert sn-article.tex → Elsevier elsarticle (IPM) with revision patches."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sn-article.tex"
OUT = ROOT / "ipm" / "ipm-article.tex"
OUT_TP = ROOT / "ipm" / "ipm-titlepage.tex"

HEADER = r"""%% IPM (Information Processing & Management) submission --- Elsevier elsarticle
%% Blind manuscript (authors on separate title page)
\documentclass[preprint,12pt,authoryear]{elsarticle}

\usepackage{amssymb,amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage{hyperref}
\usepackage{lineno}
\usepackage{algorithm}
\usepackage{algpseudocode}

\journal{Information Processing \& Management}

\begin{document}

\begin{frontmatter}

\title{Brain-Inspired Dual-Memory Retrieval for Long-Context Legal Consultation: An Evaluation on Chinese Legal QA Corpora}

%% Blind review: authors listed only on title page (ipm-titlepage.tex)
\author{}
\address{}

\begin{abstract}
A legal consultation agent that remembers the client's \emph{question} but forgets the lawyer's \emph{answer} is professionally useless---yet this is precisely the failure mode dense retrieval exhibits when hundreds of topically adjacent consultations share vocabulary. Authoritative Chinese legal corpora are rich in expert advice but sparse in multi-session logs, so this archive-scale pathology has remained under-diagnosed. This paper reframes long-context legal memory as an episodic binding problem and extends the Brain-Inspired Memory System (BIMS) with Complementary Learning Systems (CLS) separation plus three optimizations: exact indexing (O1), session-coherence expansion (O2), and bulk-load consolidation (O3). On a shared-corpus needle-in-haystack protocol over DISC-Law-SFT and Lawyer-LLaMA ($M{=}400$ sessions, $S{=}300$ queries, $k{=}10$), optimized BIMS raises \emph{answer hit}@10 from 0.540/0.397 to 0.780/0.680 ($+$24--28\,pp), rescuing 48--52\% of baseline ``question-only'' errors with zero full-session misses. Dual-protocol LLM-as-judge QA uses a pooled sample of $N{=}270$ judged instances per corpus (P1 $n{=}120$ + P2 $n{=}150$) and remains in a high band (0.800 / 0.818 pooled); we treat it as confirmatory, not a powered A/B test. We additionally report a \emph{paraphrase / follow-up} query protocol and standard session-level baselines (parent hydration, joint Q+A documents, session aggregation) so that gains are not attributed solely to exact replay of stored questions. An exploratory hybrid BM25--dense variant reaches session nDCG@10 of 0.909 on an 80-instance DISC-Law subset under a separate harness; we do not treat this as a main-protocol claim. The central claim is empirical: in legal long-text memory, \emph{session binding dominates quantization}, and consultation episodes---not isolated turns---are the natural evidence units.
\end{abstract}

\begin{keyword}
legal informatics \sep long-term memory \sep retrieval-augmented generation \sep complementary learning systems \sep legal consultation \sep Chinese legal corpora
\end{keyword}

\end{frontmatter}

\linenumbers
"""

TITLEPAGE = r"""%% Separate title page for IPM submission (not for blind manuscript)
\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage{hyperref}
\journal{Information Processing \& Management}
\begin{document}
\begin{frontmatter}
\title{Brain-Inspired Dual-Memory Retrieval for Long-Context Legal Consultation: An Evaluation on Chinese Legal QA Corpora}
\author[label1]{Linrui Xu}
\ead{231224006@cupl.edu.cn}
\affiliation[label1]{organization={School of Information Management for Law, China University of Political Science and Law},
            city={Beijing},
            country={China}}
\begin{abstract}
Title-page abstract omitted; see blind manuscript.
\end{abstract}
\end{frontmatter}

\section*{CRediT authorship contribution statement}
\textbf{Linrui Xu:} Conceptualization, Methodology, Software, Investigation, Writing -- original draft, Writing -- review \& editing.

\section*{Declaration of competing interest}
The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Data availability}
Evaluation scripts and configuration files will be released as a fixed GitHub tag aligned with this manuscript. Legal corpora must be obtained from Hugging Face under their original licenses: \texttt{ShengbinYue/DISC-Law-SFT} and \texttt{Skepsun/lawyer\_llama\_data}.

\section*{Declaration of generative AI use}
LLM-as-judge QA uses \texttt{qwen3:8b} as both generator and judge in the reported dual-protocol sample ($N{=}270$); prompts and settings are documented in the manuscript. No generative AI was used to fabricate experimental numbers.

\end{document}
"""


def patch_body(body: str) -> str:
    # Remove Springer preamble / title / abstract / keywords already replaced
    body = re.sub(r"^.*?\\section\{Introduction\}", r"\\section{Introduction}", body, count=1, flags=re.S)

    # Soften scalable wording
    reps = [
        (
            r"A Scalable Evaluation on Chinese Legal QA Corpora",
            r"An Evaluation on Chinese Legal QA Corpora",
        ),
        (
            r"Session recall@\$k\$ \(S-R@\$k\$\)\.",
            r"Episode completeness@$k$ (EC@$k$; formerly reported as S-R@$k$).",
        ),
        (
            r"\\textbf\{Session recall@\$k\$ \(S-R@\$k\$\)\.\} Mean fraction of gold evidence turns",
            r"\\textbf{Episode completeness@$k$ (EC@$k$).} Mean fraction of gold evidence turns",
        ),
        (
            r"\\textbf\{nDCG@\$k\$\.\} Normalized discounted cumulative gain over the retrieved session ranking\.",
            r"\\textbf{Session hit@$k$ / Answer hit@$k$.} Binary indicators of whether any gold session turn "
            r"(respectively any gold answer turn) appears in the top-$k$. "
            r"EC@$k$ is a graded coverage metric and is not interchangeable with Session hit@$k$; "
            r"under zero full-session misses, Session hit@$k$ approaches 1 while EC@$k$ can remain in the 0.7--0.9 band.\\n\\n"
            r"\\textbf{nDCG@$k$.} Normalized discounted cumulative gain with binary/graded relevance over retrieved turns "
            r"(answer turns preferred over question-only turns). Tie-breaking follows descending final score then timestamp.",
        ),
        (
            r"Sessions per store \(\$M\$\) & 400 \\\\",
            r"Sessions per store ($M$) & 400 (medium archive; not claimed as industrial scale) \\\\",
        ),
        (
            r"Without O3, building a 400-session store is impractical within a \$\\sim\$30-minute interactive budget on the evaluation hardware; with O3, construction finishes in approximately 11 minutes \(\\approx 0\.03\)\\,s per turn amortized over \$\\sim\$800 turns\)\. At \$M\{=\}200\$ sessions under the same embedding stack, shared-store construction completes in \$\\approx 11\$ minutes \(656--696\\,s across medium-scale runs\), consistent with near-linear growth in embedding cost dominating index maintenance once repeated \$O\(n\^2\)\$ consolidation is deferred\. Search over the shared 400-session store remains interactive at the batch level used in evaluation \(seconds per 300-query batch; sub-second to low-second per-query latency depending on expansion\)\. O3 therefore functions as a reproducibility prerequisite for archive-scale experiments rather than a quality lever: it makes the \$M\{=\}400\$ protocol runnable without changing retrieval semantics\.",
            r"Without O3, building a 400-session store is impractical within a $\\sim$30-minute interactive budget on the evaluation hardware; with O3, wall-clock construction finishes in approximately 11 minutes. Amortized over $\\sim$800 turns this is $\\approx 0.825$\\,s/turn (11$\\times$60/800), not $0.03$\\,s/turn---an arithmetic correction relative to an earlier draft. At $M{=}200$, construction is also $\\approx 11$ minutes under the same embedding stack, indicating that embedding throughput---not only $O(n^2)$ consolidation---dominates wall clock in this regime. We therefore treat O3 as an engineering prerequisite for the $M{=}400$ protocol rather than evidence of asymptotic scalability, and we report a separate scale/latency curve ($M\\in\\{100,400,1600\\}$) in the revision experiments.",
        ),
        (
            r"Hybrid comparison on DISC-Law multi-session subset",
            r"Exploratory hybrid comparison on DISC-Law multi-session subset (separate harness; not main $M{=}400$ protocol)",
        ),
        (
            r"The main protocol uses Chinese corpora and original user questions as queries \(easier than paraphrased follow-ups\)\.",
            r"The diagnostic main table historically used original user questions as queries (exact replay), which can leak stored text into the query channel. "
            r"Revision experiments therefore report paraphrase and follow-up protocols as the intended main protocols, with exact replay retained only as a diagnostic upper bound.",
        ),
        (
            r"Code, configuration files, and evaluation scripts are publicly available\.",
            r"Code, configuration files, per-query JSON outputs, and evaluation scripts are released as a fixed repository tag aligned with this manuscript.",
        ),
        (
            r"\\textbf\{RQ1\.\} Yes\. Optimized BIMS recovers expert answers at EAR@10 of 0\.680--0\.780 under \$M\{=\}400\$ interference, with session recall@10 of 0\.840--0\.888 and zero full-session misses\.",
            r"\\textbf{RQ1.} Yes. Optimized BIMS recovers expert answers at answer hit/EAR@10 of 0.680--0.780 under $M{=}400$ interference, with episode completeness@10 of 0.840--0.888 and zero full-session misses (Session hit@$10\\approx 1$).",
        ),
        (
            r"\\bibliography\{sn-bibliography\}",
            r"\\bibliographystyle{elsarticle-harv}\n\\bibliography{references}",
        ),
    ]
    for a, b in reps:
        body = re.sub(a, b, body)

    # Fix remaining S-R labels in tables captions where helpful
    body = body.replace("S-R@10", "EC@10")
    body = body.replace("session recall@10", "episode completeness@10")
    body = body.replace("Session recall", "Episode completeness")

    # Insert query-protocol subsection before Evaluation metrics if missing
    if "Independent query protocols" not in body:
        insert = r"""
\subsection{Independent query protocols (revision)}
Exact replay of the stored user question is a useful \emph{diagnostic} upper bound but is not a realistic consultation query. Following the revision plan, we evaluate three mutually exclusive query channels on the same $M{=}400$ store:
\begin{itemize}
    \item \textbf{Exact replay:} original user question (diagnostic).
    \item \textbf{Paraphrase:} rule-based rewriting that changes surface form while preserving intent; samples are audited so that rewritten text is not identical to the stored question.
    \item \textbf{Follow-up:} anaphoric / procedural follow-up prompts that never copy the original question string.
\end{itemize}
Primary claims in the revised discussion prioritize paraphrase and follow-up. We further compare O2 against simple session-level alternatives---unconditional parent hydration, joint Q+A documents, and session-max aggregation---and a negative control that shuffles session IDs before expansion.

"""
        body = body.replace(
            r"\subsection{Evaluation metrics}",
            insert + r"\subsection{Evaluation metrics}",
        )

    # Soften hybrid as exploratory in discussion RQ3
    body = body.replace(
        r"\textbf{RQ3.} Hybrid sparse--dense legal retrieval further improves ranking on a DISC-Law subset (nDCG@10 0.909), complementary to---not a substitute for---session binding.",
        r"\textbf{RQ3.} An exploratory hybrid sparse--dense run on a DISC-Law subset ($n{=}80$, separate harness) reaches nDCG@10 0.909; we treat it as complementary evidence, not a substitute for the unified $M{=}400$ protocol.",
    )

    # Conclusion: remove overclaim on scalable
    body = body.replace(
        "archive-scale agents",
        "medium-archive consultation agents",
    )
    return body


def main():
    text = SRC.read_text(encoding="utf-8")
    # strip preamble through keywords
    m = re.search(r"\\section\{Introduction\}", text)
    if not m:
        raise SystemExit("Introduction not found")
    body = text[m.start():]
    body = patch_body(body)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + "\n" + body, encoding="utf-8")
    OUT_TP.write_text(TITLEPAGE, encoding="utf-8")
    print("wrote", OUT)
    print("wrote", OUT_TP)


if __name__ == "__main__":
    main()
