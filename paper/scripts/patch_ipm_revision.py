#!/usr/bin/env python3
"""Apply IPM revision-checklist patches to ipm-article.tex."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/ipm/ipm-article.tex"

ABSTRACT = r"""\begin{abstract}
\sloppy
Recovering prior client-specific advice from a legal consultation archive is difficult because dense retrieval may rank a relevant question turn highly while omitting the lawyer's answer---an \emph{episode incompleteness} failure under same-domain interference.
BIMS-LEGAL is a dual-store system for Chinese legal consultation logs: a fast episodic turn index with session identifiers and a slower BIRCH semantic store.
Soft~O2 performs turn-level session binding by giving siblings attenuated inherited scores ($\beta s$) rather than hard score copy.
We evaluate exact replay as diagnostic and use paraphrase, follow-up, and advice-recall as primary non-exact channels.
BM25-joint is a strong lexical baseline on exact and high-overlap queries, but Soft~O2 is more effective on non-exact channels where online turn-level retrieval can recover a partial episode and then bind its answer sibling.
Soft~O2-C and gated Hybrid are reported as exploratory, gold-assisted solvability bounds for cross-session recovery, not as claims about natural unsupervised deployment.
We report corrected graded nDCG alongside Answer Hit and Episode Completeness, with paired significance tests and a failure taxonomy separating complete, incomplete, and missed-session retrieval.
\end{abstract}"""

CONTRIBUTIONS = r"""\paragraph{Contributions.}
\begin{enumerate}
    \item A dual-store memory architecture for Chinese legal consultation archives, separating fast episodic turn recovery from slower BIRCH semantic integration under same-domain interference.
    \item Soft~O2, a turn-level session soft-binding operator that addresses episode incompleteness without hard parent hydration or session-max score copy.
    \item A conservative three-candidate diagnostic of Soft~O2 score competition using the gap ratio $\rho=s_d/s_h$; default $\beta{=}0.98$ is chosen by validation sweep, not by a closed-form optimum.
    \item An honest comparison with strong lexical baselines: BM25-joint and dense joint-episode indexing are highly competitive on exact/surface-overlap channels, while Soft~O2 targets paraphrase, follow-up, and advice-recall under an online turn-level index.
    \item Exploratory Soft~O2-C / gated Hybrid results under a gold-assisted Mix solvability bound, plus corrected nDCG, failure taxonomy, and paired significance reporting.
    \item O1 exact FlatIP and O3 bulk-load consolidation as implementation operators enabling the shared-store ablations reported here.
\end{enumerate}

Section~\ref{sec:related} reviews related work.
Section~\ref{sec:method} presents BIMS-LEGAL operators and the Soft~O2 diagnostic model.
Section~\ref{sec:protocol} details corpora, corrected metrics, and protocols."""

BASE_SCORE = r"""For query $q$ with embedding $\mathbf{v}_q$, episodic base retrieval scores a memory turn $m$ by
\begin{equation}\label{eq:base-score}
\mathrm{score}(m)=\alpha\, S(\mathbf{v}_q,\mathbf{v}_m)+(1-\alpha)\, R(t_m),
\end{equation}
where $S$ is cosine similarity on $\ell_2$-normalized embeddings and
\begin{equation}
R(t_m)=\max\!\left(0,\;1-\frac{\Delta_t}{30\ \mathrm{days}}\right),
\end{equation}
with $\Delta_t$ the elapsed time between query time and the turn timestamp.
Unless stated otherwise, $\alpha{=}0.7$.
Soft~O2 and Soft~O2-C rewrite sibling scores after this fusion step; they do not replace Eq.~\eqref{eq:base-score}."""

BETA_SECTION = r"""\subsection{Soft O2 attenuation as diagnostic score competition}\label{sec:beta-theory}

The attenuation $\beta$ controls how strongly a retrieved turn promotes its session siblings.
We treat the following as a \emph{diagnostic} model, not a proof of a universal optimum.
In particular, a zero-margin hinge objective with $\beta\in(0,1]$ does not identify a meaningful quantile solution: when the rank margin is zero, the rank term is inactive for all $\beta<1$, so its weight cannot determine a closed-form optimum.

\begin{definition}[Three-candidate diagnostic model]\label{def:three-cand}
For a query whose dense retrieval directly hits a gold turn $h$ with fused score $s_h>0$, let $a$ be the gold answer sibling and let $d$ be the strongest non-session distractor in a wide candidate pool, with scores $s_a$ and $s_d$.
Soft~O2 rewrites
\begin{equation}\label{eq:soft-rewrite}
s'_h=s_h,\qquad s'_a=\max(s_a,\beta s_h),\qquad s'_d=s_d,
\end{equation}
and defines the gap ratio $\rho=s_d/s_h$.
Because $d$ is the strongest measured distractor, conditions based on $\rho$ are conservative sufficient conditions for beating that distractor, not exact top-$k$ boundary conditions for every competitor.
\end{definition}

The binding analysis is most relevant in the soft-injection regime $s_a\le\beta s_h$, where $s'_a=\beta s_h$.

\begin{proposition}[Feasible attenuation band]\label{prop:feasible}
Under Definition~\ref{def:three-cand}, assume the soft-injection case $s_a\le\beta s_h$.
Then (i)~availability against the measured distractor requires $\beta>\rho$; (ii)~rank preservation of the direct hit over the injected sibling requires $\beta<1$, and more generally $s'_a<s_h$; (iii)~when $\rho<1$, any $\beta\in(\rho,1)$ satisfies both conditions in the active injection case.
If $s_a>s_h$ already, rank preservation depends on the observed sibling score rather than on $\beta$ alone.
\end{proposition}

\begin{proof}
If $s_a\le\beta s_h$, then $s'_a=\beta s_h$.
Availability follows from $\beta s_h>s_d$, i.e.\ $\beta>\rho$.
Rank preservation requires $s'_a<s_h$, which reduces to $\beta<1$ when $s_h>0$.
\end{proof}

\paragraph{Exploratory soft-rank diagnostic.}
We also minimize an exploratory soft pairwise diagnostic on a validation grid,
\begin{equation}\label{eq:softrank}
\mathcal{L}_{\mathrm{sr}}(\beta;\rho)
=-\log\sigma\!\bigl(\tau(\beta-\rho)\bigr)
-\gamma\log\sigma\!\bigl(\tau(1-\beta)\bigr),
\end{equation}
with logistic $\sigma$, temperature $\tau$, and rank weight $\gamma$.
This visualization is useful for studying the availability--ranking trade-off, but it depends on the diagnostic pool, query channel, and chosen hyperparameters.
Default $\beta{=}0.98$ is selected from the sweep in Table~\ref{tab:beta}, where $\{0.90,0.95,0.98\}$ form a stable high-AH band.

\begin{table}[t]
\centering
\caption{Diagnostic gap-ratio statistics on LegalEp $M{=}3000$ stores (exact-channel query proxy; strongest measured non-session distractor). These are calibration diagnostics, not derivations of an optimal $\beta$.}
\label{tab:beta-theory}
\compacttab\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Corpus} & $\rho_{50}$ & $\rho_{95}$ & Cover@$0.90$ & Default $\beta$ \\
\midrule
LegalEp-DISC & $0.746$ & $0.854$ & $0.981$ & $0.98$ \\
LegalEp-Lawyer & $0.662$ & $0.815$ & $0.998$ & $0.98$ \\
\bottomrule
\end{tabular}
\end{table}"""

NDCG_METRICS = r"""Primary metrics are Answer Hit@$k$ (AH), Episode Completeness@$k$ (EC), and corrected normalized Discounted Cumulative Gain@$k$ (nDCG) at $k{=}10$.
AH is binary: whether a gold answer turn appears in the top-$k$.
EC is the mean coverage of gold turns within the target session.
Session Hit@$k$ is reported only for diagnosis.
For nDCG we use fixed graded relevance over the full gold session: answer turns $rel{=}1.0$, other same-session turns $rel{=}0.5$, all other turns $rel{=}0.0$.
The ideal denominator (IDCG) is computed from all gold-relevant turns in the target session and truncated at $k$, so every system on a query shares the same IDCG.
Earlier draft values based on retrieved-item IDCG are not comparable and have been recomputed (artifact \path{paper/ipm/figures/corrected_metrics_*.json}).
Primary contrasts use bootstrap 95\% CIs and McNemar mid-$p$ tests on paired per-query hits; where families of related hypotheses are interpreted jointly, we also report Holm-corrected $p$-values (Table~\ref{tab:sig})."""

RQ1_BLOCK = r"""\subsection{RQ1 failure taxonomy}\label{sec:rq1-failure-taxonomy}
We classify each query into \emph{complete} (answer evidence in top-$k$), \emph{incomplete} (target session partially retrieved but answer missing), or \emph{session miss} (no target-session turn in top-$k$).
Table~\ref{tab:rq1-failure} shows that incomplete retrieval is a substantial recurrent failure mode for FlatIP on primary non-exact channels, motivating Soft~O2.

\begin{table}[t]
\centering
\caption{RQ1 failure taxonomy (FlatIP, $k{=}10$, corrected metric script). Incomplete = question/session hit without answer hit on two-turn LegalEp episodes and analogous CAIL channels.}
\label{tab:rq1-failure}
\compacttab\footnotesize
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Corpus / channel} & \textbf{System} & \textbf{Complete} & \textbf{Incomplete} & \textbf{Session miss} \\
\midrule
LegalEp-DISC / u-para & FlatIP & \corrndcg{disc}{u_para}{dense_flat}{complete} & \corrndcg{disc}{u_para}{dense_flat}{incomplete} & \corrndcg{disc}{u_para}{dense_flat}{session_miss} \\
LegalEp-DISC / advice & FlatIP & \corrndcg{disc}{advice_recall}{dense_flat}{complete} & \corrndcg{disc}{advice_recall}{dense_flat}{incomplete} & \corrndcg{disc}{advice_recall}{dense_flat}{session_miss} \\
LegalEp-Lawyer / u-para & FlatIP & \corrndcg{lawyer}{u_para}{dense_flat}{complete} & \corrndcg{lawyer}{u_para}{dense_flat}{incomplete} & \corrndcg{lawyer}{u_para}{dense_flat}{session_miss} \\
CAIL / Uk-followup & FlatIP & \corrndcg{cail}{uk_followup}{dense_flat}{complete} & \corrndcg{cail}{uk_followup}{dense_flat}{incomplete} & \corrndcg{cail}{uk_followup}{dense_flat}{session_miss} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Strong lexical and joint-episode baselines}\label{sec:s1-strong-baselines}
BM25-joint indexes each question--answer episode as one lexical document and is therefore very strong on exact replay and other high-overlap channels.
Soft~O2 instead assumes an online turn-level index and uses session binding to recover missing answer turns after a dense hit.
Table~\ref{tab:bm25-tradeoff} reports this trade-off on the same store, query set, and $k{=}10$.
Soft~O2 is not claimed to dominate BM25-joint universally; its advantage is on non-exact channels and on preserving turn-level ranking flexibility.

\begin{table}[t]
\centering
\caption{BM25-joint / joint-QA vs Soft~O2 (AH@10, primary tier). Joint baselines are strongest on exact/surface overlap; Soft~O2 targets paraphrase and advice-recall.}
\label{tab:bm25-tradeoff}
\compacttab\footnotesize
\begin{tabular}{@{}llccc@{}}
\toprule
\textbf{Corpus / channel} & \textbf{BM25-joint} & \textbf{Joint QA} & \textbf{Soft O2} \\
\midrule
CAIL / U1 & $0.993$ & --- & $0.788$ \\
CAIL / Uk & $0.840$ & --- & $0.698$ \\
CAIL / U-last & $0.823$ & --- & $0.762$ \\
LegalEp-DISC / exact & $0.976$ & --- & $0.638$ \\
LegalEp-DISC / u-para & $0.592$ & --- & $0.648$ \\
LegalEp-DISC / advice & $0.432$ & --- & $0.428$ \\
LegalEp-Lawyer / exact & --- & --- & $0.752$ \\
LegalEp-Lawyer / u-para & --- & --- & $0.759$ \\
LegalEp-Lawyer / advice & $0.652$ & --- & $0.472$ \\
\bottomrule
\end{tabular}
\end{table}"""


def fill_failure_table(tex: str) -> str:
    import json

    def pct(obj, *keys, default="---"):
        cur = obj
        for k in keys:
            if cur is None or k not in cur:
                return default
            cur = cur[k]
        if isinstance(cur, (int, float)):
            return f"{100*cur:.1f}\\%"
        return default

    disc = {}
    p = ROOT / "paper/ipm/figures/corrected_metrics_disc.json"
    if p.exists():
        disc = json.loads(p.read_text()).get("channels", {})
    repl = {
        r"\corrndcg{disc}{u_para}{dense_flat}{complete}": pct(
            disc.get("u_para", {}).get("dense_flat", {}).get("failure_taxonomy"), "complete"
        ),
        r"\corrndcg{disc}{u_para}{dense_flat}{incomplete}": pct(
            disc.get("u_para", {}).get("dense_flat", {}).get("failure_taxonomy"), "incomplete"
        ),
        r"\corrndcg{disc}{u_para}{dense_flat}{session_miss}": pct(
            disc.get("u_para", {}).get("dense_flat", {}).get("failure_taxonomy"), "session_miss"
        ),
        r"\corrndcg{disc}{advice_recall}{dense_flat}{complete}": pct(
            disc.get("advice_recall", {}).get("dense_flat", {}).get("failure_taxonomy"), "complete"
        ),
        r"\corrndcg{disc}{advice_recall}{dense_flat}{incomplete}": pct(
            disc.get("advice_recall", {}).get("dense_flat", {}).get("failure_taxonomy"), "incomplete"
        ),
        r"\corrndcg{disc}{advice_recall}{dense_flat}{session_miss}": pct(
            disc.get("advice_recall", {}).get("dense_flat", {}).get("failure_taxonomy"), "session_miss"
        ),
    }
    for k, v in repl.items():
        tex = tex.replace(k, v)
    tex = re.sub(r"\\corrndcg\{[^}]+\}\{[^}]+\}\{[^}]+\}\{[^}]+\}", "---", tex)
    return tex


def main():
    tex = TEX.read_text(encoding="utf-8")
    tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda _: ABSTRACT, tex, flags=re.S)
    tex = re.sub(
        r"\\paragraph\{Contributions\.\}.*?Section~\\ref\{sec:conclusion\} concludes\.",
        lambda _: CONTRIBUTIONS + "\nSection~\\ref{sec:conclusion} concludes.",
        tex,
        flags=re.S,
    )
    tex = re.sub(
        r"For query \$q\$ with embedding.*?they do not replace Eq\.~\\eqref\{eq:base-score\}\.",
        lambda _: BASE_SCORE,
        tex,
        flags=re.S,
    )
    tex = re.sub(
        r"\\subsection\{Ranking-competition derivation of \$\\beta\$\}.*?\\end\{table\}\n\nFair controls include",
        lambda _: BETA_SECTION + "\n\nFair controls include",
        tex,
        flags=re.S,
    )
    tex = re.sub(
        r"Primary metrics are Answer Hit@\$k\$.*?McNemar mid-\$p\$ tests on paired per-query hits\.",
        lambda _: NDCG_METRICS,
        tex,
        flags=re.S,
    )
    tex = tex.replace(
        "We formalize Soft~O2 as a three-candidate ranking-competition operator and derive a corpus-adaptive $\\beta^{\\star}$",
        "Soft~O2 uses attenuated session binding with default $\\beta{=}0.98$ chosen by validation sweep",
    )
    # Insert RQ1 + BM25 block before exp-main if not present
    if "sec:rq1-failure-taxonomy" not in tex:
        tex = tex.replace(
            "\\subsection{Soft O2 on session (CAIL and LegalEp)}\\label{sec:exp-main}",
            RQ1_BLOCK + "\n\n\\subsection{Soft O2 on session (CAIL and LegalEp)}\\label{sec:exp-main}",
        )
    tex = fill_failure_table(tex)
    # Discussion / QA wording
    tex = tex.replace(
        "RQ1--RQ2 confirm that episode incompleteness dominates under same-domain interference.",
        "RQ1--RQ2 show that episode incompleteness is a substantial recurrent failure mode under same-domain interference (Table~\\ref{tab:rq1-failure}).",
    )
    tex = tex.replace(
        "The three-candidate ranking-competition analysis (Section~\\ref{sec:beta-theory}) shows that default $\\beta{=}0.98$ coincides with the soft-rank optimum under light rank regularization and lies in the high-coverage region of the fitted gap-ratio law, explaining why the empirical sweep peaks in $\\{0.9,0.95,0.98\\}$ rather than at hard hydration ($\\beta{=}1$) or aggressive attenuation.",
        "The diagnostic gap-ratio analysis (Section~\\ref{sec:beta-theory}) explains why $\\beta$ should remain close to but below unity, and why the validation sweep peaks in $\\{0.9,0.95,0.98\\}$ rather than at hard hydration ($\\beta{=}1$).",
    )
    tex = tex.replace(
        "reduces a common RAG failure mode",
        "is associated with lower rates of a common RAG failure mode",
    )
    tex = tex.replace(
        "linking complete episode retrieval to reduced hallucination",
        "linking complete episode retrieval to judged answerability under one generator--judge setup",
    )
    tex = tex.replace(
        "with Soft~O2 attenuation $\\beta$ grounded in a three-candidate ranking-competition loss whose soft-rank optimum matches the empirical operating band;",
        "with Soft~O2 attenuation $\\beta$ validated on a diagnostic gap-ratio model;",
    )
    # Related work paragraph
    if "callan1994passage" not in tex:
        tex = tex.replace(
            "Soft episode binding (Soft~O2) inherits $\\beta\\cdot s$ for siblings and keeps ranking competition among candidates.",
            "Soft episode binding (Soft~O2) inherits $\\beta\\cdot s$ for siblings and keeps ranking competition among candidates. "
            "Parent-document and hierarchical retrieval similarly attach child evidence to a parent unit \\cite{callan1994passage,dai2019context,zaheer2017deep,karpukhin2020dpr}; "
            "Soft~O2 differs by using attenuated turn-level score propagation rather than hard hydration or atomic session documents.",
        )
    # Remove acknowledgements from main body if present
    tex = re.sub(r"\\section\*\{Acknowledgements\}.*?\\section\*\{CRediT", lambda _: "\\section*{CRediT", tex, flags=re.S)
    TEX.write_text(tex, encoding="utf-8")
    print(f"[patched] {TEX}")


if __name__ == "__main__":
    main()
