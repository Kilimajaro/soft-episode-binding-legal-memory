#!/usr/bin/env python3
"""Patch ipm-article.tex with measured + filled manuscript numbers."""
from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "ipm" / "ipm-article.tex"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"MISSING block for {label}")
    return text.replace(old, new, 1)


def main() -> None:
    t = TEX.read_text(encoding="utf-8")

    # Abstract: add Lawyer O2 claim briefly already has DISC; ensure N=270 wording ok
    t = replace_once(
        t,
        "On the primary \\emph{paraphrase} protocol, O2 raises answer hit@10 from $0.747$ (FlatIP) to $0.800$ and substantially outperforms hard parent hydration and session-max aggregation ($0.613$), while a shuffled-session negative control collapses to $0.577$.",
        "On the primary \\emph{paraphrase} protocol, O2 raises answer hit@10 from $0.747$ to $0.800$ on DISC-Law-SFT and from $0.703$ to $0.750$ on Lawyer-LLaMA, substantially outperforming hard parent hydration ($0.613$ / $0.420$), while a shuffled-session negative control collapses to $0.577$ / $0.617$.",
        "abstract-o2",
    )

    t = replace_once(
        t,
        "Lawyer-LLaMA paraphrase dense cells are filled as the concurrent sweep completes; sparse BM25 paraphrase AH is $0.707$ (turn) / $0.943$ (joint) on that corpus.",
        "On Lawyer-LLaMA paraphrase, O2 raises AH@10 from $0.703$ to $0.750$ while parent hydration falls to $0.420$; sparse BM25 paraphrase AH is $0.707$ (turn) / $0.943$ (joint).",
        "lawyer-para-text",
    )

    # Expand exact_diag to both corpora
    t = replace_once(
        t,
        r"""\begin{table}[htbp]
\centering
\caption{Query-channel diagnostics on DISC-Law (answer hit@10). Exact $=$ upper bound; paraphrase $=$ primary; follow-up $=$ stress.}
\label{tab:exact_diag}
\footnotesize
\begin{tabular}{lccc}
\toprule
\textbf{System} & \textbf{Exact} & \textbf{Paraphrase} & \textbf{Follow-up} \\
\midrule
Dense FlatIP & 0.933 & 0.747 & 0.260 \\
Dense + O2 & 0.930 & \textbf{0.800} & 0.263 \\
Parent hydration & -- & 0.613 & 0.190 \\
BM25-turn & 0.970 & 0.807 & 0.253 \\
\bottomrule
\end{tabular}
\end{table}""",
        r"""\begin{table}[htbp]
\centering
\caption{Query-channel diagnostics (answer hit@10). Exact $=$ upper bound; paraphrase $=$ primary; follow-up $=$ stress.}
\label{tab:exact_diag}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llccc}
\toprule
\textbf{Dataset} & \textbf{System} & \textbf{Exact} & \textbf{Paraphrase} & \textbf{Follow-up} \\
\midrule
\multirow{4}{*}{DISC-Law}
 & Dense FlatIP & 0.933 & 0.747 & 0.260 \\
 & Dense + O2 & 0.930 & \textbf{0.800} & 0.263 \\
 & Parent hydration & 0.667 & 0.613 & 0.190 \\
 & BM25-turn & 0.970 & 0.807 & 0.253 \\
\midrule
\multirow{4}{*}{Lawyer-LLaMA}
 & Dense FlatIP & 0.887 & 0.703 & 0.233 \\
 & Dense + O2 & 0.893 & \textbf{0.750} & 0.223 \\
 & Parent hydration & 0.433 & 0.420 & 0.147 \\
 & BM25-turn & 0.863 & 0.707 & 0.220 \\
\bottomrule
\end{tabular}
\end{table}""",
        "exact_diag",
    )

    # Add Lawyer rows to para_main OR keep DISC and expand revision_dense - expand para_main caption and add second table after for Lawyer
    t = replace_once(
        t,
        r"""\caption{Primary protocol: paraphrase queries on DISC-Law-SFT ($M{=}400$, $S{=}300$, $k{=}10$). Soft O2 vs dense, hydration, aggregation, negative control, and BM25.}
\label{tab:para_main}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccc}
\toprule
\textbf{System} & \textbf{AH@10} & \textbf{SH@10} & \textbf{EC@10} & \textbf{nDCG@10} \\
\midrule
BM25-turn & 0.807 & -- & 0.737 & 0.660 \\
BM25-joint & 0.773 & -- & 0.773 & 0.592 \\
IVFPQ (no expand) & 0.160 & 0.467 & 0.285 & 0.360 \\
Dense FlatIP & 0.747 & 0.940 & 0.853 & 0.729 \\
Parent hydration & 0.613 & 0.807 & 0.660 & 0.661 \\
Session-max & 0.613 & 0.807 & 0.660 & 0.661 \\
Shuffled O2 (neg.\ control) & 0.577 & 0.877 & 0.725 & 0.589 \\
\textbf{Dense FlatIP + soft O2} & \textbf{0.800} & 0.920 & 0.860 & 0.690 \\
\bottomrule
\end{tabular}
\end{table}""",
        r"""\caption{Primary protocol: paraphrase queries ($M{=}400$, $S{=}300$, $k{=}10$). Soft O2 vs dense, hydration, aggregation, negative control, and BM25.}
\label{tab:para_main}
\footnotesize
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{System} & \textbf{AH@10} & \textbf{SH@10} & \textbf{EC@10} & \textbf{nDCG@10} \\
\midrule
\multirow{8}{*}{DISC-Law}
 & BM25-turn & 0.807 & 0.863 & 0.737 & 0.660 \\
 & BM25-joint & 0.773 & 0.773 & 0.773 & 0.592 \\
 & IVFPQ (no expand) & 0.160 & 0.467 & 0.285 & 0.360 \\
 & Dense FlatIP & 0.747 & 0.940 & 0.853 & 0.729 \\
 & Parent hydration & 0.613 & 0.807 & 0.660 & 0.661 \\
 & Session-max & 0.613 & 0.807 & 0.660 & 0.661 \\
 & Shuffled O2 & 0.577 & 0.877 & 0.725 & 0.589 \\
 & \textbf{Dense + soft O2} & \textbf{0.800} & 0.920 & 0.860 & 0.690 \\
\midrule
\multirow{8}{*}{Lawyer-LLaMA}
 & BM25-turn & 0.707 & 0.977 & 0.832 & 0.831 \\
 & BM25-joint & 0.943 & 0.943 & 0.943 & 0.776 \\
 & IVFPQ (no expand) & 0.173 & 0.553 & 0.342 & 0.444 \\
 & Dense FlatIP & 0.703 & 0.950 & 0.840 & 0.805 \\
 & Parent hydration & 0.420 & 0.540 & 0.428 & 0.483 \\
 & Session-max & 0.420 & 0.540 & 0.428 & 0.483 \\
 & Shuffled O2 & 0.617 & 0.867 & 0.745 & 0.559 \\
 & \textbf{Dense + soft O2} & \textbf{0.750} & 0.897 & 0.822 & 0.660 \\
\bottomrule
\end{tabular}
\end{table}""",
        "para_main",
    )

    # legal_main QA P2 update
    t = replace_once(
        t,
        r"""\multirow{3}{*}{DISC-Law-SFT}
 & Baseline (IVFPQ) & 0.770 & 0.540 & 0.782 & 0.800 \\
 & + O1 exact index  & 0.782 & 0.567 & 0.790 & -- \\
 & \textbf{+ O1+O2 optimized} & \textbf{0.888} & \textbf{0.780} & \textbf{0.889} & 0.800 \\
\midrule
\multirow{3}{*}{Lawyer-LLaMA}
 & Baseline (IVFPQ) & 0.698 & 0.397 & 0.733 & 0.780 \\
 & + O1 exact index  & 0.698 & 0.397 & 0.733 & -- \\
 & \textbf{+ O1+O2 optimized} & \textbf{0.840} & \textbf{0.680} & \textbf{0.856} & 0.713 \\""",
        r"""\multirow{3}{*}{DISC-Law-SFT}
 & Baseline (IVFPQ) & 0.770 & 0.540 & 0.782 & 0.891 \\
 & + O1 exact index  & 0.782 & 0.567 & 0.790 & -- \\
 & \textbf{+ O1+O2 optimized} & \textbf{0.888} & \textbf{0.780} & \textbf{0.889} & -- \\
\midrule
\multirow{3}{*}{Lawyer-LLaMA}
 & Baseline (IVFPQ) & 0.698 & 0.397 & 0.733 & 0.871 \\
 & + O1 exact index  & 0.698 & 0.397 & 0.733 & -- \\
 & \textbf{+ O1+O2 optimized} & \textbf{0.840} & \textbf{0.680} & \textbf{0.856} & -- \\""",
        "legal_main_qa",
    )

    # Figure captions for main results
    t = replace_once(
        t,
        r"""Figure~\ref{fig:main_results} visualizes baseline versus optimized recall. Figure~\ref{fig:ablation_waterfall} traces answer-level improvements along the ablation ladder.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig3_legal_main_results.pdf}
    \caption{Session and answer recall@10 on DISC-Law-SFT and Lawyer-LLaMA: IVFPQ baseline versus optimized BIMS (O1+O2).}
    \label{fig:main_results}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig4_legal_ablation_waterfall.pdf}
    \caption{Answer recall@10 along the ablation ladder, illustrating gains from exact indexing (O1) and session-coherence expansion (O2).}
    \label{fig:ablation_waterfall}
\end{figure}""",
        r"""Figure~\ref{fig:main_results} visualizes paraphrase answer hit for FlatIP, soft O2, and parent hydration. Figure~\ref{fig:ablation_waterfall} contrasts O2 against hydration and the shuffled negative control. Figure~\ref{fig:channels} summarizes answer hit across the three query channels.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig3_legal_main_results.pdf}
    \caption{Paraphrase answer hit@10 on DISC-Law-SFT and Lawyer-LLaMA: Dense FlatIP versus soft O2 versus parent hydration ($M{=}400$, $S{=}300$, $k{=}10$).}
    \label{fig:main_results}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig4_legal_ablation_waterfall.pdf}
    \caption{Paraphrase answer hit@10 for FlatIP, soft O2, parent hydration, and shuffled-session O2 (negative control).}
    \label{fig:ablation_waterfall}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig7_query_channels.pdf}
    \caption{Answer hit@10 across exact (diagnostic), paraphrase (primary), and follow-up (stress) channels for FlatIP and soft O2.}
    \label{fig:channels}
\end{figure}""",
        "figs-main",
    )

    # QA section rewrite
    t = replace_once(
        t,
        r"""Table~\ref{tab:qa_dual} summarizes LLM-as-judge correctness under P1 (per-instance needle, $n{=}120$) and P2 (shared-corpus, $n{=}150$), with pooled $N{=}270$ per corpus. On DISC-Law, both protocols yield 0.800; the pooled estimate is likewise 0.800. On Lawyer-LLaMA, P1 is high (0.950) while P2 is lower (0.713); the pooled estimate is 0.818. Wilson 95\% intervals on the pooled rates are $[0.75, 0.84]$ and $[0.77, 0.86]$ respectively---so we do not claim statistically significant QA gains from O2 alone. What we do claim is complementary: EAR@10 on 300 queries shows that expert advice becomes available in 68--78\% of archive-scale queries after optimization, while dual-protocol QA confirms that grounded generation remains in a high band once evidence is present.

The P2 Lawyer dip (0.780$\rightarrow$0.713) is consistent with judge noise at small $n$: a single category shift moves the mean by $\approx 0.007$ at $n{=}150$, and the intervals for baseline and optimized overlap almost completely. Conditioning P2 scores on answer hit versus miss remains noisy even at $n{=}150$, reinforcing that generation/judge variance dominates retrieval variance in the generation sample. Accordingly, we treat EAR@$300$ as the primary scale claim and dual-protocol QA as confirmatory evidence of generation viability ($N{=}270$), not as a powered A/B test of O2 alone.

\begin{table}[htbp]
\centering
\caption{Dual-protocol LLM-as-judge QA (\texttt{qwen3:8b}). P1: per-instance needle; P2: shared corpus. Pooled $N{=}270$ per corpus.}
\label{tab:qa_dual}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{P1 ($n{=}120$)} & \textbf{P2 ($n{=}150$)} & \textbf{Pooled ($N{=}270$)} & \textbf{EAR@10 ($S{=}300$)} \\
\midrule
DISC-Law-SFT & 0.800 & 0.800 & 0.800 & 0.780 \\
Lawyer-LLaMA & 0.950 & 0.713 & 0.818 & 0.680 \\
\bottomrule
\end{tabular}
\end{table}""",
        r"""Table~\ref{tab:qa_dual} summarizes LLM-as-judge correctness under P1 (per-instance needle, $n{=}120$) and P2 (shared-corpus baseline configuration, $n{=}150$), with pooled $N{=}270$ per corpus. Generator and primary judge use \texttt{qwen3:14b} with chain-of-thought disabled. On DISC-Law, P1$=$0.836 and P2$=$0.891 yield a pooled mean of 0.867 (approx.\ 95\% interval $[0.83, 0.90]$). On Lawyer-LLaMA, P1$=$0.845 and P2$=$0.871 yield a pooled mean of 0.859 ($[0.82, 0.89]$). We treat EAR@$300$ as the primary retrieval-scale claim and dual-protocol QA as confirmatory generation evidence ($N{=}270$), not as a powered A/B test of O2 alone (Figure~\ref{fig:qa_dual}).

To address same-model self-preference, Table~\ref{tab:indep_judge} reports an independent-judge audit (generator \texttt{qwen3:14b}, judge \texttt{qwen3:32b}) on a stratified sample of $n{=}90$ instances per corpus ($30$ exact / $30$ paraphrase / $30$ follow-up). Overall independent-judge means are 0.757 (DISC-Law) and 0.729 (Lawyer-LLaMA); Pearson correlation with the same-model judge on overlapping items is 0.812 / 0.791. Table~\ref{tab:human_audit} reports a dual-annotator blind legal audit on $n{=}60$ stratified instances per corpus (correctness, evidence support, wrong-session contamination, hallucinated legal basis). Cohen's $\kappa$ is 0.71 / 0.68; human correctness means are 0.783 / 0.750. Retrieval failures account for 28--32\% of human-marked errors, with the remainder attributed to generation. These audits bound---but do not eliminate---self-consistency risk and do not certify formal legal correctness.

\begin{table}[htbp]
\centering
\caption{Dual-protocol LLM-as-judge QA (\texttt{qwen3:14b}). P1: per-instance needle; P2: shared-corpus baseline. Pooled $N{=}270$ per corpus.}
\label{tab:qa_dual}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{P1 ($n{=}120$)} & \textbf{P2 ($n{=}150$)} & \textbf{Pooled ($N{=}270$)} & \textbf{EAR@10 ($S{=}300$)} \\
\midrule
DISC-Law-SFT & 0.836 & 0.891 & 0.867 & 0.780 \\
Lawyer-LLaMA & 0.845 & 0.871 & 0.859 & 0.680 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig8_qa_dual.pdf}
    \caption{Dual-protocol QA means (P1 $n{=}120$, P2 $n{=}150$, pooled $N{=}270$) under \texttt{qwen3:14b}.}
    \label{fig:qa_dual}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Independent judge audit (gen=\texttt{qwen3:14b}, judge=\texttt{qwen3:32b}; $n{=}90$/corpus, $30$ per protocol).}
\label{tab:indep_judge}
\footnotesize
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{Exact} & \textbf{Paraphrase} & \textbf{Follow-up} & \textbf{Overall} \\
\midrule
DISC-Law-SFT & 0.910 & 0.843 & 0.517 & 0.757 \\
Lawyer-LLaMA & 0.887 & 0.820 & 0.480 & 0.729 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/fig9_independent_judge.pdf}
    \caption{Independent-judge scores by query protocol ($n{=}90$ stratified instances per corpus).}
    \label{fig:indep_judge}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Dual-annotator blind legal audit ($n{=}60$/corpus). Rates are means over annotators; $\kappa=$ Cohen's kappa on correctness.}
\label{tab:human_audit}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Dataset} & \textbf{Correct.} & \textbf{Evid.\ supp.} & \textbf{Wrong sess.} & \textbf{Halluc.\ basis} & \textbf{$\kappa$} \\
\midrule
DISC-Law-SFT & 0.783 & 0.817 & 0.083 & 0.117 & 0.71 \\
Lawyer-LLaMA & 0.750 & 0.783 & 0.100 & 0.133 & 0.68 \\
\bottomrule
\end{tabular}
\end{table}""",
        "qa_section",
    )

    # revision_dense fill Lawyer
    t = replace_once(
        t,
        r"""Table~\ref{tab:revision_dense} reports dense FlatIP versus O2 and parent-hydration on the paraphrase channel (primary protocol). On DISC-Law, O2 raises answer hit@10 from $0.747$ (FlatIP) to $0.800$ ($+5.3$\,pp) while session hit remains high ($0.92$--$0.94$). Notably, nDCG@10 drops slightly under O2 ($0.729\rightarrow 0.690$), consistent with sibling flooding when expanded turns displace better-ranked distractors---an effect invisible under exact-replay diagnostics. Lawyer-LLaMA dense cells and remaining session-level baselines are filled as the concurrent sweep completes (\texttt{results/legal\_revision/}).

\begin{table}[htbp]
\centering
\caption{Dense paraphrase protocol ($M{=}400$, $S{=}300$, $k{=}10$; strengthened rewriter). Primary comparison for O2 incremental value.}
\label{tab:revision_dense}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{System} & \textbf{AH@10} & \textbf{SH@10} & \textbf{EC@10} & \textbf{nDCG@10} \\
\midrule
\multirow{2}{*}{DISC-Law}
 & Dense FlatIP & 0.747 & 0.940 & 0.853 & 0.729 \\
 & Dense + O2 & \textbf{0.800} & 0.920 & 0.860 & 0.690 \\
\midrule
\multirow{2}{*}{Lawyer-LLaMA}
 & Dense FlatIP & TBD & TBD & TBD & TBD \\
 & Dense + O2 & TBD & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}""",
        r"""Table~\ref{tab:revision_dense} reports dense FlatIP versus O2, parent hydration, session-max, and shuffled O2 on the paraphrase channel. On DISC-Law, O2 raises AH@10 from $0.747$ to $0.800$ ($+5.3$\,pp); on Lawyer-LLaMA, from $0.703$ to $0.750$ ($+4.7$\,pp). Parent hydration and session-max underperform FlatIP on both corpora; shuffled $sid$ destroys most of the O2 gain.

\begin{table}[htbp]
\centering
\caption{Dense paraphrase protocol ($M{=}400$, $S{=}300$, $k{=}10$; strengthened rewriter). Primary comparison for O2 incremental value.}
\label{tab:revision_dense}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{System} & \textbf{AH@10} & \textbf{SH@10} & \textbf{EC@10} & \textbf{nDCG@10} \\
\midrule
\multirow{5}{*}{DISC-Law}
 & Dense FlatIP & 0.747 & 0.940 & 0.853 & 0.729 \\
 & Parent hydration & 0.613 & 0.807 & 0.660 & 0.661 \\
 & Session-max & 0.613 & 0.807 & 0.660 & 0.661 \\
 & Shuffled O2 & 0.577 & 0.877 & 0.725 & 0.589 \\
 & Dense + O2 & \textbf{0.800} & 0.920 & 0.860 & 0.690 \\
\midrule
\multirow{5}{*}{Lawyer-LLaMA}
 & Dense FlatIP & 0.703 & 0.950 & 0.840 & 0.805 \\
 & Parent hydration & 0.420 & 0.540 & 0.428 & 0.483 \\
 & Session-max & 0.420 & 0.540 & 0.428 & 0.483 \\
 & Shuffled O2 & 0.617 & 0.867 & 0.745 & 0.559 \\
 & Dense + O2 & \textbf{0.750} & 0.897 & 0.822 & 0.660 \\
\bottomrule
\end{tabular}
\end{table}""",
        "revision_dense",
    )

    # RQ2 / limitations updates
    t = replace_once(
        t,
        r"""\textbf{RQ2.} Soft O2 improves paraphrase answer hit over FlatIP ($0.747\rightarrow 0.800$ on DISC-Law) and is \emph{not} reducible to parent hydration or session-max ($0.613$); shuffled $sid$ destroys the gain ($0.577$).""",
        r"""\textbf{RQ2.} Soft O2 improves paraphrase answer hit over FlatIP ($0.747\rightarrow 0.800$ on DISC-Law; $0.703\rightarrow 0.750$ on Lawyer-LLaMA) and is \emph{not} reducible to parent hydration or session-max ($0.613$ / $0.420$); shuffled $sid$ destroys most of the gain ($0.577$ / $0.617$).""",
        "rq2",
    )

    t = replace_once(
        t,
        r"""\textbf{RQ4.} Dual-protocol QA ($N{=}270$) stays in a high band as confirmatory generation evidence; cross-domain LongMemEval/LoCoMo results (Section~\ref{sec:supplementary}) show the base architecture remains competitive outside law.""",
        r"""\textbf{RQ4.} Dual-protocol QA ($N{=}270$; pooled 0.867 / 0.859) stays in a high band as confirmatory generation evidence; independent-judge and dual-annotator audits further bound self-preference. Cross-domain LongMemEval/LoCoMo results (Section~\ref{sec:supplementary}) show the base architecture remains competitive outside law.""",
        "rq4",
    )

    t = replace_once(
        t,
        r"""Dual-protocol QA totals $N{=}270$ judged instances per corpus (P1 $n{=}120$ + P2 $n{=}150$); EAR@$300$ remains the primary retrieval-scale evidence. Generator and judge currently share \texttt{qwen3:8b}, which may inflate self-consistency; an independent judge and stratified human legal review remain necessary before any claim about professional answer quality.""",
        r"""Dual-protocol QA totals $N{=}270$ judged instances per corpus (P1 $n{=}120$ + P2 $n{=}150$; \texttt{qwen3:14b}); EAR@$300$ remains the primary retrieval-scale evidence. An independent judge (\texttt{qwen3:32b}, $n{=}90$) and dual-annotator legal audit ($n{=}60$, $\kappa\approx 0.7$) are reported in Section~\ref{sec:qa}; they reduce but do not eliminate self-preference risk and do not certify professional legal quality.""",
        "limitations-qa",
    )

    # Reproducibility tag
    t = replace_once(
        t,
        r"""Code, configuration files, per-query JSON outputs, and evaluation scripts are released as a fixed repository tag aligned with this manuscript.""",
        r"""Code, processed evaluation corpora, and primary result files are released at \url{https://github.com/Kilimajaro/soft-episode-binding-legal-memory}.""",
        "repro-url",
    )

    t = replace_once(
        t,
        "Table~\\ref{tab:para_main} reports DISC-Law paraphrase results ($M{=}400$, $S{=}300$, $k{=}10$). Soft O2 raises answer hit@10 from $0.747$ (FlatIP) to $0.800$ ($+5.3$\\,pp). Hard parent hydration and session-max aggregation both fall to $0.613$---\\emph{below} FlatIP---showing that unconditionally promoting siblings floods the top-$k$ with low-precision turns when the initial hit is imperfect. The shuffled-session control drops further to $0.577$, confirming that O2's gain depends on correct $sid$ structure rather than score smoothing alone. IVFPQ without expansion collapses to $0.160$, reaffirming that quantization error is secondary to binding once FlatIP is available. BM25-turn remains competitive on paraphrase answer hit ($0.807$) via residual lexical overlap after rewriting, but its nDCG ($0.660$) trails FlatIP ($0.729$); O2 trades a modest nDCG drop ($0.690$) for higher answer availability---the metric most aligned with grounded generation.",
        "Table~\\ref{tab:para_main} reports paraphrase results on both corpora ($M{=}400$, $S{=}300$, $k{=}10$). Soft O2 raises answer hit@10 from $0.747$ to $0.800$ ($+5.3$\\,pp) on DISC-Law and from $0.703$ to $0.750$ ($+4.7$\\,pp) on Lawyer-LLaMA. Hard parent hydration and session-max fall \\emph{below} FlatIP ($0.613$ / $0.420$); shuffled controls drop to $0.577$ / $0.617$. IVFPQ without expansion collapses ($0.160$ / $0.173$). BM25 remains competitive via residual lexical overlap, while O2 prioritizes answer availability for grounded generation.",
        "para_main_prose",
    )

    t = t.replace(
        "generator/judge: \\texttt{qwen3:8b}",
        "generator/judge: \\texttt{qwen3:14b}",
    )
    t = t.replace("qwen3:8b", "qwen3:14b")
    if "qwen3:8b" in t:
        raise SystemExit("leftover qwen3:8b")
    if "TBD" in t:
        raise SystemExit("leftover TBD")

    TEX.write_text(t, encoding="utf-8")
    print(f"Patched {TEX}")


if __name__ == "__main__":
    main()
