#!/usr/bin/env python3
"""Rewrite IPM narrative around paraphrase-primary + soft binding novelty."""
from pathlib import Path

path = Path("/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556/paper/ipm/ipm-article.tex")
text = path.read_text(encoding="utf-8")

# --- Title ---
text = text.replace(
    r"\title{Brain-Inspired Dual-Memory Retrieval for Long-Context Legal Consultation: An Evaluation on Chinese Legal QA Corpora}",
    r"\title{Episode Binding for Legal Consultation Memory: Soft Session Coherence Beats Parent Hydration under Same-Domain Interference}",
)

# --- Abstract ---
old_abs = text[text.find(r"\begin{abstract}"): text.find(r"\end{abstract}") + len(r"\end{abstract}")]
new_abs = r"""\begin{abstract}
Legal consultation agents must recover not only a client's prior question but the lawyer's advice bound to that episode. Under hundreds of same-domain consultations, dense retrieval often hits the question turn and displaces the answer---an \emph{episode incompleteness} failure that statute and case retrieval benchmarks under-expose. We formalize legal long-text memory as episodic binding and instantiate it with \emph{soft session-coherence expansion} (O2): siblings of a retrieved turn inherit a near-parity score ($0.98\cdot s$), preserving ranking competition rather than unconditionally promoting an entire parent session. On DISC-Law-SFT and Lawyer-LLaMA with a shared $M{=}400$ store and $S{=}300$ queries, we evaluate exact, paraphrase, and follow-up query channels. On the primary \emph{paraphrase} protocol, O2 raises answer hit@10 from $0.747$ (FlatIP) to $0.800$ and substantially outperforms hard parent hydration and session-max aggregation ($0.613$), while a shuffled-session negative control collapses to $0.577$. Exact replay saturates ($>0.93$) and is treated only as a diagnostic upper bound; anaphoric follow-ups remain hard ($\approx 0.26$) for all systems. Dual-protocol LLM-as-judge QA uses $N{=}270$ judged instances per corpus (P1 $n{=}120$ + P2 $n{=}150$). The contribution is a measurable IR claim for legal informatics: \emph{soft episode binding is not reducible to parent-document hydration}, and consultation episodes---not isolated turns---are the natural evidence units under lexical interference.
\end{abstract}"""
text = text.replace(old_abs, new_abs)

# --- Keywords ---
text = text.replace(
    r"legal informatics \sep long-term memory \sep retrieval-augmented generation \sep complementary learning systems \sep legal consultation \sep Chinese legal corpora",
    r"legal information retrieval \sep consultation memory \sep episodic binding \sep session-aware retrieval \sep retrieval-augmented generation \sep Chinese legal corpora",
)

path.write_text(text, encoding="utf-8")
print("title/abstract/keywords updated")
