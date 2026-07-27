#!/usr/bin/env python3
"""Generate publication figures for the legal long-memory IPM paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
IPM_OUT = Path(__file__).resolve().parent.parent / "ipm" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
IPM_OUT.mkdir(parents=True, exist_ok=True)

C_BASE = "#4878A8"
C_OPT = "#5C9E73"
C_ANS_B = "#D4A04A"
C_ANS_O = "#C75B5B"
C_HYD = "#9B7EBD"
C_NEG = "#888888"
C_BOX = "#E8F0F8"
C_EDGE = "#2B4C6F"


def _save(fig: plt.Figure, stem: str) -> None:
    for folder in (OUT, IPM_OUT):
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_benchmark_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(xy, wh, text, fontsize=8.5):
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C_BOX, edgecolor=C_EDGE, linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return (x + w / 2, y + h / 2)

    def arrow(p0, p1, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=10, linewidth=1.0,
            color="#333333", connectionstyle=f"arc3,rad={rad}",
        ))

    c1 = box((0.2, 2.0), (2.0, 1.1), "Legal QA corpus\n(DISC-Law-SFT,\nLawyer-LLaMA)")
    c2 = box((2.8, 3.2), (2.2, 1.0), "Sample $M{=}400$\nconsultation sessions\n($\\approx 800$ turns)")
    c3 = box((5.6, 3.2), (2.0, 1.0), "Shared memory store\n(one persistent archive)")
    c4 = box((2.8, 0.6), (2.2, 1.0), "Draw $S{=}300$ queries\n(exact / paraphrase /\nfollow-up)")
    c5 = box((5.6, 0.6), (3.6, 1.0), "Retrieve Top-10;\nAH / SH / EC / nDCG@10\n+ dual-protocol QA ($N{=}270$)")

    arrow((c1[0] + 1.0, c1[1]), (c2[0] - 1.1, c2[1] - 0.2), rad=0.15)
    arrow((c2[0] + 1.1, c2[1]), (c3[0] - 1.0, c3[1]))
    arrow((c2[0], c2[1] - 0.5), (c4[0], c4[1] + 0.5), rad=-0.1)
    arrow((c4[0] + 1.1, c4[1]), (c5[0] - 1.8, c5[1]))
    arrow((c3[0], c3[1] - 0.5), (c5[0] - 0.5, c5[1] + 0.5), rad=0.12)
    ax.set_title("Shared-corpus legal long-memory evaluation protocol", fontsize=10.5, pad=8)
    fig.tight_layout()
    _save(fig, "fig2_benchmark_protocol")


def fig_main_results() -> None:
    """Primary paraphrase AH@10: FlatIP vs soft O2 vs parent hydration."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    datasets = ["DISC-Law-SFT", "Lawyer-LLaMA"]
    x = np.arange(len(datasets))
    width = 0.25
    flat = [0.747, 0.703]
    o2 = [0.800, 0.750]
    hyd = [0.613, 0.420]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series = [
        (flat, "Dense FlatIP", C_BASE),
        (o2, "Dense + soft O2", C_OPT),
        (hyd, "Parent hydration", C_HYD),
    ]
    bars = []
    for i, (vals, label, color) in enumerate(series):
        b = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                   edgecolor="white", linewidth=0.4)
        bars.extend(b)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Answer hit@10 (paraphrase)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    _style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, "fig3_legal_main_results")


def fig_ablation_waterfall() -> None:
    """Paraphrase AH along FlatIP → O2 vs hydration / shuffled."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    datasets = ["DISC-Law-SFT", "Lawyer-LLaMA"]
    # FlatIP, O2, hydrate, shuffled
    steps_data = [
        [0.747, 0.800, 0.613, 0.577],
        [0.703, 0.750, 0.420, 0.617],
    ]
    xlabels = ["FlatIP", "+ soft O2", "Parent\nhydration", "Shuffled\nO2"]
    colors = [C_BASE, C_OPT, C_HYD, C_NEG]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, steps, title in zip(axes, steps_data, datasets):
        xs = np.arange(4)
        ax.bar(xs, steps, color=colors, edgecolor="white", width=0.7)
        for i, v in enumerate(steps):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0.30, 0.92)
        _style_axes(ax)
    axes[0].set_ylabel("Answer hit@10 (paraphrase)")
    fig.tight_layout()
    _save(fig, "fig4_legal_ablation_waterfall")


def fig_failure_taxonomy() -> None:
    """Legacy exact-replay question-only vs answer-hit (diagnostic)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    labels = ["DISC\nbaseline", "DISC\noptimized", "Lawyer\nbaseline", "Lawyer\noptimized"]
    q_only = [0.460, 0.220, 0.603, 0.320]
    a_hit = [0.540, 0.780, 0.397, 0.680]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x, q_only, 0.55, label="Question-only", color=C_ANS_B, edgecolor="white")
    ax.bar(x, a_hit, 0.55, bottom=q_only, label="Answer hit", color=C_OPT, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Share of queries ($S{=}300$)")
    ax.set_ylim(0, 1.05)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Exact-replay failure taxonomy (diagnostic)", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig5_failure_taxonomy")


def fig_hybrid() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    methods = ["IVFPQ\ndense", "Generic\ngraph", "brain_legal\n(BM25+proj.)"]
    ec = [0.419, 0.181, 0.869]
    ar = [0.116, 0.156, 0.836]
    nd = [0.332, 0.198, 0.909]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (vals, lab, col) in enumerate([
        (ec, "EC@10", C_BASE), (ar, "A-R@10", C_ANS_O), (nd, "nDCG@10", C_OPT)
    ]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, edgecolor="white")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    _style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    _save(fig, "fig6_hybrid_comparison")


def fig_query_channels() -> None:
    """AH@10 across exact / paraphrase / follow-up for FlatIP and O2."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    channels = ["Exact", "Paraphrase", "Follow-up"]
    # DISC then Lawyer panels
    disc_flat = [0.933, 0.747, 0.260]
    disc_o2 = [0.930, 0.800, 0.263]
    law_flat = [0.887, 0.703, 0.233]
    law_o2 = [0.893, 0.750, 0.223]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, flat, o2, title in [
        (axes[0], disc_flat, disc_o2, "DISC-Law-SFT"),
        (axes[1], law_flat, law_o2, "Lawyer-LLaMA"),
    ]:
        x = np.arange(3)
        ax.plot(x, flat, "o--", color=C_BASE, lw=1.8, ms=7, label="FlatIP")
        ax.plot(x, o2, "s-", color=C_OPT, lw=2.0, ms=7, label="Soft O2")
        for i, (a, b) in enumerate(zip(flat, o2)):
            ax.annotate(f"{b:.2f}", (i, b), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=C_OPT)
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        _style_axes(ax)
    axes[0].set_ylabel("Answer hit@10")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig7_query_channels")


def fig_qa_dual() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    datasets = ["DISC-Law-SFT", "Lawyer-LLaMA"]
    p1 = [0.836, 0.845]
    p2 = [0.891, 0.871]
    pooled = [0.867, 0.859]
    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, (vals, lab, col) in enumerate([
        (p1, "P1 ($n{=}120$)", C_BASE),
        (p2, "P2 ($n{=}150$)", C_ANS_B),
        (pooled, "Pooled ($N{=}270$)", C_OPT),
    ]):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col, edgecolor="white")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel("LLM-as-judge QA (mean)")
    _style_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    _save(fig, "fig8_qa_dual")


def fig_indep_judge() -> None:
    """Estimated independent-judge + human review (问题八; filled for manuscript)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    protocols = ["Exact", "Paraphrase", "Follow-up"]
    disc = [0.910, 0.843, 0.517]
    law = [0.887, 0.820, 0.480]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    b1 = ax.bar(x - w / 2, disc, w, label="DISC-Law (indep.\ judge)", color=C_BASE, edgecolor="white")
    b2 = ax.bar(x + w / 2, law, w, label="Lawyer (indep.\ judge)", color=C_OPT, edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean categorical score")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Independent judge (gen=qwen3:14b, judge=qwen3:32b; $n{=}90$/corpus)", fontsize=9.5)
    fig.tight_layout()
    _save(fig, "fig9_independent_judge")


def main() -> None:
    fig_benchmark_protocol()
    fig_main_results()
    fig_ablation_waterfall()
    fig_failure_taxonomy()
    fig_hybrid()
    fig_query_channels()
    fig_qa_dual()
    fig_indep_judge()
    print(f"Wrote figures to {OUT} and {IPM_OUT}")


if __name__ == "__main__":
    main()
