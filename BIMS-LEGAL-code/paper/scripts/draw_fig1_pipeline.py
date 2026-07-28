#!/usr/bin/env python3
"""Fig.1 — BIMS-LEGAL end-to-end pipeline (IPM-style, clean academic look)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTS = [
    ROOT / "fig1_architecture.jpg",
    ROOT / "ipm" / "figures" / "fig1_architecture.jpg",
    ROOT / "ipm" / "fig1_architecture.jpg",
]


def box(ax, xy, w, h, text, fc="#EEF3F8", ec="#2F4A6A", fs=8.0, fw="normal", tc="#1a1a1a"):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor=fc, edgecolor=ec, linewidth=1.25, mutation_aspect=0.35, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, linespacing=1.3, zorder=3)


def arrow(ax, a, b, color="#445566", lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=10,
        color=color, linewidth=lw, linestyle=ls, shrinkA=1.5, shrinkB=1.5, zorder=1,
    ))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=240)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    ax.set_title(
        "Figure 1. BIMS-LEGAL: dual-store memory, O1--O3 operators, Soft O2 / Soft O2-C evaluation",
        fontsize=11.5, fontweight="bold", pad=12, loc="left",
    )

    # Background panels
    ax.add_patch(Rectangle((0.25, 4.55), 13.0, 1.95, facecolor="#F4F7FA", edgecolor="#D0D7E0", lw=0.8, zorder=0))
    ax.add_patch(Rectangle((0.25, 2.35), 13.0, 2.0, facecolor="#FBF6F6", edgecolor="#E5D4D4", lw=0.8, zorder=0))
    ax.add_patch(Rectangle((0.25, 0.2), 13.0, 1.95, facecolor="#F6F6F6", edgecolor="#D8D8D8", lw=0.8, zorder=0))

    ax.text(0.4, 6.3, "(A) Dual-store architecture + O1/O3", fontsize=9.5, fontweight="bold", color="#2F4A6A")
    ax.text(0.4, 4.15, "(B) Retrieval with Soft O2 / Soft O2-C", fontsize=9.5, fontweight="bold", color="#8B2E2E")
    ax.text(0.4, 1.95, "(C) Evaluation order (integrity-first)", fontsize=9.5, fontweight="bold", color="#444")

    # --- A ---
    box(ax, (0.45, 4.75), 2.35, 1.35,
        "Consultation turns\n(role, text, time,\nsession_id)",
        fc="#F3EDE2", ec="#7A6240", fs=7.8)
    box(ax, (3.05, 4.75), 2.55, 1.35,
        "Episodic store\nturn vectors +\nrole / time / sid",
        fc="#E7EEF6", ec="#2F4A6A", fs=7.8, fw="bold")
    box(ax, (5.85, 4.75), 2.55, 1.35,
        "Semantic store\nBIRCH Phase 1/2\nslow consolidation",
        fc="#E6F0E9", ec="#2F5D3A", fs=7.8, fw="bold")
    box(ax, (8.65, 4.75), 2.2, 1.35,
        "Index (O1)\nFlatIP / IVFPQ",
        fc="#ECE7F4", ec="#55406E", fs=7.8)
    box(ax, (11.05, 4.75), 2.0, 1.35,
        "Bulk load (O3)\nfinalize_bulk_load",
        fc="#ECE7F4", ec="#55406E", fs=7.8)
    for a, b in [((2.8, 5.42), (3.05, 5.42)), ((5.6, 5.42), (5.85, 5.42)),
                 ((8.4, 5.42), (8.65, 5.42)), ((10.85, 5.42), (11.05, 5.42))]:
        arrow(ax, a, b)

    # --- B ---
    box(ax, (0.45, 2.55), 2.35, 1.4,
        "Query channels\nU1 / Uk-followup\nU-last / U-para",
        fc="#F7EFE3", ec="#8A5A20", fs=7.8)
    box(ax, (3.05, 2.55), 2.7, 1.4,
        "Dense retrieval\nsummary + assoc.\n+ vector (+ temporal)",
        fc="#E7EEF6", ec="#2F4A6A", fs=7.8)
    box(ax, (6.0, 2.55), 3.0, 1.4,
        "Soft O2 / Soft O2-C\ns' ← max(s', β·s)\nsession or cluster",
        fc="#F8E8E8", ec="#8B2E2E", fs=8.0, fw="bold")
    box(ax, (9.25, 2.55), 3.8, 1.4,
        "Rank & evaluate\nAH@k · EC@k · nDCG@k\nbootstrap CI · McNemar",
        fc="#E8F1EA", ec="#2F5D3A", fs=7.8, fw="bold")
    for a, b in [((2.8, 3.25), (3.05, 3.25)), ((5.75, 3.25), (6.0, 3.25)),
                 ((9.0, 3.25), (9.25, 3.25))]:
        arrow(ax, a, b)
    arrow(ax, (4.35, 4.75), (4.35, 3.95), color="#2F4A6A")

    # --- C ---
    box(ax, (0.45, 0.4), 2.9, 1.3,
        "① O1--O3 ablation\nFlatIP + Soft O2\nM=400 shared store",
        fc="#E3ECF7", ec="#1F4E79", fs=7.6, fw="bold")
    box(ax, (3.55, 0.4), 2.9, 1.3,
        "② Soft O2 on session\nCAIL multi-turn +\nLegalEp paraphrase",
        fc="#FAFAFA", ec="#555", fs=7.6)
    box(ax, (6.65, 0.4), 2.9, 1.3,
        "③ Cluster pathway\nsame-session Soft O2-C\nvs Soft O2",
        fc="#FAFAFA", ec="#555", fs=7.6)
    box(ax, (9.75, 0.4), 3.3, 1.3,
        "④ Fair Mix Soft O2-C\ncross-/same-session gold\nHybrid gated binding",
        fc="#FAFAFA", ec="#555", fs=7.6)
    arrow(ax, (3.35, 1.05), (3.55, 1.05), color="#666")
    arrow(ax, (6.45, 1.05), (6.65, 1.05), color="#666")
    arrow(ax, (9.55, 1.05), (9.75, 1.05), color="#666")

    # Controls footnote strip
    ax.text(
        6.75, 0.08,
        "Controls: FlatIP · Hard hydration · BM25 · RRF · CE · shuffled sid  |  Exact replay is diagnostic only",
        ha="center", va="bottom", fontsize=7.2, color="#555", style="italic",
    )

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
