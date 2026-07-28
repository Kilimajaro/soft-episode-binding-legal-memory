#!/usr/bin/env python3
"""Publication-quality IPM-style figures for BIMS-LEGAL (Fig1–Fig5).

Palette inspired by IPM architecture figures (muted slate/teal/coral, nested
panels, generous padding). Outputs PDF+PNG under paper/ipm/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "ipm" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Aligned with Fig.1/Fig.2 architecture palette (Gemini / IPM redraw)
INK = "#1F2A37"
SLATE = "#4A5B6A"       # FlatIP / neutral
PANEL = "#F4F6F8"
PANEL_EC = "#D5DCE3"
TEAL = "#0E6B6E"        # episodic / Hard hydration (session structure)
TEAL_FILL = "#E4F2F2"
CORAL = "#C45C4A"       # Soft O2 highlight (matches Soft O2 panel)
CORAL_FILL = "#F8ECE9"
SAND = "#8A6A3D"
SAND_FILL = "#F5F0E6"
GREEN = "#2F6B4F"
GREEN_FILL = "#E8F1EB"
BLUE = "#3D6FA8"        # dense retrieval / secondary control
BLUE_FILL = "#E8EFF6"
GRAY_FILL = "#EEF1F4"
MUTED = "#9AA5B1"       # shuffled / IVFPQ


def _setup_fonts():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem: str):
    for ext in ("pdf", "png"):
        out = FIGDIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print("wrote", out)
    # also mirror fig1 jpg for legacy include path
    if stem == "fig1_architecture":
        for p in (ROOT / "ipm" / "fig1_architecture.jpg", FIGDIR / "fig1_architecture.jpg"):
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
            print("wrote", p)


def rbox(ax, xy, w, h, fc, ec, lw=1.2, pad=0.02, rs=0.04, z=2):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad={pad},rounding_size={rs}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, fs=8.2, fw="normal", c=INK, ha="center", va="center", italic=False, **kw):
    ax.text(
        x, y, text, fontsize=fs, fontweight=fw, color=c, ha=ha, va=va, zorder=5,
        fontstyle="italic" if italic else "normal", **kw,
    )


def arrow(ax, a, b, color=SLATE, lw=1.15, style="-|>", rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=ms,
        color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=3,
    ))


def panel_badge(ax, x, y, letter, fc=BLUE, ec=BLUE):
    circ = Circle((x, y), 0.18, facecolor=fc, edgecolor=ec, lw=1.0, zorder=6)
    ax.add_patch(circ)
    label(ax, x, y, letter, fs=9.5, fw="bold", c="white")


# ---------------------------------------------------------------------------
# Fig 1 — Architecture
# ---------------------------------------------------------------------------
def draw_fig1():
    fig, ax = plt.subplots(figsize=(12.2, 7.0), dpi=300)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # Outer canvas border (journal-like)
    rbox(ax, (0.15, 0.15), 11.9, 6.7, "white", PANEL_EC, lw=1.0, rs=0.06, z=0)

    # --- Panel A background ---
    rbox(ax, (0.35, 4.55), 11.5, 2.15, PANEL, PANEL_EC, lw=1.0, rs=0.05, z=1)
    panel_badge(ax, 0.62, 6.4, "A")
    label(ax, 1.0, 6.4, "Dual-store indexing", fs=10.5, fw="bold", ha="left")

    # Input
    rbox(ax, (0.55, 4.85), 2.15, 1.35, SAND_FILL, SAND, lw=1.25)
    label(ax, 1.625, 5.85, "Consultation turns", fs=8.4, fw="bold")
    label(ax, 1.625, 5.35, "role · text · time\nsession_id", fs=7.4, c=SLATE)

    # Episodic store with memory circles
    rbox(ax, (3.05, 4.85), 2.85, 1.35, TEAL_FILL, TEAL, lw=1.35)
    label(ax, 4.475, 5.95, "Episodic store", fs=8.6, fw="bold", c=TEAL)
    for i, (cx, cy) in enumerate([(3.55, 5.35), (4.15, 5.45), (4.75, 5.25), (5.35, 5.4)]):
        ax.add_patch(Circle((cx, cy), 0.18, facecolor="white", edgecolor=TEAL, lw=1.1, zorder=4))
        label(ax, cx, cy, f"$e_{i+1}$", fs=6.5, c=TEAL)
    label(ax, 4.475, 5.0, "vectors + role / time / $sid$", fs=6.8, c=SLATE)

    # Semantic store with cluster blobs
    rbox(ax, (6.2, 4.85), 2.85, 1.35, GREEN_FILL, GREEN, lw=1.35)
    label(ax, 7.625, 5.95, "Semantic store", fs=8.6, fw="bold", c=GREEN)
    for cx, cy, rx, ry in [(6.75, 5.35, 0.28, 0.2), (7.5, 5.4, 0.32, 0.22), (8.3, 5.3, 0.26, 0.18)]:
        ax.add_patch(Ellipse((cx, cy), rx * 2, ry * 2, facecolor="white", edgecolor=GREEN, lw=1.0, zorder=4))
    label(ax, 7.625, 5.0, "BIRCH Phase 1 / 2", fs=6.8, c=SLATE)

    # Index / O3
    rbox(ax, (9.35, 5.45), 2.25, 0.6, GRAY_FILL, SLATE, lw=1.1)
    label(ax, 10.475, 5.75, "Index O1: FlatIP / IVFPQ", fs=7.4)
    rbox(ax, (9.35, 4.85), 2.25, 0.5, GRAY_FILL, SLATE, lw=1.1)
    label(ax, 10.475, 5.1, "Bulk load O3", fs=7.4)

    for a, b in [((2.7, 5.52), (3.05, 5.52)), ((5.9, 5.52), (6.2, 5.52)), ((9.05, 5.52), (9.35, 5.52))]:
        arrow(ax, a, b, color=SLATE, lw=1.2)

    # --- Panel B ---
    rbox(ax, (0.35, 2.25), 11.5, 2.15, "#FBF7F5", "#E4D5CF", lw=1.0, rs=0.05, z=1)
    panel_badge(ax, 0.62, 4.1, "B", fc=CORAL, ec=CORAL)
    label(ax, 1.0, 4.1, "Retrieval with Soft O2 (episodic pathway)", fs=10.5, fw="bold", ha="left", c=CORAL)

    rbox(ax, (0.55, 2.5), 2.2, 1.35, SAND_FILL, SAND, lw=1.25)
    label(ax, 1.65, 3.55, "Query channels", fs=8.3, fw="bold")
    label(ax, 1.65, 3.0, "U1 · Uk · U-last\nU-para · advice", fs=7.2, c=SLATE)

    rbox(ax, (3.05, 2.5), 2.5, 1.35, BLUE_FILL, BLUE, lw=1.25)
    label(ax, 4.3, 3.55, "Dense retrieval", fs=8.3, fw="bold", c=BLUE)
    label(ax, 4.3, 3.0, "$\\alpha S+(1-\\alpha)R$\n(+ assoc. / temporal)", fs=7.2, c=SLATE)

    # Soft O2 highlight — key operator
    rbox(ax, (5.85, 2.5), 3.0, 1.35, CORAL_FILL, CORAL, lw=1.8)
    label(ax, 7.35, 3.6, "Soft O2 binding", fs=8.6, fw="bold", c=CORAL)
    label(ax, 7.35, 3.15, r"$s'\!\leftarrow\!\max(s',\,\beta\cdot s)$", fs=8.0, c=INK)
    label(ax, 7.35, 2.75, "no hard score copy", fs=7.0, c=SLATE, italic=True)

    rbox(ax, (9.15, 2.5), 2.45, 1.35, GREEN_FILL, GREEN, lw=1.25)
    label(ax, 10.375, 3.55, "Rank & metrics", fs=8.3, fw="bold", c=GREEN)
    label(ax, 10.375, 3.0, "AH@10 · EC@10\nCI · McNemar", fs=7.2, c=SLATE)

    for a, b in [((2.75, 3.17), (3.05, 3.17)), ((5.55, 3.17), (5.85, 3.17)), ((8.85, 3.17), (9.15, 3.17))]:
        arrow(ax, a, b, color=CORAL if a[0] > 5 else SLATE, lw=1.25)

    # vertical link episodic → dense
    arrow(ax, (4.475, 4.85), (4.3, 3.85), color=TEAL, lw=1.1, rad=0.05)

    # --- Panel C ---
    rbox(ax, (0.35, 0.35), 11.5, 1.75, PANEL, PANEL_EC, lw=1.0, rs=0.05, z=1)
    panel_badge(ax, 0.62, 1.8, "C")
    label(ax, 1.0, 1.8, "Evaluation order (shared-corpus)", fs=10.5, fw="bold", ha="left")

    stages = [
        (0.55, "(1) CAIL2024\nmulti-turn gold", BLUE_FILL, BLUE),
        (3.35, "(2) LegalEp-DISC\nepisode corpus", TEAL_FILL, TEAL),
        (6.15, "(3) LegalEp-Lawyer\nepisode corpus", GREEN_FILL, GREEN),
        (8.95, "(4) LongMemEval\n/ LoCoMo check", GRAY_FILL, SLATE),
    ]
    for x, txt, fc, ec in stages:
        rbox(ax, (x, 0.55), 2.5, 0.95, fc, ec, lw=1.15)
        label(ax, x + 1.25, 1.02, txt, fs=7.6)
    for x0, x1 in [(3.05, 3.35), (5.85, 6.15), (8.65, 8.95)]:
        arrow(ax, (x0, 1.02), (x1, 1.02), color=SLATE, lw=1.05)

    label(
        ax, 6.1, 0.42,
        "Controls on every legal corpus: FlatIP · Hard hydration · Session-max · BM25 · CE · shuffled $sid$",
        fs=6.8, c=SLATE, ha="center", italic=True,
    )

    save(fig, "fig1_architecture")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Benchmark protocol (updated V4 numbers)
# ---------------------------------------------------------------------------
def draw_fig2():
    fig, ax = plt.subplots(figsize=(11.8, 5.6), dpi=300)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    rbox(ax, (0.2, 0.2), 11.4, 5.2, "white", PANEL_EC, lw=1.0, rs=0.06, z=0)

    # Title strip
    label(ax, 0.45, 5.1, "Shared-corpus consultation-memory protocol (V4)", fs=11.5, fw="bold", ha="left")

    # Left column: corpora
    rbox(ax, (0.4, 1.55), 3.35, 3.25, PANEL, PANEL_EC, lw=1.1, rs=0.05, z=1)
    label(ax, 2.075, 4.5, "Source corpora", fs=9.5, fw="bold", c=BLUE)
    items = [
        (4.0, "CAIL2024 dialogues", "~600 gold multi-turn", BLUE_FILL, BLUE),
        (3.15, "LegalEp-DISC", "500 needles + distractors", TEAL_FILL, TEAL),
        (2.3, "LegalEp-Lawyer", "500 needles + distractors", GREEN_FILL, GREEN),
        (1.45, "Distractor pool", "same-domain archive", SAND_FILL, SAND),
    ]
    for y, t1, t2, fc, ec in items:
        rbox(ax, (0.6, y - 0.35), 2.95, 0.7, fc, ec, lw=1.1)
        label(ax, 2.075, y + 0.08, t1, fs=7.8, fw="bold")
        label(ax, 2.075, y - 0.18, t2, fs=6.6, c=SLATE)

    # Center: shared store
    rbox(ax, (4.1, 2.35), 3.5, 2.45, TEAL_FILL, TEAL, lw=1.6, rs=0.06, z=1)
    label(ax, 5.85, 4.45, "One shared store", fs=10, fw="bold", c=TEAL)
    label(ax, 5.85, 3.95, "Primary tier  $M\\approx 3000$", fs=8.2)
    label(ax, 5.85, 3.5, "gold needles + distractors\nindexed once", fs=7.6, c=SLATE)
    # small memory circles
    for i, cx in enumerate([5.15, 5.65, 6.15, 6.65]):
        ax.add_patch(Circle((cx, 2.85), 0.16, facecolor="white", edgecolor=TEAL, lw=1.0, zorder=4))
        label(ax, cx, 2.85, f"$m_{i+1}$", fs=6.2, c=TEAL)

    # Right: queries + metrics
    rbox(ax, (7.95, 3.35), 3.4, 1.45, CORAL_FILL, CORAL, lw=1.35, rs=0.05, z=1)
    label(ax, 9.65, 4.45, "Query channels", fs=9.2, fw="bold", c=CORAL)
    label(ax, 9.65, 3.95, "CAIL: U1 / Uk / U-last", fs=7.4)
    label(ax, 9.65, 3.6, "LegalEp: exact / advice / U-para", fs=7.4)

    rbox(ax, (7.95, 1.55), 3.4, 1.55, BLUE_FILL, BLUE, lw=1.35, rs=0.05, z=1)
    label(ax, 9.65, 2.75, "Retrieve Top-10", fs=9.2, fw="bold", c=BLUE)
    label(ax, 9.65, 2.25, "AH · EC · nDCG@10", fs=7.6)
    label(ax, 9.65, 1.85, "bootstrap CI · McNemar", fs=7.4, c=SLATE)

    # Bottom controls bar
    rbox(ax, (0.4, 0.35), 10.95, 1.0, GRAY_FILL, SLATE, lw=1.0, rs=0.04, z=1)
    label(ax, 5.875, 1.1, "Systems compared (turn-level)", fs=8.0, fw="bold")
    label(
        ax, 5.875, 0.72,
        "FlatIP · Soft O2 · Hard hydration · Session-max",
        fs=7.0, c=SLATE,
    )
    label(
        ax, 5.875, 0.48,
        "BM25 · dense+BM25 RRF · FlatIP+CE · shuffled $sid$",
        fs=7.0, c=SLATE,
    )

    # Arrows
    arrow(ax, (3.75, 3.55), (4.1, 3.55), color=SLATE, lw=1.3)
    arrow(ax, (7.6, 3.9), (7.95, 4.0), color=CORAL, lw=1.25)
    arrow(ax, (9.65, 3.35), (9.65, 3.1), color=BLUE, lw=1.2)
    arrow(ax, (7.6, 3.2), (7.95, 2.5), color=TEAL, lw=1.15, rad=-0.15)

    # Scale note (no overflow)
    label(
        ax, 5.85, 2.15,
        "Scale tiers: $S$ / $M$ / $L$  (gold fixed)",
        fs=6.8, c=SLATE, italic=True,
    )

    save(fig, "fig2_benchmark_protocol")
    plt.close(fig)


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PANEL_EC)
    ax.spines["bottom"].set_color(PANEL_EC)
    ax.yaxis.grid(True, color=PANEL, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SLATE, labelsize=8.5)


def _grouped_bars(ax, data, labels, colors, edges, width=0.18):
    """data: (n_cat, n_sys). Soft O2 gets a slightly stronger edge."""
    x = np.arange(data.shape[0])
    n = data.shape[1]
    offsets = (np.arange(n) - (n - 1) / 2) * width
    handles = []
    for i, (lab, c, ec) in enumerate(zip(labels, colors, edges)):
        lw = 1.15 if "Soft" in lab else 0.55
        bars = ax.bar(
            x + offsets[i], data[:, i], width=width * 0.92,
            color=c, edgecolor=ec, linewidth=lw, label=lab, zorder=3,
        )
        handles.append(bars)
        for b, v in zip(bars, data[:, i]):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.2, color=INK, clip_on=False,
            )
    ax.set_xticks(x)
    return handles


# ---------------------------------------------------------------------------
# Fig 3 — CAIL main results (primary visualization)
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def draw_fig3():
    import json
    root = _repo_root()
    cail = json.loads((root / "results/bims_legal_v4/cail_M/tier_M/results.json").read_text())
    channels = ["U1", "Uk-followup", "U-last"]
    ch_keys = ["u1_exact", "uk_followup", "u_last"]
    systems = ["FlatIP", "Soft O2", "Hard hydr.", "Shuffled O2"]
    cfg_keys = ["dense_flat", "dense_o2", "parent_hydrate", "shuffled_o2"]
    ah, ec = [], []
    for ch in ch_keys:
        cf = cail["channels"][ch]["configs"]
        ah.append([cf[k]["answer_hit@k"] for k in cfg_keys])
        ec.append([cf[k]["episode_completeness@k"] for k in cfg_keys])
    ah = np.array(ah)
    ec = np.array(ec)
    # FlatIP=slate, Soft O2=coral, Hard=teal (session), Shuffled=muted
    colors = [SLATE, CORAL, TEAL, MUTED]
    edges = [SLATE, "#8E3A30", TEAL, MUTED]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=300, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, data, title, ylab in [
        (axes[0], ah, "(a) Answer Hit@10", "Score"),
        (axes[1], ec, "(b) Episode Completeness@10", ""),
    ]:
        _grouped_bars(ax, data, systems, colors, edges, width=0.18)
        ax.set_xticklabels(channels, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
        if ylab:
            ax.set_ylabel(ylab, fontsize=9, color=INK)
        _style_axes(ax)

    axes[0].legend(
        frameon=False, fontsize=8, loc="upper left", ncol=2,
        columnspacing=1.0, handlelength=1.2,
    )
    fig.suptitle(
        "CAIL2024 multi-turn results ($M\\approx 3000$, turn-level systems)",
        fontsize=11.2, color=INK, y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig3_cail_main")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — LegalEp DISC vs Lawyer (exact / advice / para)
# ---------------------------------------------------------------------------
def draw_fig4():
    import json
    root = _repo_root()

    def load(name):
        return json.loads((root / f"results/bims_legal_v4/{name}/tier_M/results.json").read_text())

    disc_exact = load("legalep_disc_M")
    disc_advice = load("legalep_disc_advice")
    disc_para = load("legalep_disc_para")
    law_exact = load("legalep_lawyer_M")
    law_advice = load("legalep_lawyer_advice")
    law_para = load("legalep_lawyer_para")

    def row(blob, ch, keys=("dense_flat", "dense_o2", "parent_hydrate")):
        cf = blob["channels"][ch]["configs"]
        return [cf[k]["answer_hit@k"] for k in keys]

    channels = ["exact", "advice-recall", "u-para"]
    systems = ["FlatIP", "Soft O2", "Hard hydr."]
    disc_ah = np.array([
        row(disc_exact, "u1_exact" if "u1_exact" in disc_exact["channels"] else "exact"),
        row(disc_advice, "advice_recall"),
        row(disc_para, "u_para"),
    ], dtype=float)
    lawyer_ah = np.array([
        row(law_exact, "exact"),
        row(law_advice, "advice_recall"),
        row(law_para, "u_para"),
    ], dtype=float)

    colors = [SLATE, CORAL, TEAL]
    edges = [SLATE, "#8E3A30", TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=300, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, data, title in [
        (axes[0], disc_ah, "(a) LegalEp-DISC AH@10"),
        (axes[1], lawyer_ah, "(b) LegalEp-Lawyer AH@10"),
    ]:
        _grouped_bars(ax, data, systems, colors, edges, width=0.22)
        ax.set_xticklabels(channels, fontsize=9)
        ax.set_ylim(0, 0.92)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=8)
        _style_axes(ax)

    axes[0].set_ylabel("Answer Hit@10", fontsize=9, color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.2)
    fig.suptitle(
        "LegalEp ($M{=}3000$): Soft O2 gains are strongest on paraphrase and remain positive on advice-recall",
        fontsize=10.2, color=INK, y=1.01,
    )
    fig.tight_layout()
    save(fig, "fig4_legalep_main")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — Scale curve
# ---------------------------------------------------------------------------
def draw_fig5():
    M = np.array([100, 400, 1600, 6400])
    flat = np.array([0.963, 0.890, 0.833, 0.733])
    soft = np.array([0.970, 0.970, 0.917, 0.803])
    ivf = np.array([0.613, 0.463, 0.330, 0.227])
    p50_flat = np.array([67, 248, 941, 4458])
    p50_soft = np.array([63, 236, 985, 4061])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), dpi=300)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(M, soft, "o-", color=CORAL, lw=2.2, ms=7, markerfacecolor=CORAL,
            markeredgecolor="#8E3A30", markeredgewidth=0.8, label="Soft O2", zorder=4)
    ax.plot(M, flat, "s--", color=SLATE, lw=1.7, ms=6.5, markerfacecolor=SLATE,
            label="FlatIP", zorder=3)
    ax.plot(M, ivf, "^:", color=MUTED, lw=1.5, ms=6.5, markerfacecolor=MUTED,
            label="IVFPQ", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(M)
    ax.set_xticklabels([str(m) for m in M])
    ax.set_ylim(0.15, 1.05)
    ax.set_xlabel("Archive size $M$ (sessions)", fontsize=9, color=INK)
    ax.set_ylabel("AH@10", fontsize=9, color=INK)
    ax.set_title("(a) Quality vs scale (DISC exact)", fontsize=10.5, fontweight="bold",
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)

    ax = axes[1]
    ax.plot(M, p50_soft, "o-", color=CORAL, lw=2.2, ms=7, markerfacecolor=CORAL,
            markeredgecolor="#8E3A30", markeredgewidth=0.8, label="Soft O2 p50", zorder=4)
    ax.plot(M, p50_flat, "s--", color=SLATE, lw=1.7, ms=6.5, markerfacecolor=SLATE,
            label="FlatIP p50", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(M)
    ax.set_xticklabels([str(m) for m in M])
    ax.set_xlabel("Archive size $M$ (sessions)", fontsize=9, color=INK)
    ax.set_ylabel("Warm-query latency (ms)", fontsize=9, color=INK)
    ax.set_title("(b) Latency vs scale", fontsize=10.5, fontweight="bold",
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    ax.yaxis.grid(True, color=PANEL, lw=0.8, which="both")

    fig.tight_layout()
    save(fig, "fig5_scale_curve")
    plt.close(fig)


def main():
    _setup_fonts()
    # Fig.1/2 are user-supplied Gemini artwork; only regenerate result figures.
    draw_fig3()
    draw_fig4()
    draw_fig5()
    print("done")


if __name__ == "__main__":
    main()
