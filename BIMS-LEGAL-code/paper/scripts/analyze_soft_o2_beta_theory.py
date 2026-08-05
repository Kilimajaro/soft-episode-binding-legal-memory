#!/usr/bin/env python3
"""Soft-O2 β diagnostics from a three-candidate ranking-competition model (faiss-only).

coverage_quantile_diagnostic is NOT an optimizer and is NOT used to choose β;
default β=0.98 comes from the AH validation sweep.

Original one-liner:

Does not load VectorMemoryManager (slow on large workdirs). Uses:
  - vectors/vector.index + vectors/metadata.json
  - talk.txt for consecutive user/assistant episode pairing
  - corpus_manifest gold user texts to select needle episodes

Writes paper/ipm/figures/soft_o2_beta_theory.json for the manuscript.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def coverage_quantile_diagnostic(rho: np.ndarray, w_avail: float = 1.0, w_rank: float = 0.05) -> Tuple[float, dict]:
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho[np.isfinite(rho) & (rho >= 0)]
    if rho.size == 0:
        return float("nan"), {}
    pi = float(np.clip(w_avail / (w_avail + w_rank), 0.5, 0.99))
    beta = float(np.clip(np.quantile(rho, pi), 1e-3, 0.999))
    return beta, {
        "n": int(rho.size),
        "cover_quantile_pi": pi,
        "rho_p50": float(np.median(rho)),
        "rho_p90": float(np.quantile(rho, 0.90)),
        "rho_p95": float(np.quantile(rho, 0.95)),
        "rho_p98": float(np.quantile(rho, 0.98)),
        "coverage_quantile_value": beta,
        "note": "diagnostic coverage quantile only; not an optimizer; not used to choose beta",
        "w_avail": w_avail,
        "w_rank": w_rank,
    }


def beta_star_softrank(
    rho: np.ndarray, tau: float = 20.0, gamma: float = 0.05
) -> Tuple[float, dict]:
    rho = np.asarray(rho, dtype=np.float64)
    rho = rho[np.isfinite(rho) & (rho >= 0)]
    if rho.size == 0:
        return float("nan"), {}
    grid = np.linspace(0.50, 0.999, 250)
    losses = []
    for b in grid:
        avail = -np.log(_sigmoid(tau * (b - rho)) + 1e-12)
        rank = -np.log(_sigmoid(tau * (1.0 - b)) + 1e-12)
        losses.append(float(np.mean(avail) + gamma * rank))
    losses = np.asarray(losses)
    i = int(np.argmin(losses))
    return float(grid[i]), {
        "n": int(rho.size),
        "tau": tau,
        "gamma": gamma,
        "beta_star_softrank": float(grid[i]),
        "loss_min": float(losses[i]),
        "loss_at_0.90": float(losses[int(np.argmin(np.abs(grid - 0.90)))]),
        "loss_at_0.95": float(losses[int(np.argmin(np.abs(grid - 0.95)))]),
        "loss_at_0.98": float(losses[int(np.argmin(np.abs(grid - 0.98)))]),
        "loss_at_1.00": float(losses[int(np.argmin(np.abs(grid - 0.999)))]),
    }


def parse_talk_episodes(talk_path: Path) -> List[dict]:
    """Pair consecutive user/assistant lines into episodes."""
    episodes = []
    pending_user = None
    for line in talk_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        head, text = line.split("|", 1)
        try:
            meta = json.loads(head)
        except json.JSONDecodeError:
            continue
        role = (meta.get("role") or "").lower()
        tid = meta.get("tid")
        if role == "user":
            pending_user = {"tid": tid, "text": text.strip(), "role": "user"}
        elif role == "assistant" and pending_user is not None:
            episodes.append(
                {
                    "user_tid": pending_user["tid"],
                    "user_text": pending_user["text"],
                    "asst_tid": tid,
                    "asst_text": text.strip(),
                }
            )
            pending_user = None
    return episodes


def load_index(workdir: Path):
    meta = json.loads((workdir / "vectors" / "metadata.json").read_text())
    index = faiss.read_index(str(workdir / "vectors" / "vector.index"))
    # Map tid -> list of index positions (paragraph and/or sentence vectors).
    tid_to_pos: Dict[str, List[int]] = {}
    for i, m in enumerate(meta):
        tid = m.get("id") or m.get("tid")
        if not tid:
            continue
        # Prefer paragraph vectors; still keep all for reconstruct.
        tid_to_pos.setdefault(tid, []).append(int(m.get("index_pos", i)))
    return index, meta, tid_to_pos


def reconstruct_norm(index, pos: int) -> np.ndarray:
    v = np.asarray(index.reconstruct(pos), dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path(
            os.environ.get(
                "BIMS_BETA_WORKDIR",
                str(REPO / "BIMS-LEGAL-dataset/workdirs/legalep_disc_para0"),
            )
        ),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json",
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--wide_k", type=int, default=64)
    ap.add_argument("--max_n", type=int, default=500)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper/ipm/figures/soft_o2_beta_theory.json",
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    gold_texts = set()
    for s in manifest["sessions"]:
        if s.get("role") != "gold":
            continue
        for ut in s.get("user_turns") or []:
            gold_texts.add(ut.strip())
        turns = s.get("turns") or []
        if turns:
            gold_texts.add((turns[0].get("content") or "").strip())

    print(f"[load] index from {args.workdir}", flush=True)
    index, meta, tid_to_pos = load_index(args.workdir)
    episodes = parse_talk_episodes(args.workdir / "talk.txt")
    print(f"[load] ntotal={index.ntotal} episodes={len(episodes)} gold_texts={len(gold_texts)}", flush=True)

    # Restrict to gold needles by user-text match.
    gold_eps = [e for e in episodes if e["user_text"] in gold_texts][: args.max_n]
    if not gold_eps:
        # Fallback: first max_n episodes (diagnostic).
        gold_eps = episodes[: args.max_n]
        print("[warn] no gold text match; using first episodes", flush=True)
    print(f"[gaps] analyzing {len(gold_eps)} episodes", flush=True)

    # Episode membership: user+asst tids share a synthetic sid.
    tid_sid = {}
    for i, e in enumerate(episodes):
        tid_sid[e["user_tid"]] = i
        tid_sid[e["asst_tid"]] = i

    rhos, shs, sds, sas = [], [], [], []
    incompleteness = 0
    for e in gold_eps:
        h_pos = tid_to_pos.get(e["user_tid"])
        a_pos = tid_to_pos.get(e["asst_tid"])
        if not h_pos or not a_pos:
            continue
        qv = reconstruct_norm(index, h_pos[0]).reshape(1, -1)
        scores, idxs = index.search(qv, min(args.wide_k, index.ntotal))
        scores = scores.ravel()
        idxs = idxs.ravel()

        # Aggregate max score per tid.
        by_tid: Dict[str, float] = {}
        for sc, ix in zip(scores, idxs):
            if ix < 0 or ix >= len(meta):
                continue
            tid = meta[ix].get("id") or meta[ix].get("tid")
            if not tid:
                continue
            by_tid[tid] = max(by_tid.get(tid, -1e9), float(sc))

        # Ensure gold answer score present.
        av = reconstruct_norm(index, a_pos[0])
        sa = float(np.dot(qv.ravel(), av))
        by_tid[e["asst_tid"]] = max(by_tid.get(e["asst_tid"], -1e9), sa)
        sh = float(by_tid.get(e["user_tid"], np.dot(qv.ravel(), reconstruct_norm(index, h_pos[0]))))
        if sh <= 1e-8:
            continue
        sid = tid_sid.get(e["user_tid"])
        sd = -1e9
        for tid, sc in by_tid.items():
            if tid_sid.get(tid) == sid:
                continue
            sd = max(sd, sc)
        if sd < -1e8:
            continue
        rho = sd / sh
        rhos.append(rho)
        shs.append(sh)
        sds.append(sd)
        sas.append(sa)
        # FlatIP incompleteness: answer not in top-k paragraph ranking
        ranked = sorted(by_tid.items(), key=lambda x: -x[1])[: args.k]
        top = {t for t, _ in ranked}
        if e["user_tid"] in top and e["asst_tid"] not in top:
            incompleteness += 1

    rho = np.asarray(rhos, dtype=np.float64)
    print(
        f"[gaps] n={rho.size} incompleteness={incompleteness} "
        f"rho_p50={np.median(rho):.4f} rho_p95={np.quantile(rho,0.95):.4f}",
        flush=True,
    )

    b_h, st_h = coverage_quantile_diagnostic(rho, w_avail=1.0, w_rank=0.05)
    b_h2, st_h2 = coverage_quantile_diagnostic(rho, w_avail=1.0, w_rank=0.15)
    b_s, st_s = beta_star_softrank(rho, tau=20.0, gamma=0.05)
    b_s2, st_s2 = beta_star_softrank(rho, tau=20.0, gamma=0.15)

    # Feasible-band coverage of manuscript β grid.
    grid = [0.5, 0.7, 0.9, 0.95, 0.98, 1.0]
    cover = {
        str(b): float(np.mean(rho <= b)) if rho.size else float("nan") for b in grid
    }

    out = {
        "corpus": manifest.get("dataset"),
        "tier": manifest.get("tier"),
        "k": args.k,
        "n_usable": int(rho.size),
        "n_incompleteness_flatip": int(incompleteness),
        "query_proxy": "stored_gold_user_turn_embedding",
        "rho_definition": "s_d / s_h ; s_d = max non-episode score in wide FlatIP pool",
        "rho_stats": {
            "mean": float(np.mean(rho)),
            "p50": float(np.median(rho)),
            "p90": float(np.quantile(rho, 0.90)),
            "p95": float(np.quantile(rho, 0.95)),
            "p98": float(np.quantile(rho, 0.98)),
        },
        "score_stats": {
            "s_h_mean": float(np.mean(shs)),
            "s_a_mean": float(np.mean(sas)),
            "s_d_mean": float(np.mean(sds)),
        },
        "coverage_quantile_diagnostic": {
            "hinge_wrank0.05": b_h,
            "hinge_wrank0.15": b_h2,
            "softrank_gamma0.05": b_s,
            "softrank_gamma0.15": b_s2,
        },
        "hinge_detail": st_h,
        "hinge_detail_wrank0.15": st_h2,
        "softrank_detail": st_s,
        "softrank_detail_gamma0.15": st_s2,
        "empirical_cover_rho_le_beta": cover,
        "manuscript_default_beta": 0.98,
        "empirical_ah_peak_band": [0.9, 0.95, 0.98],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"coverage_quantile_diagnostic": out["coverage_quantile_diagnostic"], "rho_stats": out["rho_stats"], "cover": cover}, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
