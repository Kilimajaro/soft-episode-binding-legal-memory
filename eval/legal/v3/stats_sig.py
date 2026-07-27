#!/usr/bin/env python3
"""Significance helpers for LegalMem-MT: bootstrap CI + McNemar."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def bootstrap_ci(scores: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = np.asarray(scores, dtype=float)
    if x.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        means.append(float(x[idx].mean()))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(x.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(x.size),
        "n_boot": n_boot,
    }


def mcnemar_midp(y_a: Sequence[int], y_b: Sequence[int]) -> dict:
    """McNemar mid-p test on paired binary outcomes (1=hit, 0=miss).

    Returns b, c, mid_p (smaller => more evidence of difference).
    """
    a = np.asarray(y_a, dtype=int)
    b = np.asarray(y_b, dtype=int)
    assert a.shape == b.shape
    both = int(((a == 1) & (b == 1)).sum())
    a_only = int(((a == 1) & (b == 0)).sum())  # n12
    b_only = int(((a == 0) & (b == 1)).sum())  # n21
    neither = int(((a == 0) & (b == 0)).sum())
    n = a_only + b_only
    if n == 0:
        mid_p = 1.0
    else:
        # exact binomial mid-p under H0 p=0.5
        from math import comb
        k = min(a_only, b_only)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        # mid-p correction: subtract half the probability of the observed point
        point = comb(n, a_only) / (2 ** n)
        mid_p = min(1.0, 2 * (tail - 0.5 * point))
    return {
        "n11": both, "n12": a_only, "n21": b_only, "n00": neither,
        "mid_p": float(mid_p),
        "delta_mean": float(a.mean() - b.mean()),
    }


def paired_report(name_a: str, hits_a: Sequence[int], name_b: str, hits_b: Sequence[int], seed: int = 0) -> dict:
    return {
        "a": name_a,
        "b": name_b,
        "a_ci": bootstrap_ci(hits_a, seed=seed),
        "b_ci": bootstrap_ci(hits_b, seed=seed + 1),
        "mcnemar": mcnemar_midp(hits_a, hits_b),
    }
