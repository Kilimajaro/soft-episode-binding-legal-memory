#!/usr/bin/env python3
"""Tests for Holm step-down used in paper table regeneration (S0-5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "scripts"))

from regenerate_unified_tables import holm_adjust  # noqa: E402


def test_holm_matches_reference_ordering():
    # Checklist illustrative raw p vector (not final submission values).
    p = [8.99e-48, 5.19e-18, 5.51e-65, 0.934, 0.0239, 7.95e-5, 2.10e-15, 1.25e-4, 1.18e-10]
    h = holm_adjust(p)
    assert abs(h[4] - 0.0478) < 1e-6  # 0.0239 * 2
    assert abs(h[0] - 7.192e-47) < 1e-49
    assert abs(h[3] - 0.934) < 1e-9
    # monotone in sorted order
    order = sorted(range(len(p)), key=lambda i: p[i])
    vals = [h[i] for i in order]
    assert all(vals[i] <= vals[i + 1] + 1e-15 for i in range(len(vals) - 1))


if __name__ == "__main__":
    test_holm_matches_reference_ordering()
    print("ok test_holm_matches_reference_ordering")
