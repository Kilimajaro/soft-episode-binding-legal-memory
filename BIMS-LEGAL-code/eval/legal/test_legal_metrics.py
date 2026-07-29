#!/usr/bin/env python3
"""Unit tests for graded nDCG and failure taxonomy."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "legal"))

from legal_metrics import failure_taxonomy, metrics_at_k, ndcg_at_k_graded  # noqa: E402


def _row(tid: str, score: float = 1.0):
    return {"tid": tid, "score": score, "final_score": score}


def test_ndcg_no_relevant():
    r = metrics_at_k([], ["q1", "a1"], ["a1"], k=10)
    assert r["ndcg@k"] == 0.0


def test_ndcg_question_only_not_one():
    """Old bug: question-only hit must not yield nDCG=1."""
    retrieved = [_row("q1", 0.9), _row("d1", 0.5)]
    ndcg = ndcg_at_k_graded(retrieved, ["q1", "a1"], ["a1"], k=10)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_answer_first_is_one():
    retrieved = [_row("a1", 0.95), _row("q1", 0.9)]
    ndcg = ndcg_at_k_graded(retrieved, ["q1", "a1"], ["a1"], k=10)
    assert abs(ndcg - 1.0) < 1e-9


def test_ndcg_missing_answer_penalized():
    retrieved = [_row("q1", 0.99)] + [_row(f"d{i}", 0.5 - i * 0.01) for i in range(9)]
    ndcg_q = ndcg_at_k_graded(retrieved, ["q1", "a1"], ["a1"], k=10)
    retrieved2 = [_row("a1", 0.99), _row("q1", 0.98)]
    ndcg_both = ndcg_at_k_graded(retrieved2, ["q1", "a1"], ["a1"], k=10)
    assert ndcg_both > ndcg_q


def test_failure_taxonomy_incomplete():
    m = metrics_at_k([_row("q1")], ["q1", "a1"], ["a1"], k=10, gold_q_tids=["q1"])
    assert m["failure_mode"] == "incomplete"


def test_failure_taxonomy_complete():
    m = metrics_at_k([_row("a1"), _row("q1")], ["q1", "a1"], ["a1"], k=10, gold_q_tids=["q1"])
    assert m["failure_mode"] == "complete"


def test_failure_session_miss():
    m = metrics_at_k([_row("d1")], ["q1", "a1"], ["a1"], k=10, gold_q_tids=["q1"])
    assert m["failure_mode"] == "session_miss"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
