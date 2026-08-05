#!/usr/bin/env python3
"""Unit tests: Hybrid cluster gate uses pre-session-expansion direct hits (S0-4)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _search_method_source() -> str:
    """Return VectorMemoryManager.search source from memory_manager.py."""
    path = ROOT / "memory_manager.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VectorMemoryManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "search":
                    seg = ast.get_source_segment(text, item)
                    if not seg:
                        raise AssertionError("Could not extract VectorMemoryManager.search")
                    return seg
    raise AssertionError("VectorMemoryManager.search not found")


def test_cluster_direct_hits_recorded_before_session_expand():
    """Gate must snapshot dense hits before Soft O2 session sibling injection."""
    src = _search_method_source()
    assert "_cluster_direct_hits" in src
    assert "_expand_with_session_siblings" in src
    i_direct = src.index("self._cluster_direct_hits")
    i_expand = src.index("self._expand_with_session_siblings")
    assert i_direct < i_expand, (
        "Hybrid gate regression: _cluster_direct_hits must be set before "
        "_expand_with_session_siblings"
    )


def test_cluster_direct_hits_exclude_session_injected():
    """A tid injected only by Soft O2 session expansion must not be a cluster trigger."""
    # Lightweight behavioral fragment mirroring the fixed search order.
    cluster_direct_hits: set[str] = set()
    dense_hit = {"tid": "q_direct", "score": 0.9, "text": "q"}
    sibling = {"tid": "a_sibling", "score": 0.0, "text": "a"}

    def expand_with_session_siblings(results):
        out = list(results)
        if not any(r.get("tid") == "a_sibling" for r in out):
            inj = dict(sibling)
            inj["score"] = 0.9 * 0.98
            inj["_injected_by_session"] = True
            out.append(inj)
        return out

    all_results = [dense_hit]
    cluster_direct_hits = {r.get("tid") for r in all_results if r.get("tid") is not None}
    all_results = expand_with_session_siblings(all_results)

    assert "q_direct" in cluster_direct_hits
    assert "a_sibling" not in cluster_direct_hits
    assert any(r.get("tid") == "a_sibling" for r in all_results)


def test_answer_only_failure_mode():
    sys.path.insert(0, str(ROOT / "eval" / "legal"))
    from legal_metrics import metrics_at_k

    retrieved = [{"tid": "a1", "score": 1.0, "final_score": 1.0}]
    m = metrics_at_k(retrieved, ["q1", "a1"], ["a1"], k=10, gold_q_tids=["q1"])
    assert m["failure_mode"] == "answer_only"


if __name__ == "__main__":
    test_cluster_direct_hits_recorded_before_session_expand()
    print("ok test_cluster_direct_hits_recorded_before_session_expand")
    test_cluster_direct_hits_exclude_session_injected()
    print("ok test_cluster_direct_hits_exclude_session_injected")
    test_answer_only_failure_mode()
    print("ok test_answer_only_failure_mode")
    print("all passed")
