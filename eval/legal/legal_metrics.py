"""Graded legal retrieval metrics with fixed-gold nDCG and failure taxonomy."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set


def _tids(retrieved: Sequence[Mapping]) -> List[str]:
    return [str(r.get("tid", "")) for r in retrieved if r.get("tid")]


def graded_relevance(
    tid: str,
    gold_sess_tids: Iterable[str],
    gold_ans_tids: Iterable[str],
) -> float:
    """Answer=1.0, same-session non-answer=0.5, else 0.0."""
    t = str(tid)
    if t in {str(x) for x in gold_ans_tids}:
        return 1.0
    if t in {str(x) for x in gold_sess_tids}:
        return 0.5
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    rel = list(relevances[:k])
    if not rel:
        return 0.0
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel) if r > 0)


def idcg_at_k_graded(
    gold_sess_tids: Iterable[str],
    gold_ans_tids: Iterable[str],
    k: int,
) -> float:
    """Fixed IDCG from the full gold relevance set (not from retrieved items)."""
    ideal = []
    for _ in gold_ans_tids:
        ideal.append(1.0)
    g_sess = {str(x) for x in gold_sess_tids}
    g_ans = {str(x) for x in gold_ans_tids}
    for t in g_sess:
        if t not in g_ans:
            ideal.append(0.5)
    ideal.sort(reverse=True)
    return dcg_at_k(ideal, k)


def ndcg_at_k_graded(
    retrieved: Sequence[Mapping],
    gold_sess_tids: Iterable[str],
    gold_ans_tids: Iterable[str],
    k: int,
) -> float:
    top_tids = _tids(retrieved)[:k]
    rel = [graded_relevance(t, gold_sess_tids, gold_ans_tids) for t in top_tids]
    idcg = idcg_at_k_graded(gold_sess_tids, gold_ans_tids, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(rel, k) / idcg


def failure_taxonomy(
    retrieved: Sequence[Mapping],
    gold_sess_tids: Iterable[str],
    gold_ans_tids: Iterable[str],
    gold_q_tids: Iterable[str],
    k: int,
) -> str:
    """Mutually exclusive failure mode for RQ1 (two-turn episodes).

    Categories:
      complete      — question and answer both in top-k
      incomplete    — question in top-k, answer absent (episode incompleteness)
      answer_only   — answer in top-k without question (rare)
      session_miss  — no gold-session turn in top-k
    """
    top = set(_tids(retrieved)[:k])
    g_sess = {str(x) for x in gold_sess_tids}
    g_ans = {str(x) for x in gold_ans_tids}
    g_q = {str(x) for x in gold_q_tids}
    sess_hit = bool(top & g_sess)
    ans_hit = bool(top & g_ans)
    q_hit = bool(top & g_q) if g_q else sess_hit
    if sess_hit and ans_hit and q_hit:
        return "complete"
    if q_hit and not ans_hit:
        return "incomplete"
    if ans_hit and not q_hit:
        return "answer_only"
    return "session_miss"


def metrics_at_k(
    retrieved: Sequence[Mapping],
    gold_sess_tids: Iterable[str],
    gold_ans_tids: Iterable[str],
    k: int,
    gold_q_tids: Iterable[str] | None = None,
) -> Dict[str, float]:
    top = list(retrieved)[:k]
    tids = set(_tids(top))
    g_s = {str(x) for x in gold_sess_tids}
    g_a = {str(x) for x in gold_ans_tids}
    sess_hit = 1.0 if (tids & g_s) else 0.0
    ans_hit = 1.0 if (tids & g_a) else 0.0
    ep_comp = len(tids & g_s) / len(g_s) if g_s else 0.0
    ndcg = ndcg_at_k_graded(top, gold_sess_tids, gold_ans_tids, k)
    mrr = 0.0
    for i, t in enumerate(_tids(top)):
        if t in g_a or t in g_s:
            mrr = 1.0 / (i + 1)
            break
    out = {
        "session_hit@k": float(sess_hit),
        "answer_hit@k": float(ans_hit),
        "episode_completeness@k": float(ep_comp),
        "ndcg@k": float(ndcg),
        "mrr@k": float(mrr),
    }
    if gold_q_tids is not None:
        out["failure_mode"] = failure_taxonomy(top, gold_sess_tids, gold_ans_tids, gold_q_tids, k)
    return out


def aggregate_failure_counts(modes: Sequence[str]) -> Dict[str, float]:
    n = len(modes) or 1
    keys = ("complete", "incomplete", "answer_only", "session_miss")
    counts = {k: sum(1 for m in modes if m == k) for k in keys}
    return {k: counts[k] / n for k in keys}
