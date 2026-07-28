"""统一检索 / QA 指标（LongMemEval / LoCoMo / 法律域同款口径）。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set


def _ids(retrieved: Sequence[Any]) -> Set[str]:
    return {str(x.get("tid", "")) for x in retrieved if x.get("tid")}


def recall_at_k(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return len(_ids(retrieved) & gt) / len(gt)


def precision_at_k(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    r = _ids(retrieved)
    if not r:
        return 0.0
    gt = set(ground_truth)
    return len(r & gt) / len(r)


def f1_at_k(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    p = precision_at_k(retrieved, ground_truth)
    r = recall_at_k(retrieved, ground_truth)
    if p + r < 1e-12:
        return 0.0
    return 2.0 * p * r / (p + r)


def instance_hit(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    return 1.0 if (_ids(retrieved) & gt) else 0.0


def ndcg_at_k(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    rel = [1.0 if str(item.get("tid", "")) in gt else 0.0 for item in retrieved]
    if not rel:
        return 0.0
    dcg = sum(r / __import__("math").log2(i + 2) for i, r in enumerate(rel))
    ideal = [1.0] * min(len(gt), len(rel))
    idcg = sum(r / __import__("math").log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved: Sequence[Any], ground_truth: Sequence[str]) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    for i, item in enumerate(retrieved):
        if str(item.get("tid", "")) in gt:
            return 1.0 / (i + 1)
    return 0.0


def ordered_session_ids(retrieved: Sequence[Any], tid_to_sid: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in retrieved:
        tid = item.get("tid")
        if not tid:
            continue
        sid = tid_to_sid.get(tid)
        if sid is None:
            continue
        s = str(sid)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def session_tids_in_sessions(tid_to_sid: Mapping[str, Any], session_ids: Iterable[str]) -> List[str]:
    gold = {str(s) for s in session_ids}
    return [str(tid) for tid, sid in tid_to_sid.items() if str(sid) in gold]


def session_metrics_at_k(
    ordered_session_ids_list: Sequence[str],
    gold_session_ids: Sequence[str],
    k: int,
) -> Dict[str, float]:
    gold = {str(s) for s in gold_session_ids}
    top = list(ordered_session_ids_list[:k])
    hit = set(top) & gold
    recall = len(hit) / len(gold) if gold else 0.0
    precision = len(hit) / len(top) if top else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 1e-12 else 0.0
    rel = [1.0 if s in gold else 0.0 for s in top]
    if rel:
        dcg = sum(r / __import__("math").log2(i + 2) for i, r in enumerate(rel))
        ideal = [1.0] * min(len(gold), len(top))
        idcg = sum(r / __import__("math").log2(i + 2) for i, r in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0.0
    else:
        ndcg = 0.0
    mrr = 0.0
    for i, s in enumerate(ordered_session_ids_list[:k]):
        if s in gold:
            mrr = 1.0 / (i + 1)
            break
    hit_rate = 1.0 if hit else 0.0
    return {
        "recall@k": recall,
        "precision@k": precision,
        "f1@k": f1,
        "ndcg@k": ndcg,
        "mrr@k": mrr,
        "hit_rate": hit_rate,
    }


def turn_metrics_at_k(retrieved: Sequence[Any], gold_tids: Sequence[str], k: int) -> Dict[str, float]:
    top = list(retrieved[:k])
    gt = [str(t) for t in gold_tids]
    return {
        "recall@k": recall_at_k(top, gt),
        "precision@k": precision_at_k(top, gt),
        "f1@k": f1_at_k(top, gt),
        "ndcg@k": ndcg_at_k(top, gt),
        "mrr@k": mrr_at_k(top, gt),
        "hit_rate": instance_hit(top, gt),
    }


def evaluate_longmem_retrieval(
    retrieved: Sequence[Any],
    instance: Mapping[str, Any],
    mgr: Any,
    *,
    ks: Sequence[int] = (1, 5, 10),
    max_k: int = 10,
) -> Dict[str, Any]:
    """LongMem 格式实例：session 级 + turn 级（answer 段）指标。"""
    tid_to_sid = getattr(mgr, "_tid_to_session", {}) or {}
    gold_sessions = list(instance.get("evidence_session_ids") or instance.get("answer_session_ids") or [])
    ordered_sids = ordered_session_ids(retrieved, tid_to_sid)
    answer_tids = session_tids_in_sessions(tid_to_sid, gold_sessions)

    out: Dict[str, Any] = {"by_k": {}, "question_type": str(instance.get("question_type", ""))}
    for k in ks:
        ki = int(k)
        out["by_k"][str(ki)] = {
            "session": session_metrics_at_k(ordered_sids, gold_sessions, ki),
            "answer": turn_metrics_at_k(retrieved, answer_tids, min(ki, max_k)),
        }
    primary = str(max(int(x) for x in ks))
    out["session"] = out["by_k"][primary]["session"]
    out["answer"] = out["by_k"][primary]["answer"]
    return out


def aggregate_metric_lists(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    import numpy as np
    return float(np.mean(values))


def aggregate_retrieval_metrics(
    per_query: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not per_query:
        return {}
    rs, ps, fs, hits, ndcgs, mrrs = [], [], [], [], [], []
    for row in per_query:
        if "recall" in row:
            rs.append(row["recall"])
            ps.append(row["precision"])
            fs.append(row["f1"])
            hits.append(row.get("hit", 0.0))
            ndcgs.append(row.get("ndcg", 0.0))
            mrrs.append(row.get("mrr", 0.0))
            continue
        ret, gt = row["retrieved"], row["ground_truth"]
        rs.append(recall_at_k(ret, gt))
        ps.append(precision_at_k(ret, gt))
        fs.append(f1_at_k(ret, gt))
        hits.append(instance_hit(ret, gt))
        ndcgs.append(ndcg_at_k(ret, gt))
        mrrs.append(mrr_at_k(ret, gt))
    import numpy as np
    return {
        "recall@k": float(np.mean(rs)),
        "precision@k": float(np.mean(ps)),
        "f1@k": float(np.mean(fs)),
        "ndcg@k": float(np.mean(ndcgs)),
        "mrr@k": float(np.mean(mrrs)),
        "instance_hit_rate": float(np.mean(hits)),
    }


# 论文/Ref 对齐的主指标名（@k 后缀在汇总时加上）
SESSION_METRIC_KEYS = ("recall", "precision", "f1", "ndcg", "mrr", "hit_rate")
ANSWER_METRIC_KEYS = ("recall", "precision", "f1", "ndcg", "mrr", "hit_rate")
DEFAULT_EVAL_KS = (1, 5, 10)
PRIMARY_METRIC = "session_ndcg@10"
