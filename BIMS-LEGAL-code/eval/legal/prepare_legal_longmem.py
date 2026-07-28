#!/usr/bin/env python3
"""法律 LongMemEval-S 级 benchmark 构建（对齐 LongMem 专业设定）。

相对旧版 prepare_legal_datasets.py 的升级：
  1. 规模：默认 ~50–80 sessions / 400+ turns（LongMem-S 约 48 sessions / ~500 turns）
  2. 硬负例：BM25 语义相近的咨询作 distractor（非随机采样）
  3. 查询抽象：question 与 haystack 原文低重叠（LongMem 风格 inference 问法）
  4. 题型：single-session-user / single-session-assistant / multi-session
  5. 多轮会话：长答案分块 + 追问 filler，抬高 turn 数

输出：data/legal/{dataset}/longmemeval_s.json（与 eval_new.py 兼容）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGAL_DIR = os.path.dirname(os.path.abspath(__file__))
if LEGAL_DIR not in sys.path:
    sys.path.insert(0, LEGAL_DIR)

from legal_optim import legal_tokens  # noqa: E402
from legal_query import (  # noqa: E402
    abstract_assistant_query,
    abstract_multi_session_query,
    abstract_user_query,
    validate_abstract_query,
)
from prepare_legal_datasets import DATASET_FILES, load_pairs  # noqa: E402

_FILLER_PAIRS = [
    ("能否再具体说明一下适用条件？", "需要结合证据与情节综合判断，不能一概而论。"),
    ("谢谢律师解答。", "不客气，如有新情况可以再咨询。"),
    ("我还想了解相关程序怎么走？", "一般可先协商，不成再考虑调解、仲裁或诉讼。"),
    ("这个时效是多久？", "诉讼时效通常为三年，特殊情况依法可能中断或中止。"),
]


def _bm25_question_index(questions: List[str]):
    """在问题文本上建 BM25，用于 hard negative 挖掘。"""
    docs = [Counter(legal_tokens(q)) for q in questions]
    N = len(docs)
    df = Counter()
    doc_len = {}
    for i, dc in enumerate(docs):
        doc_len[i] = max(1, sum(dc.values()))
        for t in dc:
            df[t] += 1
    avgdl = sum(doc_len.values()) / max(N, 1)
    idf = {t: np.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    k1, b = 1.5, 0.75

    def score(qi: int, dj: int) -> float:
        if qi == dj:
            return -1.0
        q, d = docs[qi], docs[dj]
        dl = doc_len[dj]
        s = 0.0
        for term, qf in q.items():
            if term not in d:
                continue
            tf = d[term]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            s += idf.get(term, 0) * (tf * (k1 + 1)) / denom
        return s

    return score


def _split_answer(answer: str, max_chunk: int = 220) -> List[str]:
    if len(answer) <= max_chunk:
        return [answer]
    parts = re.split(r"(?<=[。；;！!？?])", answer)
    chunks, buf = [], ""
    for p in parts:
        if not p.strip():
            continue
        if len(buf) + len(p) <= max_chunk:
            buf += p
        else:
            if buf:
                chunks.append(buf.strip())
            buf = p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks if chunks else [answer[:max_chunk]]


def _build_multiturn_session(q: str, a: str, rng: np.random.Generator) -> List[Dict[str, str]]:
    """2–10 轮会话：分块答案 + 随机 filler（抬高 turn 数，模拟 LongMem 多轮）。"""
    turns = [{"role": "user", "content": q}]
    chunks = _split_answer(a)
    for i, chunk in enumerate(chunks):
        turns.append({"role": "assistant", "content": chunk})
        if i < len(chunks) - 1:
            turns.append({"role": "user", "content": "请继续说明。"})
    n_extra = int(rng.integers(0, 3))
    for _ in range(n_extra):
        fq, fa = _FILLER_PAIRS[int(rng.integers(0, len(_FILLER_PAIRS)))]
        turns.append({"role": "user", "content": fq})
        turns.append({"role": "assistant", "content": fa})
    return turns


def _pick_distractors(
    target_i: int,
    n: int,
    questions: List[str],
    rng: np.random.Generator,
    hard_ratio: float = 0.75,
) -> List[int]:
    score_fn = _bm25_question_index(questions)
    pool = [j for j in range(len(questions)) if j != target_i]
    scored = sorted(((score_fn(target_i, j), j) for j in pool), reverse=True)
    n_hard = max(1, int(n * hard_ratio))
    n_rand = max(0, n - n_hard)
    hard = [j for _, j in scored[: max(n_hard * 4, n_hard)]]
    rng.shuffle(hard)
    hard_pick = hard[:n_hard]
    rest = [j for j in pool if j not in hard_pick]
    rng.shuffle(rest)
    pick = hard_pick + rest[:n_rand]
    rng.shuffle(pick)
    return pick[:n]


def _synthetic_dates(n: int, rng: np.random.Generator, start: datetime) -> List[str]:
    dates = []
    cur = start
    for _ in range(n):
        cur += timedelta(days=int(rng.integers(1, 14)))
        dates.append(cur.strftime("%Y-%m-%d"))
    return dates


def build_longmem_instances(
    dataset_key: str,
    pairs: List[Tuple[str, str]],
    *,
    sample_size: int,
    n_sessions: int,
    seed: int,
    hard_ratio: float = 0.75,
    multi_session_frac: float = 0.22,
    assistant_frac: float = 0.38,
) -> List[Dict]:
    """构建 LongMemEval 兼容实例列表。"""
    questions = [p[0] for p in pairs]
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n < n_sessions + 5:
        raise ValueError(f"语料不足: {n} < n_sessions+5")

    target_indices = rng.choice(n, size=min(sample_size, n), replace=False)
    instances: List[Dict] = []
    base_date = datetime(2023, 1, 1)

    for k, ti in enumerate(target_indices):
        ev_q, ev_a = pairs[ti]
        qtype_roll = float(rng.random())
        if qtype_roll < multi_session_frac and n_sessions >= 10:
            qtype = "multi-session"
            ti2 = int(rng.choice([j for j in range(n) if j != ti]))
            ev_q2, ev_a2 = pairs[ti2]
            n_distract = n_sessions - 2
            distract = _pick_distractors(ti, n_distract, questions, rng, hard_ratio)
            combined: List[Tuple[List[Dict[str, str]], str, str]] = []
            for j in distract:
                combined.append((
                    _build_multiturn_session(pairs[j][0], pairs[j][1], rng),
                    f"{dataset_key}_inst{k}_s{len(combined)}",
                    "d",
                ))
            combined.append((
                _build_multiturn_session(ev_q, ev_a, rng),
                f"{dataset_key}_inst{k}_s_ev0",
                "e",
            ))
            combined.append((
                _build_multiturn_session(ev_q2, ev_a2, rng),
                f"{dataset_key}_inst{k}_s_ev1",
                "e",
            ))
            rng.shuffle(combined)
            sessions = [c[0] for c in combined]
            sids = [c[1] for c in combined]
            evidence_sids = [c[1] for c in combined if c[1].endswith("_ev0") or c[1].endswith("_ev1")]
            haystack_texts = [t["content"] for s in sessions for t in s]
            query = abstract_multi_session_query(
                [_topic(ev_q), _topic(ev_q2)], __import__("random").Random(seed + k)
            )
            answer = f"第一次咨询涉及：{ev_q[:80]}…；第二次咨询涉及：{ev_q2[:80]}…"
        elif qtype_roll < multi_session_frac + assistant_frac:
            qtype = "single-session-assistant"
            distract = _pick_distractors(ti, n_sessions - 1, questions, rng, hard_ratio)
            sessions, sids = [], []
            for idx, j in enumerate(distract):
                sid = f"{dataset_key}_inst{k}_s{idx}"
                sessions.append(_build_multiturn_session(pairs[j][0], pairs[j][1], rng))
                sids.append(sid)
            ev_pos = int(rng.integers(0, len(sessions) + 1))
            ev_session = _build_multiturn_session(ev_q, ev_a, rng)
            ev_sid = f"{dataset_key}_inst{k}_s{ev_pos}_ev"
            sessions.insert(ev_pos, ev_session)
            sids.insert(ev_pos, ev_sid)
            evidence_sids = [ev_sid]
            haystack_texts = [t["content"] for s in sessions for t in s]
            query = abstract_assistant_query(ev_q, ev_a, __import__("random").Random(seed + k + 1))
            answer = ev_a
        else:
            qtype = "single-session-user"
            distract = _pick_distractors(ti, n_sessions - 1, questions, rng, hard_ratio)
            sessions, sids = [], []
            for idx, j in enumerate(distract):
                sid = f"{dataset_key}_inst{k}_s{idx}"
                sessions.append(_build_multiturn_session(pairs[j][0], pairs[j][1], rng))
                sids.append(sid)
            ev_pos = int(rng.integers(0, len(sessions) + 1))
            ev_session = _build_multiturn_session(ev_q, ev_a, rng)
            ev_sid = f"{dataset_key}_inst{k}_s{ev_pos}_ev"
            sessions.insert(ev_pos, ev_session)
            sids.insert(ev_pos, ev_sid)
            evidence_sids = [ev_sid]
            haystack_texts = [t["content"] for s in sessions for t in s]
            query = abstract_user_query(ev_q, __import__("random").Random(seed + k + 2))
            answer = ev_a

        if not validate_abstract_query(query, haystack_texts):
            query = abstract_user_query(ev_q + "补充", __import__("random").Random(seed + k + 99))

        dates = _synthetic_dates(len(sessions), rng, base_date + timedelta(days=k * 30))
        n_turns = sum(len(s) for s in sessions)
        instances.append({
            "question_id": f"{dataset_key}_longmem_{k}",
            "question_type": qtype,
            "question": query,
            "question_date": dates[-1],
            "answer": answer,
            "answer_session_ids": evidence_sids,
            "haystack_sessions": sessions,
            "haystack_session_ids": sids,
            "haystack_dates": dates,
            "_build_meta": {
                "original_question": ev_q,
                "n_sessions": len(sessions),
                "n_turns": n_turns,
                "dataset": dataset_key,
            },
        })

    return instances


def _topic(q: str) -> str:
    q = re.sub(r"[？?！!。，,\s]+", "", q.strip())
    return q[:18] if len(q) > 18 else q


def summarize(instances: List[Dict]) -> Dict:
    ns = [len(i["haystack_sessions"]) for i in instances]
    nt = [i["_build_meta"]["n_turns"] for i in instances]
    qt = Counter(i["question_type"] for i in instances)
    return {
        "n_instances": len(instances),
        "sessions_min": min(ns),
        "sessions_max": max(ns),
        "sessions_mean": round(float(np.mean(ns)), 1),
        "turns_min": min(nt),
        "turns_max": max(nt),
        "turns_mean": round(float(np.mean(nt)), 1),
        "question_types": dict(qt),
    }


def prepare(
    dataset_key: str,
    out_dir: str,
    sample_size: int,
    n_sessions: int,
    seed: int,
    split_name: str = "longmemeval_s",
) -> str:
    pairs = load_pairs(dataset_key, max_pairs=25000)
    print(f"[{dataset_key}] loaded {len(pairs)} pairs")
    instances = build_longmem_instances(
        dataset_key, pairs, sample_size=sample_size, n_sessions=n_sessions, seed=seed,
    )
    stats = summarize(instances)
    print(f"[{dataset_key}] build stats: {json.dumps(stats, ensure_ascii=False)}")

    ds_dir = os.path.join(out_dir, dataset_key)
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, f"{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False, indent=2)
    meta_path = os.path.join(ds_dir, f"{split_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "n_sessions_target": n_sessions, "seed": seed}, f, indent=2)
    print(f"[{dataset_key}] -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="法律 LongMemEval-S 级 benchmark 构建")
    ap.add_argument("--datasets", nargs="+", default=["disc_law"], choices=list(DATASET_FILES.keys()))
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data", "legal"))
    ap.add_argument("--sample_size", type=int, default=80)
    ap.add_argument("--n_sessions", type=int, default=55, help="每实例 session 数（对齐 LongMem ~50）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_name", default="longmemeval_s")
    args = ap.parse_args()
    for ds in args.datasets:
        prepare(ds, args.out_dir, args.sample_size, args.n_sessions, args.seed, args.split_name)


if __name__ == "__main__":
    main()
