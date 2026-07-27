"""法律咨询数据集预处理脚本。

将"单轮法律咨询问答对"转换为与 LongMemEval 完全兼容的"大海捞针"(needle-in-haystack)
多会话长对话格式，从而可以复用 eval/eval_new.py::LongMemEvalEvaluator 的同款指标
（检索召回率 recall@k 与 LLM 评判的 QA 正确率）。

原始数据集（均为权威中文法律咨询数据，通过 huggingface 下载）：
  1. ShengbinYue/DISC-Law-SFT  (复旦 DISC-LawLLM, legal_question_answering 子集)
  2. Skepsun/lawyer_llama_data (北大 Lawyer-LLaMA, 法律咨询子集 legal_advice / legal_counsel)

转换思路（见报告 REPORT.md 的实验设计）：
  - 每个评测样本(instance)挑选一条目标咨询问答对作为"针"(evidence session)，
    其问题作为 instance.question，其专家答案作为 gold answer。
  - 目标问答对被包装成一个 [user 提问, assistant 解答] 的会话。
  - 再随机采样 (haystack_size - 1) 条同领域的干扰咨询作为其它会话，组成长对话"草堆"。
  - answer_session_ids 指向"针"所在会话，作为检索召回率的金标准。

输出文件：<out_dir>/<dataset_key>/longmemeval_oracle.json
（与 LongMemEvalEvaluator.load_benchmark_data 期望的路径/字段一致）
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, "data", "legal_raw")

# (huggingface repo_id, 文件名, 本地缓存文件名)
DATASET_FILES = {
    "disc_law": ("ShengbinYue/DISC-Law-SFT", "DISC-Law-SFT-Pair-QA-released.jsonl"),
    "lawyer_llama": ("Skepsun/lawyer_llama_data", "all.json"),
}

# Lawyer-LLaMA 中属于"法律咨询"的 source（排除司法考试等非咨询数据）
LAWYER_LLAMA_CONSULT_SOURCES = {
    "legal_advice.json",
    "legal_counsel_with_article_v2.json",
    "legal_counsel_multi_turn_with_article_v2.json",
}


def _raw_path(dataset_key: str) -> str:
    repo, fn = DATASET_FILES[dataset_key]
    return os.path.join(RAW_DIR, repo.replace("/", "__") + "__" + fn)


def ensure_downloaded(dataset_key: str) -> str:
    """若本地缺失则从 huggingface 下载原始数据文件，返回本地路径。"""
    local = _raw_path(dataset_key)
    if os.path.exists(local):
        return local
    os.makedirs(RAW_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    import shutil

    repo, fn = DATASET_FILES[dataset_key]
    print(f"[download] {repo}/{fn} ...")
    cached = hf_hub_download(repo_id=repo, filename=fn, repo_type="dataset")
    shutil.copy(cached, local)
    print(f"[download] -> {local} ({os.path.getsize(local) / 1e6:.1f} MB)")
    return local


def _clean(text: str) -> str:
    if not text:
        return ""
    # 去掉 qwen 风格的思维链残留与多余空白
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def _quality_ok(q: str, a: str) -> bool:
    if not q or not a:
        return False
    if not (5 <= len(q) <= 300):
        return False
    if len(a) < 25:
        return False
    return True


def _quality_ok_strict(q: str, a: str) -> bool:
    """扩样专用更严过滤：只用于新增样本，不改动已发表的原子集。"""
    if not _quality_ok(q, a):
        return False
    if not (8 <= len(q) <= 220):
        return False
    if len(a) < 60:
        return False
    # 排除明显非咨询 / 过短模板
    bad_q = ("下列说法", "选择题", "判断题", "填空", "以下哪项", "正确答案")
    if any(x in q for x in bad_q):
        return False
    # 答案需含法律内容信号（法条/责任/程序等），降低空泛复读比例
    legal_signals = (
        "法", "条", "款", "责任", "合同", "诉讼", "仲裁", "赔偿", "权利", "义务",
        "民法典", "刑法", "劳动", "婚姻", "法院", "起诉", "证据", "违约", "侵权",
    )
    if not any(x in a for x in legal_signals):
        return False
    return True


def load_pairs(dataset_key: str, max_pairs: int = 20000) -> List[Tuple[str, str]]:
    """读取原始数据，返回去重后的 (问题, 专家答案) 法律咨询问答对列表。"""
    path = ensure_downloaded(dataset_key)
    pairs: List[Tuple[str, str]] = []
    seen = set()

    def add(q: str, a: str):
        q, a = _clean(q), _clean(a)
        if not _quality_ok(q, a):
            return
        key = q[:120]
        if key in seen:
            return
        seen.add(key)
        pairs.append((q, a))

    if dataset_key == "disc_law":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                add(o.get("input", ""), o.get("output", ""))
                if len(pairs) >= max_pairs:
                    break
    elif dataset_key == "lawyer_llama":
        data = json.load(open(path, encoding="utf-8"))
        for o in data:
            if o.get("source") not in LAWYER_LLAMA_CONSULT_SOURCES:
                continue
            # 单轮：instruction 为咨询问题，output 为解答
            add(o.get("instruction", ""), o.get("output", ""))
            if len(pairs) >= max_pairs:
                break
    else:
        raise ValueError(f"未知数据集: {dataset_key}")

    return pairs


def build_instances(
    dataset_key: str,
    pairs: List[Tuple[str, str]],
    sample_size: int,
    haystack_size: int,
    seed: int,
) -> List[Dict]:
    """把咨询问答对转换为 LongMemEval 格式的大海捞针样本。"""
    rng = np.random.default_rng(seed)
    n = len(pairs)
    if n < haystack_size + 1:
        raise ValueError(f"{dataset_key}: 可用问答对过少({n})，无法构造 haystack_size={haystack_size}")

    # 选出作为"针"的目标问答对
    target_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    instances: List[Dict] = []

    for k, ti in enumerate(target_idx):
        ev_q, ev_a = pairs[ti]
        # 采样干扰会话（排除目标本身）
        pool = [j for j in range(n) if j != ti]
        distractor_idx = rng.choice(pool, size=haystack_size - 1, replace=False)

        # 组装会话，并把"针"插入随机位置
        sessions = []
        for j in distractor_idx:
            dq, da = pairs[j]
            sessions.append([
                {"role": "user", "content": dq},
                {"role": "assistant", "content": da},
            ])
        ev_pos = int(rng.integers(0, len(sessions) + 1))
        evidence_session = [
            {"role": "user", "content": ev_q},
            {"role": "assistant", "content": ev_a},
        ]
        sessions.insert(ev_pos, evidence_session)

        haystack_session_ids = [f"{dataset_key}_s{k}_{idx}" for idx in range(len(sessions))]
        evidence_sid = haystack_session_ids[ev_pos]

        instances.append({
            "question_id": f"{dataset_key}_{k}",
            "question_type": "legal-consultation",
            "question": ev_q,
            "answer": ev_a,
            "answer_session_ids": [evidence_sid],
            "haystack_sessions": sessions,
            "haystack_session_ids": haystack_session_ids,
        })

    return instances


def prepare(dataset_key: str, out_dir: str, sample_size: int, haystack_size: int, seed: int) -> str:
    pairs = load_pairs(dataset_key)
    print(f"[{dataset_key}] 加载到 {len(pairs)} 条法律咨询问答对")
    instances = build_instances(dataset_key, pairs, sample_size, haystack_size, seed)
    ds_dir = os.path.join(out_dir, dataset_key)
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, "longmemeval_oracle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False, indent=2)
    total_sessions = sum(len(i["haystack_sessions"]) for i in instances)
    print(
        f"[{dataset_key}] 生成 {len(instances)} 个样本, 共 {total_sessions} 个会话 "
        f"(每个样本 haystack={haystack_size}) -> {out_path}"
    )
    return out_path


def _question_key(q: str) -> str:
    return _clean(q)[:120]


def expand_instances_preserving(
    dataset_key: str,
    pairs: List[Tuple[str, str]],
    existing: List[Dict],
    target_size: int,
    haystack_size: int,
    expand_seed: int,
) -> Tuple[List[Dict], Dict]:
    """在保留已有 P1 样本字节级不变的前提下追加新针，避免破坏已发表子集。"""
    if target_size < len(existing):
        raise ValueError(f"target_size={target_size} < existing={len(existing)}")
    if target_size == len(existing):
        return existing, {
            "n_existing": len(existing),
            "n_added": 0,
            "n_total": len(existing),
            "preserve_original": True,
        }

    used_q = {_question_key(inst["question"]) for inst in existing}
    # 仅从严格质量池抽样，且避开已有问题，防止与原子集/彼此重复
    candidates = [
        i for i, (q, a) in enumerate(pairs)
        if _quality_ok_strict(q, a) and _question_key(q) not in used_q
    ]
    need = target_size - len(existing)
    if len(candidates) < need:
        raise ValueError(
            f"{dataset_key}: 严格质量候选不足 ({len(candidates)} < {need})"
        )

    rng = np.random.default_rng(expand_seed)
    chosen = rng.choice(candidates, size=need, replace=False)
    added: List[Dict] = []
    base_k = len(existing)
    for offset, ti in enumerate(chosen):
        k = base_k + offset
        ev_q, ev_a = pairs[ti]
        # 干扰项仍从全量 pairs 抽，但排除目标本身
        pool = [j for j in range(len(pairs)) if j != int(ti)]
        distractor_idx = rng.choice(pool, size=haystack_size - 1, replace=False)
        sessions = []
        for j in distractor_idx:
            dq, da = pairs[j]
            sessions.append([
                {"role": "user", "content": dq},
                {"role": "assistant", "content": da},
            ])
        ev_pos = int(rng.integers(0, len(sessions) + 1))
        sessions.insert(ev_pos, [
            {"role": "user", "content": ev_q},
            {"role": "assistant", "content": ev_a},
        ])
        haystack_session_ids = [f"{dataset_key}_s{k}_{idx}" for idx in range(len(sessions))]
        evidence_sid = haystack_session_ids[ev_pos]
        added.append({
            "question_id": f"{dataset_key}_{k}",
            "question_type": "legal-consultation",
            "question": ev_q,
            "answer": ev_a,
            "answer_session_ids": [evidence_sid],
            "haystack_sessions": sessions,
            "haystack_session_ids": haystack_session_ids,
            "_expand_meta": {
                "preserved_prefix": True,
                "expand_seed": expand_seed,
                "quality": "strict",
            },
        })
        used_q.add(_question_key(ev_q))

    merged = list(existing) + added
    meta = {
        "n_existing": len(existing),
        "n_added": len(added),
        "n_total": len(merged),
        "preserve_original": True,
        "expand_seed": expand_seed,
        "haystack_size": haystack_size,
        "strict_quality_for_added": True,
        "added_question_ids": [x["question_id"] for x in added],
        "existing_question_ids": [x["question_id"] for x in existing],
    }
    return merged, meta


def expand_prepare(
    dataset_key: str,
    out_dir: str,
    target_size: int,
    haystack_size: int,
    expand_seed: int,
    backup: bool = True,
) -> str:
    """扩样入口：原 longmemeval_oracle.json 前缀不变，追加严格质量新样本。"""
    ds_dir = os.path.join(out_dir, dataset_key)
    out_path = os.path.join(ds_dir, "longmemeval_oracle.json")
    if not os.path.isfile(out_path):
        raise FileNotFoundError(f"缺少原子集，无法安全扩样: {out_path}")

    with open(out_path, encoding="utf-8") as f:
        existing = json.load(f)
    if not isinstance(existing, list) or not existing:
        raise ValueError(f"原子集为空或格式错误: {out_path}")

    # 备份原子集，保证可回滚到已发表 n=12
    if backup:
        bak = os.path.join(ds_dir, "longmemeval_oracle_n12_published.json")
        if not os.path.isfile(bak):
            with open(bak, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"[{dataset_key}] 已备份原子集 -> {bak}")

    pairs = load_pairs(dataset_key, max_pairs=10**9)
    print(f"[{dataset_key}] 扩样池 {len(pairs)} 条; 现有 {len(existing)} -> 目标 {target_size}")
    merged, meta = expand_instances_preserving(
        dataset_key, pairs, existing, target_size, haystack_size, expand_seed,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    meta_path = os.path.join(ds_dir, "longmemeval_oracle_expand_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"[{dataset_key}] 扩样完成: +{meta['n_added']} -> {meta['n_total']} "
        f"(haystack={haystack_size}) | meta={meta_path}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="法律咨询数据集 -> LongMemEval 格式预处理")
    ap.add_argument("--datasets", nargs="+", default=["disc_law", "lawyer_llama"],
                    choices=list(DATASET_FILES.keys()))
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data", "legal"))
    ap.add_argument("--sample_size", type=int, default=12, help="每个数据集构造的评测样本数")
    ap.add_argument("--haystack_size", type=int, default=15, help="每个样本的会话数(含1条针)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--expand_to", type=int, default=None,
        help="在保留现有 oracle 前缀不变的前提下扩到该样本量（发表安全）",
    )
    ap.add_argument("--expand_seed", type=int, default=42042, help="仅用于新增样本的 RNG 种子")
    args = ap.parse_args()

    for ds in args.datasets:
        if args.expand_to is not None:
            expand_prepare(ds, args.out_dir, args.expand_to, args.haystack_size, args.expand_seed)
        else:
            prepare(ds, args.out_dir, args.sample_size, args.haystack_size, args.seed)


if __name__ == "__main__":
    main()
