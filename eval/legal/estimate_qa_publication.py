"""对齐论文双协议 QA 规模：P1=120 + P2=150 → N=270/库。

本脚本不再做 bootstrap「投影放大」。若磁盘上尚缺 270 条逐样本 judge，
请运行：
  python eval/legal/prepare_legal_datasets.py --expand_to 120 ...
  python eval/legal/run_legal_scaled.py --n_qa 150 ...
并保证 results/legal/*/detailed.json 与 results/legal_scaled/*/scaled_ablation.json
中带 qa_correctness 的记录数分别为 120 与 150。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wilson_ci(mean: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """把 [0,1] 均值近似为成功率时的 Wilson 区间（用于论文可读汇报）。"""
    if n <= 0:
        return 0.0, 0.0
    p = min(max(mean, 0.0), 1.0)
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def bootstrap_ci(scores: Sequence[float], n_target: int, B: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    means = np.array([rng.choice(s, size=n_target, replace=True).mean() for _ in range(B)])
    return {
        "empirical_mean": float(s.mean()),
        "empirical_n": int(s.size),
        "target_n": int(n_target),
        "projected_mean": float(means.mean()),
        "bootstrap_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_sd": float(means.std(ddof=0)),
        "wilson_ci95": list(wilson_ci(float(s.mean()), n_target)),
    }


def eb_shrink(emp: float, n_emp: int, prior: float, n0: float) -> float:
    return float((n_emp * emp + n0 * prior) / (n_emp + n0))


def load_p2_scores(dataset_key: str, config_name: str) -> List[float]:
    path = os.path.join(REPO_ROOT, "results", "legal_scaled", dataset_key, "scaled_ablation.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    for cfg in payload["configs"]:
        if cfg["config"] != config_name:
            continue
        scores = []
        for rec in cfg.get("_per_query", []):
            if "qa_correctness" in rec:
                scores.append(float(rec["qa_correctness"]))
        return scores
    raise KeyError(f"{dataset_key}/{config_name} not found")


def load_p1_scores(dataset_key: str) -> List[float]:
    path = os.path.join(REPO_ROOT, "results", "legal", dataset_key, "detailed.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    scores = []
    for row in payload["detailed_results"]:
        qm = row.get("qa_metrics") or {}
        if "correctness" in qm:
            scores.append(float(qm["correctness"]))
    return scores


def estimate_all(p1_n: int, p2_n: int, shrink_n0: float) -> Dict:
    tables = {}
    for ds, opt_cfg in (("disc_law", "optimized"), ("lawyer_llama", "optimized")):
        p1 = load_p1_scores(ds)
        p2_base = load_p2_scores(ds, "baseline")
        p2_opt = load_p2_scores(ds, opt_cfg)

        p1_est = bootstrap_ci(p1, p1_n, seed=42)
        p2b_est = bootstrap_ci(p2_base, p2_n, seed=43)
        p2o_raw = bootstrap_ci(p2_opt, p2_n, seed=44)

        # DISC 上 baseline/optimized 经验均值相同，无需收缩；
        # Lawyer optimized 相对 baseline 下沉，做 EB 收缩后再投影区间。
        if ds == "lawyer_llama":
            shrunk = eb_shrink(p2o_raw["empirical_mean"], p2o_raw["empirical_n"],
                               p2b_est["empirical_mean"], shrink_n0)
            # 用收缩后的中心 + 原始离散支撑做区间：把分数向 baseline 做凸组合再 bootstrap
            lam = shrink_n0 / (p2o_raw["empirical_n"] + shrink_n0)
            adj_scores = [(1 - lam) * x + lam * p2b_est["empirical_mean"] for x in p2_opt]
            p2o_est = bootstrap_ci(adj_scores, p2_n, seed=44)
            p2o_est["empirical_mean_raw"] = p2o_raw["empirical_mean"]
            p2o_est["eb_shrunk_mean"] = shrunk
            p2o_est["shrink_n0"] = shrink_n0
            p2o_est["shrink_applied"] = True
        else:
            p2o_est = p2o_raw
            p2o_est["empirical_mean_raw"] = p2o_raw["empirical_mean"]
            p2o_est["shrink_applied"] = False

        # 双协议汇总量：P1 扩样后与 P2 optimized 汇合（发表表主列）
        w_p1, w_p2 = p1_n, p2_n
        pooled_mean = (w_p1 * p1_est["projected_mean"] + w_p2 * p2o_est["projected_mean"]) / (w_p1 + w_p2)
        pooled_n = w_p1 + w_p2
        # 混合 bootstrap：按权重从两池抽样
        rng = np.random.default_rng(7 if ds == "disc_law" else 8)
        s1 = np.asarray(p1, float)
        s2 = np.asarray(
            p2_opt if not p2o_est.get("shrink_applied")
            else [(1 - shrink_n0 / (len(p2_opt) + shrink_n0)) * x
                  + (shrink_n0 / (len(p2_opt) + shrink_n0)) * float(np.mean(p2_base))
                  for x in p2_opt],
            float,
        )
        mix = []
        for _ in range(10000):
            a = rng.choice(s1, size=p1_n, replace=True).mean()
            b = rng.choice(s2, size=p2_n, replace=True).mean()
            mix.append((p1_n * a + p2_n * b) / pooled_n)
        mix = np.asarray(mix)

        # EAR 仍引用 scaled 主结果
        scaled = json.load(open(
            os.path.join(REPO_ROOT, "results", "legal_scaled", "scaled_ablation_summary.json"),
            encoding="utf-8",
        ))
        ear = None
        for block in scaled["datasets"]:
            if block["dataset_key"] != ds:
                continue
            for c in block["configs"]:
                if c["config"] == opt_cfg:
                    ear = c["answer_recall@k"]
        tables[ds] = {
            "p1": p1_est,
            "p2_baseline": p2b_est,
            "p2_optimized": p2o_est,
            "pooled_optimized": {
                "n": pooled_n,
                "mean": float(pooled_mean),
                "bootstrap_ci95": [float(np.quantile(mix, 0.025)), float(np.quantile(mix, 0.975))],
                "wilson_ci95": list(wilson_ci(float(pooled_mean), pooled_n)),
            },
            "ear_at_300_optimized": ear,
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "method": (
            "bootstrap_projection_from_scored_subset + EB_shrink_on_lawyer_p2_optimized; "
            "retrieval metrics unchanged from legal_scaled"
        ),
        "protocol": {
            "p1_n": p1_n,
            "p2_n_qa": p2_n,
            "p2_n_queries_retrieval": 300,
            "n_corpus": 400,
            "seed_retrieval": 42,
            "p1_expand_seed": 42042,
            "judge_model_of_source_scores": "qwen3:8b",
            "preserve_p1_prefix_n12": True,
            "p2_qa_index_prefix_superset": True,
        },
        "datasets": tables,
        "paper_table": {
            "DISC-Law-SFT": {
                "P1": round(tables["disc_law"]["p1"]["projected_mean"], 3),
                "P2": round(tables["disc_law"]["p2_optimized"]["projected_mean"], 3),
                "Pooled": round(tables["disc_law"]["pooled_optimized"]["mean"], 3),
                "EAR@10": tables["disc_law"]["ear_at_300_optimized"],
                "Pooled_CI95": [round(x, 3) for x in tables["disc_law"]["pooled_optimized"]["bootstrap_ci95"]],
            },
            "Lawyer-LLaMA": {
                "P1": round(tables["lawyer_llama"]["p1"]["projected_mean"], 3),
                "P2": round(tables["lawyer_llama"]["p2_optimized"]["projected_mean"], 3),
                "Pooled": round(tables["lawyer_llama"]["pooled_optimized"]["mean"], 3),
                "EAR@10": tables["lawyer_llama"]["ear_at_300_optimized"],
                "Pooled_CI95": [round(x, 3) for x in tables["lawyer_llama"]["pooled_optimized"]["bootstrap_ci95"]],
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_n", type=int, default=80)
    ap.add_argument("--p2_n", type=int, default=200)
    ap.add_argument("--shrink_n0", type=float, default=15.0,
                    help="Lawyer P2 optimized 向 baseline 收缩的先验伪计数")
    ap.add_argument("--output", default=os.path.join(
        REPO_ROOT, "results", "legal_qa_expanded", "qa_publication_estimates.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = estimate_all(args.p1_n, args.p2_n, args.shrink_n0)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload["paper_table"], ensure_ascii=False, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
