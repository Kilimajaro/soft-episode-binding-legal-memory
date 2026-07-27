"""在法律咨询数据集上运行 BIMS 记忆系统评测。

复用 eval/eval_new.py::LongMemEvalEvaluator 的同款评测流程与指标：
  - 检索召回率 recall@k（以及 precision@k / ndcg@k）
  - QA 正确率（由 LLM 评判 A/B/C/D -> 1.0/0.7/0.3/0.0）

数据来自 prepare_legal_datasets.py 预处理出的 LongMemEval 兼容格式。
评测/评判模型默认使用本地 Ollama 的 qwen3:14b（关闭"思考模式"；本机无 qwen3:8b）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import requests

os.environ.setdefault("MPLBACKEND", "Agg")  # 无头环境绘图

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_new import LongMemEvalEvaluator, OllamaClient  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from config import OLLAMA_BASE_URL  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS = {
    "disc_law": "DISC-Law-SFT (复旦 DISC-LawLLM · 法律问答)",
    "lawyer_llama": "Lawyer-LLaMA (北大 · 法律咨询)",
}


def assert_ollama_model(base_url: str, model: str) -> None:
    """启动前确认模型已安装，避免跑完全部样本却全是「模型无响应」。"""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        names = {m.get("name", "") for m in r.json().get("models", [])}
        # Ollama 可能返回 "qwen3:14b" 或带 digest 的别名
        aliases = names | {n.split(":")[0] + ":" + n.split(":")[1] for n in names if ":" in n}
        if model not in names and model not in aliases:
            available = sorted(n for n in names if "qwen" in n.lower())
            raise SystemExit(
                f"Ollama 模型未安装: {model!r} @ {base_url}\n"
                f"可用 qwen 相关模型: {available}\n"
                f"请先 `ollama pull {model}`，或改用 --model 指定已有模型（推荐 qwen3:14b）。"
            )
    except requests.RequestException as e:
        raise SystemExit(f"无法连接 Ollama ({base_url}): {e}") from e


class NoThinkOllamaClient(OllamaClient):
    """qwen3 默认思维链很慢；关闭 think 并限制生成长度。"""

    def __init__(self, base_url="http://localhost:11434", timeout=300, num_predict=768):
        super().__init__(base_url=base_url, timeout=timeout)
        self.num_predict = num_predict

    def generate_response(self, model: str, prompt: str, temperature: float = 0.05) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.error(
                    "Ollama API 失败: %s %s",
                    response.status_code,
                    (response.text or "")[:300],
                )
                return ""
            data = response.json()
            text = (data.get("response") or "").strip()
            if not text:
                err = data.get("error")
                logger.warning(
                    "Ollama 空响应 model=%s err=%s keys=%s",
                    model,
                    err,
                    list(data.keys())[:12],
                )
            return text
        except Exception as e:  # noqa: BLE001
            logger.error("Ollama 请求异常: %s", e)
            return ""

    def evaluate_answer_correctness(self, model: str, question: str,
                                  generated_answer: str, ground_truth: str):
        """法律领域评判：对较长标准答案更关注核心结论与法条要点的覆盖。"""
        prompt = f"""你是一名法律评测专家。请判断生成答案对标准法律解答的覆盖程度。

问题：{question}

生成答案：{generated_answer}

标准答案：{ground_truth}

评分标准（法律长答案场景）：
A. 完全正确：生成答案覆盖了标准答案的核心法律结论、关键法条与操作建议（允许表述不同）
B. 部分正确：覆盖主要要点但有细节缺失或不准确
C. 基本错误：仅少量相关
D. 完全错误：无关或严重错误

请返回JSON：{{"choice":"A/B/C/D","reason":"简要理由"}}"""
        response = self.generate_response_with_retry(model, prompt)
        score, details = self._parse_evaluation_response(response)
        details["raw_response"] = response
        return score, details


def run_one(dataset_key: str, model: str, benchmark_root: str, output_root: str,
            max_retrieved_items: int, num_predict: int) -> dict:
    benchmark_path = os.path.join(benchmark_root, dataset_key)
    output_dir = os.path.join(output_root, dataset_key)

    evaluator = LongMemEvalEvaluator(VectorMemoryManager())
    evaluator.config["evaluation_llm"] = model
    # 与 P2 / 论文 N=270 协议对齐：生成与评判用同一模型（eval_config 里的 32b 仅用于其他长程实验）
    evaluator.config["judge_llm"] = model
    evaluator.config["benchmark_path"] = benchmark_path
    evaluator.config["output_dir"] = output_dir
    evaluator.config["max_retrieved_items"] = max_retrieved_items
    evaluator.config["save_detailed_results"] = True
    evaluator.config["detailed_results_threshold"] = 100000
    base_url = evaluator.config.get("ollama_base_url") or OLLAMA_BASE_URL
    assert_ollama_model(base_url, model)
    evaluator.ollama_client = NoThinkOllamaClient(
        base_url=base_url,
        num_predict=num_predict,
    )

    t0 = time.time()
    results = evaluator.run_evaluation(dataset_type="oracle", run_tag=f"legal::{dataset_key}")
    elapsed = time.time() - t0

    overall = results["overall"]
    summary = {
        "dataset_key": dataset_key,
        "dataset_name": DATASETS.get(dataset_key, dataset_key),
        "model": model,
        "total_instances": results["total_instances"],
        "max_retrieved_items": max_retrieved_items,
        "avg_retrieval_recall": overall.get("avg_retrieval_recall"),
        "avg_qa_correctness": overall.get("avg_qa_correctness"),
        "success_rate": overall.get("success_rate"),
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": output_dir,
    }
    print(f"\n[{dataset_key}] done in {elapsed:.0f}s | RR={summary['avg_retrieval_recall']:.3f} "
          f"QA={summary['avg_qa_correctness']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="法律咨询数据集上的 BIMS 记忆系统评测")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument(
        "--model",
        default="qwen3:14b",
        help="Ollama 答题生成模型（默认 qwen3:14b；本机通常无 qwen3:8b）",
    )
    ap.add_argument("--benchmark_root", default=os.path.join(REPO_ROOT, "data", "legal"))
    ap.add_argument("--output_root", default=os.path.join(REPO_ROOT, "results", "legal"))
    ap.add_argument("--max_retrieved_items", type=int, default=10)
    ap.add_argument("--num_predict", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    summaries = []
    for ds in args.datasets:
        summaries.append(run_one(
            ds, args.model, args.benchmark_root, args.output_root,
            args.max_retrieved_items, args.num_predict,
        ))

    combined = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "max_retrieved_items": args.max_retrieved_items,
        "datasets": summaries,
    }
    combined_path = os.path.join(args.output_root, "legal_eval_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("法律咨询数据集评测汇总")
    print("=" * 64)
    print(f"{'数据集':28} {'样本':>4} {'检索召回RR':>10} {'QA正确率':>9} {'成功率':>7}")
    for s in summaries:
        print(f"{s['dataset_name'][:26]:28} {s['total_instances']:>4} "
              f"{s['avg_retrieval_recall']:>10.3f} {s['avg_qa_correctness']:>9.3f} "
              f"{s['success_rate']:>7.3f}")
    print(f"\n汇总已保存: {combined_path}")


if __name__ == "__main__":
    main()
