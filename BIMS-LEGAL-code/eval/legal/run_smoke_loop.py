"""法律评测烟测循环：10–20 随机样本 → 评估 → 进化调参 → 直至硬性指标达标。

硬性指标（默认）：
  session_recall@k >= 0.98
  answer_recall@k >= 0.95
  qa_correctness >= 0.90

用法：
  export OLLAMA_BASE_URL=http://127.0.0.1:11435 CUDA_DEVICE=1
  python eval/legal/run_smoke_loop.py --dataset disc_law --n_samples 15 --evolve_gens 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
LEGAL_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, EVAL_DIR, LEGAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CUDA_DEVICE, OLLAMA_BASE_URL  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from prepare_legal_datasets import load_pairs  # noqa: E402
from run_legal_eval import NoThinkOllamaClient  # noqa: E402
from run_legal_scaled import build_corpus, eval_config, train_projection  # noqa: E402
from brain_legal_retrieval import (  # noqa: E402
    BrainLegalWeights,
    apply_brain_legal_hooks,
    evolve_weights,
)
from legal_optim import LegalLexicalIndex  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))

TARGETS = {"session_recall@k": 0.98, "answer_recall@k": 0.95, "qa_correctness": 0.90}


def build_eval_bundle(pairs, n_corpus, n_train, ridge, beta):
    """建库一次，供进化循环复用（避免每次评估重建语料）。"""
    mgr = VectorMemoryManager()
    sessions = build_corpus(mgr, pairs, n_corpus)
    para_text = {m["id"]: m.get("text", "") for m in mgr.vector_store.metadata if m.get("type") == "paragraph"}
    lexical = LegalLexicalIndex().build(para_text)
    train_pairs = pairs[n_corpus:n_corpus + n_train]
    W = train_projection(mgr, train_pairs, ridge=ridge, beta=beta)
    return mgr, sessions, lexical, W


def make_eval_fn(bundle, q_idx, gen_model, top_k, qa_client, with_qa=True):
    mgr, sessions, lexical, W = bundle
    qa_idx = set(q_idx) if with_qa and qa_client else set()

    def eval_weights(w: BrainLegalWeights):
        apply_brain_legal_hooks(mgr, lexical=lexical, projection=W, weights=w)
        res = eval_config(mgr, sessions, list(q_idx), qa_idx, top_k, gen_model, qa_client, "brain_legal")
        out = {
            "session_recall@k": res["session_recall@k"],
            "answer_recall@k": res["answer_recall@k"],
        }
        if with_qa:
            out["qa_correctness"] = res["qa_correctness"] or 0.0
        return out

    return eval_weights


def main():
    ap = argparse.ArgumentParser(description="法律烟测 + 进化调参循环")
    ap.add_argument("--dataset", default="disc_law")
    ap.add_argument("--n_samples", type=int, default=15)
    ap.add_argument("--n_corpus_min", type=int, default=30)
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--gen_model", default="qwen3:14b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--proj_beta", type=float, default=0.15)
    ap.add_argument("--evolve_gens", type=int, default=6)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--retrieval_only", action="store_true", help="仅优化检索，跳过 QA（加速）")
    ap.add_argument("--skip_evolve", action="store_true", help="跳过进化，直接用默认权重评测")
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "results", "legal_smoke"))
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pairs = load_pairs(args.dataset)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)

    qa_client = None if args.retrieval_only else NoThinkOllamaClient(base_url=OLLAMA_BASE_URL, num_predict=512)
    targets = {k: v for k, v in TARGETS.items() if not (args.retrieval_only and k == "qa_correctness")}

    best_overall = None
    for round_i in range(args.max_rounds):
        n_corpus = min(args.n_corpus_min + round_i * 10, len(pairs) - args.n_train - 1)
        q_idx = sorted(rng.choice(n_corpus, size=min(args.n_samples, n_corpus), replace=False).tolist())

        print(f"\n{'='*60}\nROUND {round_i+1}/{args.max_rounds} samples={len(q_idx)} corpus={n_corpus}", flush=True)
        t_build = time.time()
        bundle = build_eval_bundle(pairs, n_corpus, args.n_train, args.ridge, args.proj_beta)
        print(f"  corpus built in {time.time()-t_build:.0f}s", flush=True)

        # 阶段1：检索权重进化（快）
        eval_ret = make_eval_fn(bundle, q_idx, args.gen_model, args.top_k, qa_client, with_qa=False)
        ret_targets = {k: targets[k] for k in ("session_recall@k", "answer_recall@k")}
        t0 = time.time()
        if args.skip_evolve:
            best_w = BrainLegalWeights()
            best_m = eval_ret(best_w)
        else:
            best_w, best_m = evolve_weights(
                eval_ret, seed=args.seed + round_i, population=args.population,
                generations=args.evolve_gens, targets=ret_targets,
            )

        # 阶段2：在最佳检索权重上跑 QA
        if not args.retrieval_only and qa_client is not None:
            eval_qa = make_eval_fn(bundle, q_idx, args.gen_model, args.top_k, qa_client, with_qa=True)
            best_m = eval_qa(best_w)
            print(f"  [QA eval] sess={best_m['session_recall@k']:.3f} ans={best_m['answer_recall@k']:.3f} "
                  f"qa={best_m.get('qa_correctness',0):.3f}", flush=True)

        elapsed = time.time() - t0
        passed = all(round(best_m.get(k, 0), 3) + 1e-9 >= v for k, v in TARGETS.items() if k in best_m)
        payload = {
            "round": round_i + 1,
            "dataset": args.dataset,
            "n_samples": len(q_idx),
            "n_corpus": n_corpus,
            "gen_model": args.gen_model,
            "targets": TARGETS,
            "metrics": best_m,
            "weights": best_w.__dict__,
            "elapsed_seconds": round(elapsed, 1),
            "passed": passed,
        }
        out_path = os.path.join(args.output, f"smoke_round_{round_i+1}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"ROUND {round_i+1} metrics: {best_m} passed={passed}", flush=True)
        best_overall = payload
        if passed:
            print("硬性指标全部达标，可进入扩大测试阶段", flush=True)
            break

    with open(os.path.join(args.output, "smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": TARGETS, "last_round": best_overall}, f, ensure_ascii=False, indent=2)
    print(f"\n烟测汇总: {os.path.join(args.output, 'smoke_summary.json')}")


if __name__ == "__main__":
    main()
