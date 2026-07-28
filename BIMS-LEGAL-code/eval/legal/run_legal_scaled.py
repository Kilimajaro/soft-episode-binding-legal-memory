"""法律咨询领域 —— 发表级评测：规模化 + 法律领域专用检索优化 + 完整消融 + 完整 QA。

共享语料设计：把 M 条法律咨询会话写入同一超长记忆库，再用其中 S 条会话的问题作为查询，
计算检索指标与 QA 正确率。嵌入只算一次，可在 CPU 上扩展到数百条查询。

消融配置（逐步叠加，隔离各项贡献）：
  1. baseline      ：原系统（IVFPQ 近似索引，无优化）
  2. generic       ：通用优化 = 精确内积索引 + 情景会话一致性检索
  3. legal_lexical ：generic + 法律词法/法条 BM25 hybrid 检索
  4. legal_repr    ：generic + 法律表征适配（查询线性投影 W·q，岭回归在语料外问答对上学习）
  5. full          ：generic + legal_lexical + legal_repr（全部）

指标：session_recall@k / answer_recall@k / precision@k / ndcg@k / qa_correctness。
QA 可对指定配置在**全部查询**上评测（--qa_configs，受 CPU 速度影响，默认用较快的中文模型）。
每个配置评完即增量落盘，长时任务可断点保留。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(REPO_ROOT, "eval")
LEGAL_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, EVAL_DIR, LEGAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_new import LongMemEvalEvaluator  # noqa: E402
from memory_manager import VectorMemoryManager  # noqa: E402
from config import CUDA_DEVICE, OLLAMA_BASE_URL  # noqa: E402
from prepare_legal_datasets import load_pairs, DATASET_FILES  # noqa: E402
from run_legal_eval import NoThinkOllamaClient, DATASETS  # noqa: E402
from legal_optim import LegalLexicalIndex, make_legal_augment  # noqa: E402
from brain_legal_retrieval import (  # noqa: E402
    BrainLegalWeights,
    apply_brain_legal_hooks,
    train_projection_soft,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))

_METRIC_EV = LongMemEvalEvaluator.__new__(LongMemEvalEvaluator)

# exact_index = O1 only（FlatIP，无 session expand）；generic ≈ O1+O2
ALL_CONFIGS = [
    "baseline", "exact_index", "generic",
    "legal_lexical", "legal_repr", "full", "brain_legal",
]


def build_corpus(mgr, pairs, n_corpus):
    # 保留调用方设置的 use_pq（reset 曾硬编码 True，导致 FlatIP 消融失效）
    use_pq = bool(getattr(mgr.vector_store, "use_pq", False))
    mgr.reset(use_pq=use_pq)
    mgr._bulk_load = True
    # 扩大 LRU，避免 warm_embed 后被挤出（默认 capacity=500）
    try:
        mgr.lru_cache.capacity = max(getattr(mgr.lru_cache, "capacity", 500), n_corpus * 8 + 1000)
    except Exception:
        pass
    # 批量预嵌入段落 + 分句（与 add_dialog 路径一致）
    texts = []
    for q, a in pairs[:n_corpus]:
        texts.extend([q, a])
        texts.extend(mgr._split_sentences(q))
        texts.extend(mgr._split_sentences(a))
    try:
        n_warm = mgr.warm_embed_cache(texts)
        print(
            f"  warm_embed_cache {n_warm}/{len(texts)} "
            f"use_pq={mgr.vector_store.use_pq} lru_cap={mgr.lru_cache.capacity}",
            flush=True,
        )
    except Exception as e:
        print(f"  warm_embed_cache skipped: {e}", flush=True)
    sessions = []
    for k, (q, a) in enumerate(pairs[:n_corpus]):
        sid = f"sess_{k}"
        tid_u = mgr.add_dialog("user", q, session_id=sid)
        tid_a = mgr.add_dialog("assistant", a, session_id=sid)
        sessions.append({"question": q, "answer": a, "tid_user": tid_u, "tid_assistant": tid_a})
        if (k + 1) % 50 == 0:
            print(f"  build_corpus {k+1}/{n_corpus} use_pq={mgr.vector_store.use_pq}", flush=True)
    mgr.finalize_bulk_load()
    mgr._bulk_load = False
    stored = {m.get("id") for m in mgr.vector_store.metadata if m.get("type") == "paragraph"}
    for s in sessions:
        s["evidence_session_tids"] = [t for t in (s["tid_user"], s["tid_assistant"]) if t in stored]
        s["evidence_answer_tids"] = [t for t in (s["tid_assistant"],) if t in stored]
    return sessions


def train_projection(mgr, train_pairs, ridge=1.0, beta=0.35):
    return train_projection_soft(mgr, train_pairs, ridge=ridge, beta=beta)


def _truncate_context(retrieved, max_items=4, per_item_chars=1500):
    """QA 上下文：律师解答优先；解答段尽量完整保留。"""
    ranked = sorted(
        retrieved,
        key=lambda r: (
            0 if (r.get("context_label") == "assistant") else 1,
            -float(r.get("final_score", r.get("score", 0)) or 0),
        ),
    )
    parts = []
    for i, r in enumerate(ranked[:max_items]):
        role = "律师解答" if r.get("context_label") == "assistant" else "咨询记录"
        txt = (r.get("full_dialog") or r.get("full_text") or r.get("text", "")).strip()
        if not txt:
            continue
        limit = 3000 if r.get("context_label") == "assistant" else per_item_chars
        parts.append(f"[{i + 1}][{role}]\n{txt[:limit]}")
    return "\n\n".join(parts) if parts else "（无相关记忆检索结果）"


def eval_config(mgr, sessions, q_idx, qa_idx, top_k, model, qa_client, name, t_start=None):
    mgr.lru_cache.clear()
    sr_l, ar_l, pr_l, nd_l, qa_l = [], [], [], [], []
    per_query = []
    t0 = time.time()
    for n, qi in enumerate(q_idx):
        s = sessions[qi]
        if not s["evidence_session_tids"]:
            continue
        retrieved = mgr.search(s["question"], top_k=top_k, is_temporal_task=False)
        ev_s, ev_a = s["evidence_session_tids"], s["evidence_answer_tids"]
        sr = _METRIC_EV._calculate_recall(retrieved, ev_s)
        pr = _METRIC_EV._calculate_precision(retrieved, ev_s)
        nd = _METRIC_EV._calculate_ndcg(retrieved, ev_s)
        ar = _METRIC_EV._calculate_recall(retrieved, ev_a) if ev_a else 0.0
        sr_l.append(sr); ar_l.append(ar); pr_l.append(pr); nd_l.append(nd)
        rec = {"q": s["question"][:80], "session_recall": sr, "answer_recall": ar, "ndcg": nd}
        if qi in qa_idx and qa_client is not None:
            ctx = _truncate_context(retrieved)
            prompt = (
                "你是一名资深法律顾问。下列检索记忆包含历史法律咨询中的律师专业解答。\n"
                "请严格依据【律师解答】回答问题，完整保留关键法条、法律结论与操作建议，表述准确。\n"
                "若记忆中已有完整解答，请忠实归纳该解答要点，不要编造。\n\n"
                "【检索记忆】\n" + ctx +
                "\n\n【待回答问题】\n" + s["question"] +
                "\n\n【你的专业解答】"
            )
            gen = qa_client.generate_response_with_retry(model, prompt)
            score, _ = qa_client.evaluate_answer_correctness(model, s["question"], gen, s["answer"])
            qa_l.append(score)
            rec["qa_correctness"] = score
        per_query.append(rec)
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{name}] {n+1}/{len(q_idx)} ({el:.0f}s) | sess={np.mean(sr_l):.3f} "
                  f"ans={np.mean(ar_l):.3f} ndcg={np.mean(nd_l):.3f}"
                  + (f" qa={np.mean(qa_l):.3f}(n={len(qa_l)})" if qa_l else ""), flush=True)
    return {
        "config": name,
        "n_queries_evaluated": len(sr_l),
        "session_recall@k": float(np.mean(sr_l)) if sr_l else 0.0,
        "answer_recall@k": float(np.mean(ar_l)) if ar_l else 0.0,
        "precision@k": float(np.mean(pr_l)) if pr_l else 0.0,
        "ndcg@k": float(np.mean(nd_l)) if nd_l else 0.0,
        "qa_correctness": float(np.mean(qa_l)) if qa_l else None,
        "n_qa_evaluated": len(qa_l),
        "elapsed_seconds": round(time.time() - t0, 1),
        "_per_query": per_query,
    }


def _save(out_dir, dataset_key, payload):
    with open(os.path.join(out_dir, "legal_full_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_dataset(dataset_key, gen_model, n_corpus, n_queries, n_qa, n_train, top_k, seed,
                lex_weight, ridge, proj_beta, qa_configs, output_root,
                brain_weights: BrainLegalWeights | None = None):
    rng = np.random.default_rng(seed)
    pairs = load_pairs(dataset_key)
    rng.shuffle(pairs)
    n_corpus = min(n_corpus, len(pairs))
    q_idx = list(range(min(n_corpus, n_queries)))
    qa_idx = set(q_idx[:n_qa]) if n_qa > 0 else set()
    qa_client = NoThinkOllamaClient(base_url=OLLAMA_BASE_URL, num_predict=512) if n_qa > 0 else None

    out_dir = os.path.join(output_root, dataset_key)
    os.makedirs(out_dir, exist_ok=True)
    ablation_path = os.path.join(out_dir, "legal_full_ablation.json")
    done_configs = set()
    bw_default = brain_weights or BrainLegalWeights(lex_weight=lex_weight, proj_beta=proj_beta)
    payload = {"dataset_key": dataset_key, "dataset_name": DATASETS.get(dataset_key, dataset_key),
               "gen_model": gen_model, "n_corpus": n_corpus, "n_queries": len(q_idx),
               "n_qa": n_qa, "n_train_proj": n_train, "top_k": top_k, "seed": seed,
               "lex_weight": bw_default.lex_weight, "ridge": ridge, "proj_beta": bw_default.proj_beta,
               "brain_weights": bw_default.__dict__, "configs": []}
    if os.path.isfile(ablation_path):
        try:
            with open(ablation_path, encoding="utf-8") as f:
                prev = json.load(f)
            if (prev.get("seed") == seed and prev.get("n_corpus") == n_corpus
                    and prev.get("n_queries") == len(q_idx) and prev.get("top_k") == top_k):
                payload = prev
                done_configs = {c["config"] for c in payload.get("configs", [])}
                if done_configs:
                    print(f"[{dataset_key}] resume: skip {sorted(done_configs)}", flush=True)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def add(res):
        payload["configs"].append(res)
        _save(out_dir, dataset_key, payload)
        qa = f"{res['qa_correctness']:.3f}" if res["qa_correctness"] is not None else "-"
        print(f"[{dataset_key}/{res['config']}] sess={res['session_recall@k']:.3f} "
              f"ans={res['answer_recall@k']:.3f} ndcg={res['ndcg@k']:.3f} qa={qa} "
              f"({res['elapsed_seconds']:.0f}s)", flush=True)

    # ---------- config 1: baseline (PQ) ----------
    if "baseline" not in done_configs:
        t0 = time.time()
        print(f"[{dataset_key}] build PQ corpus (M={n_corpus}) ...", flush=True)
        mgr_pq = VectorMemoryManager(); mgr_pq.vector_store.use_pq = True
        sess_pq = build_corpus(mgr_pq, pairs, n_corpus)
        print(f"[{dataset_key}] PQ build {time.time()-t0:.0f}s", flush=True)
        add(eval_config(mgr_pq, sess_pq, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "baseline" in qa_configs else None, "baseline"))
        del mgr_pq

    remaining = [c for c in ALL_CONFIGS[1:] if c not in done_configs]
    if not remaining:
        return payload

    # ---------- shared flat build for configs 2+ ----------
    t0 = time.time()
    print(f"[{dataset_key}] build exact(flat) corpus (M={n_corpus}) ...", flush=True)
    mgr = VectorMemoryManager(); mgr.vector_store.use_pq = False
    sessions = build_corpus(mgr, pairs, n_corpus)
    print(f"[{dataset_key}] flat build {time.time()-t0:.0f}s", flush=True)

    # 法律词法索引（段落级）
    para_text = {m["id"]: m.get("text", "") for m in mgr.vector_store.metadata
                 if m.get("type") == "paragraph"}
    lexical = LegalLexicalIndex().build(para_text)
    augment = make_legal_augment(mgr, lexical, weight=lex_weight, top_n=20)

    # 法律表征适配（在语料外、未被查询的法律问答对上学习投影，避免泄漏）
    t0 = time.time()
    train_pairs = pairs[n_corpus:n_corpus + n_train]
    print(f"[{dataset_key}] learning legal projection on {len(train_pairs)} disjoint pairs ...", flush=True)
    W = train_projection(mgr, train_pairs, ridge=ridge, beta=proj_beta)
    print(f"[{dataset_key}] projection learned {time.time()-t0:.0f}s (W={'ok' if W is not None else 'none'})", flush=True)

    def set_hooks(session_expand, projection, augment_fn, session_first=False, exact_boost=0.0,
                  session_coherence=0.98):
        mgr._session_expand = session_expand
        mgr._query_projection = projection
        mgr._retrieval_augment = augment_fn
        mgr._session_first_rerank = session_first
        mgr._exact_match_boost = exact_boost
        mgr._session_coherence = session_coherence

    # config 2: exact_index = O1 only（FlatIP，无 session expand）
    if "exact_index" in remaining:
        set_hooks(False, None, None, session_first=False, exact_boost=0.0)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "exact_index" in qa_configs else None, "exact_index"))
    # config 3: generic（O1+O2：精确索引 + 会话一致性）
    if "generic" in remaining:
        set_hooks(True, None, None, session_first=True, exact_boost=0.0, session_coherence=0.98)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "generic" in qa_configs else None, "generic"))
    # config 4: + legal_lexical
    if "legal_lexical" in remaining:
        set_hooks(True, None, augment, session_first=True, exact_boost=0.0)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "legal_lexical" in qa_configs else None, "legal_lexical"))
    # config 5: + legal_repr
    if "legal_repr" in remaining:
        set_hooks(True, W, None, session_first=True, exact_boost=0.0)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "legal_repr" in qa_configs else None, "legal_repr"))
    # config 6: full
    if "full" in remaining:
        set_hooks(True, W, augment, session_first=True, exact_boost=0.0)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "full" in qa_configs else None, "full"))
    # config 7: brain_legal（脑区分层综合方案 + 进化默认权重）
    if "brain_legal" in remaining:
        apply_brain_legal_hooks(mgr, lexical=lexical, projection=W, weights=bw_default)
        add(eval_config(mgr, sessions, q_idx, qa_idx, top_k, gen_model,
                        qa_client if "brain_legal" in qa_configs else None, "brain_legal"))
    set_hooks(False, None, None, session_first=False, exact_boost=0.0)
    return payload


def main():
    ap = argparse.ArgumentParser(description="法律咨询 —— 规模化 + 法律专用优化 + 完整消融 + 完整 QA")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_FILES.keys()))
    ap.add_argument("--gen_model", default="qwen3:14b", help="QA 生成/评判模型")
    ap.add_argument("--n_corpus", type=int, default=400)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--n_qa", type=int, default=150, help="P2 QA 评测查询数（论文默认 150；与 P1=120 合计 N=270/库）")
    ap.add_argument("--n_train", type=int, default=1500, help="表征适配训练问答对数（语料外）")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lex_weight", type=float, default=0.4)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--proj_beta", type=float, default=0.5, help="表征适配软融合系数 (0=不投影,1=完全投影)")
    ap.add_argument("--qa_configs", nargs="+", default=["brain_legal", "generic", "baseline"],
                    help="对哪些配置做 QA")
    ap.add_argument("--output_root", default=os.path.join(REPO_ROOT, "results", "legal_optimized"))
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    all_payloads = []
    for ds in args.datasets:
        all_payloads.append(run_dataset(
            ds, args.gen_model, args.n_corpus, args.n_queries, args.n_qa, args.n_train,
            args.top_k, args.seed, args.lex_weight, args.ridge, args.proj_beta,
            args.qa_configs, args.output_root,
        ))

    slim = []
    for p in all_payloads:
        cfgs = [{k: v for k, v in c.items() if k != "_per_query"} for c in p["configs"]]
        slim.append({**{k: v for k, v in p.items() if k != "configs"}, "configs": cfgs})
    combined = {"generated_at": datetime.now().isoformat(), "config": vars(args), "datasets": slim}
    with open(os.path.join(args.output_root, "legal_full_summary.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    print("法律咨询数据集 —— 规模化 + 法律专用优化 + 完整消融汇总")
    print("=" * 96)
    for p in slim:
        print(f"\n### {p['dataset_name']}  (M={p['n_corpus']}, queries={p['n_queries']}, k={p['top_k']})")
        print(f"{'配置':14} {'会话召回':>9} {'答案召回':>9} {'nDCG':>7} {'QA':>8}")
        for c in p["configs"]:
            qa = f"{c['qa_correctness']:.3f}(n{c['n_qa_evaluated']})" if c["qa_correctness"] is not None else "   -   "
            print(f"{c['config']:14} {c['session_recall@k']:>9.3f} {c['answer_recall@k']:>9.3f} "
                  f"{c['ndcg@k']:>7.3f} {qa:>8}")


if __name__ == "__main__":
    main()
