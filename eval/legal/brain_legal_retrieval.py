"""脑区分层法律检索综合方案（BIMS 互补学习系统实践）。

三层映射（与论文/类脑前沿对齐）：
  - 海马体情景层（Hippocampus / episodic）：精确问句匹配 + 会话事件绑定 + 问↔答共激活
  - 联想皮层（Association / parietal）：稠密语义 + 隐式关联扩展 + 法律 BM25 稀疏通道
  - 新皮层语义层（Neocortex / semantic）：簇质心关联 + 法律表征投影 W·q（慢学习）

融合权重可通过进化策略在烟测集上自动搜索（不改动核心算法，仅调融合超参）。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from legal_optim import LegalLexicalIndex, learn_legal_projection, make_legal_augment


@dataclass
class BrainLegalWeights:
    """可进化调参的融合权重（海马体层为主，语义层轻量融合）。"""
    session_coherence: float = 1.0
    exact_match_boost: float = 1.0
    lex_weight: float = 0.12
    proj_beta: float = 0.15

    def mutate(self, rng: np.random.Generator, sigma: float = 0.08) -> "BrainLegalWeights":
        d = copy.deepcopy(self)
        for name in ("session_coherence", "exact_match_boost", "lex_weight", "proj_beta"):
            v = getattr(d, name) + rng.normal(0, sigma)
            setattr(d, name, float(np.clip(v, 0.05, 1.0)))
        return d


def apply_brain_legal_hooks(
    mgr,
    *,
    lexical: Optional[LegalLexicalIndex] = None,
    projection: Optional[np.ndarray] = None,
    weights: Optional[BrainLegalWeights] = None,
):
    """将脑区分层检索挂钩安装到 VectorMemoryManager（评测专用，默认不影响 app）。"""
    w = weights or BrainLegalWeights()
    mgr.vector_store.use_pq = False
    mgr._session_expand = True
    mgr._session_coherence = w.session_coherence
    mgr._session_first_rerank = True
    mgr._exact_match_boost = w.exact_match_boost
    mgr._query_projection = projection
    augment = make_legal_augment(mgr, lexical, weight=w.lex_weight, top_n=30) if lexical else None
    mgr._retrieval_augment = augment
    mgr._brain_legal_weights = w
    return w


def train_projection_soft(mgr, train_pairs, ridge=1.0, beta=0.35) -> Optional[np.ndarray]:
    qs, as_ = [], []
    for q, a in train_pairs:
        qv = mgr._normalize_vector(mgr._get_embedding(q))
        av = mgr._normalize_vector(mgr._get_embedding(a))
        if np.linalg.norm(qv) > 1e-6 and np.linalg.norm(av) > 1e-6:
            qs.append(qv)
            as_.append(av)
    if len(qs) < 30:
        return None
    W = learn_legal_projection(np.array(qs), np.array(as_), ridge=ridge)
    return (1.0 - beta) * np.eye(W.shape[0], dtype="float32") + beta * W


def evolve_weights(
    eval_fn: Callable[[BrainLegalWeights], Dict[str, float]],
    *,
    seed: int = 42,
    population: int = 12,
    generations: int = 8,
    targets: Optional[Dict[str, float]] = None,
) -> Tuple[BrainLegalWeights, Dict[str, float]]:
    """简单进化策略：在烟测集上搜索融合权重，最大化加权适应度。"""
    targets = targets or {"session_recall@k": 0.98, "answer_recall@k": 0.95, "qa_correctness": 0.90}
    rng = np.random.default_rng(seed)
    pop: List[BrainLegalWeights] = [BrainLegalWeights()]
    for _ in range(population - 1):
        pop.append(BrainLegalWeights().mutate(rng, sigma=0.15))

    best_w, best_m, best_fit = pop[0], {}, -1.0
    for gen in range(generations):
        scored = []
        for w in pop:
            m = eval_fn(w)
            fit = 0.0
            for k, tgt in targets.items():
                v = m.get(k)
                if v is None:
                    continue
                fit += min(v / tgt, 1.0)
            scored.append((fit, w, m))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_fit:
            best_fit, best_w, best_m = scored[0]
        print(f"  [evolve gen {gen+1}/{generations}] best_fit={scored[0][0]:.3f} "
              f"sess={scored[0][2].get('session_recall@k',0):.3f} "
              f"ans={scored[0][2].get('answer_recall@k',0):.3f} "
              f"qa={scored[0][2].get('qa_correctness',0):.3f}", flush=True)
        if best_fit >= len(targets) * 0.99:
            break
        elites = [x[1] for x in scored[: max(2, population // 4)]]
        pop = elites[:]
        while len(pop) < population:
            parent = elites[rng.integers(0, len(elites))]
            pop.append(parent.mutate(rng, sigma=0.06))
    return best_w, best_m
