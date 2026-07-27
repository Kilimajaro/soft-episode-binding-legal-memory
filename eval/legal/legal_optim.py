"""针对法律文本特征的检索优化模块。

包含两类法律领域专用优化（均可与通用优化叠加，做完整消融）：

1) 法律词法/法条关键索引（LegalLexicalIndex）—— 法律稀疏检索
   法律咨询文本高度依赖**法条引用**（如《中华人民共和国刑法》第二百六十四条）与**法律术语**
   （盗窃罪、合同诈骗、人身安全保护令…）。通用稠密嵌入（nomic）对这些低频但决定性的法律 token
   往往加权不足。这里抽取"法条引用 / 法条编号 / 法律术语(借助中文字符二元组)"建立 BM25 稀疏索引，
   与稠密检索做 **hybrid 融合**（dense ∪ lexical，再按融合分重排），把命中法律关键项的答案段召回。

2) 法律表征适配（learn_legal_projection）—— 轻量"法律表征模型微调"
   真正微调 8B 级嵌入模型在 CPU 上不可行；这里在**冻结的通用嵌入之上**学习一个 768×768 的线性
   投影 W（岭回归闭式解），用法律(问, 答)嵌入对训练，使"问"的表征朝其"答"的表征对齐：W·q ≈ a。
   检索时把查询嵌入投影到法律答案空间（query-side 适配），直接提升 answer_recall。
   训练数据与评测查询**严格不相交**（用语料外的法律问答对训练），避免信息泄漏。
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# 法条引用：《xxx法/解释/规定》[第…条][之…]
_STATUTE_RE = re.compile(r"《[^》]{2,40}》")
# 法条编号：第…条 / 第…条之…
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零0-9]{1,8}条(?:之[一二三四五六七八九十0-9]{1,3})?")
# 常见罪名/法律行为后缀，捕获 "*罪" 等法律术语
_CRIME_RE = re.compile(r"[\u4e00-\u9fa5]{1,8}罪")


def _char_bigrams(text: str):
    t = re.sub(r"\s+", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def legal_tokens(text: str):
    """法律感知分词：字符二元组(覆盖通用语义) + 法条引用/编号/罪名(法律关键 token，加权前缀)。"""
    if not text:
        return []
    toks = _char_bigrams(text)
    for m in _STATUTE_RE.findall(text):
        toks.append("§" + m)          # 法条出处
    for m in _ARTICLE_RE.findall(text):
        toks.append("†" + m)          # 法条编号
    for m in _CRIME_RE.findall(text):
        toks.append("¶" + m)          # 罪名
    return toks


class LegalLexicalIndex:
    """法律关键项 BM25 稀疏索引（法条/编号/罪名 + 中文字符二元组）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.df = Counter()
        self.doc_tokens = {}
        self.doc_len = {}
        self.N = 0
        self.avgdl = 1.0
        self.idf = {}

    def build(self, tid_text: dict):
        for tid, text in tid_text.items():
            toks = legal_tokens(text)
            self.doc_tokens[tid] = Counter(toks)
            self.doc_len[tid] = max(1, len(toks))
            for t in set(toks):
                self.df[t] += 1
        self.N = max(1, len(self.doc_tokens))
        self.avgdl = sum(self.doc_len.values()) / self.N
        for t, df in self.df.items():
            self.idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        return self

    def _score(self, q_counter: Counter, tid: str) -> float:
        dt = self.doc_tokens.get(tid)
        if not dt:
            return 0.0
        dl = self.doc_len[tid]
        s = 0.0
        for t, _ in q_counter.items():
            f = dt.get(t, 0)
            if not f:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top(self, query: str, top_n: int = 20):
        qc = Counter(legal_tokens(query))
        scored = [(tid, self._score(qc, tid)) for tid in self.doc_tokens]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]


def learn_legal_projection(q_embs, a_embs, ridge: float = 1.0) -> np.ndarray:
    """岭回归闭式解学习线性投影 W，使 W·q ≈ a（法律表征适配 / 轻量微调）。
    q_embs, a_embs: (N, d) 已归一化嵌入。返回 (d, d) 的 float32 矩阵。"""
    Q = np.asarray(q_embs, dtype=np.float64)
    A = np.asarray(a_embs, dtype=np.float64)
    d = Q.shape[1]
    # 最小化 ||Q Wt - A||^2 + ridge||Wt||^2  =>  Wt = (Q^T Q + ridge I)^-1 Q^T A
    G = Q.T @ Q + ridge * np.eye(d)
    Wt = np.linalg.solve(G, Q.T @ A)   # (d,d): 行向量 q @ Wt ≈ a
    return Wt.T.astype(np.float32)       # 列约定：W @ q ≈ a


def make_legal_augment(mgr, lexical_index: LegalLexicalIndex, weight: float = 0.4, top_n: int = 20):
    """构造检索增强闭包（挂到 mgr._retrieval_augment）：
    把法律稀疏检索的高分候选并入稠密候选(dense ∪ lexical)，并按 (1-w)·dense + w·lexical 融合打分。"""
    def augment(query, qv, all_results):
        existing = {r.get("tid") for r in all_results}
        lex_top = dict(lexical_index.top(query, top_n))
        maxlex = max(lex_top.values()) if lex_top else 1.0
        qv_n = np.asarray(qv, dtype="float32")
        # 并入法律稀疏检索召回的新候选
        for tid, ls in lex_top.items():
            if tid in existing:
                continue
            node = mgr.para_tree.get(tid)
            if node is None or getattr(node, "para_vector", None) is None:
                continue
            dense = float(np.dot(qv_n, node.para_vector))
            existing.add(tid)
            all_results.append({
                "tid": tid, "text": node.text, "full_text": node.text, "type": "paragraph",
                "score": dense, "timestamp": getattr(node, "timestamp", ""),
                "memory_id": tid, "_lex_norm": (ls / maxlex if maxlex > 0 else 0.0),
            })
        # 融合打分：dense 与归一化 lexical 加权
        for r in all_results:
            ls = r.get("_lex_norm")
            if ls is None:
                raw = lex_top.get(r.get("tid"), 0.0)
                ls = raw / maxlex if maxlex > 0 else 0.0
            r["score"] = (1 - weight) * float(r.get("score", 0) or 0) + weight * ls
        return all_results
    return augment
