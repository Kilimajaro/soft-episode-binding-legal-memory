"""加固版法律评测：查询改写 + 大语料，削弱 exact-match 捷径。"""
from __future__ import annotations

import random
import re
from typing import List, Tuple


_PARAPHRASE_PREFIX = (
    "请问", "想咨询一下", "麻烦解答", "关于这个问题：", "我遇到的情况是：",
)
_PARAPHRASE_SUFFIX = (
    "，应该怎么处理？", "，法律上如何认定？", "，请帮忙分析。", "，有什么建议？",
)

# 常见法律口语替换：降低与原文的连续字符重叠（投稿前修订：paraphrase 主协议）
_REPLACEMENTS = (
    (r"怎么办", "应如何处理"),
    (r"怎么处理", "法律上有何救济"),
    (r"可以吗", "是否合法"),
    (r"能不能", "是否有权"),
    (r"是否构成", "是否成立"),
    (r"我该", "当事人应当"),
    (r"^我", "当事人"),
    (r"公司", "用人单位"),
    (r"老板", "用人单位负责人"),
    (r"工资", "劳动报酬"),
    (r"合同", "协议"),
    (r"离婚", "婚姻解除"),
    (r"借款", "借贷"),
    (r"交通事故", "道路交通事故"),
    (r"赔偿", "损害赔偿"),
)


def _overlap_ratio(a: str, b: str) -> float:
    """最长公共子串长度 / max(len)。"""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 1.0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best / max(n, m)


def paraphrase_query(q: str, rng: random.Random) -> str:
    """强规则改写：禁止整句复用；控制与原文最长公共子串比例。"""
    orig = q.strip()
    if len(orig) < 8:
        return orig

    def _apply_replacements(text: str) -> str:
        out = text
        # 随机打乱替换顺序，避免固定模式
        reps = list(_REPLACEMENTS)
        rng.shuffle(reps)
        for pat, rep in reps:
            out2 = re.sub(pat, rep, out, count=1)
            if out2 != out:
                out = out2
        return out

    candidates = []
    # A: 关键词骨架 + 模板（打乱语序，去掉首尾大段原文）
    core = orig
    if len(core) > 36:
        mid = core[len(core) // 4 : len(core) * 3 // 4]
    else:
        mid = core[2:-2] if len(core) > 8 else core
    mid = _apply_replacements(mid)
    templates = [
        "请从法律角度说明：{c}，对应的权利义务与处理路径是什么？",
        "若事实大致为「{c}」，律师通常会给出何种结论与依据？",
        "就下列情形寻求法律意见：{c}。请给出可操作建议。",
        "把问题改述为咨询请求：关于{c}，应当如何依法应对？",
    ]
    candidates.append(rng.choice(templates).format(c=mid))

    # B: 替换 + 前后缀，但删除原文中的连续 8+ 字片段（抽样删）
    t = _apply_replacements(orig)
    if len(t) > 24:
        cut = rng.randint(6, min(14, len(t) // 3))
        t = t[cut:] if rng.random() < 0.5 else t[:-cut]
    candidates.append(rng.choice(_PARAPHRASE_PREFIX) + t + rng.choice(_PARAPHRASE_SUFFIX))

    # C: 角色/指代改写 + 问句重组
    t = _apply_replacements(re.sub(r"^我", "当事人", orig, count=1))
    candidates.append(f"针对「{_apply_replacements(t[: max(12, len(t)//2)])}」等情节，法律结论是什么？")

    # 选择与原文重叠最低的候选；若仍过高则强制模板骨架
    best = min(candidates, key=lambda s: _overlap_ratio(s, orig))
    if _overlap_ratio(best, orig) > 0.55 or best.strip() == orig:
        skeleton = mid[: max(10, min(28, len(mid)))]
        best = f"请分析与「{skeleton}」相关的法律责任与可行救济途径。"
    return best


def harden_pairs(
    pairs: List[Tuple[str, str]],
    *,
    seed: int = 42,
    frac: float = 1.0,
) -> List[Tuple[str, str]]:
    """对查询做改写，答案不变。"""
    rng = random.Random(seed)
    out = []
    for q, a in pairs:
        if rng.random() <= frac:
            out.append((paraphrase_query(q, rng), a))
        else:
            out.append((q, a))
    return out
