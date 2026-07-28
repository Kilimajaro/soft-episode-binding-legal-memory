"""LongMemEval 风格查询抽象：问题不得与 haystack 原文高度重叠。"""
from __future__ import annotations

import random
import re
from typing import List, Tuple


def _topic_snippet(q: str, max_len: int = 18) -> str:
    q = re.sub(r"[？?！!。，,\s]+", "", q.strip())
    if len(q) <= max_len:
        return q
    return q[: max_len - 1] + "…"


def abstract_user_query(original_q: str, rng: random.Random) -> str:
    """single-session-user：问「之前咨询过的问题」而非原文。"""
    topic = _topic_snippet(original_q)
    templates = [
        "在之前的法律咨询中，关于「{t}」这一事项，当时咨询的核心问题是什么？",
        "回顾历史对话里与{t}相关的那次咨询，用户当时主要想确认什么？",
        "我在较早的对话中问过{t}方面的问题，请概括那次提问的要点。",
        "关于{t}，在过往咨询记录里用户最初是怎么描述自己的情况的？",
    ]
    return rng.choice(templates).format(t=topic)


def abstract_assistant_query(original_q: str, answer: str, rng: random.Random) -> str:
    """single-session-assistant：问律师解答中的结论/依据，而非复述原问。"""
    topic = _topic_snippet(original_q)
    # 从答案抽一句法条/结论作锚点（若有）
    anchor = ""
    for pat in (r"《[^》]{2,30}》[^。]{0,40}", r"构成[^。]{2,20}罪", r"应当[^。]{4,30}", r"可以[^。]{4,30}"):
        m = re.search(pat, answer)
        if m:
            anchor = m.group(0)[:35]
            break
    if anchor:
        templates = [
            "在关于{t}的咨询中，律师曾给出怎样的法律结论或处理建议？",
            "回顾那次{t}相关的解答，律师对关键法律后果是怎么说的？",
            "历史记录里针对{t}问题，律师意见中提到的「{a}」属于什么性质的建议？",
        ]
        tpl = rng.choice(templates)
        return tpl.format(t=topic, a=anchor)
    return rng.choice([
        f"在关于{topic}的咨询中，律师给出的主要法律意见是什么？",
        f"回顾与{topic}相关的那次解答，律师建议当事人采取什么措施？",
        f"历史对话里针对{topic}问题，律师是如何界定各方权利义务的？",
    ])


def abstract_multi_session_query(topics: List[str], rng: random.Random) -> str:
    """multi-session：跨两次咨询的综合回忆题。"""
    t1, t2 = topics[0], topics[1]
    return rng.choice([
        f"我在不同时间分别咨询过「{t1}」和「{t2}」两件事，这两次咨询分别讨论的核心问题是什么？",
        f"历史记录中有关于{t1}与{t2}的两次独立咨询，请分别说明每次咨询的主题。",
        f"回顾与{t1}、{t2}相关的两次法律咨询，用户各自关注的主要争议点是什么？",
    ])


def lexical_overlap_ratio(query: str, haystack_texts: List[str]) -> float:
    """查询与 haystack 最长公共子串占比（越大越「作弊」）。"""
    q = re.sub(r"\s+", "", query)
    if len(q) < 6:
        return 0.0
    best = 0
    for text in haystack_texts:
        t = re.sub(r"\s+", "", text)
        for ln in range(min(len(q), 40), 5, -1):
            for i in range(len(q) - ln + 1):
                sub = q[i : i + ln]
                if sub in t:
                    best = max(best, ln / len(q))
                    break
            if best > 0.5:
                return best
    return best


def validate_abstract_query(query: str, haystack_texts: List[str], max_overlap: float = 0.35) -> bool:
    return lexical_overlap_ratio(query, haystack_texts) <= max_overlap
