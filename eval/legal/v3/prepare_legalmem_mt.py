#!/usr/bin/env python3
"""LegalMem-MT corpus builder (protocol V3).

Gold needles = authentic CAIL2024 multi-turn dialogues ONLY.
Distractors = DISC-Law / Lawyer-LLaMA pairs used solely as same-domain noise
(never as primary evaluation gold).

Writes under data/legal/legalmem_mt/:
  gold_dialogs.json
  distractors_pairs.json
  corpus_manifest_{S,M,L}.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "legal" / "legalmem_mt"

PRELIM_TRAIN = REPO / "data/cail法律咨询初赛数据集/法律咨询-初赛-数据集/初赛数据/train.json"
PRELIM_TEST = REPO / "data/cail法律咨询初赛数据集/法律咨询-初赛-数据集/初赛数据/test.json"
FINAL_TEST = REPO / "data/cail法律咨询复赛数据集/法律咨询对话生成-复赛阶段-数据集/test.json"

# Target total sessions (gold + distractors) per scale tier.
SCALE_TARGETS = {"S": 0, "M": 3000, "L": 10000}  # S = gold only


def _complete_turns(item: dict) -> List[Dict[str, str]]:
    turns = []
    for t in item.get("conversation") or []:
        role = (t.get("role") or "").strip()
        content = (t.get("content") or "").strip()
        if role and content:
            turns.append({"role": role, "content": content})
    lab = item.get("label")
    if isinstance(lab, dict):
        content = (lab.get("content") or "").strip()
        if content:
            turns.append({
                "role": (lab.get("role") or "assistant").strip(),
                "content": content,
            })
    return turns


def load_cail_gold() -> List[dict]:
    dialogs = []
    for tag, path in [
        ("prelim", PRELIM_TRAIN),
        ("prelim", PRELIM_TEST),
        ("final", FINAL_TEST),
    ]:
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, item in enumerate(data):
            turns = _complete_turns(item)
            users = [t["content"] for t in turns if t["role"] == "user"]
            assts = [t["content"] for t in turns if t["role"] == "assistant"]
            if len(users) < 1 or len(assts) < 1 or len(turns) < 3:
                continue
            dialogs.append({
                "id": f"cail_{tag}_{item.get('id', i)}",
                "source": f"cail_{tag}",
                "role": "gold",
                "turns": turns,
                "user_turns": users,
                "assistant_turns": assts,
                "n_turns": len(turns),
                "n_user": len(users),
            })
    return dialogs


def load_distractor_pairs(max_n: int, seed: int) -> List[Tuple[str, str]]:
    import sys
    sys.path.insert(0, str(REPO / "eval" / "legal"))
    from prepare_legal_datasets import load_pairs  # noqa: E402

    rng = random.Random(seed)
    pairs = []
    for key in ("disc_law", "lawyer_llama"):
        pairs.extend(load_pairs(key))
    rng.shuffle(pairs)
    # filter empties / ultra-short
    clean = [(q.strip(), a.strip()) for q, a in pairs if len(q.strip()) >= 8 and len(a.strip()) >= 20]
    return clean[:max_n]


def build_manifest(gold: List[dict], distractors: List[Tuple[str, str]], tier: str, seed: int) -> dict:
    """Assemble session list for a scale tier."""
    sessions = []
    for d in gold:
        sessions.append({
            "session_id": d["id"],
            "role": "gold",
            "source": d["source"],
            "turns": d["turns"],
            "user_turns": d["user_turns"],
            "assistant_turns": d["assistant_turns"],
        })

    target = SCALE_TARGETS[tier]
    if tier == "S":
        n_dist = 0
    else:
        n_dist = max(0, target - len(sessions))
    n_dist = min(n_dist, len(distractors))
    rng = random.Random(seed + hash(tier) % 997)
    chosen = distractors[: n_dist] if n_dist <= len(distractors) else distractors
    # already shuffled upstream; take prefix then re-shuffle for tier
    pool = distractors[:]
    rng.shuffle(pool)
    for i, (q, a) in enumerate(pool[:n_dist]):
        sid = f"dist_{tier}_{i:05d}"
        sessions.append({
            "session_id": sid,
            "role": "distractor",
            "source": "disc_or_lawyer",
            "turns": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "user_turns": [q],
            "assistant_turns": [a],
        })

    gold_ids = [s["session_id"] for s in sessions if s["role"] == "gold"]
    return {
        "tier": tier,
        "seed": seed,
        "n_sessions": len(sessions),
        "n_gold": len(gold_ids),
        "n_distractor": len(sessions) - len(gold_ids),
        "n_turns": sum(len(s["turns"]) for s in sessions),
        "gold_session_ids": gold_ids,
        "sessions": sessions,
        "protocol": "LegalMem-MT-v3",
        "notes": (
            "Gold = CAIL multi-turn only. Distractors = single-pair Q+A from "
            "DISC/Lawyer used solely as same-domain interference."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gold = load_cail_gold()
    (out / "gold_dialogs.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"gold dialogs: {len(gold)} "
          f"mean_turns={sum(d['n_turns'] for d in gold)/len(gold):.2f} "
          f"mean_user={sum(d['n_user'] for d in gold)/len(gold):.2f}")

    need = max(SCALE_TARGETS[t] for t in args.tiers) + 1000
    distractors = load_distractor_pairs(need, args.seed)
    (out / "distractors_pairs.json").write_text(
        json.dumps(distractors, ensure_ascii=False), encoding="utf-8",
    )
    print(f"distractor pairs cached: {len(distractors)}")

    for tier in args.tiers:
        man = build_manifest(gold, distractors, tier, args.seed)
        path = out / f"corpus_manifest_{tier}.json"
        path.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
        print(
            f"tier {tier}: sessions={man['n_sessions']} "
            f"gold={man['n_gold']} dist={man['n_distractor']} turns={man['n_turns']} → {path}"
        )


if __name__ == "__main__":
    main()
