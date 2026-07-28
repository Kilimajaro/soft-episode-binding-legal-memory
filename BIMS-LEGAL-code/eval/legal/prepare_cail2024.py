#!/usr/bin/env python3
"""Prepare CAIL2024 legal consultation dialogues for shared-corpus memory eval.

Converts conversation+label multi-turn JSON into session records compatible with
run_revision_protocol / FlatIP indexing (one session_id per dialogue; turns
indexed separately with evidence = assistant turns).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

PRELIM_TRAIN = REPO_ROOT / "data/cail法律咨询初赛数据集/法律咨询-初赛-数据集/初赛数据/train.json"
PRELIM_TEST = REPO_ROOT / "data/cail法律咨询初赛数据集/法律咨询-初赛-数据集/初赛数据/test.json"
FINAL_TEST = REPO_ROOT / "data/cail法律咨询复赛数据集/法律咨询对话生成-复赛阶段-数据集/test.json"

OUT_DIR = REPO_ROOT / "data" / "legal" / "cail2024"


def _complete_turns(item: dict) -> List[Tuple[str, str]]:
    """Return ordered (role, text) turns including label as final assistant turn."""
    turns = []
    for t in item.get("conversation") or []:
        role = (t.get("role") or "").strip()
        content = (t.get("content") or "").strip()
        if role and content:
            turns.append((role, content))
    lab = item.get("label")
    if isinstance(lab, dict):
        role = (lab.get("role") or "assistant").strip()
        content = (lab.get("content") or "").strip()
        if content:
            turns.append((role, content))
    return turns


def load_cail_dialogs(paths: List[Path], tag: str) -> List[dict]:
    dialogs = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for i, item in enumerate(data):
            turns = _complete_turns(item)
            if len(turns) < 2:
                continue
            users = [c for r, c in turns if r == "user"]
            assts = [c for r, c in turns if r == "assistant"]
            if not users or not assts:
                continue
            dialogs.append({
                "id": f"{tag}_{item.get('id', i)}",
                "turns": turns,
                "users": users,
                "assistants": assts,
                "question": users[0],
                "answer": assts[-1],
                "query_user": users[-1],  # natural follow-up / last user turn
            })
    return dialogs


def save_prepared(dialogs: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Compact form used by eval runner
    payload = []
    for d in dialogs:
        payload.append({
            "id": d["id"],
            "turns": [{"role": r, "content": c} for r, c in d["turns"]],
            "question": d["question"],
            "answer": d["answer"],
            "query_user": d["query_user"],
            "n_user": len(d["users"]),
            "n_assistant": len(d["assistants"]),
        })
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload)} dialogs → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)

    prelim = load_cail_dialogs([PRELIM_TRAIN, PRELIM_TEST], "prelim")
    final = load_cail_dialogs([FINAL_TEST], "final")
    save_prepared(prelim, out / "cail_prelim.json")
    save_prepared(final, out / "cail_final.json")

    # Also emit simple (q,a) pairs for legacy loaders: first user + last assistant
    for name, dialogs in [("cail_prelim", prelim), ("cail_final", final)]:
        pairs = [(d["question"], d["answer"]) for d in dialogs]
        (out / f"{name}_pairs.json").write_text(
            json.dumps(pairs, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{name}: n={len(dialogs)} mean_turns="
              f"{sum(len(d['turns']) for d in dialogs)/max(len(dialogs),1):.2f}")


if __name__ == "__main__":
    main()
