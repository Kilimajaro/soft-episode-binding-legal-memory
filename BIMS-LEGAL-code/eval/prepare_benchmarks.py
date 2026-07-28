#!/usr/bin/env python3
"""下载并准备 LongMemEval / LoCoMo 基准数据（LongMemEval 兼容 JSON 格式）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LONGMEMEVAL_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCAL_FALLBACKS = {
    "longmemeval_oracle": os.environ.get(
        "LONGMEMEVAL_ORACLE_PATH",
        os.path.join(REPO_ROOT, "..", "..", "baseline", "longmemeval_oracle.json"),
    ),
    "longmemeval_s": os.environ.get(
        "LONGMEMEVAL_S_PATH",
        os.path.join(REPO_ROOT, "..", "..", "baseline", "longmemeval_s_cleaned.json"),
    ),
    "locomo10": os.environ.get("LOCOMO10_PATH", ""),
}


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  skip (exists): {dest}")
        return
    print(f"  downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  download failed: {e}")
        raise


def prepare_longmemeval(out_dir: str, splits: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        fname = f"longmemeval_{split}.json"
        dest = os.path.join(out_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"  skip (exists): {dest}")
        elif split == "oracle" and os.path.isfile(LOCAL_FALLBACKS["longmemeval_oracle"]):
            import shutil
            shutil.copy2(LOCAL_FALLBACKS["longmemeval_oracle"], dest)
            print(f"  copied local fallback -> {dest}")
        elif split == "s" and os.path.isfile(LOCAL_FALLBACKS["longmemeval_s"]):
            import shutil
            shutil.copy2(LOCAL_FALLBACKS["longmemeval_s"], dest)
            print(f"  copied local fallback -> {dest}")
        else:
            url = LONGMEMEVAL_URLS[split]
            _download(url, dest)
        n = len(json.load(open(dest, encoding="utf-8")))
        print(f"  {fname}: {n} instances")


def _role_from_speaker(speaker: str, speaker_a: str) -> str:
    return "user" if speaker == speaker_a else "assistant"


def convert_locomo_to_longmemeval(locomo_path: str, out_path: str) -> int:
    """将 LoCoMo locomo10.json 转为 LongMemEval 兼容实例列表。"""
    with open(locomo_path, encoding="utf-8") as f:
        samples = json.load(f)

    instances = []
    for sample in samples:
        sid = sample.get("sample_id", "unknown")
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "Speaker A")
        speaker_b = conv.get("speaker_b", "Speaker B")

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")
             and not k.endswith("_summary")],
            key=lambda x: int(x.split("_")[1]),
        )
        haystack_sessions = []
        haystack_session_ids = []
        dia_to_haystack_sid: dict[str, str] = {}

        for sk in session_keys:
            turns = conv.get(sk, [])
            if not isinstance(turns, list):
                continue
            hsid = f"{sid}_{sk}"
            haystack_session_ids.append(hsid)
            session_turns = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                dia_id = turn.get("dia_id", "")
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                role = _role_from_speaker(turn.get("speaker", speaker_a), speaker_a)
                session_turns.append({"role": role, "content": text})
                if dia_id:
                    dia_to_haystack_sid[dia_id] = hsid
            if session_turns:
                haystack_sessions.append(session_turns)

        for qi, qa in enumerate(sample.get("qa", [])):
            question = (qa.get("question") or "").strip()
            answer = qa.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            answer = str(answer).strip()
            if not question or not answer:
                continue
            evidence = qa.get("evidence") or []
            answer_session_ids = list({
                dia_to_haystack_sid[d]
                for d in evidence
                if d in dia_to_haystack_sid
            })
            instances.append({
                "question_id": f"locomo_{sid}_q{qi}",
                "question_type": str(qa.get("category", "locomo")),
                "question": question,
                "answer": answer,
                "answer_session_ids": answer_session_ids,
                "haystack_sessions": haystack_sessions,
                "haystack_session_ids": haystack_session_ids,
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False, indent=2)
    return len(instances)


def prepare_locomo(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "locomo10.json")
    out = os.path.join(out_dir, "longmemeval_oracle.json")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        n = len(json.load(open(out, encoding="utf-8")))
        print(f"  skip (exists): {out} ({n} instances)")
        return
    if not os.path.isfile(raw) or os.path.getsize(raw) == 0:
        locomo_fb = LOCAL_FALLBACKS.get("locomo10") or ""
        if locomo_fb and os.path.isfile(locomo_fb):
            import shutil
            shutil.copy2(locomo_fb, raw)
            print(f"  copied local locomo -> {raw}")
        else:
            _download(LOCOMO_URL, raw)
    n = convert_locomo_to_longmemeval(raw, out)
    print(f"  longmemeval_oracle.json (from LoCoMo): {n} instances")


def main():
    ap = argparse.ArgumentParser(description="准备 LongMemEval / LoCoMo 基准")
    ap.add_argument("--longmem_splits", nargs="+", default=["oracle", "s"])
    ap.add_argument("--longmem_dir", default=os.path.join(REPO_ROOT, "data", "longmemeval"))
    ap.add_argument("--locomo_dir", default=os.path.join(REPO_ROOT, "data", "locomo"))
    ap.add_argument("--skip_longmem", action="store_true")
    ap.add_argument("--skip_locomo", action="store_true")
    args = ap.parse_args()

    if not args.skip_longmem:
        print("=== LongMemEval ===")
        prepare_longmemeval(args.longmem_dir, args.longmem_splits)
    if not args.skip_locomo:
        print("=== LoCoMo ===")
        prepare_locomo(args.locomo_dir)
    print("done")


if __name__ == "__main__":
    main()
