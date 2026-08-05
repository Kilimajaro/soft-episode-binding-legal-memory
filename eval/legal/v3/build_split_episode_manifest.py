#!/usr/bin/env python3
"""Build Split-Episode / Mix manifests for Soft O2-C evaluation.

Protocols
---------
qa_split / mid_split (legacy CSCE):
  Every gold dialogue is split into Sa (query) / Sb (evidence) with distinct
  session_ids.

mix (fair same-store protocol):
  A fraction ``split_ratio`` of gold is split (cross_session); the rest stay
  intact (same_session). All methods share unrestricted dense retrieval on the
  same store and query list. Queries carry ``evidence_scope``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _turns_by_role(session: dict) -> Tuple[List[dict], List[dict]]:
    users, assts = [], []
    for t in session["turns"]:
        role = t.get("role")
        item = {"role": role, "content": t["content"]}
        if role == "user":
            users.append(item)
        elif role == "assistant":
            assts.append(item)
    if not users and session.get("user_turns"):
        users = [{"role": "user", "content": c} for c in session["user_turns"]]
    if not assts and session.get("assistant_turns"):
        assts = [{"role": "assistant", "content": c} for c in session["assistant_turns"]]
    return users, assts


def _pack_session(sid: str, role: str, source: str, turns: List[dict], orig_id: str, half: str) -> dict:
    users = [t["content"] for t in turns if t["role"] == "user"]
    assts = [t["content"] for t in turns if t["role"] == "assistant"]
    return {
        "session_id": sid,
        "role": role,
        "source": source,
        "turns": turns,
        "user_turns": users,
        "assistant_turns": assts,
        "split_pair_id": orig_id,
        "split_half": half,
        "orig_session_id": orig_id,
    }


def split_gold_qa(session: dict) -> Tuple[dict, dict]:
    """qa_split: Sa = user turns, Sb = assistant turns."""
    users, assts = _turns_by_role(session)
    if not users or not assts:
        raise ValueError(
            f"cannot qa_split session {session.get('session_id')}: "
            f"users={len(users)} assts={len(assts)}"
        )
    oid = str(session["session_id"])
    sa = _pack_session(f"{oid}__Sa", "gold", session.get("source", "split"), users, oid, "Sa")
    sb = _pack_session(f"{oid}__Sb", "gold_evidence", session.get("source", "split"), assts, oid, "Sb")
    return sa, sb


def split_gold_mid(session: dict) -> Tuple[dict, dict]:
    """mid_split: first half / second half (requires >= 4 turns)."""
    turns = list(session["turns"])
    if len(turns) < 4:
        return split_gold_qa(session)
    mid = len(turns) // 2
    for i in range(mid - 1, 0, -1):
        if turns[i]["role"] == "user":
            mid = i
            break
    left, right = turns[:mid], turns[mid:]
    if not left or not right:
        return split_gold_qa(session)
    if not any(t["role"] == "assistant" for t in right):
        return split_gold_qa(session)
    oid = str(session["session_id"])
    sa = _pack_session(f"{oid}__Sa", "gold", session.get("source", "split"), left, oid, "Sa")
    sb = _pack_session(f"{oid}__Sb", "gold_evidence", session.get("source", "split"), right, oid, "Sb")
    return sa, sb


def _channel_pack(
    *,
    session_id: str,
    gold_session_id: str,
    split_pair_id: str | None,
    channel: str,
    qtext: str,
    user_idx: int,
    n_gold_answers: int,
    evidence_scope: str,
    protocol: str,
) -> dict:
    return {
        "session_id": session_id,
        "gold_session_id": gold_session_id,
        "split_pair_id": split_pair_id,
        "channel": channel,
        "query": qtext,
        "user_idx": user_idx,
        "evidence_mode": "gold_session_all_assistant",
        "n_gold_answers": n_gold_answers,
        "evidence_scope": evidence_scope,
        "protocol": protocol,
    }


def _load_source_queries(src_manifest: Path) -> Dict[str, Dict[str, dict]]:
    """Map session_id -> channel -> query row from sibling queries.json if present."""
    qpath = src_manifest.parent / "queries.json"
    if not qpath.exists():
        return {}
    rows = json.loads(qpath.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        sid = str(r.get("session_id") or "")
        ch = r.get("channel")
        if not sid or not ch:
            continue
        out.setdefault(sid, {})[ch] = r
    return out


def _query_text_for(
    session: dict,
    channel: str,
    users: List[str],
    rng: random.Random,
    source_queries: Dict[str, Dict[str, dict]],
) -> Tuple[str, int]:
    """Prefer prebuilt LegalEp/LegalMem channel text; fallback to user turns."""
    oid = str(session.get("orig_session_id") or session.get("session_id") or "")
    # Strip Sa/Sb suffix if present
    if oid.endswith("__Sa") or oid.endswith("__Sb"):
        oid = oid.rsplit("__", 1)[0]
    src = (source_queries.get(oid) or {}).get(channel)
    if src and src.get("query"):
        return str(src["query"]), int(src.get("user_idx") or 0)
    if channel == "advice_recall" and users:
        core = users[0][:28]
        templates = [
            "上次咨询里，律师对「{core}」最终是怎么建议的？",
            "关于此前那个问题（涉及：{core}），结论与依据是什么？",
            "请根据档案中该次咨询，说明律师给出的处理意见。",
        ]
        return rng.choice(templates).format(core=core), 0
    if channel in ("u_last",) and users:
        return users[-1], len(users) - 1
    if channel in ("uk_followup",) and len(users) >= 2:
        idx = rng.randint(1, len(users) - 1)
        return users[idx], idx
    return users[0], 0


def build_queries_from_users(
    *,
    session_id: str,
    gold_session_id: str,
    users: List[str],
    n_gold_answers: int,
    channels: List[str],
    rng: random.Random,
    evidence_scope: str,
    protocol: str,
    split_pair_id: str | None = None,
    source_session: dict | None = None,
    source_queries: Dict[str, Dict[str, dict]] | None = None,
) -> List[dict]:
    if not users:
        return []
    out = []
    source_queries = source_queries or {}
    src_sess = source_session or {"session_id": session_id, "orig_session_id": session_id}

    for channel in channels:
        # Lookup key in source queries.json
        lookup = {
            "u1_exact": "exact",
            "exact": "exact",
            "advice_recall": "advice_recall",
            "advice-recall": "advice_recall",
            "u_para": "u_para",
            "uk_followup": "uk_followup",
            "u_last": "u_last",
        }.get(channel, channel)
        qtext, user_idx = _query_text_for(src_sess, lookup, users, rng, source_queries)
        out.append(
            _channel_pack(
                session_id=session_id,
                gold_session_id=gold_session_id,
                split_pair_id=split_pair_id,
                channel=channel,
                qtext=qtext,
                user_idx=user_idx,
                n_gold_answers=n_gold_answers,
                evidence_scope=evidence_scope,
                protocol=protocol,
            )
        )
    return out


def build_csce_queries(
    sa: dict,
    sb: dict,
    channels: List[str],
    rng: random.Random,
    *,
    source_session: dict | None = None,
    source_queries: Dict[str, Dict[str, dict]] | None = None,
    protocol: str = "split_episode_csce",
) -> List[dict]:
    """Cross-session queries: Sa query session, Sb gold."""
    return build_queries_from_users(
        session_id=sa["session_id"],
        gold_session_id=sb["session_id"],
        users=sa["user_turns"],
        n_gold_answers=len(sb["assistant_turns"]),
        channels=channels,
        rng=rng,
        evidence_scope="cross_session",
        protocol=protocol,
        split_pair_id=sa["split_pair_id"],
        source_session=source_session,
        source_queries=source_queries,
    )


def build_same_session_queries(
    session: dict,
    channels: List[str],
    rng: random.Random,
    *,
    source_queries: Dict[str, Dict[str, dict]] | None = None,
    protocol: str = "csce_mix",
) -> List[dict]:
    """Intact gold: query and answers share session_id."""
    users, assts = _turns_by_role(session)
    if not users:
        return []
    sid = str(session["session_id"])
    return build_queries_from_users(
        session_id=sid,
        gold_session_id=sid,
        users=[t["content"] if isinstance(t, dict) else t for t in users],
        n_gold_answers=len(assts),
        channels=channels,
        rng=rng,
        evidence_scope="same_session",
        protocol=protocol,
        split_pair_id=None,
        source_session=session,
        source_queries=source_queries,
    )


def _dedupe_gold(gold: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for s in gold:
        sid = str(s.get("session_id"))
        if sid in seen:
            continue
        seen.add(sid)
        out.append(s)
    return out


def convert_manifest(
    man: dict,
    *,
    mode: str,
    channels: List[str],
    seed: int,
    max_gold: int = 0,
    split_ratio: float = 0.5,
    split_style: str = "qa_split",
    source_queries: Dict[str, Dict[str, dict]] | None = None,
) -> Tuple[dict, List[dict], dict]:
    rng = random.Random(seed)
    source_queries = source_queries or {}
    gold = _dedupe_gold([s for s in man["sessions"] if s.get("role") == "gold"])
    distractors = [s for s in man["sessions"] if s.get("role") != "gold"]
    if max_gold > 0:
        gold = gold[:max_gold]

    split_fn = split_gold_qa if split_style == "qa_split" else split_gold_mid
    new_sessions: List[dict] = []
    queries: List[dict] = []
    pair_map: Dict[str, dict] = {}
    skipped = 0
    n_same = 0
    n_cross = 0

    if mode == "mix":
        protocol = "csce_mix"
        order = list(gold)
        rng.shuffle(order)
        n_split = int(round(len(order) * float(split_ratio)))
        to_split = set(id(s) for s in order[:n_split])

        for s in gold:
            if id(s) in to_split:
                try:
                    sa, sb = split_fn(s)
                except ValueError:
                    skipped += 1
                    ss = dict(s)
                    ss.setdefault("split_pair_id", None)
                    ss.setdefault("split_half", None)
                    ss.setdefault("orig_session_id", s["session_id"])
                    new_sessions.append(ss)
                    queries.extend(
                        build_same_session_queries(
                            ss, channels, rng, source_queries=source_queries, protocol=protocol
                        )
                    )
                    n_same += 1
                    continue
                new_sessions.append(sa)
                new_sessions.append(sb)
                pair_map[sa["split_pair_id"]] = {
                    "sa": sa["session_id"],
                    "sb": sb["session_id"],
                    "n_sa_turns": len(sa["turns"]),
                    "n_sb_turns": len(sb["turns"]),
                    "evidence_scope": "cross_session",
                }
                queries.extend(
                    build_csce_queries(
                        sa, sb, channels, rng,
                        source_session=s,
                        source_queries=source_queries,
                        protocol=protocol,
                    )
                )
                n_cross += 1
            else:
                ss = dict(s)
                ss.setdefault("split_pair_id", None)
                ss.setdefault("split_half", None)
                ss.setdefault("orig_session_id", s["session_id"])
                new_sessions.append(ss)
                queries.extend(
                    build_same_session_queries(
                        ss, channels, rng, source_queries=source_queries, protocol=protocol
                    )
                )
                n_same += 1
    else:
        protocol = "split_episode_csce"
        split_fn = split_gold_qa if mode == "qa_split" else split_gold_mid
        for s in gold:
            try:
                sa, sb = split_fn(s)
            except ValueError:
                skipped += 1
                continue
            new_sessions.append(sa)
            new_sessions.append(sb)
            pair_map[sa["split_pair_id"]] = {
                "sa": sa["session_id"],
                "sb": sb["session_id"],
                "n_sa_turns": len(sa["turns"]),
                "n_sb_turns": len(sb["turns"]),
                "evidence_scope": "cross_session",
            }
            queries.extend(
                build_csce_queries(
                    sa, sb, channels, rng,
                    source_session=s,
                    source_queries=source_queries,
                    protocol=protocol,
                )
            )
            n_cross += 1

    for s in distractors:
        ss = dict(s)
        ss.setdefault("split_pair_id", None)
        ss.setdefault("split_half", None)
        ss.setdefault("orig_session_id", s["session_id"])
        new_sessions.append(ss)

    notes = (
        "CSCE Mix: fraction of gold split into Sa/Sb (cross_session); remainder "
        "intact (same_session). Fair eval uses unrestricted dense for all operators; "
        "glue_split_pairs at eval makes Soft O2-C solvable on cross_session pairs."
        if mode == "mix"
        else (
            "CSCE Split-Episode: all gold dialogues split into Sa/Sb. "
            "Use unrestricted dense for fair Soft O2 vs Soft O2-C; optional seed_* "
            "configs are diagnostic only."
        )
    )

    out_man = {
        "tier": man.get("tier", "M"),
        "seed": seed,
        "protocol": protocol,
        "split_mode": mode,
        "split_style": split_style if mode == "mix" else mode,
        "split_ratio": float(split_ratio) if mode == "mix" else 1.0,
        "n_sessions": len(new_sessions),
        "n_gold": sum(1 for s in new_sessions if s.get("role") == "gold"),
        "n_gold_evidence": sum(1 for s in new_sessions if s.get("role") == "gold_evidence"),
        "n_distractor": sum(1 for s in new_sessions if s.get("role") not in ("gold", "gold_evidence")),
        "n_turns": sum(len(s["turns"]) for s in new_sessions),
        "n_split_pairs": len(pair_map),
        "n_same_session_gold": n_same,
        "n_cross_session_gold": n_cross,
        "n_skipped_gold": skipped,
        "gold_session_ids": [s["session_id"] for s in new_sessions if s.get("role") == "gold"],
        "split_pairs": pair_map,
        "channels": channels,
        "notes": notes,
        "sessions": new_sessions,
        "source_manifest_keys": {k: man.get(k) for k in ("protocol", "notes", "tier") if k in man},
    }
    return out_man, queries, pair_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Source corpus_manifest_*.json")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSCE artifacts")
    ap.add_argument(
        "--mode",
        choices=["qa_split", "mid_split", "mix"],
        default="mix",
        help="mix = fair partial split; qa_split/mid_split = full split",
    )
    ap.add_argument(
        "--split_style",
        choices=["qa_split", "mid_split"],
        default="qa_split",
        help="How to split gold when mode=mix (or ignored for full-split modes)",
    )
    ap.add_argument(
        "--split_ratio",
        type=float,
        default=0.5,
        help="Fraction of gold to split under mode=mix",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=["advice_recall"],
        help="Query channels to emit (prefer advice_recall for LegalEp Mix)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_gold", type=int, default=0, help="0=all gold")
    args = ap.parse_args()

    src = Path(args.manifest)
    man = json.loads(src.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_queries = _load_source_queries(src)

    out_man, queries, pairs = convert_manifest(
        man,
        mode=args.mode,
        channels=args.channels,
        seed=args.seed,
        max_gold=args.max_gold,
        split_ratio=args.split_ratio,
        split_style=args.split_style,
        source_queries=source_queries,
    )
    man_path = out_dir / "corpus_manifest_M.json"
    q_path = out_dir / "queries.json"
    pair_path = out_dir / "split_pairs.json"
    man_path.write_text(json.dumps(out_man, ensure_ascii=False, indent=2), encoding="utf-8")
    q_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    pair_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[csce] wrote {man_path} sessions={out_man['n_sessions']} "
        f"pairs={out_man['n_split_pairs']} same={out_man.get('n_same_session_gold')} "
        f"cross={out_man.get('n_cross_session_gold')} queries={len(queries)} "
        f"skipped={out_man['n_skipped_gold']} src_q_sids={len(source_queries)}",
        flush=True,
    )
    print(f"[csce] queries -> {q_path}", flush=True)


if __name__ == "__main__":
    main()
