#!/usr/bin/env python3
"""LegalMem-MT Soft O2-C / hybrid ablation.

Narrative-aligned operator stack:
  - ep_flat:     episodic FlatIP only
  - sess_o2:     Soft O2 on same-session siblings (existing Soft O2)
  - cluster_o2:  Soft O2-C on same-BIRCH-cluster siblings (new)
  - hybrid_o2:   sess_o2 then Soft O2-C (legal dual-binding)
  - hybrid_xsess: sess_o2 + gated Soft O2-C (cross-session only, direct-hit trigger, budget=2)
  - bridge_o2:     Module A — same-session cross-cluster bridge after sess_o2
  - sess_rerank_c: Module B — cluster-coherent rerank in wide pool (no injection)
  - sess_suppress_c: Module C — confuser suppression in hit cluster
  - cluster_route: Module D — top-M cluster routing boost on dense hits
  - sess_residual_c: Module E — cluster expand only when session incomplete
  - birch_flat:  BIRCH associative retrieval without either Soft O2
  - birch_c_o2:  BIRCH associative retrieval + Soft O2-C

Summary retrieval is disabled throughout so the slow pathway means BIRCH
clusters only. Soft O2-C inherits scores within a cluster the same way Soft O2
inherits within a session: soft_sc = max_hit_score × β_c.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval" / "legal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_manager import VectorMemoryManager  # noqa: E402
from stats_sig import bootstrap_ci, paired_report  # noqa: E402
from run_legalmem_mt import (  # noqa: E402
    index_turn_store,
    metrics_at_k,
    resolve_queries,
    uniquify_session_ids,
)

# sklearn/threadpoolctl emits noisy "Exception ignored" traces during BIRCH.
class _FilterThreadpoolCtlStderr:
    _DROP_SUBSTR = (
        "threadpoolctl",
        "_ThreadpoolInfo",
        "match_module_callback",
        "Exception ignored on calling ctypes callback",
        "AttributeError: 'NoneType' object has no attribute 'split'",
    )

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""
        self._drop_traceback = False

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            text = line + "\n"
            if any(x in text for x in self._DROP_SUBSTR):
                self._drop_traceback = True
                continue
            if self._drop_traceback:
                stripped = text.lstrip()
                if text.startswith("  File ") or stripped.startswith("self.") or stripped.startswith("module ") or stripped.startswith("config ") or stripped.startswith("^"):
                    continue
                if text.strip() == "Traceback (most recent call last):":
                    continue
                self._drop_traceback = False
            self._stream.write(text)

    def flush(self):
        if self._buf:
            self._stream.write(self._buf)
            self._buf = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stderr = _FilterThreadpoolCtlStderr(sys.stderr)

SMOKE_CONFIGS = [
    "sess_o2",
    "bridge_o2",
    "sess_rerank_c",
    "sess_suppress_c",
    "cluster_route",
    "sess_residual_c",
]

CONFIGS = [
    "ep_flat",
    "sess_o2",
    "cluster_o2",
    "hybrid_o2",
    "hybrid_xsess",
    "seed_flat",
    "seed_sess_o2",
    "seed_cluster_o2",
    *SMOKE_CONFIGS[1:],
    "birch_flat",
    "birch_c_o2",
]


def gold_qa_same_cluster_rate(mgr: VectorMemoryManager, meta: dict) -> dict:
    """Diagnostic: how often a session's user+assistant gold turns share a cluster."""
    if not getattr(mgr, "_tid_to_cluster", None):
        mgr._rebuild_cluster_membership()
    n = 0
    same = 0
    for sid, g in meta.items():
        tids = list(g.get("all_tids") or [])
        if len(tids) < 2:
            continue
        # LegalEp: [user, assistant]; CAIL: multiple turns — use first user + last assistant when possible
        u = tids[0]
        a = (g.get("ans_tids") or [tids[-1]])[-1]
        if u == a:
            continue
        n += 1
        cu = mgr._tid_to_cluster.get(u)
        ca = mgr._tid_to_cluster.get(a)
        if cu is not None and cu == ca:
            same += 1
    return {
        "n_pairs": n,
        "same_cluster": same,
        "rate": (same / n) if n else 0.0,
        "n_clusters": len(mgr._cluster_members or {}),
    }


def csce_sa_sb_co_cluster_rate(mgr: VectorMemoryManager, meta: dict, queries: List[dict]) -> dict:
    """Diagnostic: fraction of CSCE queries where some Sa tid shares a cluster with some Sb gold tid."""
    if not getattr(mgr, "_tid_to_cluster", None):
        mgr._rebuild_cluster_membership()
    n = 0
    same = 0
    seen = set()
    for q in queries:
        gsid = q.get("gold_session_id")
        sid = q.get("session_id")
        if not gsid or not sid:
            continue
        key = (sid, gsid)
        if key in seen:
            continue
        seen.add(key)
        sa = meta.get(sid) or {}
        sb = meta.get(gsid) or {}
        sa_tids = list(sa.get("all_tids") or [])
        sb_tids = list(sb.get("ans_tids") or sb.get("all_tids") or [])
        if not sa_tids or not sb_tids:
            continue
        n += 1
        sa_c = {mgr._tid_to_cluster.get(t) for t in sa_tids if t in mgr._tid_to_cluster}
        sb_c = {mgr._tid_to_cluster.get(t) for t in sb_tids if t in mgr._tid_to_cluster}
        sa_c.discard(None)
        sb_c.discard(None)
        if sa_c & sb_c:
            same += 1
    return {
        "n_pairs": n,
        "same_cluster": same,
        "rate": (same / n) if n else 0.0,
        "n_clusters": len(mgr._cluster_members or {}),
    }


def glue_split_pairs(mgr: VectorMemoryManager, sessions: List[dict]) -> dict:
    """Force each Split-Episode Sa∪Sb tid set into one BIRCH cluster.

    Soft O2 still cannot cross session_id; Soft O2-C becomes solvable for CSCE.
    Returns diagnostic counts.
    """
    if not getattr(mgr, "_tid_to_cluster", None):
        mgr._rebuild_cluster_membership()
    pairs: Dict[str, List[str]] = {}
    for s in sessions:
        pid = s.get("split_pair_id")
        half = s.get("split_half")
        if not pid or half not in ("Sa", "Sb"):
            continue
        pairs.setdefault(pid, [])
        # collect tids belonging to this session
        sid = s["session_id"]
        for tid, tsid in list(mgr._tid_to_session.items()):
            if tsid == sid:
                pairs[pid].append(tid)
    glued = 0
    for pid, tids in pairs.items():
        tids = list(dict.fromkeys(tids))
        if len(tids) < 2:
            continue
        # Prefer an existing cluster that already covers most tids; else new id
        votes: Dict[str, int] = {}
        for tid in tids:
            cid = mgr._tid_to_cluster.get(tid)
            if cid is not None:
                votes[cid] = votes.get(cid, 0) + 1
        if votes:
            target = max(votes.items(), key=lambda x: x[1])[0]
        else:
            target = f"split_glue_{pid}"
            mgr._cluster_members[target] = []
        # Remove tids from old clusters
        for tid in tids:
            old = mgr._tid_to_cluster.get(tid)
            if old is not None and old in mgr._cluster_members:
                mgr._cluster_members[old] = [x for x in mgr._cluster_members[old] if x != tid]
                if not mgr._cluster_members[old]:
                    mgr._cluster_members.pop(old, None)
                    mgr._cluster_centroids.pop(old, None)
            mgr._tid_to_cluster[tid] = target
            mgr._cluster_members.setdefault(target, [])
            if tid not in mgr._cluster_members[target]:
                mgr._cluster_members[target].append(tid)
        glued += 1
    # Persist so apply_cfg can restore after rebuilding from full_kg.
    mgr._glued_cluster_members = {k: list(v) for k, v in (mgr._cluster_members or {}).items()}
    mgr._glued_tid_to_cluster = dict(mgr._tid_to_cluster or {})
    return {"n_pairs_glued": glued, "n_clusters": len(mgr._cluster_members or {})}


def load_csce_queries(manifest_path: Path, channels: List[str]) -> List[dict]:
    """Load Split-Episode / Mix queries.json next to the manifest."""
    qpath = manifest_path.parent / "queries.json"
    if not qpath.exists():
        return []
    rows = json.loads(qpath.read_text(encoding="utf-8"))
    want = set(channels)
    ok_proto = {"split_episode_csce", "csce_mix"}
    out = []
    for r in rows:
        if r.get("protocol") not in ok_proto:
            continue
        if r.get("channel") not in want:
            continue
        out.append(r)
    return out


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def stratify_ah(queries: List[dict], hits: List[int]) -> dict:
    """Per evidence_scope Answer Hit means + counts."""
    buckets: Dict[str, List[int]] = {}
    for q, h in zip(queries, hits):
        scope = q.get("evidence_scope") or (
            "cross_session" if q.get("gold_session_id") and q.get("gold_session_id") != q.get("session_id") else "same_session"
        )
        buckets.setdefault(scope, []).append(int(h))
    return {
        scope: {"n": len(vals), "answer_hit@k": _mean(vals)}
        for scope, vals in sorted(buckets.items())
    }


def resolve_gold_meta(q: dict, meta: dict) -> tuple[dict, List[str]]:
    """Return (episode_meta_for_metrics, answer_tids) honoring CSCE gold_session_id."""
    gsid = q.get("gold_session_id")
    sid = q["session_id"]
    if gsid:
        if gsid not in meta:
            raise KeyError(
                f"CSCE gold_session_id={gsid!r} missing from meta "
                f"(query session_id={sid!r}); refusing Sa fallback"
            )
        gold = meta[gsid]
        ans = list(gold.get("ans_tids") or gold.get("all_tids") or [])
        return gold, ans
    gold = meta[sid]
    if q.get("evidence_mode") == "after_query" and len(gold["ans_tids"]) > q.get("user_idx", 0):
        ans = gold["ans_tids"][q["user_idx"] :]
    else:
        ans = gold["ans_tids"]
    return gold, ans


def apply_cfg(
    mgr: VectorMemoryManager,
    name: str,
    *,
    beta_sess: float,
    beta_cluster: float,
    full_kg: dict,
    cluster_max_siblings: int,
) -> dict:
    if hasattr(mgr, "clear_search_cache"):
        mgr.clear_search_cache()
    else:
        mgr.lru_cache.clear()

    mgr._retrieval_augment = None
    mgr._query_projection = None
    mgr._exact_match_boost = 0.0
    mgr.summary_nodes = {}
    mgr._session_expand = False
    mgr._session_first_rerank = False
    mgr._session_coherence = float(beta_sess)
    mgr._cluster_expand = False
    mgr._cluster_coherence = float(beta_cluster)
    mgr._cluster_max_siblings = int(cluster_max_siblings)
    mgr._cluster_cross_session_only = False
    mgr._cluster_trigger_on_soft = True
    mgr._cluster_budget = 0
    mgr._cluster_prefer_assistant = False
    mgr._cluster_pin_session_slots = False
    mgr._cluster_bridge_o2 = False
    mgr._cluster_rerank_only = False
    mgr._cluster_suppress_confusers = False
    mgr._cluster_route_top_m = 0
    mgr._cluster_residual_fallback = False
    mgr._cluster_centroids = {}

    use_birch_retrieval = name in ("birch_flat", "birch_c_o2")
    if use_birch_retrieval:
        mgr.knowledge_graph = full_kg
        mgr._ablation_no_assoc = False
        mgr._rebuild_cluster_membership()
    else:
        # Soft O2-C still needs membership maps, but search must not pull
        # centroid/assoc candidates unless birch_* configs are requested.
        mgr.knowledge_graph = {}
        mgr._ablation_no_assoc = True
        mgr._cluster_members = {}
        mgr._tid_to_cluster = {}
        mgr._cluster_centroids = {}
        for cid, node in full_kg.items():
            pids = list(getattr(node, "paragraph_ids", None) or [])
            if not pids:
                continue
            mgr._cluster_members[cid] = pids
            for pid in pids:
                mgr._tid_to_cluster[pid] = cid
            vec = getattr(node, "vector", None)
            if vec is not None:
                mgr._cluster_centroids[cid] = vec

    # CSCE glue maps must survive apply_cfg rebuild from full_kg.
    glued_m = getattr(mgr, "_glued_cluster_members", None)
    glued_t = getattr(mgr, "_glued_tid_to_cluster", None)
    if glued_m is not None and glued_t is not None:
        mgr._cluster_members = {k: list(v) for k, v in glued_m.items()}
        mgr._tid_to_cluster = dict(glued_t)

    sess_configs = (
        "sess_o2", "hybrid_o2", "hybrid_xsess",
        "bridge_o2", "sess_rerank_c", "sess_suppress_c", "cluster_route", "sess_residual_c",
        "seed_sess_o2",
    )
    if name in sess_configs:
        mgr._session_expand = True
        mgr._session_first_rerank = True
    if name in ("cluster_o2", "hybrid_o2", "hybrid_xsess", "birch_c_o2", "seed_cluster_o2"):
        mgr._cluster_expand = True
    if name == "cluster_o2":
        # Fair Soft O2-C: recover cross-session siblings; prefer assistants.
        mgr._cluster_cross_session_only = True
        mgr._cluster_trigger_on_soft = True
        mgr._cluster_budget = 12
        mgr._cluster_prefer_assistant = True
        mgr._cluster_max_siblings = max(int(mgr._cluster_max_siblings), 12)
    if name == "hybrid_xsess":
        mgr._cluster_cross_session_only = True
        mgr._cluster_trigger_on_soft = False
        mgr._cluster_budget = 8
        mgr._cluster_prefer_assistant = True
        mgr._cluster_pin_session_slots = False
        mgr._cluster_max_siblings = max(int(mgr._cluster_max_siblings), 10)
    if name == "bridge_o2":
        mgr._cluster_bridge_o2 = True
    elif name == "sess_rerank_c":
        mgr._cluster_rerank_only = True
    elif name == "sess_suppress_c":
        mgr._cluster_suppress_confusers = True
    elif name == "cluster_route":
        mgr._cluster_route_top_m = 5
    elif name == "sess_residual_c":
        mgr._cluster_residual_fallback = True

    seed_restricted = name in ("seed_flat", "seed_sess_o2", "seed_cluster_o2")
    return {
        "summary_enabled": False,
        "birch_retrieval": use_birch_retrieval,
        "session_expand": bool(mgr._session_expand),
        "cluster_expand": bool(mgr._cluster_expand),
        "beta_sess": float(mgr._session_coherence),
        "beta_cluster": float(mgr._cluster_coherence),
        "cluster_max_siblings": int(mgr._cluster_max_siblings),
        "cluster_cross_session_only": bool(mgr._cluster_cross_session_only),
        "cluster_trigger_on_soft": bool(mgr._cluster_trigger_on_soft),
        "cluster_budget": int(mgr._cluster_budget),
        "cluster_pin_session_slots": bool(mgr._cluster_pin_session_slots),
        "cluster_bridge_o2": bool(mgr._cluster_bridge_o2),
        "cluster_rerank_only": bool(mgr._cluster_rerank_only),
        "cluster_suppress_confusers": bool(mgr._cluster_suppress_confusers),
        "cluster_route_top_m": int(mgr._cluster_route_top_m),
        "cluster_residual_fallback": bool(mgr._cluster_residual_fallback),
        "seed_restricted": seed_restricted,
        "n_cluster_nodes_for_binding": len(mgr._cluster_members or {}),
    }


def eval_config(
    mgr,
    meta,
    queries,
    config,
    top_k,
    *,
    beta_sess,
    beta_cluster,
    full_kg,
    cluster_max_siblings,
):
    cfg_meta = apply_cfg(
        mgr,
        config,
        beta_sess=beta_sess,
        beta_cluster=beta_cluster,
        full_kg=full_kg,
        cluster_max_siblings=cluster_max_siblings,
    )
    metric_keys = ["session_hit@k", "answer_hit@k", "episode_completeness@k", "ndcg@k", "mrr@k"]
    acc = {k: [] for k in metric_keys}
    t0 = time.time()
    seed_restricted = bool(cfg_meta.get("seed_restricted"))
    for n, q in enumerate(queries):
        gold, ans = resolve_gold_meta(q, meta)
        if seed_restricted:
            mgr._seed_restrict_session_id = q.get("session_id")
        else:
            mgr._seed_restrict_session_id = None
        try:
            retrieved = mgr.search(q["query"], top_k=max(top_k * 3, 30), is_temporal_task=False)[:top_k]
        finally:
            mgr._seed_restrict_session_id = None
        # Hard enforce seed protocol after operators (catches any residual leak).
        if seed_restricted:
            qsid = q.get("session_id")
            if config in ("seed_flat", "seed_sess_o2"):
                retrieved = [
                    r for r in retrieved
                    if mgr._session_id_for_tid(r.get("tid")) == qsid
                ][:top_k]
            elif config == "seed_cluster_o2":
                sa_tids = list(mgr._session_members.get(qsid, []) or [])
                allowed_cids = {
                    mgr._tid_to_cluster[t]
                    for t in sa_tids
                    if t in (mgr._tid_to_cluster or {})
                }
                kept = []
                for r in retrieved:
                    tid = mgr._paragraph_tid(r.get("tid"))
                    sid = mgr._session_id_for_tid(tid)
                    cid = (mgr._tid_to_cluster or {}).get(tid)
                    if sid == qsid or (cid is not None and cid in allowed_cids):
                        kept.append(r)
                retrieved = kept[:top_k]
        m = metrics_at_k(retrieved, gold["all_tids"], ans, top_k)
        for k in metric_keys:
            acc[k].append(m[k])
        if (n + 1) % 50 == 0:
            print(
                f"  [{config}] {n+1}/{len(queries)} "
                f"AH={sum(acc['answer_hit@k'])/len(acc['answer_hit@k']):.3f}",
                flush=True,
            )
    hits = [int(x) for x in acc["answer_hit@k"]]
    return {
        "config": config,
        "n": len(queries),
        "session_hit@k": _mean(acc["session_hit@k"]),
        "answer_hit@k": _mean(acc["answer_hit@k"]),
        "episode_completeness@k": _mean(acc["episode_completeness@k"]),
        "ndcg@k": _mean(acc["ndcg@k"]),
        "mrr@k": _mean(acc["mrr@k"]),
        "ah_ci": bootstrap_ci(hits),
        "per_query_ah": hits,
        "ah_by_scope": stratify_ah(queries, hits),
        "elapsed_seconds": round(time.time() - t0, 1),
        "operator": cfg_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tier", default="")
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--beta_sess", type=float, default=0.98)
    ap.add_argument("--beta_cluster", type=float, default=0.90)
    ap.add_argument("--cluster_max_siblings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--para_cache", default="")
    ap.add_argument("--force_queries_json", action="store_true")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--glue_split_pairs", action="store_true",
                    help="CSCE: merge each Sa∪Sb tid set into one cluster after BIRCH")
    ap.add_argument("--out_dir", default=str(REPO / "results" / "bims_legal_cluster_o2"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tier = args.tier or man.get("tier", "X")
    sessions = man["sessions"]
    n_sid_fix = uniquify_session_ids(sessions)
    if n_sid_fix:
        print(f"[fix] uniquified {n_sid_fix} duplicate session_id occurrences", flush=True)

    man_path = Path(args.manifest)
    mix_or_csce = man.get("protocol") in ("split_episode_csce", "csce_mix")
    csce_qs = load_csce_queries(man_path, args.channels) if mix_or_csce else []
    if csce_qs:
        queries = csce_qs
        print(
            f"[csce] loaded {len(queries)} queries protocol={man.get('protocol')} "
            f"split_ratio={man.get('split_ratio')}",
            flush=True,
        )
    else:
        queries = resolve_queries(man, args)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    by_ch: Dict[str, List[dict]] = {}
    for q in queries:
        by_ch.setdefault(q["channel"], []).append(q)

    print(
        f"[index] turn store sessions={len(sessions)} "
        f"turns={sum(len(s['turns']) for s in sessions)}",
        flush=True,
    )
    mgr_turn, meta_turn = index_turn_store(sessions)
    full_kg = dict(mgr_turn.knowledge_graph)
    mgr_turn._rebuild_cluster_membership()
    glue_diag = {}
    if args.glue_split_pairs or mix_or_csce:
        # Glue Sa∪Sb for cross-session pairs so Soft O2-C is solvable.
        glue_diag = glue_split_pairs(mgr_turn, sessions)
        print(f"[csce] glue_split_pairs={glue_diag}", flush=True)
    qa_diag = gold_qa_same_cluster_rate(mgr_turn, meta_turn)
    print(f"[diag] gold Q-A same-cluster rate={qa_diag}", flush=True)
    csce_diag = {}
    if any(q.get("gold_session_id") and q.get("gold_session_id") != q.get("session_id") for q in queries):
        csce_diag = csce_sa_sb_co_cluster_rate(mgr_turn, meta_turn, queries)
        print(f"[diag] CSCE Sa-Sb co-cluster rate={csce_diag}", flush=True)

    out_dir = Path(args.out_dir) / Path(args.manifest).parent.name / f"tier_{tier}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": man.get("protocol") or "LegalMem-MT-v3-cluster-o2-ablation",
        "tier": tier,
        "manifest": str(Path(args.manifest)),
        "split_ratio": man.get("split_ratio"),
        "n_sessions": man.get("n_sessions"),
        "n_gold": man.get("n_gold"),
        "n_distractor": man.get("n_distractor"),
        "n_turns": man.get("n_turns"),
        "n_same_session_gold": man.get("n_same_session_gold"),
        "n_cross_session_gold": man.get("n_cross_session_gold"),
        "beta_sess": args.beta_sess,
        "beta_cluster": args.beta_cluster,
        "cluster_max_siblings": args.cluster_max_siblings,
        "glue_split_pairs": bool(glue_diag),
        "glue_diag": glue_diag,
        "gold_qa_same_cluster": qa_diag,
        "csce_sa_sb_co_cluster": csce_diag,
        "knowledge_nodes_indexed": len(full_kg),
        "note": (
            "Fair Mix / CSCE: unrestricted dense for ep_flat/sess_o2/cluster_o2/hybrid_xsess. "
            "hybrid_xsess = Soft O2 + gated Soft O2-C (cross-session only, direct-hit, "
            "assistant-prefer). glue_split_pairs consolidates Sa∪Sb clusters. "
            "seed_* configs remain diagnostic only."
        ),
        "channels": {},
        "comparisons": {},
    }
    for ch, qs in by_ch.items():
        print(f"=== channel {ch} n={len(qs)} ===", flush=True)
        payload["channels"][ch] = {"configs": {}}
        for cfg in args.configs:
            print(f"[{ch}/{cfg}] start", flush=True)
            res = eval_config(
                mgr_turn,
                meta_turn,
                qs,
                cfg,
                args.top_k,
                beta_sess=args.beta_sess,
                beta_cluster=args.beta_cluster,
                full_kg=full_kg,
                cluster_max_siblings=args.cluster_max_siblings,
            )
            payload["channels"][ch]["configs"][cfg] = res
            (out_dir / "results.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[{ch}/{cfg}] AH={res['answer_hit@k']:.3f} "
                f"EC={res['episode_completeness@k']:.3f} nDCG={res['ndcg@k']:.3f} "
                f"scope={res.get('ah_by_scope')}",
                flush=True,
            )

        cfgs = payload["channels"][ch]["configs"]
        comps = {}
        for a, b, key in [
            ("sess_o2", "ep_flat", "sess_o2_vs_ep_flat"),
            ("cluster_o2", "ep_flat", "cluster_o2_vs_ep_flat"),
            ("hybrid_o2", "ep_flat", "hybrid_o2_vs_ep_flat"),
            ("hybrid_o2", "sess_o2", "hybrid_o2_vs_sess_o2"),
            ("hybrid_xsess", "ep_flat", "hybrid_xsess_vs_ep_flat"),
            ("hybrid_xsess", "sess_o2", "hybrid_xsess_vs_sess_o2"),
            ("hybrid_xsess", "hybrid_o2", "hybrid_xsess_vs_hybrid_o2"),
            ("cluster_o2", "sess_o2", "cluster_o2_vs_sess_o2"),
            ("hybrid_xsess", "cluster_o2", "hybrid_xsess_vs_cluster_o2"),
            ("seed_cluster_o2", "seed_flat", "seed_cluster_o2_vs_seed_flat"),
            ("seed_cluster_o2", "seed_sess_o2", "seed_cluster_o2_vs_seed_sess_o2"),
            ("seed_sess_o2", "seed_flat", "seed_sess_o2_vs_seed_flat"),
            ("bridge_o2", "sess_o2", "bridge_o2_vs_sess_o2"),
            ("sess_rerank_c", "sess_o2", "sess_rerank_c_vs_sess_o2"),
            ("sess_suppress_c", "sess_o2", "sess_suppress_c_vs_sess_o2"),
            ("cluster_route", "sess_o2", "cluster_route_vs_sess_o2"),
            ("sess_residual_c", "sess_o2", "sess_residual_c_vs_sess_o2"),
            ("hybrid_o2", "cluster_o2", "hybrid_o2_vs_cluster_o2"),
            ("birch_c_o2", "birch_flat", "birch_c_o2_vs_birch_flat"),
            ("birch_flat", "ep_flat", "birch_flat_vs_ep_flat"),
        ]:
            if a in cfgs and b in cfgs:
                comps[key] = paired_report(
                    a, cfgs[a]["per_query_ah"],
                    b, cfgs[b]["per_query_ah"],
                )
        # Stratified paired tests on cross_session / same_session subsets
        def _scope_of(q: dict) -> str:
            if q.get("evidence_scope") in ("cross_session", "same_session"):
                return q["evidence_scope"]
            if q.get("gold_session_id") and q["gold_session_id"] != q.get("session_id"):
                return "cross_session"
            return "same_session"

        scope_comps = {}
        for scope in ("cross_session", "same_session"):
            idxs = [i for i, q in enumerate(qs) if _scope_of(q) == scope]
            if not idxs:
                continue
            for a, b, key in [
                ("hybrid_xsess", "sess_o2", f"hybrid_xsess_vs_sess_o2__{scope}"),
                ("cluster_o2", "sess_o2", f"cluster_o2_vs_sess_o2__{scope}"),
                ("sess_o2", "ep_flat", f"sess_o2_vs_ep_flat__{scope}"),
                ("hybrid_xsess", "ep_flat", f"hybrid_xsess_vs_ep_flat__{scope}"),
            ]:
                if a not in cfgs or b not in cfgs:
                    continue
                ha = [cfgs[a]["per_query_ah"][i] for i in idxs]
                hb = [cfgs[b]["per_query_ah"][i] for i in idxs]
                scope_comps[key] = paired_report(a, ha, b, hb)
        comps.update(scope_comps)
        payload["comparisons"][ch] = comps

    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {(out_dir / 'results.json')}", flush=True)


if __name__ == "__main__":
    main()