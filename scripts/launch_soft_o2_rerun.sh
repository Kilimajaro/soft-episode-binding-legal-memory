#!/usr/bin/env bash
# Re-run Soft O2 / shuffled_o2 / β after search-cache fix.
# Merges patched cells into the existing V4 results.json files (keeps FlatIP etc.).
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
mkdir -p results/bims_legal_v4/logs
BETAS="${BETAS:-0.5 0.7 0.9 0.95 0.98 1.0}"
LOG=results/bims_legal_v4/logs/soft_o2_rerun.log
PY="${PYTHON:-/home/cdll/anaconda3/bin/python3}"

run_env() {
  local gpu="$1" ollama="$2" work="$3"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/results/bims_legal_v4/$work"
  export EMBED_DISK_CACHE_DIR="$ROOT/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS=8
  export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT/embed_cache" "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  rm -f "$BIMS_DATA_ROOT/talk.txt"
  rm -rf "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  mkdir -p "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
}

merge_o2() {
  # merge_o2 <fresh_results.json> <target_results.json> [channels...]
  "$PY" - "$1" "$2" "${@:3}" <<'PY'
import json, sys
from pathlib import Path
fresh_p, target_p = Path(sys.argv[1]), Path(sys.argv[2])
only_chs = sys.argv[3:]
fresh = json.loads(fresh_p.read_text(encoding="utf-8"))
if target_p.is_file():
    target = json.loads(target_p.read_text(encoding="utf-8"))
else:
    target = {
        "protocol": fresh.get("protocol"),
        "tier": fresh.get("tier"),
        "n_sessions": fresh.get("n_sessions"),
        "n_gold": fresh.get("n_gold"),
        "n_distractor": fresh.get("n_distractor"),
        "n_turns": fresh.get("n_turns"),
        "channels": {},
        "comparisons": {},
        "beta_sweep": {},
    }
target.setdefault("channels", {})
target.setdefault("comparisons", {})
chs = only_chs or list((fresh.get("channels") or {}).keys())
patched = []
for ch in chs:
    src = (fresh.get("channels") or {}).get(ch) or {}
    src_cfgs = src.get("configs") or {}
    if not src_cfgs:
        continue
    dst = target["channels"].setdefault(ch, {"configs": {}})
    dst_cfgs = dst.setdefault("configs", {})
    for cfg, cell in src_cfgs.items():
        if cfg in ("dense_o2", "shuffled_o2"):
            dst_cfgs[cfg] = cell
            patched.append(f"{ch}/{cfg}")
    # refresh paired comps against whatever is already in target
    comps = {}
    if "dense_o2" in dst_cfgs and "dense_flat" in dst_cfgs:
        try:
            sys.path.insert(0, str(Path("eval/legal/v3").resolve()))
            from stats_sig import paired_report
            comps["o2_vs_flat"] = paired_report(
                "dense_o2", dst_cfgs["dense_o2"]["per_query_ah"],
                "dense_flat", dst_cfgs["dense_flat"]["per_query_ah"],
            )
            if "parent_hydrate" in dst_cfgs:
                comps["o2_vs_hard"] = paired_report(
                    "dense_o2", dst_cfgs["dense_o2"]["per_query_ah"],
                    "parent_hydrate", dst_cfgs["parent_hydrate"]["per_query_ah"],
                )
            if "dense_ce" in dst_cfgs:
                comps["o2_vs_ce"] = paired_report(
                    "dense_o2", dst_cfgs["dense_o2"]["per_query_ah"],
                    "dense_ce", dst_cfgs["dense_ce"]["per_query_ah"],
                )
        except Exception as e:
            comps["o2_vs_flat_error"] = str(e)
    if comps:
        target.setdefault("comparisons", {})[ch] = {
            **(target.get("comparisons", {}).get(ch) or {}),
            **comps,
        }
bs = fresh.get("beta_sweep") or {}
if bs and any(v is not None for v in bs.values()):
    target["beta_sweep"] = bs
    patched.append("beta_sweep")
target["soft_o2_cache_fix"] = {
    "merged_from": str(fresh_p),
    "patched": patched,
}
target_p.parent.mkdir(parents=True, exist_ok=True)
target_p.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[merge] {fresh_p} -> {target_p} patched={patched}", flush=True)
PY
}

echo "[soft-o2-rerun] start $(date -Is)" | tee -a "$LOG"

# ---- GPU0: CAIL Soft O2 + shuffled + β(uk_followup) ----
(
  run_env 0 http://127.0.0.1:11434 workdir_o2fix0
  echo "[gpu0] CAIL o2+shuffled+beta $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/cail_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalmem_mt/corpus_manifest_M.json --tier M \
    --channels u1_exact uk_followup u_last \
    --configs dense_o2 shuffled_o2 \
    --betas $BETAS \
    --beta_channel uk_followup \
    --out_dir results/bims_legal_v4/cail_o2fix \
    > results/bims_legal_v4/logs/cail_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/cail_o2fix/tier_M/results.json \
    results/bims_legal_v4/cail_M/tier_M/results.json
  # Also refresh dense_o2 cells inside cail_ce / cail_beta for manuscript sync
  if [[ -f results/bims_legal_v4/cail_ce/tier_M/results.json ]]; then
    merge_o2 \
      results/bims_legal_v4/cail_o2fix/tier_M/results.json \
      results/bims_legal_v4/cail_ce/tier_M/results.json \
      u1_exact uk_followup u_last
  fi
  if [[ -f results/bims_legal_v4/cail_beta/tier_M/results.json ]]; then
    merge_o2 \
      results/bims_legal_v4/cail_o2fix/tier_M/results.json \
      results/bims_legal_v4/cail_beta/tier_M/results.json \
      uk_followup
  fi
  echo "[gpu0] CAIL done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/soft_o2_gpu0.pid

# ---- GPU1: Lawyer exact + Lawyer para + DISC para ----
(
  run_env 1 http://127.0.0.1:11435 workdir_o2fix1
  echo "[gpu1] Lawyer exact o2+shuffled+beta $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/legalep_lawyer_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json --tier M \
    --channels exact \
    --configs dense_o2 shuffled_o2 \
    --force_queries_json \
    --betas $BETAS \
    --out_dir results/bims_legal_v4/legalep_lawyer_o2fix \
    > results/bims_legal_v4/logs/lawyer_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/legalep_lawyer_o2fix/tier_M/results.json \
    results/bims_legal_v4/legalep_lawyer_M/tier_M/results.json
  if [[ -f results/bims_legal_v4/legalep_lawyer_beta/tier_M/results.json ]]; then
    merge_o2 \
      results/bims_legal_v4/legalep_lawyer_o2fix/tier_M/results.json \
      results/bims_legal_v4/legalep_lawyer_beta/tier_M/results.json \
      exact
  fi

  echo "[gpu1] Lawyer para o2+shuffled+beta $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/legalep_lawyer_para_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json --tier M \
    --channels u_para \
    --configs dense_o2 shuffled_o2 \
    --force_queries_json \
    --para_cache data/legal/legalep_v4/legalep_lawyer/paraphrase_cache.json \
    --betas $BETAS \
    --out_dir results/bims_legal_v4/legalep_lawyer_para_o2fix \
    > results/bims_legal_v4/logs/lawyer_para_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/legalep_lawyer_para_o2fix/tier_M/results.json \
    results/bims_legal_v4/legalep_lawyer_para/tier_M/results.json

  echo "[gpu1] DISC para o2+shuffled+beta $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/legalep_disc_para_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalep_v4/legalep_disc/corpus_manifest_M.json --tier M \
    --channels u_para \
    --configs dense_o2 shuffled_o2 \
    --force_queries_json \
    --para_cache data/legal/legalep_v4/legalep_disc/paraphrase_cache.json \
    --betas $BETAS \
    --out_dir results/bims_legal_v4/legalep_disc_para_o2fix \
    > results/bims_legal_v4/logs/disc_para_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/legalep_disc_para_o2fix/tier_M/results.json \
    results/bims_legal_v4/legalep_disc_para/tier_M/results.json

  echo "[gpu1] Lawyer/DISC para done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/soft_o2_gpu1.pid

echo "[soft-o2-rerun] pids gpu0=$(cat results/bims_legal_v4/logs/soft_o2_gpu0.pid) gpu1=$(cat results/bims_legal_v4/logs/soft_o2_gpu1.pid)" | tee -a "$LOG"
wait
"$PY" paper/scripts/fill_v4_tables.py || true
echo "[soft-o2-rerun] ALL_DONE $(date -Is)" | tee -a "$LOG"
