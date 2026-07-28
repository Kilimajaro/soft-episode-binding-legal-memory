#!/usr/bin/env bash
# Re-run failed/incomplete V4 extras after session_id + CE-skip fixes.
# Gaps: CAIL BM25 (assert), CAIL beta (never started), Lawyer CE (partial results.json),
# plus CAIL main + CAIL CE (Soft O2 contaminated by 100 duplicate session_ids).
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
mkdir -p results/bims_legal_v4/logs
CE_MODEL="${CE_MODEL:-$ROOT/results/bims_legal_v4/models/bge-reranker-v2-m3}"
BETAS="0.5 0.7 0.9 0.95 0.98 1.0"
LOG=results/bims_legal_v4/logs/rerun_gaps.log

complete() {
  # usage: complete results.json "ch1 ch2" "cfg1 cfg2"
  local path="$1" chs="$2" cfgs="$3"
  python3 - "$path" "$chs" "$cfgs" <<'PY'
import json, sys
from pathlib import Path
path, chs, cfgs = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
if not path.is_file():
    raise SystemExit(1)
try:
    d = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
ch_map = d.get("channels") or {}
for ch in chs:
    block = ch_map.get(ch) or {}
    have = (block.get("configs") or {}) if isinstance(block, dict) else {}
    for cfg in cfgs:
        if cfg not in have:
            raise SystemExit(1)
raise SystemExit(0)
PY
}

run_env() {
  local gpu="$1" ollama="$2" work="$3"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/results/bims_legal_v4/$work"
  # Shared merged embed cache across all prior workdirs (fastest hit rate).
  export EMBED_DISK_CACHE_DIR="$ROOT/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS=8
  export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT/embed_cache"
  # Fresh talk/vectors for this run; keep shared embed sqlite intact.
  rm -f "$BIMS_DATA_ROOT/talk.txt"
  rm -rf "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  mkdir -p "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
}

echo "[rerun] start $(date -Is)" | tee -a "$LOG"

# Drop incomplete Lawyer CE so it is not skipped
LAW_CE=results/bims_legal_v4/legalep_lawyer_ce/tier_M/results.json
if [[ -f "$LAW_CE" ]] && ! complete "$LAW_CE" "exact advice_recall" "dense_flat dense_ce dense_o2"; then
  echo "[rerun] removing incomplete $LAW_CE" | tee -a "$LOG"
  rm -f "$LAW_CE"
fi

# ---- GPU0: Lawyer CE (full) ----
(
  run_env 0 http://127.0.0.1:11434 workdir_gpu0
  echo "[gpu0] Lawyer CE $(date -Is)" | tee -a "$LOG"
  if ! complete results/bims_legal_v4/legalep_lawyer_ce/tier_M/results.json "exact advice_recall" "dense_flat dense_ce dense_o2"; then
    python -u eval/legal/v3/run_legalmem_mt.py \
      --manifest data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json --tier M \
      --channels exact advice_recall \
      --configs dense_flat dense_ce dense_o2 \
      --force_queries_json \
      --ce_model "$CE_MODEL" \
      --out_dir results/bims_legal_v4/legalep_lawyer_ce \
      > results/bims_legal_v4/logs/lawyer_ce.log 2>&1
  fi
  echo "[gpu0] Lawyer CE done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/rerun_gpu0.pid

# ---- GPU1: CAIL BM25 only (CPU/Ollama; dense β/main/CE launched separately) ----
# Dense jobs: scripts/launch_cail_dense_parallel.sh (workdir_gpu1_dense, can overlap BM25).
(
  run_env 1 http://127.0.0.1:11435 workdir_gpu1
  echo "[gpu1] CAIL BM25 $(date -Is)" | tee -a "$LOG"
  if ! complete results/bims_legal_v4/cail_bm25/tier_M/results.json "u1_exact uk_followup u_last" "bm25_turn bm25_joint dense_rrf"; then
    rm -f results/bims_legal_v4/cail_bm25/tier_M/results.json
    python -u eval/legal/v3/run_legalmem_mt.py \
      --manifest data/legal/legalmem_mt/corpus_manifest_M.json --tier M \
      --channels u1_exact uk_followup u_last \
      --configs bm25_turn bm25_joint dense_rrf \
      --out_dir results/bims_legal_v4/cail_bm25 \
      > results/bims_legal_v4/logs/cail_bm25.log 2>&1
  fi
  echo "[gpu1] CAIL BM25 done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/rerun_gpu1.pid

echo "[rerun] pids gpu0=$(cat results/bims_legal_v4/logs/rerun_gpu0.pid) gpu1=$(cat results/bims_legal_v4/logs/rerun_gpu1.pid)" | tee -a "$LOG"
echo "[rerun] for CAIL β/main/CE use: bash scripts/launch_cail_dense_parallel.sh" | tee -a "$LOG"
wait
python3 paper/scripts/fill_v4_tables.py || true
echo "[rerun] ALL_DONE $(date -Is)" | tee -a "$LOG"
