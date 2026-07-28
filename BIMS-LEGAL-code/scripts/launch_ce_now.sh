#!/usr/bin/env bash
# Parallel FlatIP+CE (dense_ce) — do not wait for BM25 queue.
# Reason CE was idle: launch_final_extras.sh runs CE only after BM25+beta.
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
mkdir -p results/bims_legal_v4/logs
CE_MODEL="${CE_MODEL:-$ROOT/results/bims_legal_v4/models/bge-reranker-v2-m3}"
LOG=results/bims_legal_v4/logs/ce_parallel.log
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

run_env() {
  local gpu="$1" ollama="$2" work="$3"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/results/bims_legal_v4/$work"
  export EMBED_DISK_CACHE_DIR="$BIMS_DATA_ROOT/embed_cache/vectors.sqlite"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS=8
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT/embed_cache"
}

echo "[ce] parallel launch model=$CE_MODEL $(date -Is)" | tee -a "$LOG"

# ---- GPU0: DISC CE (reuse warm gpu0 embed cache) ----
(
  run_env 0 http://127.0.0.1:11434 workdir_gpu0
  echo "[gpu0] DISC CE start $(date -Is)" | tee -a "$LOG"
  if [[ ! -f results/bims_legal_v4/legalep_disc_ce/tier_M/results.json ]]; then
    python -u eval/legal/v3/run_legalmem_mt.py \
      --manifest data/legal/legalep_v4/legalep_disc/corpus_manifest_M.json --tier M \
      --channels exact advice_recall \
      --configs dense_flat dense_ce dense_o2 \
      --force_queries_json \
      --ce_model "$CE_MODEL" \
      --out_dir results/bims_legal_v4/legalep_disc_ce \
      > results/bims_legal_v4/logs/disc_ce.log 2>&1
  fi
  echo "[gpu0] DISC CE done $(date -Is)" | tee -a "$LOG"

  echo "[gpu0] Lawyer CE start $(date -Is)" | tee -a "$LOG"
  if [[ ! -f results/bims_legal_v4/legalep_lawyer_ce/tier_M/results.json ]]; then
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
echo $! > results/bims_legal_v4/logs/ce_gpu0.pid

# ---- GPU1: CAIL CE (reuse warm gpu1 embed cache) ----
(
  run_env 1 http://127.0.0.1:11435 workdir_gpu1
  echo "[gpu1] CAIL CE start $(date -Is)" | tee -a "$LOG"
  if [[ ! -f results/bims_legal_v4/cail_ce/tier_M/results.json ]]; then
    python -u eval/legal/v3/run_legalmem_mt.py \
      --manifest data/legal/legalmem_mt/corpus_manifest_M.json --tier M \
      --channels u1_exact uk_followup u_last \
      --configs dense_flat dense_ce dense_o2 \
      --ce_model "$CE_MODEL" \
      --out_dir results/bims_legal_v4/cail_ce \
      > results/bims_legal_v4/logs/cail_ce.log 2>&1
  fi
  echo "[gpu1] CAIL CE done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/ce_gpu1.pid

echo "[ce] pids gpu0=$(cat results/bims_legal_v4/logs/ce_gpu0.pid) gpu1=$(cat results/bims_legal_v4/logs/ce_gpu1.pid) $(date -Is)" | tee -a "$LOG"
wait
python3 paper/scripts/fill_v4_tables.py || true
echo "[ce] ALL_DONE $(date -Is)" | tee -a "$LOG"
