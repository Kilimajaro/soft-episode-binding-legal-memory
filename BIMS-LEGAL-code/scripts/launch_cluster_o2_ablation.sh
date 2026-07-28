#!/usr/bin/env bash
# Soft O2-C / hybrid ablation launcher (reuses shared embed cache).
#
# Usage:
#   bash scripts/launch_cluster_o2_ablation.sh              # all: cail + disc + lawyer
#   bash scripts/launch_cluster_o2_ablation.sh cail
#   bash scripts/launch_cluster_o2_ablation.sh disc
#   bash scripts/launch_cluster_o2_ablation.sh lawyer
#   bash scripts/launch_cluster_o2_ablation.sh disc_remaining  # u_para + advice only
# Optional 2nd arg overrides channels for a single-dataset run.
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
mkdir -p results/bims_legal_cluster_o2/logs /tmp/cursor/logs
PY="${PYTHON:-python}"
CACHE="$ROOT/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
TARGET="${1:-all}"
CHANNELS_OVERRIDE="${2:-}"

run_env() {
  local gpu="$1" ollama="$2" work="$3"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/results/bims_legal_cluster_o2/$work"
  export EMBED_DISK_CACHE_DIR="$CACHE"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS="${EMBED_OLLAMA_WORKERS:-10}"
  export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT"
  rm -f "$BIMS_DATA_ROOT/talk.txt"
  rm -rf "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  mkdir -p "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
}

run_cail() {
  local ch="${CHANNELS_OVERRIDE:-u1_exact uk_followup u_last}"
  run_env 0 http://127.0.0.1:11434 workdir_cail
  echo "[gpu0] CAIL cluster-o2 channels=[$ch] $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
  # shellcheck disable=SC2086
  "$PY" -u eval/legal/v3/run_cluster_o2_ablation.py \
    --manifest data/legal/legalmem_mt/corpus_manifest_M.json --tier M \
    --channels $ch \
    --configs ep_flat sess_o2 cluster_o2 hybrid_o2 birch_flat birch_c_o2 \
    --out_dir results/bims_legal_cluster_o2 \
    > results/bims_legal_cluster_o2/logs/cail.log 2>&1
  echo "[gpu0] CAIL done $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
}

run_disc() {
  local ch="${CHANNELS_OVERRIDE:-exact u_para advice_recall}"
  run_env 1 http://127.0.0.1:11435 workdir_disc
  echo "[gpu1] DISC cluster-o2 channels=[$ch] $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
  # shellcheck disable=SC2086
  "$PY" -u eval/legal/v3/run_cluster_o2_ablation.py \
    --manifest data/legal/legalep_v4/legalep_disc/corpus_manifest_M.json --tier M \
    --channels $ch --force_queries_json \
    --para_cache data/legal/legalep_v4/legalep_disc/paraphrase_cache.json \
    --configs ep_flat sess_o2 cluster_o2 hybrid_o2 birch_flat birch_c_o2 \
    --out_dir results/bims_legal_cluster_o2 \
    > results/bims_legal_cluster_o2/logs/disc.log 2>&1
  echo "[gpu1] DISC done $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
}

run_lawyer() {
  local ch="${CHANNELS_OVERRIDE:-exact u_para advice_recall}"
  run_env 1 http://127.0.0.1:11435 workdir_lawyer
  echo "[gpu1] Lawyer cluster-o2 channels=[$ch] $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
  # shellcheck disable=SC2086
  "$PY" -u eval/legal/v3/run_cluster_o2_ablation.py \
    --manifest data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json --tier M \
    --channels $ch --force_queries_json \
    --para_cache data/legal/legalep_v4/legalep_lawyer/paraphrase_cache.json \
    --configs ep_flat sess_o2 cluster_o2 hybrid_o2 birch_flat birch_c_o2 \
    --out_dir results/bims_legal_cluster_o2 \
    > results/bims_legal_cluster_o2/logs/lawyer.log 2>&1
  echo "[gpu1] Lawyer done $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
}

echo "[cluster-o2] start target=$TARGET $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log

case "$TARGET" in
  cail)
    run_cail
    ;;
  disc)
    run_disc
    ;;
  disc_remaining)
    CHANNELS_OVERRIDE="${CHANNELS_OVERRIDE:-u_para advice_recall}"
    run_disc
    ;;
  lawyer)
    run_lawyer
    ;;
  all)
    run_cail &
    echo $! > results/bims_legal_cluster_o2/logs/cail.pid
    (
      run_disc
      run_lawyer
    ) &
    echo $! > results/bims_legal_cluster_o2/logs/disc_lawyer.pid
    echo "[cluster-o2] pids cail=$(cat results/bims_legal_cluster_o2/logs/cail.pid) disc_lawyer=$(cat results/bims_legal_cluster_o2/logs/disc_lawyer.pid)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
    wait
    ;;
  *)
    echo "Unknown target: $TARGET (expected: all|cail|disc|disc_remaining|lawyer)" >&2
    exit 1
    ;;
esac

echo "[cluster-o2] DONE target=$TARGET $(date -Is)" | tee -a results/bims_legal_cluster_o2/logs/launch.log
