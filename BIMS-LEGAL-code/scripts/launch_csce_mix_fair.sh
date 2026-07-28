#!/usr/bin/env bash
# Fair Mix protocol: Soft O2 vs Soft O2-C/hybrid on the same store (unrestricted dense).
# Usage:
#   bash scripts/launch_csce_mix_fair.sh build
#   bash scripts/launch_csce_mix_fair.sh smoke   # max_q=80, LegalEp-DISC
#   bash scripts/launch_csce_mix_fair.sh all     # full three corpora
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
PY="${PYTHON:-python}"
CACHE="$ROOT/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
OUT_ROOT=results/bims_legal_csce_mix
DATA_ROOT=data/legal/csce_mix
SPLIT_RATIO="${SPLIT_RATIO:-0.7}"
TARGET="${1:-all}"
MAX_Q="${MAX_QUERIES:-0}"
CONFIGS="${CONFIGS:-ep_flat sess_o2 cluster_o2 hybrid_xsess}"

mkdir -p "$OUT_ROOT/logs" "$DATA_ROOT"

build_one() {
  local src="$1" name="$2" chans="$3"
  local style="${4:-qa_split}"
  local dest="$DATA_ROOT/$name"
  mkdir -p "$dest"
  # shellcheck disable=SC2086
  "$PY" -u eval/legal/v3/build_split_episode_manifest.py \
    --manifest "$src" \
    --out_dir "$dest" \
    --mode mix \
    --split_style "$style" \
    --split_ratio "$SPLIT_RATIO" \
    --channels $chans \
    --seed 42
}

build_all() {
  build_one data/legal/legalep_v4/legalep_disc/corpus_manifest_M.json legalep_disc "advice_recall" qa_split
  build_one data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json legalep_lawyer "advice_recall" qa_split
  build_one data/legal/legalmem_mt/corpus_manifest_M.json legalmem_mt "uk_followup" mid_split
}

run_one() {
  local gpu="$1" ollama="$2" work="$3" name="$4" chans="$5"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/$OUT_ROOT/$work"
  export EMBED_DISK_CACHE_DIR="$CACHE"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS="${EMBED_OLLAMA_WORKERS:-10}"
  export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT"
  rm -rf "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  mkdir -p "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  local extra=()
  if [[ "$MAX_Q" != "0" ]]; then
    extra+=(--max_queries "$MAX_Q")
  fi
  echo "[mix] $name gpu=$gpu configs=[$CONFIGS] $(date -Is)" | tee -a "$OUT_ROOT/logs/launch.log"
  # shellcheck disable=SC2086
  "$PY" -u eval/legal/v3/run_cluster_o2_ablation.py \
    --manifest "$DATA_ROOT/$name/corpus_manifest_M.json" \
    --tier M \
    --channels $chans \
    --configs $CONFIGS \
    --glue_split_pairs \
    --cluster_max_siblings "${CLUSTER_MAX_SIBLINGS:-16}" \
    --beta_cluster "${BETA_CLUSTER:-0.95}" \
    --out_dir "$OUT_ROOT" \
    "${extra[@]}" \
    > "$OUT_ROOT/logs/${name}.log" 2>&1
  echo "[mix] $name done $(date -Is)" | tee -a "$OUT_ROOT/logs/launch.log"
}

case "$TARGET" in
  build)
    build_all
    ;;
  smoke)
    build_all
    MAX_Q="${MAX_QUERIES:-100}"
    run_one 0 http://127.0.0.1:11434 workdir_disc_smoke legalep_disc "advice_recall"
    ;;
  all)
    build_all
    (
      run_one 0 http://127.0.0.1:11434 workdir_disc legalep_disc "advice_recall"
      run_one 0 http://127.0.0.1:11434 workdir_lawyer legalep_lawyer "advice_recall"
    ) &
    pid0=$!
    (
      run_one 1 http://127.0.0.1:11435 workdir_cail legalmem_mt "uk_followup"
    ) &
    pid1=$!
    echo "[mix] pids disc_lawyer=$pid0 cail=$pid1" | tee -a "$OUT_ROOT/logs/launch.log"
    wait $pid0 $pid1
    echo "[mix] ALL DONE $(date -Is)" | tee -a "$OUT_ROOT/logs/launch.log"
    ;;
  *)
    echo "usage: $0 {build|smoke|all}" >&2
    exit 1
    ;;
esac
