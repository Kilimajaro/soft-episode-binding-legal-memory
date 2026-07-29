#!/usr/bin/env bash
# Launch corrected-nDCG recomputes for Lawyer + CAIL after DISC finishes.
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/soft-episode-binding-legal-memory"
VM="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
EMBED="$VM/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE="${CUDA_DEVICE:-1}"
export USE_EMBED_DISK_CACHE=1
export EMBED_DISK_CACHE_DIR="$EMBED"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

run_one() {
  local name="$1" manifest="$2" channels="$3" expect="$4"
  local work="$ROOT/results/bims_recompute/${name}_clean"
  local out="$ROOT/paper/ipm/figures/corrected_metrics_${name}.json"
  rm -rf "$work"; mkdir -p "$work"
  echo "[run] $name expect_flat_ah=$expect"
  python3 paper/scripts/recompute_corrected_metrics.py \
    --manifest "$manifest" \
    --workdir "$work" \
    --embed_cache "$EMBED" \
    --channels $channels \
    --configs dense_flat dense_o2 \
    --skip_finalize \
    --expect_flat_ah "$expect" \
    --out "$out" 2>&1 | tee "/tmp/recompute_${name}_clean.log"
}

case "${1:-all}" in
  lawyer)
    run_one lawyer BIMS-LEGAL-dataset/legalep_v4/legalep_lawyer/corpus_manifest_M.json "u_para advice_recall" 0.568
    ;;
  cail)
    run_one cail BIMS-LEGAL-dataset/cail_v4/corpus_manifest_M.json "uk_followup" 0.270
    ;;
  *)
    echo "usage: $0 lawyer|cail"
    exit 2
    ;;
esac
