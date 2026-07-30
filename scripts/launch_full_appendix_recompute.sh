#!/usr/bin/env bash
# Full appendix-grid corrected-metric recomputes (FlatIP / Soft O2 / Hard / Shuffled).
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/soft-episode-binding-legal-memory"
VM="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
EMBED="${EMBED_DISK_CACHE_DIR:-$VM/results/bims_legal_v4/embed_cache_shared/vectors.sqlite}"
cd "$ROOT"

run_one() {
  local gpu="$1" name="$2" manifest="$3"; shift 3
  local channels=("$@")
  local work="$ROOT/results/bims_recompute/${name}_full"
  local out="$ROOT/paper/ipm/figures/corrected_metrics_${name}.json"
  local log="/tmp/recompute_full_${name}.log"
  mkdir -p "$work"
  echo "[launch] gpu=$gpu name=$name -> $out"
  # reuse_index: reuse saved FlatIP if eval_meta.json + ntotal>0 exist.
  # merge: keep already-computed channel/config cells; fill missing ones.
  CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE="$gpu" \
  USE_EMBED_DISK_CACHE=1 EMBED_DISK_CACHE_DIR="$EMBED" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -u paper/scripts/recompute_corrected_metrics.py \
    --manifest "$manifest" \
    --workdir "$work" \
    --embed_cache "$EMBED" \
    --channels "${channels[@]}" \
    --configs dense_flat dense_o2 parent_hydrate shuffled_o2 \
    --skip_finalize \
    --reuse_index \
    --merge \
    --out "$out" \
    2>&1 | tee "$log"
}

case "${1:-help}" in
  disc)
    run_one "${GPU:-1}" disc \
      BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json \
      exact u_para advice_recall
    ;;
  lawyer)
    run_one "${GPU:-0}" lawyer \
      BIMS-LEGAL-dataset/legalep_v4/legalep_lawyer/corpus_manifest_M.json \
      exact u_para advice_recall
    ;;
  cail)
    run_one "${GPU:-1}" cail \
      BIMS-LEGAL-dataset/legalmem_mt/corpus_manifest_M.json \
      u1_exact uk_followup u_last
    ;;
  all-parallel)
    # GPU0=lawyer, GPU1=disc then cail. Use absolute self-path (tmux login shells reset $0).
    self="$ROOT/scripts/launch_full_appendix_recompute.sh"
    tmux -f /exec-daemon/tmux.portal.conf kill-session -t recompute-full-lawyer 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf kill-session -t recompute-full-disc 2>/dev/null || true
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s recompute-full-lawyer -c "$ROOT" -- \
      bash -lc "GPU=0 bash '$self' lawyer; echo LAWYER_DONE; exec bash"
    tmux -f /exec-daemon/tmux.portal.conf new-session -d -s recompute-full-disc -c "$ROOT" -- \
      bash -lc "GPU=1 bash '$self' disc && GPU=1 bash '$self' cail; echo DISC_CAIL_DONE; exec bash"
    echo "started tmux: recompute-full-lawyer / recompute-full-disc"
    ;;
  *)
    echo "usage: $0 disc|lawyer|cail|all-parallel"
    exit 2
    ;;
esac
