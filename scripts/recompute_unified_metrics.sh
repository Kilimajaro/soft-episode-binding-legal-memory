#!/usr/bin/env bash
# Recompute unified corrected_metrics_*.json (with per_query_ah) from saved FlatIP indexes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMBED="${EMBED_DISK_CACHE_DIR:-/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556/results/bims_legal_v4/embed_cache_shared/vectors.sqlite}"
cd "$ROOT"
name="${1:?usage: $0 lawyer|disc|cail}"
case "$name" in
  lawyer) manifest=BIMS-LEGAL-dataset/legalep_v4/legalep_lawyer/corpus_manifest_M.json; ch=(exact u_para advice_recall);;
  disc) manifest=BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json; ch=(exact u_para advice_recall);;
  cail) manifest=BIMS-LEGAL-dataset/legalmem_mt/corpus_manifest_M.json; ch=(u1_exact uk_followup u_last);;
  *) echo bad name; exit 2;;
esac
work="$ROOT/results/bims_recompute/${name}_full"
out="$ROOT/paper/ipm/figures/corrected_metrics_${name}.json"
USE_EMBED_DISK_CACHE=1 EMBED_DISK_CACHE_DIR="$EMBED" \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -u paper/scripts/recompute_corrected_metrics.py \
  --manifest "$manifest" --workdir "$work" --embed_cache "$EMBED" \
  --channels "${ch[@]}" --configs dense_flat dense_o2 parent_hydrate shuffled_o2 \
  --skip_finalize --reuse_index --out "$out"
