#!/usr/bin/env bash
# Minimal IPM reproduction smoke test (no full retrain / full grid).
# Validates package layout, metric unit tests, and presence of main-table JSON artifacts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/4] package layout"
test -f memory_manager.py
test -d BIMS-LEGAL-code
test -d BIMS-LEGAL-dataset
test -f paper/ipm/ipm-article.tex
test -f paper/ipm/ipm-article-anonymous.tex
test -f paper/ipm/ipm-titlepage.tex

echo "[2/4] Soft O2 / nDCG metric unit tests (if present)"
if [[ -f eval/legal/test_legal_metrics.py ]]; then
  python3 -m pytest -q eval/legal/test_legal_metrics.py
elif [[ -f eval/legal/test_metrics_ndcg.py ]]; then
  python3 -m pytest -q eval/legal/test_metrics_ndcg.py
else
  python3 - <<'PY'
from pathlib import Path
# Lightweight sanity: graded nDCG IDCG must be system-independent in legal_metrics if importable.
import importlib.util
cands = list(Path('eval').rglob('*legal_metrics*.py')) + list(Path('eval').rglob('*metrics*.py'))
print('metric modules found:', len(cands))
assert cands, 'no metrics module found'
print('OK (module presence)')
PY
fi

echo "[3/4] primary result artifacts"
ART_ROOT="BIMS-LEGAL-dataset"
if [[ ! -d "$ART_ROOT" ]]; then ART_ROOT="."; fi
# Accept either packaged or in-tree primary_results
found=0
for p in \
  primary_results/legal_scaled_o1o2 \
  primary_results/bims_legal_v4 \
  primary_results/bims_legal_csce_mix \
  BIMS-LEGAL-dataset/primary_results/legal_scaled_o1o2 \
  BIMS-LEGAL-dataset/primary_results/bims_legal_v4 \
  BIMS-LEGAL-dataset/primary_results/bims_legal_csce_mix \
  paper/ipm/figures
 do
  if [[ -e "$p" ]]; then
    echo "  found $p"
    found=$((found+1))
  fi
done
test "$found" -ge 2

echo "[4/4] manuscript compile check (tectonic if available)"
if command -v tectonic >/dev/null 2>&1; then
  (cd paper/ipm && tectonic -X compile ipm-article-anonymous.tex >/tmp/ipm_smoke_tectonic.log 2>&1) \
    && echo "  anonymous PDF OK" \
    || { echo "  tectonic failed; see /tmp/ipm_smoke_tectonic.log"; exit 1; }
else
  echo "  tectonic not installed; skipped PDF smoke"
fi

echo "IPM smoke test passed."
