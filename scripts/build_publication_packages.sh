#!/usr/bin/env bash
# Sync development tree into BIMS-LEGAL-code / BIMS-LEGAL-dataset publication folders.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/sync_canonical_code.sh"
CODE="$ROOT/BIMS-LEGAL-code"
DATA="$ROOT/BIMS-LEGAL-dataset"

rsync -a --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='data/' \
  "$ROOT/eval/" "$CODE/eval/" 2>/dev/null || true
rsync -a \
  "$ROOT/memory_manager.py" "$ROOT/config.py" "$ROOT/app.py" "$ROOT/embed_backend.py" \
  "$ROOT/ablation_eval.py" "$ROOT/requirements.txt" \
  "$CODE/" 2>/dev/null || true
rsync -a "$ROOT/paper/scripts/" "$CODE/paper/scripts/"
rsync -a "$ROOT/BIMS-LEGAL-dataset/" "$DATA/" 2>/dev/null || true
echo "[build_publication_packages] synced eval, core modules, and paper scripts."
