#!/usr/bin/env bash
# Canonical source of truth: repository root memory_manager.py and eval/.
# Publication package BIMS-LEGAL-code/ is a synced mirror.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$ROOT/BIMS-LEGAL-code"
mkdir -p "$CODE/eval/legal/v3" "$CODE/paper/scripts"
cp "$ROOT/memory_manager.py" "$CODE/memory_manager.py"
cp "$ROOT/config.py" "$CODE/config.py" 2>/dev/null || true
cp "$ROOT/embed_backend.py" "$CODE/embed_backend.py" 2>/dev/null || true
cp "$ROOT/requirements.txt" "$CODE/requirements.txt" 2>/dev/null || true
rsync -a "$ROOT/eval/" "$CODE/eval/"
rsync -a "$ROOT/paper/scripts/" "$CODE/paper/scripts/"
# zero-diff check for critical files
diff -q "$ROOT/memory_manager.py" "$CODE/memory_manager.py"
diff -q "$ROOT/eval/legal/v3/run_legalmem_mt.py" "$CODE/eval/legal/v3/run_legalmem_mt.py"
echo "[sync_canonical_code] root -> BIMS-LEGAL-code OK (critical diffs empty)"
