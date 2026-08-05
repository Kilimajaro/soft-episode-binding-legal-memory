#!/usr/bin/env bash
# Lightweight IPM pre-submission smoke checks (no GPU / full eval required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== [1/5] Sync canonical code into BIMS-LEGAL-code =="
bash scripts/sync_canonical_code.sh

echo "== [2/5] Hybrid gate order unit test =="
python3 -m pytest -q eval/legal/test_hybrid_gate.py

echo "== [3/5] Holm adjustment unit test =="
python3 -m pytest -q paper/scripts/test_holm_adjust.py

echo "== [4/5] Soft O2 metric / failure taxonomy tests =="
python3 -m pytest -q eval/legal/test_legal_metrics.py

echo "== [5/5] Import + Soft O2-C clustering API smoke =="
python3 - <<'PY'
from memory_manager import VectorMemoryManager, ClusteringLayer
import inspect
import numpy as np

src = inspect.getsource(VectorMemoryManager.search)
assert src.index("self._cluster_direct_hits") < src.index("self._expand_with_session_siblings")
assert hasattr(ClusteringLayer, "encode_with_target_k")
X = np.random.RandomState(0).randn(20, 8).astype("float32")
labels, centers = ClusteringLayer().encode_with_target_k(X, target_k=4)
assert len(labels) == 20 and len(centers) == 4
# Construction should succeed without hitting the network.
_ = VectorMemoryManager()
print("vector-memory-smoke-ok", int(labels.max()) + 1)
PY

echo "IPM smoke checks passed."
