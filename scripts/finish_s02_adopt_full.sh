#!/usr/bin/env bash
# Adopt *_full PQ rebuild as sole Soft O2 metric source (S0-2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMBED="${EMBED_CACHE:-/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556/results/bims_legal_v4/embed_cache_shared/vectors.sqlite}"
DEV="${DEV_ROOT:-/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556}"
cd "$ROOT"
mkdir -p /tmp
exec > >(tee -a /tmp/finish_s02_adopt_full.log) 2>&1

wait_pq() {
  local path="$1"; shift
  local channels=("$@")
  echo "[finish] wait $path $(date -Is)"
  while true; do
    if [[ -f "$path" ]]; then
      if PATH_JSON="$path" CHANNELS="${channels[*]}" python3 - <<'PY'
import json, os
from pathlib import Path
d = json.loads(Path(os.environ["PATH_JSON"]).read_text())
need = os.environ["CHANNELS"].split()
cfgs = ["dense_flat", "dense_o2", "parent_hydrate", "shuffled_o2"]
ok = all(
    c in d.get("channels", {})
    and all(x in d["channels"][c] and "per_query_ah" in d["channels"][c][x] for x in cfgs)
    for c in need
)
raise SystemExit(0 if ok else 1)
PY
      then
        echo "[finish] complete $path"
        break
      fi
    fi
    sleep 30
  done
}

wait_pq /tmp/corrected_metrics_cail_pq.json u1_exact uk_followup u_last

echo "[finish] disc with --reuse_index $(date -Is)"
rm -f /tmp/corrected_metrics_disc_pq.json
set +e
USE_EMBED_DISK_CACHE=1 EMBED_DISK_CACHE_DIR="$EMBED" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
BIMS_DATA_ROOT="$ROOT/results/bims_recompute/disc_full" \
python3 -u paper/scripts/recompute_corrected_metrics.py \
  --manifest BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json \
  --workdir results/bims_recompute/disc_full --embed_cache "$EMBED" \
  --channels exact u_para advice_recall --configs dense_flat dense_o2 parent_hydrate shuffled_o2 \
  --skip_finalize --reuse_index --out /tmp/corrected_metrics_disc_pq.json 2>&1 | tee /tmp/recompute_pq_disc_reuse.log
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 || ! -f /tmp/corrected_metrics_disc_pq.json ]]; then
  echo "[finish] disc reuse failed rc=$rc; full rebuild $(date -Is)"
  USE_EMBED_DISK_CACHE=1 EMBED_DISK_CACHE_DIR="$EMBED" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  BIMS_DATA_ROOT="$ROOT/results/bims_recompute/disc_full" \
  python3 -u paper/scripts/recompute_corrected_metrics.py \
    --manifest BIMS-LEGAL-dataset/legalep_v4/legalep_disc/corpus_manifest_M.json \
    --workdir results/bims_recompute/disc_full --embed_cache "$EMBED" \
    --channels exact u_para advice_recall --configs dense_flat dense_o2 parent_hydrate shuffled_o2 \
    --skip_finalize --out /tmp/corrected_metrics_disc_pq.json 2>&1 | tee /tmp/recompute_pq_disc_rebuild.log
fi
wait_pq /tmp/corrected_metrics_disc_pq.json exact u_para advice_recall

echo "[finish] merge+tables+figs $(date -Is)"
python3 paper/scripts/merge_per_query_into_corrected.py --name lawyer --src /tmp/corrected_metrics_lawyer_pq.json --force
python3 paper/scripts/merge_per_query_into_corrected.py --name disc --src /tmp/corrected_metrics_disc_pq.json --force
python3 paper/scripts/merge_per_query_into_corrected.py --name cail --src /tmp/corrected_metrics_cail_pq.json --force
python3 paper/scripts/regenerate_unified_tables.py
python3 paper/scripts/sync_prose_from_corrected.py
python3 - <<'PY'
import sys
sys.path.insert(0, "paper/scripts")
from draw_ipm_figures import _setup_fonts, draw_fig3, draw_fig4
_setup_fonts()
draw_fig3()
draw_fig4()
print("figures ok")
PY
(
  cd paper/ipm
  tectonic --keep-logs ipm-article.tex
  python3 ../scripts/make_anonymous_tex.py
  tectonic --keep-logs ipm-article-anonymous.tex
)

python3 paper/scripts/assert_manuscript_consistency.py

python3 - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "paper/scripts")
from regenerate_unified_tables import PRIMARY_FAMILY, load_corrected

holm = json.loads(Path("paper/ipm/figures/holm_primary_family.json").read_text())
assert "bims_legal_v4" not in str(holm.get("source", "")).lower(), holm.get("source")
by = {r["label"]: r for r in holm["rows"]}
for lab, js, ch in PRIMARY_FAMILY:
    b = load_corrected(js)["channels"][ch]
    d = float(b["dense_o2"]["answer_hit@k"]) - float(b["dense_flat"]["answer_hit@k"])
    hd = float(by[lab]["delta_ah"])
    print(lab, f"table={d:+.3f}", f"holm={hd:+.3f}")
    assert abs(d - hd) < 1e-9, (lab, d, hd)
print("OK consistency; holm source=", holm.get("source"))
PY

mkdir -p "$DEV/paper/ipm/figures" "$DEV/paper/scripts"
cp -a paper/scripts/draw_ipm_figures.py "$DEV/paper/scripts/"
cp -a paper/ipm/figures/corrected_metrics_*.json paper/ipm/figures/holm_primary_family.json "$DEV/paper/ipm/figures/"
cp -a paper/ipm/figures/fig3_cail_main.* paper/ipm/figures/fig4_legalep_main.* "$DEV/paper/ipm/figures/"
cp -a paper/ipm/ipm-article.tex paper/ipm/ipm-article.pdf "$DEV/paper/ipm/" || true
cp -a paper/ipm/ipm-article-anonymous.tex paper/ipm/ipm-article-anonymous.pdf "$DEV/paper/ipm/" || true

echo FINISH_S02_ADOPT_DONE | tee /tmp/finish_s02.done
