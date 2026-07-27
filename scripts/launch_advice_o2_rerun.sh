#!/usr/bin/env bash
# Clean Soft O2 / shuffled_o2 on LegalEp advice-recall (both corpora).
set -euo pipefail
ROOT="/home/cdll/llm-dev/THU/lora/Vector-Memory-Is-All-You-Need-cursor-motify-legal-eval-7556"
cd "$ROOT"
mkdir -p results/bims_legal_v4/logs
LOG=results/bims_legal_v4/logs/advice_o2_rerun.log
PY="${PYTHON:-/home/cdll/anaconda3/bin/python3}"

run_env() {
  local gpu="$1" ollama="$2" work="$3"
  export CUDA_DEVICE="$gpu"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export OLLAMA_BASE_URL="$ollama"
  export BIMS_DATA_ROOT="$ROOT/results/bims_legal_v4/$work"
  export EMBED_DISK_CACHE_DIR="$ROOT/results/bims_legal_v4/embed_cache_shared/vectors.sqlite"
  export USE_EMBED_DISK_CACHE=1
  export USE_EMBED_BATCH=1
  export EMBED_OLLAMA_WORKERS=8
  export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  mkdir -p "$BIMS_DATA_ROOT/embed_cache" "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  rm -f "$BIMS_DATA_ROOT/talk.txt"
  rm -rf "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
  mkdir -p "$BIMS_DATA_ROOT/vectors" "$BIMS_DATA_ROOT/knowledge"
}

merge_o2() {
  "$PY" - "$1" "$2" "${@:3}" <<'PY'
import json, sys
from pathlib import Path
fresh_p, target_p = Path(sys.argv[1]), Path(sys.argv[2])
only_chs = sys.argv[3:]
fresh = json.loads(fresh_p.read_text(encoding="utf-8"))
target = json.loads(target_p.read_text(encoding="utf-8")) if target_p.is_file() else {
    "protocol": fresh.get("protocol"), "tier": fresh.get("tier"),
    "n_sessions": fresh.get("n_sessions"), "n_gold": fresh.get("n_gold"),
    "n_distractor": fresh.get("n_distractor"), "n_turns": fresh.get("n_turns"),
    "channels": {}, "comparisons": {}, "beta_sweep": {},
}
target.setdefault("channels", {}); target.setdefault("comparisons", {})
chs = only_chs or list((fresh.get("channels") or {}).keys())
patched = []
for ch in chs:
    src_cfgs = ((fresh.get("channels") or {}).get(ch) or {}).get("configs") or {}
    if not src_cfgs:
        continue
    dst = target["channels"].setdefault(ch, {"configs": {}})
    dst_cfgs = dst.setdefault("configs", {})
    for cfg, cell in src_cfgs.items():
        if cfg in ("dense_o2", "shuffled_o2"):
            dst_cfgs[cfg] = cell
            patched.append(f"{ch}/{cfg}")
    comps = {}
    if "dense_o2" in dst_cfgs and "dense_flat" in dst_cfgs:
        try:
            sys.path.insert(0, str(Path("eval/legal/v3").resolve()))
            from stats_sig import paired_report
            comps["o2_vs_flat"] = paired_report(
                "dense_o2", dst_cfgs["dense_o2"]["per_query_ah"],
                "dense_flat", dst_cfgs["dense_flat"]["per_query_ah"],
            )
            if "parent_hydrate" in dst_cfgs:
                comps["o2_vs_hard"] = paired_report(
                    "dense_o2", dst_cfgs["dense_o2"]["per_query_ah"],
                    "parent_hydrate", dst_cfgs["parent_hydrate"]["per_query_ah"],
                )
        except Exception as e:
            comps["o2_vs_flat_error"] = str(e)
    if comps:
        target.setdefault("comparisons", {})[ch] = {
            **(target.get("comparisons", {}).get(ch) or {}),
            **comps,
        }
target["soft_o2_cache_fix"] = {"merged_from": str(fresh_p), "patched": patched}
target_p.parent.mkdir(parents=True, exist_ok=True)
target_p.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[merge] {fresh_p} -> {target_p} patched={patched}", flush=True)
PY
}

echo "[advice-o2] start $(date -Is)" | tee -a "$LOG"

(
  run_env 0 http://127.0.0.1:11434 workdir_advice_o2_0
  echo "[gpu0] DISC advice o2+shuffled $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/legalep_disc_advice_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalep_v4/legalep_disc/corpus_manifest_M.json --tier M \
    --channels advice_recall \
    --configs dense_o2 shuffled_o2 \
    --force_queries_json \
    --out_dir results/bims_legal_v4/legalep_disc_advice_o2fix \
    > results/bims_legal_v4/logs/disc_advice_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/legalep_disc_advice_o2fix/tier_M/results.json \
    results/bims_legal_v4/legalep_disc_advice/tier_M/results.json \
    advice_recall
  echo "[gpu0] DISC advice done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/advice_o2_gpu0.pid

(
  run_env 1 http://127.0.0.1:11435 workdir_advice_o2_1
  echo "[gpu1] Lawyer advice o2+shuffled $(date -Is)" | tee -a "$LOG"
  rm -rf results/bims_legal_v4/legalep_lawyer_advice_o2fix
  "$PY" -u eval/legal/v3/run_legalmem_mt.py \
    --manifest data/legal/legalep_v4/legalep_lawyer/corpus_manifest_M.json --tier M \
    --channels advice_recall \
    --configs dense_o2 shuffled_o2 \
    --force_queries_json \
    --out_dir results/bims_legal_v4/legalep_lawyer_advice_o2fix \
    > results/bims_legal_v4/logs/lawyer_advice_o2fix.log 2>&1
  merge_o2 \
    results/bims_legal_v4/legalep_lawyer_advice_o2fix/tier_M/results.json \
    results/bims_legal_v4/legalep_lawyer_advice/tier_M/results.json \
    advice_recall
  echo "[gpu1] Lawyer advice done $(date -Is)" | tee -a "$LOG"
) &
echo $! > results/bims_legal_v4/logs/advice_o2_gpu1.pid

echo "[advice-o2] pids gpu0=$(cat results/bims_legal_v4/logs/advice_o2_gpu0.pid) gpu1=$(cat results/bims_legal_v4/logs/advice_o2_gpu1.pid)" | tee -a "$LOG"
wait
"$PY" paper/scripts/fill_v4_tables.py || true
"$PY" paper/scripts/draw_ipm_figures.py || true
echo "[advice-o2] ALL_DONE $(date -Is)" | tee -a "$LOG"
