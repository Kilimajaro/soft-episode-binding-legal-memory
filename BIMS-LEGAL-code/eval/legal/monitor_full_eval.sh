#!/usr/bin/env bash
# 每小时检查 legal_full 完整实验进度，写入 monitor_status.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="$REPO_ROOT/results/legal_full"
LOG="$OUT_ROOT/legal_full_run.log"
STATUS="$OUT_ROOT/monitor_status.log"
INTERVAL="${MONITOR_INTERVAL_SEC:-3600}"

log_status() {
  {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] hourly check"
    echo "============================================================"

    if pgrep -f "run_legal_scaled.py.*results/legal_full" >/dev/null 2>&1; then
      echo "process: RUNNING ($(pgrep -f 'run_legal_scaled.py.*results/legal_full' | tr '\n' ' '))"
      ps -o etime,pcpu,pmem -p "$(pgrep -f 'run_legal_scaled.py.*results/legal_full' | head -1)" 2>/dev/null || true
    else
      echo "process: NOT RUNNING (needs attention)"
    fi

    if pgrep -f "OLLAMA_HOST=127.0.0.1:11435 ollama serve" >/dev/null 2>&1 || \
       curl -sf http://127.0.0.1:11435/api/tags >/dev/null 2>&1; then
      echo "ollama_gpu1: UP (11435)"
    else
      echo "ollama_gpu1: DOWN (needs attention)"
    fi

    if [ -f "$LOG" ]; then
      echo "last_log_lines:"
      tail -5 "$LOG" | sed 's/^/  /'
    fi

    python3 - <<PY
import json, os
root = "$OUT_ROOT"
for ds in ["disc_law", "lawyer_llama"]:
    p = os.path.join(root, ds, "legal_full_ablation.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        cfgs = d.get("configs", [])
        print(f"{ds}: {len(cfgs)}/5 configs done")
        for c in cfgs:
            qa = c.get("qa_correctness")
            n = c.get("n_qa_evaluated", 0)
            qastr = f"{qa:.3f}(n={n})" if qa is not None else "-"
            print(f"  {c['config']:14} sess={c['session_recall@k']:.3f} ans={c['answer_recall@k']:.3f} qa={qastr}")
    else:
        print(f"{ds}: not started")
summary = os.path.join(root, "legal_full_summary.json")
print("summary:", "COMPLETE" if os.path.isfile(summary) else "pending")
PY

    echo ""
  } >> "$STATUS"
}

mkdir -p "$OUT_ROOT"
log_status

while true; do
  sleep "$INTERVAL"
  log_status
  if [ -f "$OUT_ROOT/legal_full_summary.json" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] experiment COMPLETE — monitor exiting" >> "$STATUS"
    exit 0
  fi
  if ! pgrep -f "run_legal_scaled.py.*results/legal_full" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] process stopped but summary missing — monitor keeps watching" >> "$STATUS"
  fi
done
