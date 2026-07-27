#!/usr/bin/env bash
# 全面实验：brain_legal 配置，分别在 qwen3:14b 与 qwen3.6:27b 上各跑一遍。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
export CUDA_DEVICE="${CUDA_DEVICE:-1}"

COMMON=(
  python3 eval/legal/run_legal_scaled.py
  --datasets disc_law lawyer_llama
  --n_corpus 400 --n_queries 200 --n_qa 200 --n_train 1500
  --top_k 10 --seed 42
  --qa_configs brain_legal generic baseline
  --proj_beta 0.15 --lex_weight 0.12
)

echo "=== Full eval: qwen3:14b ==="
"${COMMON[@]}" --gen_model qwen3:14b \
  --output_root results/legal_optimized/qwen3_14b \
  2>&1 | tee results/legal_optimized/qwen3_14b_run.log

echo "=== Full eval: qwen3.6:27b ==="
"${COMMON[@]}" --gen_model qwen3.6:27b \
  --output_root results/legal_optimized/qwen3_6_27b \
  2>&1 | tee results/legal_optimized/qwen3_6_27b_run.log

echo "=== Done ==="
