#!/usr/bin/env bash
# 法律投稿修订实验启动脚本（正确嵌入缓存 / FlatIP use_pq 修复）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EMBED_BACKEND="${EMBED_BACKEND:-ollama}"
export USE_EMBED_BATCH=1
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
export USE_EMBED_DISK_CACHE=1
export EMBED_DISK_CACHE_DIR="${EMBED_DISK_CACHE_DIR:-$ROOT/results/legal_revision/embed_cache.sqlite}"
export PYTHONPATH=".:eval:eval/legal:eval/frontier:${PYTHONPATH:-}"
mkdir -p results/legal_revision

MODE="${1:-revision}"
case "$MODE" in
  revision)
    nohup python3 -u eval/legal/run_revision_protocol.py \
      --datasets disc_law lawyer_llama \
      --n_corpus 400 --n_queries 300 --top_k 10 --seed 42 \
      --protocols paraphrase followup exact \
      --configs dense_flat dense_o2 parent_hydrate joint_qa session_max shuffled_o2 baseline_pq \
      --output_root results/legal_revision \
      > results/legal_revision/revision_protocol.log 2>&1 &
    echo $! | tee results/legal_revision/revision.pid
    ;;
  scale)
    nohup python3 -u eval/legal/run_scale_curve.py \
      --dataset disc_law --Ms 100 400 1600 --n_queries 100 --repeats 3 \
      --output results/legal_revision/scale_curve_disc_law.json \
      > results/legal_revision/scale_curve.log 2>&1 &
    echo $! | tee results/legal_revision/scale.pid
    ;;
  multiseed)
    nohup python3 -u eval/legal/run_multiseed.py \
      --datasets disc_law lawyer_llama --n_corpus 400 --n_queries 200 \
      --seeds 42 43 44 45 46 \
      > results/legal_revision/multiseed.log 2>&1 &
    echo $! | tee results/legal_revision/multiseed.pid
    ;;
  hardneg)
    nohup python3 -u eval/legal/run_hard_neg.py \
      --datasets disc_law lawyer_llama --n_corpus 400 \
      > results/legal_revision/hard_neg.log 2>&1 &
    echo $! | tee results/legal_revision/hardneg.pid
    ;;
  bm25)
    nohup python3 -u eval/legal/run_bm25_rrf.py \
      --datasets disc_law lawyer_llama --modes bm25_turn bm25_joint dense_rrf \
      > results/legal_revision/bm25_rrf.log 2>&1 &
    echo $! | tee results/legal_revision/bm25.pid
    ;;
  *)
    echo "usage: $0 {revision|scale|multiseed|hardneg|bm25}"
    exit 1
    ;;
esac
