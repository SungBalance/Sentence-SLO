#!/usr/bin/env bash
# Single-GPU sequential sweep: max_num_seqs ∈ {32, 64, 128} × request_rate ∈
# {0, 4, 8, 16}, N trials each, 3 SSLO modes (baseline / sslo / sslo_adaptive).
#
# Run inside the sk-sslo-vllm container from /workspace/mlsys.
# Usage:
#   bash exp/sslo_test/run_full_sweep_seq.sh [num_runs=2]
set -euo pipefail

NUM_RUNS="${1:-2}"

MODEL="Qwen/Qwen3.5-35B-A3B"
NUM_PROMPTS=256
GENERATION_MAX_TOKENS=4096
BASE_SEED=42
BASE_OUTPUT="exp/sslo_test/output_sweep"
MAX_NUM_SEQS_VALUES=(32 64 128)
REQUEST_RATES=(0 4 8 16)
TENSOR_PARALLEL_SIZE=1

# Comma-separated mode list. Edit here to control which SSLO variants run.
# Choices: baseline, sslo, sslo_offload, sslo_adaptive, sslo_adaptive_offload
MODES="baseline,sslo,sslo_adaptive"

export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "  Sequential sweep on GPU 1"
echo "  model=${MODEL}  tp=${TENSOR_PARALLEL_SIZE}"
echo "  modes=${MODES}  num_runs=${NUM_RUNS}"
echo "  configs = ${#REQUEST_RATES[@]} rates × ${#MAX_NUM_SEQS_VALUES[@]} seqs"
echo "  generation_max_tokens=${GENERATION_MAX_TOKENS}"
echo "=========================================="

for RATE in "${REQUEST_RATES[@]}"; do
  for SEQS in "${MAX_NUM_SEQS_VALUES[@]}"; do
    OUT="${BASE_OUTPUT}/rate_${RATE}_seqs_${SEQS}"
    echo
    echo "----- rate=${RATE}  seqs=${SEQS}  -> ${OUT} -----"
    bash exp/sslo_test/run_repeat.sh \
      "${NUM_RUNS}" "${SEQS}" "${MODEL}" \
      "${NUM_PROMPTS}" "${GENERATION_MAX_TOKENS}" \
      "${OUT}" "${RATE}" "${BASE_SEED}" "${MODES}" "${TENSOR_PARALLEL_SIZE}"
  done
done

echo
echo "=========================================="
echo "  Sweep complete. Aggregating..."
echo "=========================================="
python3 exp/sslo_test/analysis/aggregate_sweep.py \
  --base-output "${BASE_OUTPUT}" \
  --num-runs "${NUM_RUNS}"
