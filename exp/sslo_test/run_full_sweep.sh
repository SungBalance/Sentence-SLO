#!/usr/bin/env bash
# Sweep experiment: max_num_seqs ∈ {32, 64, 128} × request_rate ∈ {0, 4, 8, 16}
# × N trials × 3 modes (baseline/sslo/sslo_adaptive).
#
# Output layout:
#   ${BASE_OUTPUT}/rate_${rate}_seqs_${seqs}/run_{1..N}/seqs_${seqs}/
#
# Run inside the sk-sslo-vllm container from /workspace/mlsys.
set -euo pipefail

NUM_RUNS="${1:-3}"

MODEL="Qwen/Qwen3-8B"
NUM_PROMPTS=256
GENERATION_MAX_TOKENS=512
BASE_SEED=42
BASE_OUTPUT="exp/sslo_test/output_sweep"

MAX_NUM_SEQS_VALUES=(32 64 128)
REQUEST_RATES=(0 4 8 16)

for RATE in "${REQUEST_RATES[@]}"; do
  for SEQS in "${MAX_NUM_SEQS_VALUES[@]}"; do
    OUTPUT_ROOT="${BASE_OUTPUT}/rate_${RATE}_seqs_${SEQS}"
    echo
    echo "=========================================="
    echo "  rate=${RATE} req/s   max_num_seqs=${SEQS}"
    echo "  → ${OUTPUT_ROOT}"
    echo "=========================================="
    bash exp/sslo_test/run_repeat.sh \
      "${NUM_RUNS}" "${SEQS}" "${MODEL}" \
      "${NUM_PROMPTS}" "${GENERATION_MAX_TOKENS}" \
      "${OUTPUT_ROOT}" "${RATE}" "${BASE_SEED}"
  done
done

echo
echo "=========================================="
echo "  Sweep complete. Aggregating..."
echo "=========================================="
python3 exp/sslo_test/analysis/aggregate_sweep.py \
  --base-output "${BASE_OUTPUT}" \
  --num-runs "${NUM_RUNS}"
