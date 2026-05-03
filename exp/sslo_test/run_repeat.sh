#!/usr/bin/env bash
# Run one SSLO test config N times with separate output dirs, for noise/error
# margin estimation. Usage:
#
#   run_repeat.sh <num_runs> <max_num_seqs> <model> [num_prompts=256] \
#                 [generation_max_tokens=512] [output_root=exp/sslo_test/output] \
#                 [request_rate=0] [base_seed=42]
#
# Each run writes to ${output_root}/run_{i}/seqs_${max_num_seqs}/.
# request_rate is forwarded to run_single.sh; the seed is base_seed + i so each
# run gets a different Poisson sample (when request_rate > 0).
# Run inside the sk-sslo-vllm container.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <num_runs> <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output] [request_rate=0] [base_seed=42]" >&2
  exit 2
fi

NUM_RUNS="$1"
MAX_NUM_SEQS="$2"
MODEL="$3"
NUM_PROMPTS="${4:-256}"
GENERATION_MAX_TOKENS="${5:-512}"
OUTPUT_ROOT="${6:-exp/sslo_test/output}"
REQUEST_RATE="${7:-0}"
BASE_SEED="${8:-42}"

for ((i=1; i<=NUM_RUNS; i++)); do
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/run_${i}"
  RUN_SEED=$((BASE_SEED + i))
  echo "===== run ${i}/${NUM_RUNS} → ${RUN_OUTPUT_DIR}/seqs_${MAX_NUM_SEQS} (rate=${REQUEST_RATE} seed=${RUN_SEED}) ====="
  rm -rf "${RUN_OUTPUT_DIR}/seqs_${MAX_NUM_SEQS}"
  bash exp/sslo_test/run_single.sh \
    "${MAX_NUM_SEQS}" "${MODEL}" \
    "${NUM_PROMPTS}" "${GENERATION_MAX_TOKENS}" "${RUN_OUTPUT_DIR}" \
    "${REQUEST_RATE}" "${RUN_SEED}"
done

python3 exp/sslo_test/analysis/aggregate_repeats.py \
  --output-root "${OUTPUT_ROOT}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --num-runs "${NUM_RUNS}"
