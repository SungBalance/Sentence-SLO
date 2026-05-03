#!/usr/bin/env bash
# Run one SSLO test config N times with separate output dirs, for noise/error
# margin estimation. Usage:
#
#   run_repeat.sh <num_runs> <max_num_seqs> <model> [num_prompts=256] \
#                 [generation_max_tokens=512] [output_root=exp/sslo_test/output]
#
# Each run writes to ${output_root}/run_{i}/seqs_${max_num_seqs}/.
# Run inside the sk-sslo-vllm container.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <num_runs> <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output]" >&2
  exit 2
fi

NUM_RUNS="$1"
MAX_NUM_SEQS="$2"
MODEL="$3"
NUM_PROMPTS="${4:-256}"
GENERATION_MAX_TOKENS="${5:-512}"
OUTPUT_ROOT="${6:-exp/sslo_test/output}"

for ((i=1; i<=NUM_RUNS; i++)); do
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/run_${i}"
  echo "===== run ${i}/${NUM_RUNS} → ${RUN_OUTPUT_DIR}/seqs_${MAX_NUM_SEQS} ====="
  rm -rf "${RUN_OUTPUT_DIR}/seqs_${MAX_NUM_SEQS}"
  bash exp/sslo_test/run_single.sh \
    "${MAX_NUM_SEQS}" "${MODEL}" \
    "${NUM_PROMPTS}" "${GENERATION_MAX_TOKENS}" "${RUN_OUTPUT_DIR}"
done

python3 exp/sslo_test/analysis/aggregate_repeats.py \
  --output-root "${OUTPUT_ROOT}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --num-runs "${NUM_RUNS}"
