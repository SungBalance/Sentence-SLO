#!/usr/bin/env bash
# Full sweep on 2 GPUs in parallel.
#   Cells: max_num_seqs ∈ {32, 64, 128} × request_rate ∈ {0, 4, 8, 16}
#   GPU 0: rates {0, 16}     GPU 1: rates {4, 8}
# Split chosen to balance per-GPU runtime: high-pressure (burst rate=0) +
# low-pressure (rate=16) on one GPU, mid-pressure (4, 8) on the other.
#
# Output layout: ${BASE_OUTPUT}/rate_${rate}_seqs_${seqs}/run_{1..N}/seqs_${seqs}/
# Each GPU writes its own log to /tmp/sweep_gpu{0,1}.log.
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
RATES_GPU0=(0 16)
RATES_GPU1=(4 8)

run_subset() {
  local gpu="$1"
  shift
  local rates=("$@")
  for rate in "${rates[@]}"; do
    for seqs in "${MAX_NUM_SEQS_VALUES[@]}"; do
      local out="${BASE_OUTPUT}/rate_${rate}_seqs_${seqs}"
      echo "[GPU${gpu}] rate=${rate}  seqs=${seqs}  -> ${out}"
      CUDA_VISIBLE_DEVICES="${gpu}" bash exp/sslo_test/run_repeat.sh \
        "${NUM_RUNS}" "${seqs}" "${MODEL}" \
        "${NUM_PROMPTS}" "${GENERATION_MAX_TOKENS}" \
        "${out}" "${rate}" "${BASE_SEED}"
    done
  done
}

echo "=========================================="
echo "  Parallel sweep: GPU0 rates=${RATES_GPU0[*]}, GPU1 rates=${RATES_GPU1[*]}"
echo "  configs/GPU = ${#RATES_GPU0[@]} rates × ${#MAX_NUM_SEQS_VALUES[@]} seqs"
echo "=========================================="

run_subset 0 "${RATES_GPU0[@]}" > /tmp/sweep_gpu0.log 2>&1 &
PID0=$!
run_subset 1 "${RATES_GPU1[@]}" > /tmp/sweep_gpu1.log 2>&1 &
PID1=$!

echo "GPU0 pid=${PID0}, GPU1 pid=${PID1}"
wait ${PID0}
RC0=$?
wait ${PID1}
RC1=$?
echo "GPU0 exit=${RC0}, GPU1 exit=${RC1}"

if [[ "${RC0}" -ne 0 || "${RC1}" -ne 0 ]]; then
  echo "One or both GPU streams failed; aggregator will run on whatever is present."
fi

python3 exp/sslo_test/analysis/aggregate_sweep.py \
  --base-output "${BASE_OUTPUT}" \
  --num-runs "${NUM_RUNS}"
