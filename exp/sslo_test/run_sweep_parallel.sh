#!/usr/bin/env bash
# Launch run_sweep.sh on 4 GPUs in parallel, one request rate per GPU,
# and aggregate when all complete.
#
# Run from exp/sslo_test/ (paths are relative to this folder).
#
# Usage:
#   bash run_sweep_parallel.sh
#
# Env vars:
#   NUM_RUNS=3  number of trials per cell (forwarded to run_sweep.sh)
set -euo pipefail

NUM_RUNS="${NUM_RUNS:-3}"
export NUM_RUNS

CHUNK_UNITS=(sentence paragraph)
MODES="baseline,sslo,sslo_adaptive"
BASE_OUTPUT="output_sweep"
REQUEST_RATES=(0 2 8 16)

pids=()
for gpu_id in 0 1 2 3; do
  rate="${REQUEST_RATES[${gpu_id}]}"
  bash ./run_sweep.sh "${gpu_id}" "${rate}" \
    > "/tmp/sweep_gpu${gpu_id}.log" 2>&1 &
  pids+=($!)
done
echo "pids: ${pids[*]}"

any_failed=0
for pid in "${pids[@]}"; do
  wait "${pid}"; rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "WARNING: background sweep pid=${pid} exit=${rc}"
    any_failed=1
  fi
done
if [[ "${any_failed}" -ne 0 ]]; then
  echo "WARNING: one or more GPU streams failed; aggregator will run on whatever is present."
fi

echo
echo "=========================================="
echo "  Sweep complete. Aggregating..."
echo "=========================================="
for unit in "${CHUNK_UNITS[@]}"; do
  echo
  echo "--- chunk_unit=${unit} ---"
  python3 ./analysis/aggregate_sweep.py \
    --base-output "${BASE_OUTPUT}/${unit}" \
    --num-runs "${NUM_RUNS}" \
    --modes "${MODES}"
done
