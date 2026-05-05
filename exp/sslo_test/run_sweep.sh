#!/usr/bin/env bash
# Unified full sweep: chunk_unit × max_num_seqs × request_rate × N trials × modes.
#
# Usage:
#   run_sweep.sh [num_runs=3]
#
# Env vars:
#   PARALLEL=0    0=sequential (single GPU), 4=4-GPU parallel using
#                 GPU_RATE_ASSIGNMENTS (manual rate→GPU map below).
#
# Run inside the sk-sslo container from /workspace/mlsys.
set -euo pipefail

NUM_RUNS="${1:-3}"
PARALLEL="${PARALLEL:-0}"

MODEL="Qwen/Qwen3.5-35B-A3B"
NUM_PROMPTS=256
GENERATION_MAX_TOKENS=4096
MAX_MODEL_LEN=0   # 0 = auto: vLLM derives from the model's HF config max
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
SECONDS_PER_WORD=0.28
MODES="baseline,sslo,sslo_adaptive"

CHUNK_UNITS=(sentence)
MAX_NUM_SEQS_VALUES=(16 32 64 128)
REQUEST_RATES=(0 4 16 32 64 128)

# Manual GPU → rates assignment (one space-separated rate list per GPU).
# Goal: balance wallclock — rate=4 is the slowest (256/4 = 64s arrival),
# rate=0 the fastest (instant arrival), the rest scale by 1/rate. Pair so
# the four GPUs finish around the same time.
GPU_RATE_ASSIGNMENTS=(
  "4"          # GPU 0 (slowest alone)
  "16 128"     # GPU 1
  "32 64"      # GPU 2
  "0"          # GPU 3 (fastest alone)
)

BASE_OUTPUT="exp/sslo_test/output_sweep"
BASE_SEED=42
GPU_READY_FREE_FRAC=0.95
GPU_READY_TIMEOUT_S=60
GPU_READY_SLEEP_S=5

# ---------------------------------------------------------------------------
# gpu_wait_ready <gpu_index>
# Polls nvidia-smi until free memory > GPU_READY_FREE_FRAC * total, or timeout.
# ---------------------------------------------------------------------------
gpu_wait_ready() {
  local gpu_index="${1:-1}"
  local deadline=$(( $(date +%s) + GPU_READY_TIMEOUT_S ))
  while true; do
    local line
    line=$(nvidia-smi --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits -i "${gpu_index}" 2>/dev/null | head -1) || true
    local free total
    free=$(echo "${line}" | awk -F',' '{print $1+0}')
    total=$(echo "${line}" | awk -F',' '{print $2+0}')
    if [[ "${total}" -gt 0 ]]; then
      local ready
      ready=$(awk -v f="${free}" -v t="${total}" -v frac="${GPU_READY_FREE_FRAC}" \
        'BEGIN { print (f > t * frac) ? "1" : "0" }')
      if [[ "${ready}" == "1" ]]; then
        echo "[gpu_wait] GPU${gpu_index}: ${free}/${total} MiB free — ready"
        return 0
      fi
      echo "[gpu_wait] GPU${gpu_index}: ${free}/${total} MiB free — waiting..."
    fi
    if [[ $(date +%s) -ge ${deadline} ]]; then
      echo "WARNING: GPU${gpu_index} memory wait timed out; proceeding anyway"
      return 0
    fi
    sleep "${GPU_READY_SLEEP_S}"
  done
}

# ---------------------------------------------------------------------------
# run_cell <unit> <gpu> <rate> <seqs> <run_index>
# ---------------------------------------------------------------------------
run_cell() {
  local unit="$1"
  local gpu="$2"
  local rate="$3"
  local seqs="$4"
  local run_index="$5"

  local out_dir="${BASE_OUTPUT}/${unit}/seqs_${seqs}/rate_${rate}/run_${run_index}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  IFS=',' read -ra modes_arr <<< "${MODES}"
  # Accumulate "mode:status" pairs for the JSON writer below
  local status_pairs=()
  local last_rc=0
  local early_exit=0

  for mode in "${modes_arr[@]}"; do
    if [[ "${mode}" != "baseline" ]]; then
      gpu_wait_ready "${gpu:-1}"
    fi

    last_rc=0
    OUTPUT_DIR="${out_dir}" \
    CUDA_VISIBLE_DEVICES="${gpu:-1}" \
    CHUNK_UNIT="${unit}" \
    REQUEST_RATE="${rate}" \
    REQUEST_RATE_SEED=$(( BASE_SEED + run_index )) \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SECONDS_PER_WORD="${SECONDS_PER_WORD}" \
    bash exp/sslo_test/run_test.sh "${mode}" "${seqs}" "${MODEL}" \
      || last_rc=$?

    status_pairs+=("${mode}:${last_rc}")
    if [[ "${last_rc}" -eq 0 ]]; then
      _SSLO_OUT_DIR="${out_dir}" _SSLO_MODE="${mode}" python3 - <<'PYEOF'
import json, os
out_dir = os.environ['_SSLO_OUT_DIR']
mode = os.environ['_SSLO_MODE']
file_map = [
    (f'requests_{mode}.jsonl',     'requests.jsonl'),
    (f'chunks_{mode}.jsonl',       'chunks.jsonl'),
    (f'_stats_{mode}.jsonl',       'scheduler_stats.jsonl'),
    (f'_offload_log_{mode}.jsonl', 'offload_log.jsonl'),
]
for src_name, dst_name in file_map:
    src = os.path.join(out_dir, src_name)
    if not os.path.exists(src):
        continue
    dst = os.path.join(out_dir, dst_name)
    with open(src) as f_in, open(dst, 'a') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            merged = {'mode': mode, **row}
            f_out.write(json.dumps(merged) + '\n')
    os.remove(src)
PYEOF
    fi

    if [[ "${last_rc}" -ne 0 ]]; then
      echo "WARNING: mode=${mode} exited ${last_rc}; skipping remaining modes"
      early_exit=1
      break
    fi
  done

  # Write run_status.json: build a python dict literal from "mode:status" pairs
  local py_dict="{"
  local sep=""
  for pair in "${status_pairs[@]}"; do
    local k="${pair%%:*}"
    local v="${pair##*:}"
    py_dict="${py_dict}${sep}\"${k}\": ${v}"
    sep=", "
  done
  py_dict="${py_dict}}"
  python3 -c "import json; print(json.dumps(${py_dict}, indent=2))" \
    > "${out_dir}/run_status.json"

  python3 exp/sslo_test/analyze.py \
    --output-dir "${out_dir}" \
    --max-num-seqs "${seqs}"
}

# ---------------------------------------------------------------------------
# run_subset <unit> <gpu> <rates...>
# ---------------------------------------------------------------------------
run_subset() {
  local unit="$1"
  local gpu="$2"
  shift 2
  local rates=("$@")

  for rate in "${rates[@]}"; do
    for seqs in "${MAX_NUM_SEQS_VALUES[@]}"; do
      for (( i=1; i<=NUM_RUNS; i++ )); do
        echo
        echo "[${unit}${gpu:+ GPU${gpu}}] rate=${rate} seqs=${seqs} run=${i}/${NUM_RUNS}"
        run_cell "${unit}" "${gpu}" "${rate}" "${seqs}" "${i}"
      done
    done
  done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "${PARALLEL}" != "0" && "${PARALLEL}" != "4" ]]; then
  echo "ERROR: PARALLEL=${PARALLEL} is not supported. Supported values: 0, 4." >&2
  exit 1
fi

for unit in "${CHUNK_UNITS[@]}"; do
  echo
  echo "=========================================="
  echo "  chunk_unit=${unit}"
  echo "=========================================="

  if [[ "${PARALLEL}" == "0" ]]; then
    run_subset "${unit}" "" "${REQUEST_RATES[@]}"
  else
    pids=()
    for (( gpu_id=0; gpu_id<${#GPU_RATE_ASSIGNMENTS[@]}; gpu_id++ )); do
      read -ra gpu_rates <<< "${GPU_RATE_ASSIGNMENTS[gpu_id]}"
      echo "  GPU${gpu_id} rates=${gpu_rates[*]:-<none>}"
      if [[ ${#gpu_rates[@]} -gt 0 ]]; then
        run_subset "${unit}" "${gpu_id}" "${gpu_rates[@]}" \
          > "/tmp/sweep_${unit}_gpu${gpu_id}.log" 2>&1 &
        pids+=($!)
      fi
    done
    echo "  pids: ${pids[*]:-<none>}"
    any_failed=0
    for pid in "${pids[@]}"; do
      wait "${pid}"; rc=$?
      if [[ "${rc}" -ne 0 ]]; then
        echo "  WARNING: background sweep pid=${pid} exit=${rc}"
        any_failed=1
      fi
    done
    if [[ "${any_failed}" -ne 0 ]]; then
      echo "  WARNING: one or more GPU streams failed; aggregator will run on whatever is present."
    fi
  fi
done

echo
echo "=========================================="
echo "  Sweep complete. Aggregating..."
echo "=========================================="
for unit in "${CHUNK_UNITS[@]}"; do
  echo
  echo "--- chunk_unit=${unit} ---"
  python3 exp/sslo_test/analysis/aggregate_sweep.py \
    --base-output "${BASE_OUTPUT}/${unit}" \
    --num-runs "${NUM_RUNS}" \
    --modes "${MODES}"
done
