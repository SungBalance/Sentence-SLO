#!/usr/bin/env bash
# Unified full sweep: chunk_unit × max_num_seqs × request_rate × N trials × modes.
#
# Usage:
#   run_sweep.sh [num_runs=3] [--label NAME]
#
# Env-var overrides (all optional; bare names — no _OVERRIDE suffix):
#   MODEL=Qwen/Qwen3.5-35B-A3B
#   NUM_PROMPTS=256
#   GENERATION_MAX_TOKENS=4096
#   MAX_MODEL_LEN=0                  (0 = auto, vLLM uses model HF config max)
#   TENSOR_PARALLEL_SIZE=1
#   GPU_MEMORY_UTILIZATION=0.95
#   SECONDS_PER_WORD=0.28
#   MODES=baseline,sslo,sslo_adaptive
#   CHUNK_UNITS="sentence"           (space-separated)
#   MAX_NUM_SEQS_VALUES="16 32 64 128"
#   REQUEST_RATES="0 4 16 32 64 128"
#   LABEL=                           (default-named subdir if empty)
#   PARALLEL=0                       (0=sequential, 4=split across 4 GPUs)
#
# --label NAME (or LABEL env) writes outputs to
#   exp/sslo_test/output_sweep/<NAME>/...
# so multiple comparison sweeps can sit side by side and feed one
# summary.csv at exp/sslo_test/output_sweep/summary.csv.
#
# Run inside the sk-sslo container from /workspace/mlsys.
set -euo pipefail

LABEL="${LABEL:-}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)    LABEL="$2"; shift 2 ;;
    --label=*)  LABEL="${1#--label=}"; shift ;;
    *)          POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

NUM_RUNS="${1:-3}"
PARALLEL="${PARALLEL:-0}"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
SECONDS_PER_WORD="${SECONDS_PER_WORD:-0.28}"
MODES="${MODES:-baseline,sslo,sslo_adaptive}"

read -ra CHUNK_UNITS_ARR        <<< "${CHUNK_UNITS:-sentence}"
read -ra MAX_NUM_SEQS_VALUES    <<< "${MAX_NUM_SEQS_VALUES:-16 32 64 128}"
read -ra REQUEST_RATES          <<< "${REQUEST_RATES:-0 4 16 32 64 128}"

# Manual GPU → rates assignment for PARALLEL=4. Goal: balance wallclock —
# rate=4 is slowest (256/4 = 64s arrival), rate=0 fastest (instant); pair
# so the four GPUs finish around the same time.
GPU_RATE_ASSIGNMENTS=(
  "4"          # GPU 0 (slowest alone)
  "16 128"     # GPU 1
  "32 64"      # GPU 2
  "0"          # GPU 3 (fastest alone)
)

SWEEP_ROOT="exp/sslo_test/output_sweep"
BASE_OUTPUT="${SWEEP_ROOT}/${LABEL:-default}"
BASE_SEED=42
GPU_READY_FREE_FRAC=0.95
GPU_READY_TIMEOUT_S=60
GPU_READY_SLEEP_S=5

# ---------------------------------------------------------------------------
# gpu_wait_ready <gpu_index> — poll nvidia-smi until free memory exceeds
# GPU_READY_FREE_FRAC * total or GPU_READY_TIMEOUT_S elapses.
# ---------------------------------------------------------------------------
gpu_wait_ready() {
  local gpu_index="${1:-1}"
  local deadline=$(( $(date +%s) + GPU_READY_TIMEOUT_S ))
  while true; do
    local line free total
    line=$(nvidia-smi --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits -i "${gpu_index}" 2>/dev/null | head -1) || true
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
# write_run_status <out_dir> <mode:rc> [<mode:rc> ...]  →  out_dir/run_status.json
# ---------------------------------------------------------------------------
write_run_status() {
  local out_dir="$1"; shift
  local body="" sep=""
  for pair in "$@"; do
    body+="${sep}\"${pair%%:*}\": ${pair##*:}"
    sep=", "
  done
  printf '{%s}\n' "${body}" > "${out_dir}/run_status.json"
}

# ---------------------------------------------------------------------------
# run_cell <unit> <gpu> <rate> <seqs> <run_index>
# ---------------------------------------------------------------------------
run_cell() {
  local unit="$1" gpu="$2" rate="$3" seqs="$4" run_index="$5"
  local out_dir="${BASE_OUTPUT}/${unit}/seqs_${seqs}/rate_${rate}/run_${run_index}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  IFS=',' read -ra modes_arr <<< "${MODES}"
  local status_pairs=() last_rc=0

  for mode in "${modes_arr[@]}"; do
    [[ "${mode}" != "baseline" ]] && gpu_wait_ready "${gpu:-1}"

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
      python3 exp/sslo_test/_consolidate_mode_outputs.py "${out_dir}" "${mode}"
    else
      echo "WARNING: mode=${mode} exited ${last_rc}; skipping remaining modes"
      break
    fi
  done

  write_run_status "${out_dir}" "${status_pairs[@]}"
  python3 exp/sslo_test/analyze.py \
    --output-dir "${out_dir}" \
    --max-num-seqs "${seqs}" \
    --chunk-unit "${unit}" \
    --request-rate "${rate}" \
    --model "${MODEL}" \
    --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${LABEL:+--label "${LABEL}"}
}

# ---------------------------------------------------------------------------
# run_subset <unit> <gpu> <rates...>
# ---------------------------------------------------------------------------
run_subset() {
  local unit="$1" gpu="$2"
  shift 2
  for rate in "$@"; do
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
  echo "ERROR: PARALLEL=${PARALLEL} is not supported (use 0 or 4)." >&2
  exit 1
fi

for unit in "${CHUNK_UNITS_ARR[@]}"; do
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
      wait "${pid}" || { echo "  WARNING: pid=${pid} exit=$?"; any_failed=1; }
    done
    [[ "${any_failed}" -ne 0 ]] && echo "  WARNING: aggregator will use whatever cells are present."
  fi
done

echo
echo "=========================================="
echo "  Sweep complete. Aggregating..."
echo "=========================================="
for unit in "${CHUNK_UNITS_ARR[@]}"; do
  echo
  echo "--- chunk_unit=${unit} ---"
  python3 exp/sslo_test/analysis/aggregate_sweep.py \
    --base-output "${BASE_OUTPUT}/${unit}" \
    --num-runs "${NUM_RUNS}" \
    --modes "${MODES}"
done

echo
echo "--- writing summary.csv across all labels under ${SWEEP_ROOT}/ ---"
python3 exp/sslo_test/analysis/sweep_summary_csv.py \
  --sweep-root "${SWEEP_ROOT}" \
  --output "${SWEEP_ROOT}/summary.csv"
