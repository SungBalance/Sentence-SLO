#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Run this launcher on the host; all project commands execute inside containers.
CONTAINER_REPO="/workspace/mlsys"
BENCH_CONTAINER="sk-sslo-vllm"
ANALYSIS_CONTAINER="sk-sslo-vllm"

EXP_ROOT="${CONTAINER_REPO}/exp"
OUTPUT_ROOT="${EXP_ROOT}/measure_internal_slack/outputs"

# Benchmark options.
NUM_PROMPTS=256
MAX_MODEL_LEN=8192
GENERATION_MAX_TOKENS=8192
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
WARMUP_REQUESTS=1
SECONDS_PER_WORD=0.28

# Batch size sweep.
MAX_NUM_SEQS_VALUES=(16 32 64 96 128 160 192 224 256)

# Dataset options.
DATASET_NAME="koala"

# Chunk units — separate inference passes (one per unit, same GPU).
CHUNK_UNITS=(sentence paragraph)

# Two models, one per GPU, run in parallel per batch size.
MODELS=(
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-27B"
)
MODEL_GPUS=(0 1)

slugify() {
  local value="$1"
  value="${value//\//__}"
  value="${value// /_}"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '_')"
  printf '%s' "${value}"
}

run_on_gpu() {
  local container="$1"
  local gpu="$2"
  shift 2
  docker exec \
    -e CUDA_VISIBLE_DEVICES="${gpu}" \
    -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    "${container}" bash -lc '
    set -euo pipefail
    cd /workspace/mlsys
    export HF_HOME=/cache
    export HF_HUB_CACHE=/cache/hub
    "$@"
  ' bash "$@"
}

run_in_container() {
  local container="$1"
  shift
  docker exec "${container}" bash -lc '
    set -euo pipefail
    cd /workspace/mlsys
    export HF_HOME=/cache
    export HF_HUB_CACHE=/cache/hub
    "$@"
  ' bash "$@"
}

require_container() {
  local container="$1"
  if ! docker inspect "${container}" >/dev/null 2>&1; then
    echo "Required container is not available: ${container}" >&2
    exit 1
  fi
}

require_container "${BENCH_CONTAINER}"

# Delete previous output so results are clean.
run_in_container "${BENCH_CONTAINER}" rm -rf "${OUTPUT_ROOT}"

DATASET_SLUG="$(slugify "${DATASET_NAME}")"


# Run all batch sizes sequentially; within each batch size run models in parallel.
for MAX_NUM_SEQS in "${MAX_NUM_SEQS_VALUES[@]}"; do
  echo "=== Batch size: max_num_seqs=${MAX_NUM_SEQS} ==="
  PIDS=()

  for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    GPU="${MODEL_GPUS[$i]}"
    MODEL_SLUG="$(slugify "${MODEL}")"

    # Run both chunk units sequentially on this GPU (background per model).
    (
      for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
        CHUNK_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}/${DATASET_SLUG}/batch_${MAX_NUM_SEQS}/${CHUNK_UNIT}"
        run_on_gpu "${BENCH_CONTAINER}" "${GPU}" \
          python3 "${EXP_ROOT}/measure_internal_slack/benchmark.py" \
          --model "${MODEL}" \
          --dataset-name "${DATASET_NAME}" \
          --num-prompts "${NUM_PROMPTS}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
          --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          --warmup-requests "${WARMUP_REQUESTS}" \
          --chunk-unit "${CHUNK_UNIT}" \
          --seconds-per-word "${SECONDS_PER_WORD}" \
          --output-dir "${CHUNK_DIR}"
      done
    ) &
    PIDS+=($!)
  done

  for PID in "${PIDS[@]}"; do
    wait "${PID}" || {
      echo "Inference failed (max_num_seqs=${MAX_NUM_SEQS}, PID=${PID})" >&2
      exit 1
    }
  done
done

# Stage 2: cross-cutting distribution analysis.
ANALYSIS_OUTPUT="${OUTPUT_ROOT}/analysis"
run_in_container "${ANALYSIS_CONTAINER}" \
  python3 "${EXP_ROOT}/measure_internal_slack/analyze_slack_dist.py" \
  --data-dir "${OUTPUT_ROOT}" \
  --output-dir "${ANALYSIS_OUTPUT}"

echo "=== Experiment complete. Results in ${OUTPUT_ROOT} ==="
