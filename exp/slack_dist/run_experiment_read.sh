#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

# Run this launcher on the host; all project commands execute in containers.
CONTAINER_REPO="/workspace/mlsys"
BENCH_CONTAINER="sk-sslo"
ANALYSIS_CONTAINER="sk-sslo"
CUDA_VISIBLE_DEVICES_VALUE="0,1"

EXP_ROOT="${CONTAINER_REPO}/exp"
OUTPUT_ROOT="${EXP_ROOT}/output"

# Benchmark options.
NUM_PROMPTS=256
MAX_MODEL_LEN=8192
GENERATION_MAX_TOKENS=8192
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
WARMUP_REQUESTS=1
REQUEST_RATE="inf"
REQUEST_BURSTINESS=1.0
MAX_NUM_SEQS_VALUES=(
  "16"
  "32"
  "64"
  "128"
)

# Dataset options.
DATASET_NAME="HuggingFaceH4/Koala-test-set"
DATASET_SPLIT="test"

# Human reading speed used for slack analysis.
SECONDS_PER_WORD=0.28

# Chunk groups to compare. Paragraph chunks use blank-line boundaries.
CHUNK_UNITS=(
  "sentence"
  "paragraph"
)

SLACK_MODES=(
  "previous_chunk"
  "cumulative"
)

MODELS=(
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-27B"
)

slugify() {
  local value="$1"
  value="${value//\//__}"
  value="${value// /_}"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '_')"
  printf '%s' "${value}"
}

run_in_container() {
  local container="$1"
  shift

  docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${container}" bash -lc '
    set -euo pipefail
    cd /workspace/mlsys
    export HF_HOME=/cache
    export HF_HUB_CACHE=/cache/hub
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
require_container "${ANALYSIS_CONTAINER}"

delete_model_dataset_root() {
  local model_dataset_root="$1"

  docker exec "${ANALYSIS_CONTAINER}" bash -lc '
    set -euo pipefail
    rm -rf "$1"
  ' bash "${model_dataset_root}"
}

# Build dataset arguments once so every model run sees the same inputs.
DATASET_SLUG="$(slugify "${DATASET_NAME}")"
DATASET_ARGS=(--dataset-name "${DATASET_NAME}" --dataset-split "${DATASET_SPLIT}")

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(slugify "${MODEL}")"
  MODEL_DATASET_ROOT="${OUTPUT_ROOT}/${MODEL_SLUG}/${DATASET_SLUG}"
  delete_model_dataset_root "${MODEL_DATASET_ROOT}"

  for MAX_NUM_SEQS in "${MAX_NUM_SEQS_VALUES[@]}"; do
    BATCH_ROOT="${MODEL_DATASET_ROOT}/batch_${MAX_NUM_SEQS}"
    CHUNK_OUTPUT_ARGS=()

    for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
      CHUNK_OUTPUT_ROOT="${BATCH_ROOT}/${CHUNK_UNIT}"
      TEXT_DIR="${CHUNK_OUTPUT_ROOT}/text_outputs"
      run_in_container "${ANALYSIS_CONTAINER}" mkdir -p "${TEXT_DIR}"
      CHUNK_OUTPUT_ARGS+=(--chunk-output-group "${CHUNK_UNIT}=${TEXT_DIR}")
    done

    # Stage 1: one LLM inference pass per max_num_seqs, with both chunk collectors.
    run_in_container "${BENCH_CONTAINER}" python3 "${EXP_ROOT}/benchmark.py" \
      --model "${MODEL}" \
      "${DATASET_ARGS[@]}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --warmup-requests "${WARMUP_REQUESTS}" \
      --request-rate "${REQUEST_RATE}" \
      --request-burstiness "${REQUEST_BURSTINESS}" \
      "${CHUNK_OUTPUT_ARGS[@]}" \
      --seconds-per-word "${SECONDS_PER_WORD}"

    for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
      CHUNK_OUTPUT_ROOT="${BATCH_ROOT}/${CHUNK_UNIT}"
      TEXT_DIR="${CHUNK_OUTPUT_ROOT}/text_outputs"
      for SLACK_MODE in "${SLACK_MODES[@]}"; do
        MODE_ROOT="${CHUNK_OUTPUT_ROOT}/${SLACK_MODE}"
        RESULTS_DIR="${MODE_ROOT}/results"
        run_in_container "${ANALYSIS_CONTAINER}" mkdir -p "${RESULTS_DIR}"

        # Stage 2: compute one human reading slack mode for this batch setting.
        run_in_container "${ANALYSIS_CONTAINER}" python3 "${EXP_ROOT}/analyze_results.py" \
          --analysis-target human \
          --text-output-dir "${TEXT_DIR}" \
          --slack-mode "${SLACK_MODE}" \
          --output-dir "${RESULTS_DIR}"
      done
    done
  done

  for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
    COMPARISON_ROOT="${MODEL_DATASET_ROOT}/batch_size_comparison/${CHUNK_UNIT}"
    for SLACK_MODE in "${SLACK_MODES[@]}"; do
      COMPARISON_DIR="${COMPARISON_ROOT}/${SLACK_MODE}"
      BATCH_RESULT_ARGS=()
      for MAX_NUM_SEQS in "${MAX_NUM_SEQS_VALUES[@]}"; do
        BATCH_RESULT_ARGS+=(
          --batch-result
          "${MAX_NUM_SEQS}=${MODEL_DATASET_ROOT}/batch_${MAX_NUM_SEQS}/${CHUNK_UNIT}/${SLACK_MODE}/results"
        )
      done

      # Stage 3: compare one chunk unit + slack mode across max_num_seqs runs.
      run_in_container "${ANALYSIS_CONTAINER}" python3 "${EXP_ROOT}/compare_read_batch_results.py" \
        --slack-mode "${SLACK_MODE}" \
        --output-dir "${COMPARISON_DIR}" \
        "${BATCH_RESULT_ARGS[@]}"
    done
  done
done
