#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

# Run this launcher on the host; all project commands execute in containers.
CONTAINER_REPO="/workspace/mlsys"
BENCH_CONTAINER="sk-sslo"
ANALYSIS_CONTAINER="sk-sslo"
CUDA_VISIBLE_DEVICES_VALUE="0"

EXP_ROOT="${CONTAINER_REPO}/exp"
OUTPUT_ROOT="${EXP_ROOT}/output"

# Benchmark options.
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.95
WARMUP_REQUESTS=1
REQUEST_RATE="inf"
REQUEST_BURSTINESS=1.0

# Dataset options. DATASET_PATH is used for the output slug when it is set.
DATASET_NAME="hf"
DATASET_PATH="Aeala/ShareGPT_Vicuna_unfiltered"

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

copy_directory_files() {
  local source_dir="$1"
  local target_dir="$2"

  docker exec "${ANALYSIS_CONTAINER}" bash -lc '
    set -euo pipefail
    source_dir="$1"
    target_dir="$2"
    mkdir -p "${target_dir}"
    shopt -s nullglob
    files=("${source_dir}"/*)
    if ((${#files[@]})); then
      cp -f "${files[@]}" "${target_dir}/"
    fi
  ' bash "${source_dir}" "${target_dir}"
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

# Build dataset arguments once so every model run sees the same inputs.
DATASET_SLUG_SOURCE="${DATASET_PATH:-${DATASET_NAME}}"
DATASET_SLUG="$(slugify "${DATASET_SLUG_SOURCE}")"
DATASET_ARGS=(--dataset-name "${DATASET_NAME}")
if [[ -n "${DATASET_PATH}" ]]; then
  DATASET_ARGS+=(--dataset-path "${DATASET_PATH}")
fi

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(slugify "${MODEL}")"
  MODEL_DATASET_ROOT="${OUTPUT_ROOT}/${MODEL_SLUG}/${DATASET_SLUG}"
  SOURCE_MODE="${SLACK_MODES[0]}"
  CHUNK_OUTPUT_ARGS=()

  for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
    CHUNK_OUTPUT_ROOT="${MODEL_DATASET_ROOT}/${CHUNK_UNIT}"
    SOURCE_TEXT_DIR="${CHUNK_OUTPUT_ROOT}/${SOURCE_MODE}/text_outputs"
    run_in_container "${ANALYSIS_CONTAINER}" mkdir -p "${SOURCE_TEXT_DIR}"
    CHUNK_OUTPUT_ARGS+=(--chunk-output-group "${CHUNK_UNIT}=${SOURCE_TEXT_DIR}")
  done

  # Stage 1: one LLM inference pass, with multiple chunk collectors attached.
  run_in_container "${BENCH_CONTAINER}" python3 "${EXP_ROOT}/benchmark.py" \
    --model "${MODEL}" \
    "${DATASET_ARGS[@]}" \
    --num-prompts "${NUM_PROMPTS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --request-rate "${REQUEST_RATE}" \
    --request-burstiness "${REQUEST_BURSTINESS}" \
    "${CHUNK_OUTPUT_ARGS[@]}" \
    --seconds-per-word "${SECONDS_PER_WORD}"

  for CHUNK_UNIT in "${CHUNK_UNITS[@]}"; do
    CHUNK_OUTPUT_ROOT="${MODEL_DATASET_ROOT}/${CHUNK_UNIT}"
    SOURCE_TEXT_DIR="${CHUNK_OUTPUT_ROOT}/${SOURCE_MODE}/text_outputs"
    for SLACK_MODE in "${SLACK_MODES[@]}"; do
      MODE_ROOT="${CHUNK_OUTPUT_ROOT}/${SLACK_MODE}"
      TEXT_DIR="${MODE_ROOT}/text_outputs"
      RESULTS_DIR="${MODE_ROOT}/results"
      run_in_container "${ANALYSIS_CONTAINER}" mkdir -p \
        "${TEXT_DIR}" "${RESULTS_DIR}"

      if [[ "${SLACK_MODE}" != "${SOURCE_MODE}" ]]; then
        # Reuse source text outputs so only mode-specific analysis is repeated.
        copy_directory_files "${SOURCE_TEXT_DIR}" "${TEXT_DIR}"
      fi

      # Stage 2: compute and plot one human reading slack mode.
      run_in_container "${ANALYSIS_CONTAINER}" python3 "${EXP_ROOT}/analyze_results.py" \
        --analysis-target human \
        --text-output-dir "${TEXT_DIR}" \
        --slack-mode "${SLACK_MODE}" \
        --output-dir "${RESULTS_DIR}"
    done
  done
done
