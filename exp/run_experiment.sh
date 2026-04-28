#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

# Run this launcher on the host; all project commands execute in containers.
CONTAINER_REPO="/workspace/mlsys"
BENCH_CONTAINER="sk-sslo"
TTS_CONTAINER="sk-sslo-omni"
ANALYSIS_CONTAINER="sk-sslo"
CUDA_VISIBLE_DEVICES_VALUE="0"

EXP_ROOT="${CONTAINER_REPO}/exp"
OUTPUT_ROOT="${EXP_ROOT}/output"

# Benchmark options.
NUM_PROMPTS=48
MAX_MODEL_LEN=4096
GENERATION_MAX_TOKENS=1024
GPU_MEMORY_UTILIZATION=0.95
WARMUP_REQUESTS=1
REQUEST_RATE="inf"
REQUEST_BURSTINESS=1.0
MAX_CHUNKS_PER_REQUEST=48

# Dataset options. DATASET_PATH is used for the output slug when it is set.
DATASET_NAME="hf"
DATASET_PATH="Aeala/ShareGPT_Vicuna_unfiltered"

# Human reading speed used for slack analysis.
SECONDS_PER_WORD=0.28

# TTS options.
TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_BATCH_SIZE=128
TTS_LANGUAGE="English"
TTS_SPEAKER="Ryan"
TTS_INSTRUCT=""
TTS_MAX_NEW_TOKENS=2048
TTS_MAX_SEGMENT_CHARS=240
TTS_MAX_RETRIES=4
TTS_STAGE_CONFIG_TEMPLATE="${EXP_ROOT}/configs/qwen3_tts_omni_batch.yaml"
TTS_STAGE_CONFIG="${OUTPUT_ROOT}/_runtime/qwen3_tts_omni.yaml"
TTS_DEVICES="0"
TTS_TENSOR_PARALLEL_SIZE=1
TTS_GPU_MEMORY_UTILIZATION=0.95
TTS_CODE2WAV_GPU_MEMORY_UTILIZATION=0.20
MAX_NEW_AUDIO_CHUNKS=""

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
require_container "${TTS_CONTAINER}"
require_container "${ANALYSIS_CONTAINER}"

render_tts_stage_config() {
  docker exec "${ANALYSIS_CONTAINER}" bash -lc '
    set -euo pipefail
    template_path="$1"
    output_path="$2"
    batch_size="$3"
    devices="$4"
    tensor_parallel_size="$5"
    total_gpu_memory_utilization="$6"
    code2wav_gpu_memory_utilization="$7"
    python3 - "${template_path}" "${output_path}" "${batch_size}" \
      "${devices}" "${tensor_parallel_size}" \
      "${total_gpu_memory_utilization}" "${code2wav_gpu_memory_utilization}" <<'"'"'PY'"'"'
from pathlib import Path
import sys

import yaml

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
batch_size = int(sys.argv[3])
devices = sys.argv[4]
tensor_parallel_size = int(sys.argv[5])
total_gpu_memory_utilization = float(sys.argv[6])
code2wav_gpu_memory_utilization = float(sys.argv[7])

with template_path.open() as f:
    config = yaml.safe_load(f)

stages = config["stage_args"]
if len(stages) != 2:
    raise ValueError("Expected exactly two TTS stages.")

for stage in stages:
    stage["runtime"]["devices"] = devices
    stage["engine_args"]["tensor_parallel_size"] = tensor_parallel_size
    stage["engine_args"]["max_num_seqs"] = batch_size

same_devices = stages[0]["runtime"]["devices"] == stages[1]["runtime"]["devices"]
if same_devices:
    stage1_utilization = code2wav_gpu_memory_utilization
    stage0_utilization = total_gpu_memory_utilization - stage1_utilization
    if stage0_utilization <= 0:
        raise ValueError("TTS stage 0 gpu_memory_utilization must be positive.")
else:
    stage0_utilization = total_gpu_memory_utilization
    stage1_utilization = total_gpu_memory_utilization

stages[0]["engine_args"]["gpu_memory_utilization"] = round(stage0_utilization, 4)
stages[1]["engine_args"]["gpu_memory_utilization"] = round(stage1_utilization, 4)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(
    "Rendered TTS config: "
    f"devices={devices}, "
    f"tensor_parallel_size={tensor_parallel_size}, "
    f"batch/max_num_seqs={batch_size}, "
    f"stage0_gpu_memory_utilization={stage0_utilization:.4f}, "
    f"stage1_gpu_memory_utilization={stage1_utilization:.4f}"
)
PY
  ' bash "${TTS_STAGE_CONFIG_TEMPLATE}" "${TTS_STAGE_CONFIG}" \
    "${TTS_BATCH_SIZE}" "${TTS_DEVICES}" "${TTS_TENSOR_PARALLEL_SIZE}" \
    "${TTS_GPU_MEMORY_UTILIZATION}" "${TTS_CODE2WAV_GPU_MEMORY_UTILIZATION}"
}

render_tts_stage_config

# Build dataset arguments once so every model run sees the same inputs.
DATASET_SLUG_SOURCE="${DATASET_PATH:-${DATASET_NAME}}"
DATASET_SLUG="$(slugify "${DATASET_SLUG_SOURCE}")"
DATASET_ARGS=(--dataset-name "${DATASET_NAME}")
if [[ -n "${DATASET_PATH}" ]]; then
  DATASET_ARGS+=(--dataset-path "${DATASET_PATH}")
fi

for MODEL in "${MODELS[@]}"; do
  MODEL_SLUG="$(slugify "${MODEL}")"
  MODEL_OUTPUT_ROOT="${OUTPUT_ROOT}/${MODEL_SLUG}/${DATASET_SLUG}"
  SOURCE_MODE="${SLACK_MODES[0]}"
  SOURCE_TEXT_DIR="${MODEL_OUTPUT_ROOT}/${SOURCE_MODE}/text_outputs"
  SOURCE_AUDIO_DIR="${MODEL_OUTPUT_ROOT}/${SOURCE_MODE}/audio_durations"
  run_in_container "${ANALYSIS_CONTAINER}" mkdir -p \
    "${SOURCE_TEXT_DIR}" "${SOURCE_AUDIO_DIR}"

  # Stage 1: LLM inference and per-chunk text timeline.
  run_in_container "${BENCH_CONTAINER}" python3 "${EXP_ROOT}/benchmark.py" \
    --model "${MODEL}" \
    "${DATASET_ARGS[@]}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --request-rate "${REQUEST_RATE}" \
    --request-burstiness "${REQUEST_BURSTINESS}" \
    --max-chunks-per-request "${MAX_CHUNKS_PER_REQUEST}" \
    --output-dir "${SOURCE_TEXT_DIR}" \
    --seconds-per-word "${SECONDS_PER_WORD}"

  EXTRA_ARGS=()
  ANALYZE_EXTRA_ARGS=()
  if [[ -n "${MAX_NEW_AUDIO_CHUNKS}" ]]; then
    EXTRA_ARGS+=(--max-chunks "${MAX_NEW_AUDIO_CHUNKS}" --allow-partial-output)
    ANALYZE_EXTRA_ARGS+=(--allow-missing-durations)
  fi

  # Stage 2: TTS duration generation for the chunk rows.
  run_in_container "${TTS_CONTAINER}" python3 "${EXP_ROOT}/audio_duration.py" \
    --input-dir "${SOURCE_TEXT_DIR}" \
    --output-dir "${SOURCE_AUDIO_DIR}" \
    --stage-configs-path "${TTS_STAGE_CONFIG}" \
    --tts-model "${TTS_MODEL}" \
    --batch-size "${TTS_BATCH_SIZE}" \
    --language "${TTS_LANGUAGE}" \
    --speaker "${TTS_SPEAKER}" \
    --instruct "${TTS_INSTRUCT}" \
    --max-new-tokens "${TTS_MAX_NEW_TOKENS}" \
    --tts-max-segment-chars "${TTS_MAX_SEGMENT_CHARS}" \
    --tts-max-retries "${TTS_MAX_RETRIES}" \
    "${EXTRA_ARGS[@]}"

  for SLACK_MODE in "${SLACK_MODES[@]}"; do
    MODE_ROOT="${MODEL_OUTPUT_ROOT}/${SLACK_MODE}"
    TEXT_DIR="${MODE_ROOT}/text_outputs"
    AUDIO_DIR="${MODE_ROOT}/audio_durations"
    RESULTS_DIR="${MODE_ROOT}/results"
    run_in_container "${ANALYSIS_CONTAINER}" mkdir -p \
      "${TEXT_DIR}" "${AUDIO_DIR}" "${RESULTS_DIR}"

    if [[ "${SLACK_MODE}" != "${SOURCE_MODE}" ]]; then
      # Reuse source stage outputs so only analysis is repeated per mode.
      copy_directory_files "${SOURCE_TEXT_DIR}" "${TEXT_DIR}"
      copy_directory_files "${SOURCE_AUDIO_DIR}" "${AUDIO_DIR}"
    fi

    # Stage 3: compute and plot one slack mode.
    run_in_container "${ANALYSIS_CONTAINER}" python3 "${EXP_ROOT}/analyze_results.py" \
      --text-output-dir "${TEXT_DIR}" \
      --audio-duration-dir "${AUDIO_DIR}" \
      --slack-mode "${SLACK_MODE}" \
      --output-dir "${RESULTS_DIR}" \
      "${ANALYZE_EXTRA_ARGS[@]}"
  done
done
