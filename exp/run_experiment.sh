#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

# All paths below are container-side paths.
cd /workspace/mlsys

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${EXP_ROOT}/output"

# Benchmark options.
NUM_PROMPTS=1024
MAX_MODEL_LEN=4096
GENERATION_MAX_TOKENS=1024
GPU_MEMORY_UTILIZATION=0.95

# Dataset options. DATASET_PATH is used for the output slug when it is set.
DATASET_NAME="hf"
DATASET_PATH="Aeala/ShareGPT_Vicuna_unfiltered"

# Human reading speed used for slack analysis.
SECONDS_PER_WORD=0.28

# TTS options.
TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_BATCH_SIZE=4
TTS_LANGUAGE="English"
TTS_SPEAKER="Ryan"
TTS_INSTRUCT=""
TTS_MAX_NEW_TOKENS=2048
TTS_MAX_SEGMENT_CHARS=240
TTS_STAGE_CONFIG="${EXP_ROOT}/configs/qwen3_tts_omni_batch_tp2.yaml"
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

copy_named_files() {
  local source_dir="$1"
  local target_dir="$2"
  shift 2

  mkdir -p "${target_dir}"
  for filename in "$@"; do
    cp -f "${source_dir}/${filename}" "${target_dir}/${filename}"
  done
}

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
  mkdir -p "${SOURCE_TEXT_DIR}" "${SOURCE_AUDIO_DIR}"

  SUMMARY_JSON="${SOURCE_TEXT_DIR}/summary.json"
  REQUESTS_JSONL="${SOURCE_TEXT_DIR}/requests.jsonl"
  CHUNKS_JSONL="${SOURCE_TEXT_DIR}/chunks.jsonl"
  CHUNKS_CSV="${SOURCE_TEXT_DIR}/chunks.csv"

  # Stage 1: LLM inference and per-chunk text timeline.
  python3 "${EXP_ROOT}/benchmark.py" \
    --backend vllm \
    --async-engine \
    --apply-chat-template \
    --model "${MODEL}" \
    "${DATASET_ARGS[@]}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
    --output-json "${SUMMARY_JSON}" \
    --slack-output-jsonl "${REQUESTS_JSONL}" \
    --chunk-output-jsonl "${CHUNKS_JSONL}" \
    --chunk-output-csv "${CHUNKS_CSV}" \
    --seconds-per-word "${SECONDS_PER_WORD}"

  DURATIONS_JSONL="${SOURCE_AUDIO_DIR}/durations.jsonl"
  DURATIONS_CSV="${SOURCE_AUDIO_DIR}/durations.csv"
  DURATION_CACHE_JSONL="${SOURCE_AUDIO_DIR}/duration_cache.jsonl"

  EXTRA_ARGS=()
  ANALYZE_EXTRA_ARGS=()
  if [[ -n "${MAX_NEW_AUDIO_CHUNKS}" ]]; then
    EXTRA_ARGS+=(--max-chunks "${MAX_NEW_AUDIO_CHUNKS}" --allow-partial-output)
    ANALYZE_EXTRA_ARGS+=(--allow-missing-durations)
  fi

  # Stage 2: TTS duration generation for the chunk rows.
  python3 "${EXP_ROOT}/audio_duration.py" \
    --input-jsonl "${CHUNKS_JSONL}" \
    --output-jsonl "${DURATIONS_JSONL}" \
    --output-csv "${DURATIONS_CSV}" \
    --cache-jsonl "${DURATION_CACHE_JSONL}" \
    --stage-configs-path "${TTS_STAGE_CONFIG}" \
    --tts-model "${TTS_MODEL}" \
    --batch-size "${TTS_BATCH_SIZE}" \
    --language "${TTS_LANGUAGE}" \
    --speaker "${TTS_SPEAKER}" \
    --instruct "${TTS_INSTRUCT}" \
    --max-new-tokens "${TTS_MAX_NEW_TOKENS}" \
    --tts-max-segment-chars "${TTS_MAX_SEGMENT_CHARS}" \
    "${EXTRA_ARGS[@]}"

  for SLACK_MODE in "${SLACK_MODES[@]}"; do
    MODE_ROOT="${MODEL_OUTPUT_ROOT}/${SLACK_MODE}"
    TEXT_DIR="${MODE_ROOT}/text_outputs"
    AUDIO_DIR="${MODE_ROOT}/audio_durations"
    RESULTS_DIR="${MODE_ROOT}/results"
    mkdir -p "${TEXT_DIR}" "${AUDIO_DIR}" "${RESULTS_DIR}"

    if [[ "${SLACK_MODE}" != "${SOURCE_MODE}" ]]; then
      # Reuse source stage outputs so only analysis is repeated per mode.
      copy_named_files \
        "${SOURCE_TEXT_DIR}" \
        "${TEXT_DIR}" \
        requests.jsonl chunks.jsonl chunks.csv summary.json
      copy_named_files \
        "${SOURCE_AUDIO_DIR}" \
        "${AUDIO_DIR}" \
        durations.jsonl durations.csv duration_cache.jsonl
    fi

    # Stage 3: compute and plot one slack mode.
    python3 "${EXP_ROOT}/analyze_results.py" \
      --chunks-jsonl "${TEXT_DIR}/chunks.jsonl" \
      --durations-jsonl "${AUDIO_DIR}/durations.jsonl" \
      --slack-mode "${SLACK_MODE}" \
      --output-dir "${RESULTS_DIR}" \
      "${ANALYZE_EXTRA_ARGS[@]}"
  done
done
