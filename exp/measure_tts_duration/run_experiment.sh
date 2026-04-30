#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

# Run this launcher on the host; all execution happens in sk-sslo-omni.
CONTAINER_NAME="sk-sslo-omni"
CONTAINER_REPO="/workspace/mlsys"
EXP_ROOT="${CONTAINER_REPO}/exp/measure_tts_duration"
OUTPUT_ROOT="${EXP_ROOT}/output"
CUDA_VISIBLE_DEVICES_VALUE="0"

# Dialogue dataset options.
DATASET_NAME="HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT="train_sft"
MAX_DIALOGUES=256

# TTS models to compare.
MODELS=(
  "hexgrad/Kokoro-82M"
  "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

# Kokoro options.
KOKORO_LANG_CODE="a"
KOKORO_VOICE="af_heart"

# Qwen Base options.
QWEN_BATCH_SIZE=64
QWEN_LANGUAGE="English"
QWEN_MAX_NEW_TOKENS=2048
QWEN_REF_AUDIO="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
QWEN_REF_TEXT="Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

slugify() {
  local value="$1"
  value="${value//\//__}"
  value="${value// /_}"
  value="$(printf '%s' "${value}" | tr -c 'A-Za-z0-9._-' '_')"
  printf '%s' "${value}"
}

run_in_container() {
  docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${CONTAINER_NAME}" bash -lc '
    set -euo pipefail
    cd /workspace/mlsys
    export HF_HOME=/cache
    export HF_HUB_CACHE=/cache/hub
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    "$@"
  ' bash "$@"
}

require_container() {
  if ! docker inspect "$1" >/dev/null 2>&1; then
    echo "Required container is not available: $1" >&2
    exit 1
  fi
}

ensure_dependencies() {
  # Install only the runtime pieces this experiment needs in sk-sslo-omni.
  docker exec "${CONTAINER_NAME}" bash -lc '
    set -euo pipefail
    export HF_HOME=/cache
    export HF_HUB_CACHE=/cache/hub
    python3 -m pip install -q "datasets>=2.18.0" "kokoro==0.9.4" "qwen-tts==0.1.1" soundfile
    if ! command -v espeak-ng >/dev/null 2>&1; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng
    fi
  '
}

require_container "${CONTAINER_NAME}"
ensure_dependencies

DATASET_SLUG="$(slugify "${DATASET_NAME}")"
DATASET_ROOT="${OUTPUT_ROOT}/${DATASET_SLUG}"

# Stage 1: prepare shared sentence/paragraph chunks once per dataset.
run_in_container python3 "${EXP_ROOT}/prepare_dataset_chunks.py" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-split "${DATASET_SPLIT}" \
  --max-dialogues "${MAX_DIALOGUES}" \
  --output-root "${DATASET_ROOT}"

for CHUNK_UNIT in sentence paragraph; do
  CHUNKS_DIR="${DATASET_ROOT}/${CHUNK_UNIT}/text_chunks"
  COMBINED_INPUT_ARGS=()

  for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="$(slugify "${MODEL}")"
    MODEL_ROOT="${DATASET_ROOT}/${CHUNK_UNIT}/${MODEL_SLUG}"
    AUDIO_DIR="${MODEL_ROOT}/audio_durations"
    RESULTS_DIR="${MODEL_ROOT}/results"
    run_in_container mkdir -p "${AUDIO_DIR}" "${RESULTS_DIR}"

    MODEL_ARGS=()
    if [[ "${MODEL}" == "hexgrad/Kokoro-82M" ]]; then
      MODEL_ARGS=(
        --kokoro-lang-code "${KOKORO_LANG_CODE}"
        --kokoro-voice "${KOKORO_VOICE}"
      )
    elif [[ "${MODEL}" == "Qwen/Qwen3-TTS-12Hz-1.7B-Base" ]]; then
      MODEL_ARGS=(
        --batch-size "${QWEN_BATCH_SIZE}"
        --qwen-language "${QWEN_LANGUAGE}"
        --qwen-max-new-tokens "${QWEN_MAX_NEW_TOKENS}"
        --qwen-ref-audio "${QWEN_REF_AUDIO}"
        --qwen-ref-text "${QWEN_REF_TEXT}"
      )
    else
      echo "Unsupported model in launcher: ${MODEL}" >&2
      exit 1
    fi

    # Stage 2: measure audio duration for one model on one chunk unit.
    run_in_container python3 "${EXP_ROOT}/measure_audio_duration.py" \
      --input-dir "${CHUNKS_DIR}" \
      --output-dir "${AUDIO_DIR}" \
      --tts-model "${MODEL}" \
      "${MODEL_ARGS[@]}"

    # Stage 3: build the per-model word-count duration table.
    run_in_container python3 "${EXP_ROOT}/summarize_word_stats.py" \
      --input-jsonl "${AUDIO_DIR}/durations.jsonl" \
      --output-dir "${RESULTS_DIR}"

    COMBINED_INPUT_ARGS+=(--input-jsonl "${AUDIO_DIR}/durations.jsonl")
  done

  # Cross-model table for this chunk unit.
  run_in_container python3 "${EXP_ROOT}/summarize_word_stats.py" \
    "${COMBINED_INPUT_ARGS[@]}" \
    --output-dir "${DATASET_ROOT}/${CHUNK_UNIT}/combined_results"
done
