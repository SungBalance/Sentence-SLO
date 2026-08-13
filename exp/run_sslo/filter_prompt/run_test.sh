#!/usr/bin/env bash
# Run ONE SSLO mode for ONE config and write its JSONLs.
#
# Usage:
#   run_test.sh <run_kind> <max_num_seqs> <model>
#
# Required positional args:
#   $1  run_kind   (baseline | sslo | sslo_offload | sslo_adaptive | sslo_adaptive_offload | sslo_mlp)
#   $2  max_num_seqs
#   $3  model
#
# Env vars (with defaults):
#   NUM_PROMPTS=256
#   GENERATION_MAX_TOKENS=512
#   MAX_MODEL_LEN=8192
#   TENSOR_PARALLEL_SIZE=1
#   GPU_MEMORY_UTILIZATION=0.95
#   OUTPUT_DIR                    (required, no default)
#   REQUEST_RATE=0
#   REQUEST_RATE_SEED=42
#   CHUNK_UNIT=sentence
#   SECONDS_PER_WORD=0.28
#   CUDA_VISIBLE_DEVICES=1
#   SSLO_KV_OFFLOAD_EXTRA='{"cpu_bytes_to_use": 17179869184}'
#
# Run inside the sk-sslo container from /workspace/mlsys.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <run_kind> <max_num_seqs> <model>" >&2
  exit 1
fi

run_kind="$1"
max_num_seqs="$2"
model="$3"

: "${OUTPUT_DIR:?OUTPUT_DIR env var is required}"

NUM_PROMPTS="${NUM_PROMPTS:-4096}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"  # 0 = auto (vLLM uses model config max)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
REQUEST_RATE_SEED="${REQUEST_RATE_SEED:-42}"
# REQUEST_RATES: comma- or space-separated list of Poisson rates. All
# rates are swept inside a single engine; SSLO state is reset between
# rates. Defaults to single rate "4" when neither REQUEST_RATES nor
# REQUEST_RATE is set; REQUEST_RATE (single value, legacy) is honored
# if REQUEST_RATES is unset.
if [[ -n "${REQUEST_RATES:-}" ]]; then
  RATES_ARG="${REQUEST_RATES}"
else
  RATES_ARG="${REQUEST_RATE:-4}"
fi
# Measurement window is now completion-count gated (warmup=max_num_seqs*2,
# measurement=max_num_seqs*4). No safety timeout — caller picks
# reachable (cap, rate) combinations.
CHUNK_UNIT="${CHUNK_UNIT:-sentence}"
SECONDS_PER_WORD="${SECONDS_PER_WORD:-0.28}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SSLO_KV_OFFLOAD_EXTRA="${SSLO_KV_OFFLOAD_EXTRA:-{\"cpu_bytes_to_use\": 17179869184}}"
# Consume-time selection. Defaults to fixed seconds-per-word ("read");
# set CONSUME_MODE=tts plus TTS_PROFILE_PATH + TTS_MODEL to drive
# chunk_consume_time + conversion_time from a measured TTS profile CSV.
CONSUME_MODE="${CONSUME_MODE:-read}"
TTS_PROFILE_PATH="${TTS_PROFILE_PATH:-}"
TTS_MODEL="${TTS_MODEL:-}"
if [[ "${CONSUME_MODE}" == "tts" ]]; then
  if [[ -z "${TTS_PROFILE_PATH}" || -z "${TTS_MODEL}" ]]; then
    echo "CONSUME_MODE=tts requires both TTS_PROFILE_PATH and TTS_MODEL." >&2
    exit 1
  fi
  CONSUME_ARGS=(--consume-mode tts
                --tts-profile-path "${TTS_PROFILE_PATH}"
                --tts-model "${TTS_MODEL}")
else
  CONSUME_ARGS=()
fi
# Thinking control for unified Instruct+Thinking models (Qwen3.5).
# ENABLE_THINKING=1 → --enable-thinking; 0/unset (default) → flag omitted.
ENABLE_THINKING="${ENABLE_THINKING:-0}"
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  THINKING_FLAG="--enable-thinking"
else
  THINKING_FLAG=""
fi
# Sampling overrides (empty = use model HF defaults). For heterogeneous
# model families pin these to keep cells comparable.
SAMPLING_ARGS=""
[[ -n "${TEMPERATURE:-}" ]]        && SAMPLING_ARGS+=" --temperature ${TEMPERATURE}"
[[ -n "${TOP_P:-}" ]]              && SAMPLING_ARGS+=" --top-p ${TOP_P}"
[[ -n "${TOP_K:-}" ]]              && SAMPLING_ARGS+=" --top-k ${TOP_K}"
[[ -n "${MIN_P:-}" ]]              && SAMPLING_ARGS+=" --min-p ${MIN_P}"
[[ -n "${PRESENCE_PENALTY:-}" ]]   && SAMPLING_ARGS+=" --presence-penalty ${PRESENCE_PENALTY}"
[[ -n "${REPETITION_PENALTY:-}" ]] && SAMPLING_ARGS+=" --repetition-penalty ${REPETITION_PENALTY}"
# Dataset selection + code-gen exclusion.
DATASET_NAME="${DATASET_NAME:-koala}"
DATASET_SEED="${DATASET_SEED:-42}"
if [[ "${EXCLUDE_CODE:-0}" == "1" ]]; then
  EXCLUDE_CODE_FLAG="--exclude-code"
else
  EXCLUDE_CODE_FLAG=""
fi
if [[ "${CONVERSATION_ONLY:-0}" == "1" ]]; then
  CONVERSATION_ONLY_FLAG="--conversation-only"
else
  CONVERSATION_ONLY_FLAG=""
fi
if [[ "${ENGLISH_ONLY:-1}" == "1" ]]; then
  ENGLISH_ONLY_FLAG="--english-only"
else
  ENGLISH_ONLY_FLAG=""
fi
MAX_RESPONSE_CHUNK_CHARS="${MAX_RESPONSE_CHUNK_CHARS:-1000}"

export HF_HOME=/cache
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CHUNK_UNIT
export CUDA_VISIBLE_DEVICES
export SSLO_KV_OFFLOAD_EXTRA

export SSLO_STATS_LOG_PATH="${OUTPUT_DIR}/scheduler_stats.jsonl"
if [[ "${run_kind}" == *offload* ]]; then
  export SSLO_OFFLOAD_LOG_PATH="${OUTPUT_DIR}/offload_log.jsonl"
fi

python3 exp/run_sslo/filter_prompt/run_test.py \
  --run-kind "${run_kind}" \
  --model "${model}" \
  --max-num-seqs "${max_num_seqs}" \
  --num-prompts "${NUM_PROMPTS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --output-dir "${OUTPUT_DIR}" \
  --request-rates "${RATES_ARG}" \
  --request-rate-seed "${REQUEST_RATE_SEED}" \
  --summary-csv "${SUMMARY_CSV:-}" \
  --repeat "${REPEAT:-1}" \
  --seconds-per-word "${SECONDS_PER_WORD}" \
  --dataset-name "${DATASET_NAME}" \
  --dataset-seed "${DATASET_SEED}" \
  ${EXCLUDE_CODE_FLAG} \
  ${CONVERSATION_ONLY_FLAG} \
  ${ENGLISH_ONLY_FLAG} \
  --max-response-chunk-chars "${MAX_RESPONSE_CHUNK_CHARS}" \
  ${THINKING_FLAG} \
  ${SAMPLING_ARGS} \
  "${CONSUME_ARGS[@]}"
