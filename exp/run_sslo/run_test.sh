#!/usr/bin/env bash
# Run ONE SSLO mode for ONE config and write its JSONLs.
#
# Usage:
#   run_test.sh <run_kind> <max_num_seqs> <model>
#
# Required positional args:
#   $1  run_kind   (baseline | sslo | sslo_offload | sslo_adaptive | sslo_adaptive_offload)
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

NUM_PROMPTS="${NUM_PROMPTS:-256}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"  # 0 = auto (vLLM uses model config max)
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
REQUEST_RATE="${REQUEST_RATE:-0}"
REQUEST_RATE_SEED="${REQUEST_RATE_SEED:-42}"
CHUNK_UNIT="${CHUNK_UNIT:-sentence}"
SECONDS_PER_WORD="${SECONDS_PER_WORD:-0.28}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SSLO_KV_OFFLOAD_EXTRA="${SSLO_KV_OFFLOAD_EXTRA:-{\"cpu_bytes_to_use\": 17179869184}}"

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CHUNK_UNIT
export CUDA_VISIBLE_DEVICES
export SSLO_KV_OFFLOAD_EXTRA

if [[ "${run_kind}" == sslo* ]]; then
  export SSLO_STATS_LOG_PATH="${OUTPUT_DIR}/_stats_${run_kind}.jsonl"
fi
if [[ "${run_kind}" == *offload* ]]; then
  export SSLO_OFFLOAD_LOG_PATH="${OUTPUT_DIR}/_offload_log_${run_kind}.jsonl"
fi

python3 exp/run_sslo/run_test.py \
  --run-kind "${run_kind}" \
  --model "${model}" \
  --max-num-seqs "${max_num_seqs}" \
  --num-prompts "${NUM_PROMPTS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --output-dir "${OUTPUT_DIR}" \
  --request-rate "${REQUEST_RATE}" \
  --request-rate-seed "${REQUEST_RATE_SEED}" \
  --seconds-per-word "${SECONDS_PER_WORD}"
