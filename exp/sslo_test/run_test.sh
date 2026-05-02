#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

MODEL="Qwen/Qwen3-8B"
DATASET_NAME="koala"
NUM_PROMPTS=256
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=64
GENERATION_MAX_TOKENS=512
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
CHUNK_UNIT="sentence"
SECONDS_PER_WORD=0.28
OUTPUT_DIR="exp/sslo_test/output"

python3 exp/sslo_test/run_test.py \
  --model "${MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --chunk-unit "${CHUNK_UNIT}" \
  --seconds-per-word "${SECONDS_PER_WORD}" \
  --output-dir "${OUTPUT_DIR}"
