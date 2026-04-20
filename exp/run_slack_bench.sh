#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

cd /workspace/mlsys

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUTPUT_DIR}"

NUM_PROMPTS=1024
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.95
SECONDS_PER_WORD=0.28
MODELS=(
  "Qwen/Qwen3.5-35B-A3B"
  "Qwen/Qwen3.5-27B"
)

for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE_NAME="${MODEL//\//__}"
  MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_SAFE_NAME}"
  mkdir -p "${MODEL_OUTPUT_DIR}"

  OUTPUT_JSON="${MODEL_OUTPUT_DIR}/slack_results.json"
  SLACK_OUTPUT_JSONL="${MODEL_OUTPUT_DIR}/slack_results.jsonl"
  POSTPROCESS_JSONL="${MODEL_OUTPUT_DIR}/slack_chunks_postprocessed.jsonl"
  POSTPROCESS_CSV="${MODEL_OUTPUT_DIR}/slack_chunks_postprocessed.csv"

  python3 exp/run_slack_bench.py \
    --backend vllm \
    --async-engine \
    --apply-chat-template \
    --model "${MODEL}" \
    --dataset-name hf \
    --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
    --num-prompts "${NUM_PROMPTS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --async-engine \
    --output-json "${OUTPUT_JSON}" \
    --slack-output-jsonl "${SLACK_OUTPUT_JSONL}"

  python3 exp/postprocess_slack.py \
    --input-jsonl "${SLACK_OUTPUT_JSONL}" \
    --summary-json "${OUTPUT_JSON}" \
    --output-jsonl "${POSTPROCESS_JSONL}" \
    --output-csv "${POSTPROCESS_CSV}" \
    --model "${MODEL}" \
    --seconds-per-word "${SECONDS_PER_WORD}"
done
