#!/usr/bin/env bash
# Stage 1: extract hidden states and boundary labels from Qwen3.5.
# Runs inside the sk-sslo Docker container.
# Edit MODEL, DATASET, NUM_PROMPTS, and OUTPUT_DIR as needed.

set -euo pipefail

MODEL="Qwen/Qwen3.5-7B"
DATASET="koala"
NUM_PROMPTS=200
MAX_NEW_TOKENS=512
BATCH_SIZE=4
OUTPUT_DIR="length_predictor/features/${DATASET}"

docker exec sk-sslo bash -lc "
  cd /workspace/mlsys
  export HF_HOME=/cache
  export HF_HUB_CACHE=/cache/hub
  python3 length_predictor/prepare_features.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --num-prompts ${NUM_PROMPTS} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR}
"
