#!/usr/bin/env bash
# Stage 2: train LengthPredictorHead on pre-extracted features.
# Runs inside the sk-sslo Docker container with all available GPUs.

set -euo pipefail

FEATURES_DIR="length_predictor/features/koala"
OUTPUT_DIR="length_predictor/checkpoints/koala"
HIDDEN_SIZE=4096
INNER_SIZE=256
BATCH_SIZE=512
LR=1e-3
EPOCHS=10

docker exec sk-sslo bash -lc "
  cd /workspace/mlsys
  NUM_GPUS=\$(nvidia-smi -L | wc -l)
  echo \"Launching on \${NUM_GPUS} GPU(s)\"
  accelerate launch --num_processes \${NUM_GPUS} \
    length_predictor/train.py \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --hidden-size ${HIDDEN_SIZE} \
    --inner-size ${INNER_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --epochs ${EPOCHS}
"
