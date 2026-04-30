#!/usr/bin/env bash
set -euo pipefail

# Host launcher — runs profiler inside sk-sslo container.
# Edit constants below to change model or sweep.

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

CONTAINER="sk-sslo"
CONTAINER_REPO="/workspace/mlsys"
SCRIPT="${CONTAINER_REPO}/exp/measure_KV_overhead/measure_kv_overhead.py"

MODEL="Qwen/Qwen3-8B"
BLOCK_SIZE=16
DTYPE="bfloat16"
NUM_BLOCKS="1 2 4 8 16 32 64 128 256 512 1024 2048"
WARMUP_RUNS=3
TIMED_RUNS=10
OUTPUT_DIR="${CONTAINER_REPO}/exp/measure_KV_overhead/outputs"

docker exec \
    -e HF_HOME="${HF_HOME}" \
    -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
    "${CONTAINER}" \
    python3 "${SCRIPT}" \
        --model "${MODEL}" \
        --block-size "${BLOCK_SIZE}" \
        --dtype "${DTYPE}" \
        --num-blocks ${NUM_BLOCKS} \
        --warmup-runs "${WARMUP_RUNS}" \
        --timed-runs "${TIMED_RUNS}" \
        --output-dir "${OUTPUT_DIR}"
