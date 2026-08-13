#!/usr/bin/env bash
# Build per-(language, task) prompt pools for the chunk-length study.
# Run inside the sk-sslo-vllm container from /workspace/mlsys.
set -euo pipefail
cd /workspace/mlsys

export HF_HOME=/cache

LOAD_PER_DATASET="${LOAD_PER_DATASET:-6000}"
TARGET_PER_CATEGORY="${TARGET_PER_CATEGORY:-512}"
TOP_LANGS="${TOP_LANGS:-3}"

python3 exp/chunk_length_study/build_category_pools.py \
  --load-per-dataset "${LOAD_PER_DATASET}" \
  --target-per-category "${TARGET_PER_CATEGORY}" \
  --top-langs "${TOP_LANGS}"
