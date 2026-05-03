#!/usr/bin/env bash
# Sweep SSLO test across multiple max_num_seqs values, then aggregate.
#
# Run inside the sk-sslo container. Edit the constants below to change the
# model or the value list.
set -euo pipefail

MODEL="Qwen/Qwen3-8B"
NUM_PROMPTS=256
GENERATION_MAX_TOKENS=512
OUTPUT_ROOT="exp/sslo_test/output"
MAX_NUM_SEQS_VALUES=(32 64 128 256)

# Clean previous sweep outputs (do NOT touch this file's siblings outside
# the seqs_* dirs and the top-level sweep_summary.json).
rm -rf "${OUTPUT_ROOT}"/seqs_* "${OUTPUT_ROOT}/sweep_summary.json"

for N in "${MAX_NUM_SEQS_VALUES[@]}"; do
  bash exp/sslo_test/run_single.sh \
    "$N" "$MODEL" "$NUM_PROMPTS" "$GENERATION_MAX_TOKENS" "$OUTPUT_ROOT"
done

python3 exp/sslo_test/analyze.py --sweep-root "${OUTPUT_ROOT}"
