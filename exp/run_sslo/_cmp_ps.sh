#!/usr/bin/env bash
# Scratch: baseline vs progress_serve comparison, 9B/read/cap128, rate ladder.
set -euo pipefail
cd /workspace/mlsys

run_kind="$1"; gpu="$2"
OUTDIR="exp/run_sslo/output_cmp_ps/${run_kind}/run_1"
rm -rf "exp/run_sslo/output_cmp_ps/${run_kind}"
mkdir -p "$OUTDIR"

export HF_HOME=/cache
export NUM_PROMPTS=1600
export GENERATION_MAX_TOKENS=2048
export MAX_MODEL_LEN=0
export CHUNK_UNIT=sentence
export DATASET_NAME=koala
export DATASET_SEED=42
export EXCLUDE_CODE=1
export CONVERSATION_ONLY=1
export ENGLISH_ONLY=1
export MAX_RESPONSE_CHUNK_CHARS=1000
export SECONDS_PER_WORD=0.28
export REQUEST_RATES="8 16 24"
export REQUEST_RATE_SEED=44
export REPEAT=1
export SUMMARY_CSV="exp/run_sslo/output_cmp_ps/summary.csv"
export CONSUME_MODE=read
export TTS_MODEL=""
export OUTPUT_DIR="$OUTDIR"
export CUDA_VISIBLE_DEVICES="$gpu"

bash exp/run_sslo/run_test.sh "$run_kind" 128 Qwen/Qwen3.5-9B
