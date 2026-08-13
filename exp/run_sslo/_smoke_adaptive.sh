#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mlsys
OUTDIR="exp/run_sslo/output_smoke_adaptive/progress_serve_adaptive/run_1"
rm -rf exp/run_sslo/output_smoke_adaptive; mkdir -p "$OUTDIR"
export HF_HOME=/cache
export NUM_PROMPTS=800 GENERATION_MAX_TOKENS=2048 MAX_MODEL_LEN=0
export CHUNK_UNIT=sentence DATASET_NAME=koala DATASET_SEED=42
export EXCLUDE_CODE=1 CONVERSATION_ONLY=1 ENGLISH_ONLY=1
export MAX_RESPONSE_CHUNK_CHARS=1000 SECONDS_PER_WORD=0.28
export REQUEST_RATES="40" REQUEST_RATE_SEED=44 REPEAT=1
export SUMMARY_CSV="exp/run_sslo/output_smoke_adaptive/summary.csv"
export CONSUME_MODE=read TTS_MODEL="" OUTPUT_DIR="$OUTDIR" CUDA_VISIBLE_DEVICES=0
bash exp/run_sslo/run_test.sh progress_serve_adaptive 256 Qwen/Qwen3.5-9B
