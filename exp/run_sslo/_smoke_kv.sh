#!/usr/bin/env bash
# Smoke: 1 cell to verify KV block fields land in jsonl + summary.json + summary.csv.
set -euo pipefail
cd /workspace/mlsys

OUTDIR="exp/run_sslo/output_smoke_kv/sslo_mlp/run_1"
rm -rf exp/run_sslo/output_smoke_kv
mkdir -p "$OUTDIR"

export HF_HOME=/cache
export TTS_PROFILE_PATH="exp/run_sslo/profiles/word_count_duration_stats.csv"
export NUM_PROMPTS=512
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
export REQUEST_RATES="8"
export REQUEST_RATE_SEED=44
export REPEAT=1
export SUMMARY_CSV="exp/run_sslo/output_smoke_kv/summary.csv"
export SSLO_ADAPTIVE_BATCHING=0
export CONSUME_MODE=read
export TTS_MODEL=""
export OUTPUT_DIR="$OUTDIR"
export CUDA_VISIBLE_DEVICES=0

bash exp/run_sslo/run_test.sh sslo_mlp 128 Qwen/Qwen3.5-9B 2>&1 | tail -40
