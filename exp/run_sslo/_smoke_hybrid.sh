#!/usr/bin/env bash
# Worst-case: cells where adaptive locked into a small batch before hybrid.
#   $1 = model_hf, $2 = cap, $3 = rate, $4 = gpu
set -euo pipefail
cd /workspace/mlsys
model="$1"; cap="$2"; rate="$3"; gpu="$4"
slug=$(echo "$model" | tr '/' '_')
OUTDIR="exp/run_sslo/output_smoke_hybrid/${slug}_cap${cap}_r${rate}/progress_serve_adaptive/run_1"
rm -rf "exp/run_sslo/output_smoke_hybrid/${slug}_cap${cap}_r${rate}"; mkdir -p "$OUTDIR"
export HF_HOME=/cache
export NUM_PROMPTS=1200 GENERATION_MAX_TOKENS=2048 MAX_MODEL_LEN=0
export CHUNK_UNIT=sentence DATASET_NAME=koala DATASET_SEED=42
export EXCLUDE_CODE=1 CONVERSATION_ONLY=1 ENGLISH_ONLY=1
export MAX_RESPONSE_CHUNK_CHARS=1000 SECONDS_PER_WORD=0.28
export REQUEST_RATES="$rate" REQUEST_RATE_SEED=44 REPEAT=1
export SUMMARY_CSV="exp/run_sslo/output_smoke_hybrid/summary.csv"
export CONSUME_MODE=read TTS_MODEL="" OUTPUT_DIR="$OUTDIR" CUDA_VISIBLE_DEVICES="$gpu"
bash exp/run_sslo/run_test.sh progress_serve_adaptive "$cap" "$model"
