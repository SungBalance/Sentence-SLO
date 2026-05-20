#!/usr/bin/env bash
# Simplified SSLO (no critical branch, no abatch effect): worst case
# 1 run for 9B / 27B / 35B-A3B in parallel on GPUs 0/1/2.
set -euo pipefail
cd /workspace/mlsys
export HF_TOKEN=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

ROOT=exp/run_sslo/output/simplified_worst
rm -rf "$ROOT"; mkdir -p "$ROOT"

declare -A JOBS=(
  ["9b"]="Qwen/Qwen3.5-9B 256 128 0"
  ["27b"]="Qwen/Qwen3.5-27B 48 128 1"
  ["35b"]="Qwen/Qwen3.5-35B-A3B 128 64 2"
)

launch() {
  local key=$1
  read -r model cap rate gpu <<< "${JOBS[$key]}"
  local outdir="$ROOT/$key/sslo_mlp/run_1"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] $key model=$model cap=$cap rate=$rate"
  OUTPUT_DIR="$outdir" REQUEST_RATE="$rate" REQUEST_RATE_SEED=43 \
    MEASUREMENT_WINDOW_S=180 \
    NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 MIN_P=0.0 \
    PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 \
    bash exp/run_sslo/run_test.sh sslo_mlp "$cap" "$model" \
    > "$outdir/run.log" 2>&1
}

for key in 9b 27b 35b; do launch "$key" & done
wait
echo "All 3 done."
