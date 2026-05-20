#!/usr/bin/env bash
# Critical partition strategy comparison on 9B/35B worst.
#   strategies: pressure_desc, tokens_asc
#   models: 9B (256/128) gpu 0/1, 35B (128/64) gpu 2/3
# 4 runs in parallel.
set -euo pipefail
cd /workspace/mlsys
export HF_TOKEN=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

ROOT=exp/run_sslo/output/crit_strategy_cmp
rm -rf "$ROOT"; mkdir -p "$ROOT"

declare -A JOBS=(
  ["9b_pdesc"]="Qwen/Qwen3.5-9B 256 128 0 pressure_desc"
  ["9b_tasc"]="Qwen/Qwen3.5-9B 256 128 1 tokens_asc"
  ["35b_pdesc"]="Qwen/Qwen3.5-35B-A3B 128 64 2 pressure_desc"
  ["35b_tasc"]="Qwen/Qwen3.5-35B-A3B 128 64 3 tokens_asc"
)

launch() {
  local key=$1
  read -r model cap rate gpu strat <<< "${JOBS[$key]}"
  local outdir="$ROOT/$key/sslo_mlp/run_1"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] $key model=$model cap=$cap rate=$rate strat=$strat"
  MLP_CRITICAL_PARTITION_STRATEGY="$strat" \
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

for key in 9b_pdesc 9b_tasc 35b_pdesc 35b_tasc; do launch "$key" & done
wait
echo "All 4 done."
