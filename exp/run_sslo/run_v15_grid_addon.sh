#!/usr/bin/env bash
# Add-on cells for v15_grid:
#   Qwen3.5-9B cap=256          3 repeats × 2 modes = 6 jobs (~110min each)
#   Qwen3.5-35B-A3B cap=32      3 repeats × 2 modes = 6 jobs (~20min each)
# 12 jobs / 4 GPUs via LPT static assignment to minimise idle time:
#   GPU 0: 9B cap=256 base rep1 + 9B cap=256 base rep3   (220min)
#   GPU 1: 9B cap=256 sslo rep1 + 9B cap=256 sslo rep3   (220min)
#   GPU 2: 9B cap=256 base rep2 + 35B cap=32 base × 3    (170min)
#   GPU 3: 9B cap=256 sslo rep2 + 35B cap=32 sslo × 3    (170min)
# Critical path ≈ 220min ≈ 3.7h.
#
# Appends rows to existing per-model summary.csv (file-locked).
set -euo pipefail
cd /workspace/mlsys

ROOT=exp/run_sslo/output/v15_grid

RATES="0.5 1 2 4 8 12 16 20"

launch() {
  local model_slug=$1 model=$2 cap=$3 mode=$4 repeat=$5 gpu=$6
  local outdir="$ROOT/$model_slug/cap${cap}/$mode/run_${repeat}"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] $model_slug cap=$cap mode=$mode repeat=$repeat"
  OUTPUT_DIR="$outdir" REQUEST_RATES="$RATES" REQUEST_RATE_SEED="$((43 + repeat))" \
    REPEAT="$repeat" SUMMARY_CSV="$ROOT/$model_slug/summary.csv" \
    NUM_PROMPTS=4096 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 \
    CONVERSATION_ONLY=1 ENGLISH_ONLY=1 MAX_RESPONSE_CHUNK_CHARS=1000 \
    bash exp/run_sslo/run_test.sh "$mode" "$cap" "$model" \
    > "$outdir/run.log" 2>&1
}

gpu0() {
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 baseline 1 0
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 baseline 3 0
}
gpu1() {
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 sslo_mlp 1 1
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 sslo_mlp 3 1
}
gpu2() {
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 baseline 2 2
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 baseline 1 2
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 baseline 2 2
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 baseline 3 2
}
gpu3() {
  launch Qwen3.5-9B Qwen/Qwen3.5-9B 256 sslo_mlp 2 3
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 sslo_mlp 1 3
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 sslo_mlp 2 3
  launch Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B 32 sslo_mlp 3 3
}

echo "===== addon: 9B cap=256 + 35B cap=32 (LPT, 12 jobs) ====="
gpu0 &
gpu1 &
gpu2 &
gpu3 &
wait
echo "All v15 grid addon jobs done."
