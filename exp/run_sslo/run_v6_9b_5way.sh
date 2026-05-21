#!/usr/bin/env bash
# 5-way 9B comparison (cap=256 r=16 window=180s seed=43):
#   baseline                        — GPU0 (run first, then run 5th)
#   sslo abatch=T allow=F (default) — GPU1
#   sslo abatch=F allow=F           — GPU2
#   sslo abatch=T allow=T           — GPU3
#   sslo abatch=F allow=T           — GPU0 (after baseline)
set -euo pipefail
cd /workspace/mlsys
# HF_TOKEN must be set in the environment by the caller (gated lmsys dataset).

ROOT=exp/run_sslo/output/v6_9b_5way
rm -rf "$ROOT"; mkdir -p "$ROOT"

base_env() {
  echo "REQUEST_RATE=16 REQUEST_RATE_SEED=43 MEASUREMENT_WINDOW_S=180
        NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048
        CHUNK_UNIT=sentence DATASET_NAME=combine DATASET_SEED=42
        EXCLUDE_CODE=1 CONVERSATION_ONLY=1"
}

launch() {
  local name=$1 gpu=$2 kind=$3
  shift 3
  local outdir="$ROOT/$name/run_1"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] $name kind=$kind extra=$*"
  env "$@" \
    OUTPUT_DIR="$outdir" REQUEST_RATE=16 REQUEST_RATE_SEED=43 \
    MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    bash exp/run_sslo/run_test.sh "$kind" 256 Qwen/Qwen3.5-9B \
    > "$outdir/run.log" 2>&1
}

# Round 1: 4 concurrent
launch baseline                 0 baseline &
P0=$!
launch abatchT_allowF_default   1 sslo_mlp &
P1=$!
launch abatchF_allowF           2 sslo_mlp SSLO_ADAPTIVE_BATCHING=0 &
P2=$!
launch abatchT_allowT           3 sslo_mlp ALLOW_ADMIT_CRITICAL=1 &
P3=$!

wait $P0
echo "baseline done; launching 5th on GPU0"
launch abatchF_allowT 0 sslo_mlp SSLO_ADAPTIVE_BATCHING=0 ALLOW_ADMIT_CRITICAL=1 &
P4=$!

wait $P1 $P2 $P3 $P4
echo "All 5 jobs done."
