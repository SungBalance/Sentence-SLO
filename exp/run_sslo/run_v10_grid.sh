#!/usr/bin/env bash
# v10 grid: 9B + 35B-A3B  ×  batch  ×  rate  ×  (baseline | sslo_mlp).
#   9B  : batch {16,32,64,128}  rate {1,2,4,8}  → 32 runs on GPU 0,1
#   35B : batch {64,128,256,512} rate {1,2,4,8} → 32 runs on GPU 2,3
# Window=180s, seed=43, conv-only, 4K pool.
set -euo pipefail
cd /workspace/mlsys
# HF_TOKEN must be exported by the caller.

ROOT=exp/run_sslo/output/v10_grid
rm -rf "$ROOT"; mkdir -p "$ROOT"

launch() {
  local model_slug=$1 model=$2 cap=$3 rate=$4 mode=$5 gpus=$6
  local outdir="$ROOT/$model_slug/cap${cap}_r${rate}/${mode}/run_1"
  mkdir -p "$outdir"
  echo "[gpu=$gpus] $model_slug cap=$cap rate=$rate mode=$mode"
  OUTPUT_DIR="$outdir" REQUEST_RATE="$rate" REQUEST_RATE_SEED=43 \
    MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpus" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    bash exp/run_sslo/run_test.sh "$mode" "$cap" "$model" \
    > "$outdir/run.log" 2>&1
}

# ----- 9B jobs (GPU 0, 1) -----
jobs_9b=()
for cap in 16 32 64 128; do
  for rate in 1 2 4 8; do
    for mode in baseline sslo_mlp; do
      jobs_9b+=("Qwen3.5-9B|Qwen/Qwen3.5-9B|$cap|$rate|$mode")
    done
  done
done

# ----- 35B-A3B jobs (GPU 2, 3) -----
jobs_35b=()
for cap in 64 128 256 512; do
  for rate in 1 2 4 8; do
    for mode in baseline sslo_mlp; do
      jobs_35b+=("Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|$cap|$rate|$mode")
    done
  done
done

echo "9B jobs: ${#jobs_9b[@]} ; 35B jobs: ${#jobs_35b[@]}"

# Run two parallel sub-shells per model (one per assigned GPU).
launch_group() {
  local gpu=$1; shift
  local -n joblist=$1
  local g_idx=$2
  local stride=$3
  for (( i=g_idx; i<${#joblist[@]}; i+=stride )); do
    IFS='|' read -r slug model cap rate mode <<< "${joblist[i]}"
    launch "$slug" "$model" "$cap" "$rate" "$mode" "$gpu"
  done
}

# 9B on GPU 0 (even-idx) + GPU 1 (odd-idx).
( launch_group 0 jobs_9b 0 2 ) &
P0=$!
( launch_group 1 jobs_9b 1 2 ) &
P1=$!
# 35B on GPU 2 (even-idx) + GPU 3 (odd-idx).
( launch_group 2 jobs_35b 0 2 ) &
P2=$!
( launch_group 3 jobs_35b 1 2 ) &
P3=$!

wait $P0 $P1 $P2 $P3
echo "All jobs done."
