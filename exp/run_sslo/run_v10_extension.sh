#!/usr/bin/env bash
# v10 extension: rate {0.5, 12} × (9B cap{16,32,64,128} + 35B cap{64,128,256,512})
# Plus retry of 9B cap=128 r=8 baseline (CUBLAS-failed in v10 main).
# 33 runs total. 4 GPU round-robin.
set -euo pipefail
cd /workspace/mlsys
# HF_TOKEN must be exported by the caller.

ROOT=exp/run_sslo/output/v10_grid

launch() {
  local model_slug=$1 model=$2 cap=$3 rate=$4 mode=$5 gpu=$6
  local outdir="$ROOT/$model_slug/cap${cap}_r${rate}/${mode}/run_1"
  rm -rf "$outdir"; mkdir -p "$outdir"
  echo "[gpu=$gpu] $model_slug cap=$cap rate=$rate mode=$mode"
  OUTPUT_DIR="$outdir" REQUEST_RATE="$rate" REQUEST_RATE_SEED=43 \
    MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    bash exp/run_sslo/run_test.sh "$mode" "$cap" "$model" \
    > "$outdir/run.log" 2>&1
}

# 9B retry (1 job) + 9B rate {0.5, 12} × cap × mode = 1 + 16 = 17
jobs_9b=("Qwen3.5-9B|Qwen/Qwen3.5-9B|128|8|baseline")
for cap in 16 32 64 128; do
  for rate in 0.5 12; do
    for mode in baseline sslo_mlp; do
      jobs_9b+=("Qwen3.5-9B|Qwen/Qwen3.5-9B|$cap|$rate|$mode")
    done
  done
done

# 35B rate {0.5, 12} × cap × mode = 16
jobs_35b=()
for cap in 64 128 256 512; do
  for rate in 0.5 12; do
    for mode in baseline sslo_mlp; do
      jobs_35b+=("Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|$cap|$rate|$mode")
    done
  done
done

echo "9B jobs: ${#jobs_9b[@]} ; 35B jobs: ${#jobs_35b[@]}"

launch_group() {
  local gpu=$1
  local -n joblist=$2
  local g_idx=$3
  local stride=$4
  for (( i=g_idx; i<${#joblist[@]}; i+=stride )); do
    IFS='|' read -r slug model cap rate mode <<< "${joblist[i]}"
    launch "$slug" "$model" "$cap" "$rate" "$mode" "$gpu"
  done
}

( launch_group 0 jobs_9b  0 2 ) &
P0=$!
( launch_group 1 jobs_9b  1 2 ) &
P1=$!
( launch_group 2 jobs_35b 0 2 ) &
P2=$!
( launch_group 3 jobs_35b 1 2 ) &
P3=$!
wait $P0 $P1 $P2 $P3
echo "All jobs done."
