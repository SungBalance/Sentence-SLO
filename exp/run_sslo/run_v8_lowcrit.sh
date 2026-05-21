#!/usr/bin/env bash
# 9B cap=256 × rate{4,8} × modes{baseline, sslo_mlp, sslo_mlp_noabatch}
# 6 jobs, 4 GPU round-robin.
set -euo pipefail
cd /workspace/mlsys
# HF_TOKEN must be set in the environment by the caller (gated lmsys dataset).

ROOT=exp/run_sslo/output/v8_9b_lowcrit
rm -rf "$ROOT"; mkdir -p "$ROOT"

launch() {
  local cap=$1 rate=$2 mode=$3 gpu=$4
  local outdir="$ROOT/cap${cap}_r${rate}/${mode}/run_1"
  mkdir -p "$outdir"
  local abatch_env=""
  local kind=$mode
  if [[ $mode == "sslo_mlp_noabatch" ]]; then
    abatch_env="SSLO_ADAPTIVE_BATCHING=0"
    kind=sslo_mlp
  fi
  echo "[gpu=$gpu] cap=$cap rate=$rate mode=$mode"
  env $abatch_env \
    OUTPUT_DIR="$outdir" REQUEST_RATE="$rate" REQUEST_RATE_SEED=43 \
    MEASUREMENT_WINDOW_S=180 NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    bash exp/run_sslo/run_test.sh "$kind" "$cap" Qwen/Qwen3.5-9B \
    > "$outdir/run.log" 2>&1
}

jobs=()
for rate in 4 8; do
  for mode in baseline sslo_mlp sslo_mlp_noabatch; do
    jobs+=("256|$rate|$mode")
  done
done

NUM_GPUS=4
for (( g=0; g<NUM_GPUS; g++ )); do
  (
    for (( i=g; i<${#jobs[@]}; i+=NUM_GPUS )); do
      IFS='|' read -r cap rate mode <<< "${jobs[i]}"
      launch "$cap" "$rate" "$mode" "$g"
    done
  ) &
done
wait
echo "All done."
