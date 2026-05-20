#!/usr/bin/env bash
# 3-way sweep for best/worst case on 27B with conversation_only.
#   modes: baseline, sslo_mlp (abatch=T), sslo_mlp_noabatch (abatch=F)
#   cases: best=cap32 r64, worst=cap48 r128
#   runs : 3 each (seed=43,44,45)
# Total 18 jobs distributed across 4 GPUs round-robin.
set -euo pipefail

cd /workspace/mlsys
export HF_TOKEN=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

CASES=(
  "32 64 best"
  "48 128 worst"
)
MODES=(baseline sslo_mlp sslo_mlp_noabatch)
RUNS=(1 2 3)
NUM_GPUS=4

ROOT="${OUTPUT_ROOT:-exp/run_sslo/output/conv_3way}"
rm -rf "$ROOT"
mkdir -p "$ROOT"

launch() {
  local cap=$1 rate=$2 label=$3 mode=$4 run=$5 gpu=$6
  local seed=$((42 + run))
  local outdir="$ROOT/$label/${cap}_${rate}/${mode}/run_${run}"
  mkdir -p "$outdir"
  local abatch_env=""
  local kind=$mode
  if [[ $mode == "sslo_mlp_noabatch" ]]; then
    abatch_env="SSLO_ADAPTIVE_BATCHING=0"
    kind=sslo_mlp
  fi
  echo "[gpu=$gpu] launch $label $cap $rate $mode run=$run seed=$seed"
  env $abatch_env \
    OUTPUT_DIR="$outdir" REQUEST_RATE="$rate" REQUEST_RATE_SEED="$seed" \
    MEASUREMENT_WINDOW_S=180 \
    NUM_PROMPTS=4000 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 CONVERSATION_ONLY=1 \
    TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 MIN_P=0.0 \
    PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 \
    bash exp/run_sslo/run_test.sh "$kind" "$cap" Qwen/Qwen3.5-27B \
    > "$outdir/run.log" 2>&1
}

# Build flat job list
jobs=()
for case in "${CASES[@]}"; do
  read -r cap rate label <<< "$case"
  for mode in "${MODES[@]}"; do
    for run in "${RUNS[@]}"; do
      jobs+=("$cap|$rate|$label|$mode|$run")
    done
  done
done
echo "Total jobs: ${#jobs[@]}"

# Round-robin assign jobs to GPUs, run each GPU's jobs sequentially in
# parallel sub-shells.
for (( g=0; g<NUM_GPUS; g++ )); do
  (
    for (( i=g; i<${#jobs[@]}; i+=NUM_GPUS )); do
      IFS='|' read -r cap rate label mode run <<< "${jobs[i]}"
      launch "$cap" "$rate" "$label" "$mode" "$run" "$g"
    done
  ) &
done
wait
echo "All 18 jobs done."
