#!/usr/bin/env bash
# v15 grid for Qwen3.5-35B-A3B:
#   max_num_seqs ∈ {64, 128, 256, 512}
#   rates per cell: {0.5, 1, 2, 4, 8, 12, 16, 20}  (swept under one engine)
#   modes: {baseline, sslo_mlp}
#   repeats: 3 per (cap, mode), outer-most repeat loop
# 4 GPUs round-robin. Each (cap, mode, repeat) job sweeps 8 rates inside
# one engine. Per-rate output: v15_grid/Qwen3.5-35B-A3B/cap{C}/{mode}/run_{R}/rate_{r}/.
# Per-model summary CSV appended after every rate.
#
# Output dir: exp/run_sslo/output/v15_grid/Qwen3.5-35B-A3B/
# (separate root from 9B so both can coexist).
set -euo pipefail
cd /workspace/mlsys

ROOT=exp/run_sslo/output/v15_grid
mkdir -p "$ROOT/Qwen3.5-35B-A3B"
# Wipe only the 35B subtree, leave 9B intact.
rm -rf "$ROOT/Qwen3.5-35B-A3B"
mkdir -p "$ROOT/Qwen3.5-35B-A3B"

RATES="0.5 1 2 4 8 12 16 20"
MODES=(baseline sslo_mlp)
REPEATS=3

launch_job() {
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

build_jobs_for_repeat() {
  local model_slug=$1 model=$2 repeat=$3; shift 3
  local caps=("$@")
  local jobs=()
  for cap in "${caps[@]}"; do
    for mode in "${MODES[@]}"; do
      jobs+=("$model_slug|$model|$cap|$mode|$repeat")
    done
  done
  printf '%s\n' "${jobs[@]}"
}

worker() {
  local gpu=$1
  local job
  while IFS= read -r job; do
    [[ -z "$job" ]] && continue
    IFS='|' read -r ms m c md rp <<<"$job"
    launch_job "$ms" "$m" "$c" "$md" "$rp" "$gpu"
  done
}

run_phase() {
  local label=$1; shift
  local jobs=("$@")
  echo "===== phase: $label  (${#jobs[@]} jobs across 4 GPUs) ====="
  local q0=() q1=() q2=() q3=()
  for i in "${!jobs[@]}"; do
    case $((i % 4)) in
      0) q0+=("${jobs[i]}");;
      1) q1+=("${jobs[i]}");;
      2) q2+=("${jobs[i]}");;
      3) q3+=("${jobs[i]}");;
    esac
  done
  printf '%s\n' "${q0[@]}" | worker 0 &
  printf '%s\n' "${q1[@]}" | worker 1 &
  printf '%s\n' "${q2[@]}" | worker 2 &
  printf '%s\n' "${q3[@]}" | worker 3 &
  wait
  echo "===== phase done: $label ====="
}

for repeat in $(seq 1 "$REPEATS"); do
  mapfile -t JOBS < <(build_jobs_for_repeat \
      Qwen3.5-35B-A3B Qwen/Qwen3.5-35B-A3B "$repeat" 64 128 256 512)
  run_phase "35B-A3B repeat=$repeat" "${JOBS[@]}"
done

echo "All v15 grid (35B-A3B, 3 repeats) jobs done."
