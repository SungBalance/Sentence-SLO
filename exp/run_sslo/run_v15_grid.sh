#!/usr/bin/env bash
# v15: new completion-gated, rate-sweep-on-shared-engine method.
# 9B only (35B paused). Full 9B grid:
#   max_num_seqs ∈ {16, 32, 64, 128}
#   rates per cell: {0.5, 1, 2, 4, 8, 12, 16, 20}  (swept under one engine)
#   modes: {baseline, sslo_mlp}
#   repeats: 3 per (cap, mode)  — OUTER loop, so repeat 1 finishes for every
#                                  (cap, mode) before repeat 2 starts.
# 4 GPUs in parallel. Each (cap, mode, repeat) is one job that internally
# sweeps the 8 rates on a shared engine. Output layout:
#   v15_grid/{model}/cap{cap}/{mode}/run_{repeat}/rate_{r}/
# Per-model summary CSV is updated after EACH rate (file-locked append):
#   v15_grid/{model}/summary.csv
set -euo pipefail
cd /workspace/mlsys

ROOT=exp/run_sslo/output/v15_grid
rm -rf "$ROOT"; mkdir -p "$ROOT"

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

# Build the list of (cap, mode) jobs for one repeat. Each entry:
#   "<model_slug>|<model>|<cap>|<mode>|<repeat>"
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

# OUTER loop = repeat. Inside each repeat: run all (cap, mode) jobs for
# 9B across 4 GPUs. Repeat 1 finishes for the whole grid before repeat 2
# starts — first-pass results land early so spot-checks are cheap.
for repeat in $(seq 1 "$REPEATS"); do
  mapfile -t JOBS < <(build_jobs_for_repeat \
      Qwen3.5-9B Qwen/Qwen3.5-9B "$repeat" 16 32 64 128)
  run_phase "9B repeat=$repeat" "${JOBS[@]}"
done

echo "All v15 grid (9B, 3 repeats) jobs done."
