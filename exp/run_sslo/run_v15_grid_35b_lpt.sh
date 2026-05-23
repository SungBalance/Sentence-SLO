#!/usr/bin/env bash
# 35B-A3B grid — LPT GPU assignment (no idle GPU per repeat).
#
# Static job-to-GPU map (per cap-minute estimates, cap=64 ~47 / cap=128
# ~70 / cap=256 ~120 / cap=512 ~215):
#   GPU 0: cap=512 baseline                                       (215 min)
#   GPU 1: cap=512 sslo_mlp                                       (215 min)
#   GPU 2: cap=256 baseline → cap=128 baseline → cap=64 baseline  (237 min)
#   GPU 3: cap=256 sslo_mlp → cap=128 sslo_mlp → cap=64 sslo_mlp  (237 min)
# Critical path ≈ 237 min vs round-robin's ≈ 285 min (~17% faster).
#
# Outputs go to v15_grid/Qwen3.5-35B-A3B/cap{C}/{mode}/run_{R}/rate_{r}/
# and the per-model summary.csv is appended after every rate.
#
# Env: START_REPEAT (default 1), END_REPEAT (default 3).
# Cells already in summary.csv are NOT skipped — caller is responsible
# for removing prior per-cell output dirs if rerunning is intended.
set -euo pipefail
cd /workspace/mlsys

ROOT=exp/run_sslo/output/v15_grid
mkdir -p "$ROOT/Qwen3.5-35B-A3B"

RATES="0.5 1 2 4 8 12 16 20"
START_REPEAT=${START_REPEAT:-1}
END_REPEAT=${END_REPEAT:-3}

launch() {
  local cap=$1 mode=$2 repeat=$3 gpu=$4
  local outdir="$ROOT/Qwen3.5-35B-A3B/cap${cap}/$mode/run_${repeat}"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] cap=$cap mode=$mode repeat=$repeat"
  OUTPUT_DIR="$outdir" REQUEST_RATES="$RATES" REQUEST_RATE_SEED="$((43 + repeat))" \
    REPEAT="$repeat" SUMMARY_CSV="$ROOT/Qwen3.5-35B-A3B/summary.csv" \
    NUM_PROMPTS=4096 GENERATION_MAX_TOKENS=2048 \
    CHUNK_UNIT=sentence CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 \
    CONVERSATION_ONLY=1 ENGLISH_ONLY=1 MAX_RESPONSE_CHUNK_CHARS=1000 \
    bash exp/run_sslo/run_test.sh "$mode" "$cap" Qwen/Qwen3.5-35B-A3B \
    > "$outdir/run.log" 2>&1
}

gpu0() { local rp=$1; launch 512 baseline "$rp" 0; }
gpu1() { local rp=$1; launch 512 sslo_mlp "$rp" 1; }
gpu2() {
  local rp=$1
  launch 256 baseline "$rp" 2
  launch 128 baseline "$rp" 2
  launch 64  baseline "$rp" 2
}
gpu3() {
  local rp=$1
  launch 256 sslo_mlp "$rp" 3
  launch 128 sslo_mlp "$rp" 3
  launch 64  sslo_mlp "$rp" 3
}

for repeat in $(seq "$START_REPEAT" "$END_REPEAT"); do
  echo "===== phase: 35B-A3B repeat=$repeat (LPT) ====="
  gpu0 "$repeat" &
  gpu1 "$repeat" &
  gpu2 "$repeat" &
  gpu3 "$repeat" &
  wait
  echo "===== phase done: 35B-A3B repeat=$repeat ====="
done

echo "All v15 grid (35B-A3B, LPT, repeats $START_REPEAT..$END_REPEAT) jobs done."
