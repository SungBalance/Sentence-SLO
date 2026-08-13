#!/usr/bin/env bash
# Run_1 sweep for output_sweep_noabatch (post KV-block-recording change).
# 30 cells per model × 2 models × 1 run = 60 engine loads.
# 2 phases: 9B run_1, 35B run_1.
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="exp/run_sslo/output_sweep_noabatch"
ROOT="${OUTPUT_ROOT}/sentence"
mkdir -p "$ROOT"

export HF_HOME=/cache
export TTS_PROFILE_PATH="exp/run_sslo/profiles/word_count_duration_stats.csv"
export NUM_PROMPTS=4000
export GENERATION_MAX_TOKENS=2048
export MAX_MODEL_LEN=0
export CHUNK_UNIT=sentence
export DATASET_NAME=koala
export DATASET_SEED=42
export EXCLUDE_CODE=1
export CONVERSATION_ONLY=1
export ENGLISH_ONLY=1
export MAX_RESPONSE_CHUNK_CHARS=1000
export SECONDS_PER_WORD=0.28
export REQUEST_RATES="8 16 24 32"
export REQUEST_RATE_SEED=44
export REPEAT=1
export SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
export SSLO_ADAPTIVE_BATCHING=0

NUM_GPUS=4

# Cells: "model_slug|model_hf|consume_mode|tts_slug|tts_model|cap|mode|cost"
build_cells_for_model() {
  local model_slug=$1 model_hf=$2 model_factor=$3
  for mode in baseline sslo_mlp; do
    for cap in 32 64 128 256 512; do
      local cost=$(( cap * model_factor ))
      echo "$model_slug|$model_hf|read|none|-|$cap|$mode|$cost"
      echo "$model_slug|$model_hf|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|$cap|$mode|$cost"
      echo "$model_slug|$model_hf|tts|Supertone__supertonic-3|Supertone/supertonic-3|$cap|$mode|$cost"
    done
  done
}

worker() {
  local gpu=$1 run_idx=$2
  while IFS='|' read -r model_slug model_hf consume_mode tts_slug tts_model_raw cap mode _cost; do
    [[ -z "$model_slug" ]] && continue
    local tts_model=""
    [[ "$tts_model_raw" != "-" ]] && tts_model="$tts_model_raw"
    local outdir="${ROOT}/${model_slug}/${consume_mode}/${tts_slug}/cap${cap}/${mode}/run_${run_idx}"
    mkdir -p "$outdir"
    echo "[gpu=$gpu] ${model_slug}/${consume_mode}/${tts_slug} cap=${cap} ${mode} run_${run_idx}"
    env CONSUME_MODE="$consume_mode" \
      TTS_MODEL="$tts_model" \
      OUTPUT_DIR="$outdir" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      bash exp/run_sslo/run_test.sh "$mode" "$cap" "$model_hf" \
      > "$outdir/run.log" 2>&1
  done
}

run_phase() {
  local phase_name="$1" run_idx="$2"; shift 2
  local cells=("$@")
  echo "===== ${phase_name}: ${#cells[@]} cells across ${NUM_GPUS} GPUs (LPT) ====="
  declare -a gpu_cost gpu_queue
  for ((g=0; g<NUM_GPUS; g++)); do gpu_cost[g]=0; gpu_queue[g]=""; done
  local sorted=()
  while IFS= read -r line; do sorted+=("$line"); done < <(
    printf '%s\n' "${cells[@]}" | awk -F'|' '{print $NF, $0}' | sort -rn | cut -d' ' -f2-)
  for cell in "${sorted[@]}"; do
    cost="${cell##*|}"
    min_g=0
    for ((g=1; g<NUM_GPUS; g++)); do
      (( gpu_cost[g] < gpu_cost[min_g] )) && min_g=$g
    done
    gpu_cost[min_g]=$(( ${gpu_cost[min_g]} + cost ))
    if [[ -z "${gpu_queue[min_g]}" ]]; then
      gpu_queue[min_g]="$cell"
    else
      gpu_queue[min_g]+=$'\n'"$cell"
    fi
  done
  for ((g=0; g<NUM_GPUS; g++)); do
    n=0; [[ -n "${gpu_queue[g]}" ]] && n=$(printf '%s\n' "${gpu_queue[g]}" | grep -c .)
    echo "  GPU${g}: cost=${gpu_cost[g]} cells=${n}"
  done
  local pids=()
  for ((g=0; g<NUM_GPUS; g++)); do
    printf '%s\n' "${gpu_queue[g]}" | worker "$g" "$run_idx" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: worker pid=$pid exited non-zero" >&2
  done
  echo "===== ${phase_name} complete ====="
  echo
}

NINE_B_CELLS=()
while IFS= read -r line; do NINE_B_CELLS+=("$line"); done < <(
  build_cells_for_model "Qwen3.5-9B" "Qwen/Qwen3.5-9B" 1)

THIRTY5_B_CELLS=()
while IFS= read -r line; do THIRTY5_B_CELLS+=("$line"); done < <(
  build_cells_for_model "Qwen3.5-35B-A3B" "Qwen/Qwen3.5-35B-A3B" 3)

run_phase "Phase 1 (9B run_1)"   1 "${NINE_B_CELLS[@]}"
run_phase "Phase 2 (35B run_1)"  1 "${THIRTY5_B_CELLS[@]}"

echo "===== ALL RUN_1 DONE ====="
