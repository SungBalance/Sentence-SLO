#!/usr/bin/env bash
# 9B + 35B sslo_mlp with SSLO_ADAPTIVE_BATCHING=0.
# Phase 1: caps [32, 64, 128, 256] x 3 consume x 2 models = 24 cells x 4 rates.
# Phase 2: cap [512] x 3 consume x 2 models = 6 cells x 4 rates.
# Rates: [8, 16, 24, 32]. REPEATS=1.
# LPT-balanced across 4 GPUs within each phase.
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="exp/run_sslo/output_noabatch"
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

# Cell format: "model_slug|model_hf|consume_mode|tts_slug|tts_model|cap|cost"
# cost ~ cap × model_factor (35B is ~3x slower than 9B).

PHASE1_CELLS=(
  # 9B (cost = cap)
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|256|800"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|128|500"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|64|350"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|32|250"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|256|800"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|128|500"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|64|350"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|32|250"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|256|800"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|128|500"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|64|350"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|32|250"
  # 35B-A3B (cost = cap × 3)
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|256|2400"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|128|1500"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|64|1050"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|32|750"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|256|2400"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|128|1500"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|64|1050"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|32|750"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|256|2400"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|128|1500"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|64|1050"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|32|750"
)

PHASE2_CELLS=(
  # cap=512
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|512|1600"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|512|1600"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|512|1600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|512|4800"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|512|4800"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|512|4800"
)

worker() {
  local gpu=$1
  while IFS='|' read -r model_slug model_hf consume_mode tts_slug tts_model_raw cap _cost; do
    [[ -z "$model_slug" ]] && continue
    local tts_model=""
    [[ "$tts_model_raw" != "-" ]] && tts_model="$tts_model_raw"
    local outdir="${ROOT}/${model_slug}/${consume_mode}/${tts_slug}/cap${cap}/sslo_mlp/run_1"
    mkdir -p "$outdir"
    echo "[gpu=$gpu] ${model_slug}/${consume_mode}/${tts_slug} cap=${cap} sslo_mlp(noabatch)"
    env CONSUME_MODE="$consume_mode" \
      TTS_MODEL="$tts_model" \
      OUTPUT_DIR="$outdir" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      bash exp/run_sslo/run_test.sh sslo_mlp "$cap" "$model_hf" \
      > "$outdir/run.log" 2>&1
  done
}

run_phase() {
  local phase_name="$1"; shift
  local cells=("$@")
  echo "===== ${phase_name}: ${#cells[@]} cells across ${NUM_GPUS} GPUs (LPT) ====="
  declare -a gpu_cost gpu_queue
  for ((g=0; g<NUM_GPUS; g++)); do gpu_cost[g]=0; gpu_queue[g]=""; done

  # LPT: sort cells desc by cost first, then assign
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
    printf '%s\n' "${gpu_queue[g]}" | worker "$g" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: worker pid=$pid exited non-zero" >&2
  done
  echo "===== ${phase_name} complete ====="
  echo
}

run_phase "Phase 1 (caps 32/64/128/256)" "${PHASE1_CELLS[@]}"
run_phase "Phase 2 (cap 512)" "${PHASE2_CELLS[@]}"

echo "===== ALL DONE ====="
