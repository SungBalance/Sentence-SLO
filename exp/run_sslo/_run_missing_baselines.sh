#!/usr/bin/env bash
# Fill in missing baseline cells for output_sweep_noabatch:
# Phase 1: caps [32, 64, 128, 256] x baseline x rate=32 only (24 cells x 1 rate)
# Phase 2: cap=512 x baseline x rates [8, 16, 24, 32] (6 cells x 4 rates)
# Outputs directly to output_sweep_noabatch/.
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="exp/run_sslo/output_sweep_noabatch"
ROOT="${OUTPUT_ROOT}/sentence"
mkdir -p "$ROOT"

export HF_HOME=/cache HF_HUB_CACHE=/cache/hub
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
export REQUEST_RATE_SEED=44
export REPEAT=1
export SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"

NUM_GPUS=4

# Cell format: "model_slug|model_hf|consume_mode|tts_slug|tts_model|cap|rates_csv|cost"
PHASE1_CELLS=(
  # rate=32 only for cap [32, 64, 128, 256] — 24 cells (2 models × 4 caps × 3 consume)
  # 9B (cost ~ cap)
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|256|32|200"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|128|32|150"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|64|32|100"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|32|32|80"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|256|32|200"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|128|32|150"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|64|32|100"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|32|32|80"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|256|32|200"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|128|32|150"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|64|32|100"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|32|32|80"
  # 35B (cost ~ 3x cap)
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|256|32|600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|128|32|450"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|64|32|300"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|32|32|240"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|256|32|600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|128|32|450"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|64|32|300"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|32|32|240"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|256|32|600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|128|32|450"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|64|32|300"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|32|32|240"
)

PHASE2_CELLS=(
  # cap=512 baseline at all 4 rates — 6 cells
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|512|8 16 24 32|1200"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|512|8 16 24 32|1200"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|512|8 16 24 32|1200"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|read|none|-|512|8 16 24 32|3600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|512|8 16 24 32|3600"
  "Qwen3.5-35B-A3B|Qwen/Qwen3.5-35B-A3B|tts|Supertone__supertonic-3|Supertone/supertonic-3|512|8 16 24 32|3600"
)

worker() {
  local gpu=$1
  while IFS='|' read -r model_slug model_hf consume_mode tts_slug tts_model_raw cap rates _cost; do
    [[ -z "$model_slug" ]] && continue
    local tts_model=""
    [[ "$tts_model_raw" != "-" ]] && tts_model="$tts_model_raw"
    local outdir="${ROOT}/${model_slug}/${consume_mode}/${tts_slug}/cap${cap}/baseline/run_1"
    mkdir -p "$outdir"
    echo "[gpu=$gpu] ${model_slug}/${consume_mode}/${tts_slug} cap=${cap} baseline rates=[${rates}]"
    env CONSUME_MODE="$consume_mode" \
      TTS_MODEL="$tts_model" \
      OUTPUT_DIR="$outdir" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      REQUEST_RATES="$rates" \
      bash exp/run_sslo/run_test.sh baseline "$cap" "$model_hf" \
      > "$outdir/run.log" 2>&1
  done
}

run_phase() {
  local phase_name="$1"; shift
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
    printf '%s\n' "${gpu_queue[g]}" | worker "$g" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: worker pid=$pid exited non-zero" >&2
  done
  echo "===== ${phase_name} complete ====="
  echo
}

run_phase "Phase 1 (caps 32-256 rate=32)" "${PHASE1_CELLS[@]}"
run_phase "Phase 2 (cap=512 all rates)" "${PHASE2_CELLS[@]}"

echo "===== ALL MISSING BASELINES DONE ====="
