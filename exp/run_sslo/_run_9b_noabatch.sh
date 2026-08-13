#!/usr/bin/env bash
# One-off: 9B sslo_mlp with SSLO_ADAPTIVE_BATCHING=0, all caps x 3 consume modes
# x 5 rates x 1 run (12 cells across 4 GPUs, LPT).
# Purpose: isolate adaptive_batching effect at 9B (low-contention regime where
# SSLO regressed in main sweep).
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="exp/run_sslo/output_9b_noabatch"
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
export REQUEST_RATES="8 12 16 20 24"
export REQUEST_RATE_SEED=44
export REPEAT=1
export SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv"
# Key: disable adaptive_batching
export SSLO_ADAPTIVE_BATCHING=0

# Cells: "model_slug|model_hf|consume_mode|tts_slug|tts_model|cap|cost"
# All sslo_mlp, 4 caps x 3 consume modes = 12 cells.
# cost ~ cap (heavier caps run longer at saturation).
CELLS=(
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|256|900"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|128|600"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|64|400"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|read|none|-|32|300"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|256|900"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|128|600"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|64|400"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|hexgrad__Kokoro-82M|hexgrad/Kokoro-82M|32|300"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|256|900"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|128|600"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|64|400"
  "Qwen3.5-9B|Qwen/Qwen3.5-9B|tts|Supertone__supertonic-3|Supertone/supertonic-3|32|300"
)

NUM_GPUS=4
declare -a gpu_cost gpu_queue
for ((g=0; g<NUM_GPUS; g++)); do gpu_cost[g]=0; gpu_queue[g]=""; done

# LPT: assign each cell to currently-lightest GPU
for cell in "${CELLS[@]}"; do
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

echo "===== 9B sslo_mlp adaptive_batching=False 12 cells across $NUM_GPUS GPUs (LPT) ====="
for ((g=0; g<NUM_GPUS; g++)); do
  n=0; [[ -n "${gpu_queue[g]}" ]] && n=$(printf '%s\n' "${gpu_queue[g]}" | grep -c .)
  echo "  GPU${g}: cost=${gpu_cost[g]} cells=${n}"
done

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

pids=()
for ((g=0; g<NUM_GPUS; g++)); do
  printf '%s\n' "${gpu_queue[g]}" | worker "$g" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid" || echo "WARNING: worker pid=$pid exited non-zero" >&2
done
echo "===== 9B noabatch sweep complete ====="
