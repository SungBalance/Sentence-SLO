#!/usr/bin/env bash
# Six-cell smoke test, 4-GPU parallel.
#
# Each cell = (CONSUME_CELL, MODE) on its own GPU. Cells are distributed
# round-robin across GPUs (default 4). Each cell internally sweeps all
# RATES through one shared engine.
#
# Output: $OUTPUT_ROOT/sentence/$MODEL_SLUG/<consume_mode>/<tts_slug>/cap<N>/<mode>/run_<i>/rate_<r>/
# (tts_slug = "none" for consume_mode=read, else TTS_MODEL with '/' -> '__')
#
# Env overrides:
#   CONSUME_CELLS  space-sep entries "read" or "tts:<HF_id>"
#                  (default "read tts:hexgrad/Kokoro-82M tts:Supertone/supertonic-3")
#   MODES          comma-sep                  (default "baseline,sslo_mlp")
#   MODEL_SLUG, MODEL, CAP, REPEAT, RATES
#   NUM_GPUS       parallel workers           (default 4)
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="${OUTPUT_ROOT:-exp/run_sslo/output_smoke}"
ROOT="${OUTPUT_ROOT}/sentence"
mkdir -p "$ROOT"

MODEL_SLUG="${MODEL_SLUG:-Qwen3.5-9B}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CAP="${CAP:-128}"
MODES="${MODES:-baseline,sslo_mlp}"
IFS=',' read -ra MODE_ARR <<< "${MODES}"
REPEAT="${REPEAT:-1}"
RATES="${RATES:-12 16}"
CONSUME_CELLS=(${CONSUME_CELLS:-read tts:hexgrad/Kokoro-82M tts:Supertone/supertonic-3})
NUM_GPUS="${NUM_GPUS:-4}"

TTS_PROFILE_PATH="${TTS_PROFILE_PATH:-exp/run_sslo/profiles/word_count_duration_stats.csv}"
NUM_PROMPTS="${NUM_PROMPTS:-4000}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
DATASET_NAME="${DATASET_NAME:-koala}"
DATASET_SEED="${DATASET_SEED:-42}"
EXCLUDE_CODE="${EXCLUDE_CODE:-1}"
CONVERSATION_ONLY="${CONVERSATION_ONLY:-1}"
ENGLISH_ONLY="${ENGLISH_ONLY:-1}"
MAX_RESPONSE_CHUNK_CHARS="${MAX_RESPONSE_CHUNK_CHARS:-1000}"
SECONDS_PER_WORD="${SECONDS_PER_WORD:-0.28}"

parse_cell() {
  local cell="$1"
  if [[ "$cell" == "read" ]]; then
    echo "read none -"
  elif [[ "$cell" == tts:* ]]; then
    local tts_model="${cell#tts:}"
    echo "tts ${tts_model//\//__} ${tts_model}"
  else
    echo "ERROR: CONSUME_CELLS entry must be 'read' or 'tts:<HF id>'; got '$cell'" >&2
    return 1
  fi
}

run_one_cell() {
  local cell="$1" mode="$2" gpu="$3"
  local consume_mode tts_slug tts_model_raw
  read -r consume_mode tts_slug tts_model_raw < <(parse_cell "$cell") || return 1
  local tts_model=""
  [[ "$tts_model_raw" != "-" ]] && tts_model="$tts_model_raw"
  local outdir="${ROOT}/${MODEL_SLUG}/${consume_mode}/${tts_slug}/cap${CAP}/${mode}/run_${REPEAT}"
  mkdir -p "$outdir"
  echo "[smoke gpu=${gpu}] ${MODEL_SLUG}/${consume_mode}/${tts_slug} cap=${CAP} mode=${mode} rates=${RATES}"

  env CONSUME_MODE="$consume_mode" \
    TTS_MODEL="$tts_model" \
    TTS_PROFILE_PATH="${TTS_PROFILE_PATH}" \
    OUTPUT_DIR="$outdir" \
    REQUEST_RATES="$RATES" \
    REQUEST_RATE_SEED="$((43 + REPEAT))" \
    REPEAT="$REPEAT" \
    SUMMARY_CSV="${OUTPUT_ROOT}/summary.csv" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    GENERATION_MAX_TOKENS="$GENERATION_MAX_TOKENS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    CHUNK_UNIT=sentence \
    CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_SEED="$DATASET_SEED" \
    EXCLUDE_CODE="$EXCLUDE_CODE" \
    CONVERSATION_ONLY="$CONVERSATION_ONLY" \
    ENGLISH_ONLY="$ENGLISH_ONLY" \
    MAX_RESPONSE_CHUNK_CHARS="$MAX_RESPONSE_CHUNK_CHARS" \
    SECONDS_PER_WORD="$SECONDS_PER_WORD" \
    bash exp/run_sslo/run_test.sh "$mode" "$CAP" "$MODEL" \
    > "$outdir/run.log" 2>&1
}

# Build (cell, mode) job list.
JOBS=()
for mode in "${MODE_ARR[@]}"; do
  for cell in "${CONSUME_CELLS[@]}"; do
    JOBS+=("$cell|$mode")
  done
done

# Round-robin assign to GPUs.
declare -a queues
for ((g=0; g<NUM_GPUS; g++)); do queues[g]=""; done
for i in "${!JOBS[@]}"; do
  g=$((i % NUM_GPUS))
  if [[ -z "${queues[g]}" ]]; then
    queues[g]="${JOBS[i]}"
  else
    queues[g]+=$'\n'"${JOBS[i]}"
  fi
done
echo "===== smoke: ${#JOBS[@]} cells across ${NUM_GPUS} GPUs ====="
for ((g=0; g<NUM_GPUS; g++)); do
  n=0
  [[ -n "${queues[g]}" ]] && n=$(printf '%s\n' "${queues[g]}" | grep -c .)
  echo "  GPU${g}: ${n} cells"
done

worker() {
  local gpu=$1
  while IFS= read -r job; do
    [[ -z "$job" ]] && continue
    IFS='|' read -r cell mode <<<"$job"
    run_one_cell "$cell" "$mode" "$gpu"
  done
}

pids=()
for ((g=0; g<NUM_GPUS; g++)); do
  printf '%s\n' "${queues[g]}" | worker "$g" &
  pids+=($!)
done

any_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "WARNING: worker pid=$pid exited non-zero" >&2
    any_failed=1
  fi
done

python3 exp/run_sslo/analysis/sweep_analysis.py csv \
  --sweep-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.csv"

echo
if [[ "$any_failed" -ne 0 ]]; then
  echo "smoke finished with cell failures."
  exit 1
fi
echo "smoke complete. ${#JOBS[@]} cells under ${ROOT}/${MODEL_SLUG}/"
