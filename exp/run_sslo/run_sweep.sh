#!/usr/bin/env bash
# Full sweep launcher.
#
# One Python process per cell = (model, cap, consume_cell, mode, repeat).
# Each cell iterates ALL rates on a shared engine via REQUEST_RATES, so a
# given engine is loaded exactly once per cell.
#
# Output layout:
#   $OUTPUT_ROOT/sentence/<MODEL_SLUG>/<consume_mode>/<tts_slug>/cap<N>/<mode>/run_<i>/rate_<r>/
# where tts_slug is "none" for consume_mode=read, else the TTS HF id with
# '/' replaced by '__'. The model identity is recorded in each cell's
# sslo_config.json and surfaced as a column by sweep_analysis.py csv.
#
# Execution plan: 4 sequential phases, LPT-balanced inside each phase.
#   Phase 1: MODEL_SPECS[0] (default 9B)  run_1
#   Phase 2: MODEL_SPECS[1] (default 35B) run_1
#   Phase 3: MODEL_SPECS[0]               run_2..REPEATS
#   Phase 4: MODEL_SPECS[1]               run_2..REPEATS
#
# Env overrides:
#   MODEL_SPECS      space-sep "<slug>:<HF_id>" pairs (default 9B + 35B-A3B; must be 2)
#   CAPS             max_num_seqs values (space-sep)  (default "32 64 128 256")
#   MODES            modes (comma-sep)            (default baseline,progress_serve)
#                    selectable: baseline, progress_serve,
#                    progress_serve_adaptive, progress_serve_offload
#                    (e.g. MODES=baseline,progress_serve_offload)
#   REPEATS          number of repeats                (default 3)
#   RATES            rate ladder (space-sep)          (default "8 12 16 20 24")
#   CONSUME_CELLS    entries: "read" or "tts:<HF id>" (default read + 2 TTS models)
#   TTS_PROFILE_PATH path to profile CSV              (default exp/run_sslo/profiles/word_count_duration_stats.csv)
#   OUTPUT_ROOT      sweep destination dir            (default exp/run_sslo/output_sweep)
#   NUM_GPUS         parallel workers                 (default 4)
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="${OUTPUT_ROOT:-exp/run_sslo/output_sweep}"
ROOT="${OUTPUT_ROOT}/sentence"
mkdir -p "$ROOT"

MODEL_SPECS=(${MODEL_SPECS:-Qwen3.5-9B:Qwen/Qwen3.5-9B Qwen3.5-35B-A3B:Qwen/Qwen3.5-35B-A3B})
CAPS=(${CAPS:-32 64 128 256})
# Modes are env-selectable (comma-sep). Default stays baseline,progress_serve;
# add progress_serve_offload / progress_serve_adaptive when sweeping those.
MODES_CSV="${MODES:-baseline,progress_serve}"
IFS=',' read -ra MODE_ARR <<< "${MODES_CSV}"
REPEATS="${REPEATS:-1}"
RATES="${RATES:-8 12 16 20 24}"
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

# Parse "read" or "tts:<HF_id>" → echoes "<consume_mode> <tts_slug> <tts_model>"
# Uses "-" placeholder for empty tts_model (read mode) to keep token count fixed.
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

# Relative LPT cost. Only ordering matters; pick something that ranks
# 35B > 9B and scales with cap.
cell_cost() {
  local model_slug="$1" cap="$2"
  local base
  case "$model_slug" in
    *9B*)   base=9 ;;
    *35B*)  base=22 ;;
    *)      base=12 ;;
  esac
  # base minutes × 100 + cap so equal-base cells sort cap-desc.
  echo $((base * 100 + cap))
}

launch_job() {
  local model_slug="$1" model="$2" cap="$3" cell="$4" mode="$5" repeat="$6" gpu="$7"
  local consume_mode tts_slug tts_model_raw
  read -r consume_mode tts_slug tts_model_raw < <(parse_cell "$cell") || return 1
  local tts_model=""
  [[ "$tts_model_raw" != "-" ]] && tts_model="$tts_model_raw"
  local outdir="${ROOT}/${model_slug}/${consume_mode}/${tts_slug}/cap${cap}/${mode}/run_${repeat}"
  mkdir -p "$outdir"
  echo "[gpu=$gpu] ${model_slug}/${consume_mode}/${tts_slug} cap=${cap} mode=${mode} repeat=${repeat}"
  env CONSUME_MODE="$consume_mode" \
    TTS_MODEL="$tts_model" \
    TTS_PROFILE_PATH="${TTS_PROFILE_PATH}" \
    OUTPUT_DIR="$outdir" \
    REQUEST_RATES="$RATES" \
    REQUEST_RATE_SEED="$((43 + repeat))" \
    REPEAT="$repeat" \
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
    bash exp/run_sslo/run_test.sh "$mode" "$cap" "$model" \
    > "$outdir/run.log" 2>&1
}

# Build cells for (model_slug, model_hf, repeats…) — one line per cell:
#   "cost|model_slug|model|cap|cell|mode|repeat"
build_phase_jobs() {
  local model_slug="$1" model_hf="$2"; shift 2
  local rs=("$@")
  for cap in "${CAPS[@]}"; do
    for cell in "${CONSUME_CELLS[@]}"; do
      for mode in "${MODE_ARR[@]}"; do
        for repeat in "${rs[@]}"; do
          printf '%d|%s|%s|%d|%s|%s|%d\n' \
            "$(cell_cost "$model_slug" "$cap")" \
            "$model_slug" "$model_hf" "$cap" "$cell" "$mode" "$repeat"
        done
      done
    done
  done
}

# LPT: sort jobs cost-desc, greedy assign to currently lightest GPU.
run_phase() {
  local phase_name="$1"; shift
  local jobs=("$@")
  if (( ${#jobs[@]} == 0 )); then
    echo "[phase ${phase_name}] (no jobs)"
    return 0
  fi
  echo
  echo "===== phase ${phase_name}: ${#jobs[@]} cells across ${NUM_GPUS} GPUs (LPT) ====="
  local sorted
  sorted=$(printf '%s\n' "${jobs[@]}" | sort -t'|' -k1,1nr)
  declare -a gpu_cost gpu_queue
  for ((g=0; g<NUM_GPUS; g++)); do gpu_cost[g]=0; gpu_queue[g]=""; done
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    local cost="${entry%%|*}" job="${entry#*|}"
    local min_g=0
    for ((g=1; g<NUM_GPUS; g++)); do
      (( ${gpu_cost[g]} < ${gpu_cost[min_g]} )) && min_g=$g
    done
    gpu_cost[min_g]=$((${gpu_cost[min_g]} + cost))
    if [[ -z "${gpu_queue[min_g]}" ]]; then
      gpu_queue[min_g]="$job"
    else
      gpu_queue[min_g]+=$'\n'"$job"
    fi
  done <<< "$sorted"
  for ((g=0; g<NUM_GPUS; g++)); do
    local n_cells=0
    [[ -n "${gpu_queue[g]}" ]] && n_cells=$(printf '%s\n' "${gpu_queue[g]}" | grep -c .)
    echo "  GPU${g}: cost=${gpu_cost[g]} cells=${n_cells}"
  done
  worker() {
    local gpu=$1
    while IFS= read -r job; do
      [[ -z "$job" ]] && continue
      IFS='|' read -r ms mh c cell m rp <<<"$job"
      launch_job "$ms" "$mh" "$c" "$cell" "$m" "$rp" "$gpu"
    done
  }
  local pids=()
  for ((g=0; g<NUM_GPUS; g++)); do
    printf '%s\n' "${gpu_queue[g]}" | worker "$g" &
    pids+=($!)
  done
  local any_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "WARNING: worker pid=$pid exited non-zero" >&2
      any_failed=1
    fi
  done
  [[ "$any_failed" -ne 0 ]] && echo "phase ${phase_name} finished with worker failures"
  echo "===== phase ${phase_name} complete ====="
}

# ----- 4-phase plan ----------------------------------------------------
if (( ${#MODEL_SPECS[@]} != 2 )); then
  echo "ERROR: Phase plan requires exactly 2 entries in MODEL_SPECS; got ${#MODEL_SPECS[@]}." >&2
  exit 1
fi
m0_slug="${MODEL_SPECS[0]%%:*}"; m0_hf="${MODEL_SPECS[0]#*:}"
m1_slug="${MODEL_SPECS[1]%%:*}"; m1_hf="${MODEL_SPECS[1]#*:}"

later_repeats=()
for r in $(seq 2 "$REPEATS"); do later_repeats+=("$r"); done

mapfile -t p1 < <(build_phase_jobs "$m0_slug" "$m0_hf" 1)
mapfile -t p2 < <(build_phase_jobs "$m1_slug" "$m1_hf" 1)
p3=()
p4=()
if (( ${#later_repeats[@]} > 0 )); then
  mapfile -t p3 < <(build_phase_jobs "$m0_slug" "$m0_hf" "${later_repeats[@]}")
  mapfile -t p4 < <(build_phase_jobs "$m1_slug" "$m1_hf" "${later_repeats[@]}")
fi

phase3_label="2..${REPEATS}"
[[ "$REPEATS" -le 1 ]] && phase3_label="(skipped)"

# START_PHASE (default 1) lets you skip already-completed phases. Set to 3
# to run only run_2..REPEATS (preserving existing run_1 outputs).
START_PHASE="${START_PHASE:-1}"
(( START_PHASE <= 1 )) && run_phase "1 (${m0_slug} run_1)" "${p1[@]}"
(( START_PHASE <= 2 )) && run_phase "2 (${m1_slug} run_1)" "${p2[@]}"
(( START_PHASE <= 3 )) && run_phase "3 (${m0_slug} run_${phase3_label})" "${p3[@]}"
(( START_PHASE <= 4 )) && run_phase "4 (${m1_slug} run_${phase3_label})" "${p4[@]}"

echo
echo "===== sweep complete ====="
python3 exp/run_sslo/analysis/sweep_analysis.py csv \
  --sweep-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.csv"
