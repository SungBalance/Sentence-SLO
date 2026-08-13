#!/usr/bin/env bash
# Profile per-chunk TTS conversion_time / audio_duration / RTF for both
# Kokoro and Supertonic-3 on the chunks copied into input_chunks.jsonl.
#
# Usage:
#   cp exp/run_sslo/output/v15_grid/.../rate_<R>/chunks.jsonl \
#      exp/measure_tts_duration/input_chunks.jsonl
#   bash exp/measure_tts_duration/run_experiment.sh
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sk-sslo-vllm}"
CONTAINER_REPO="/workspace/mlsys"
EXP_ROOT="${CONTAINER_REPO}/exp/measure_tts_duration"
OUTPUT_ROOT="${EXP_ROOT}/output"
INPUT_JSONL="${EXP_ROOT}/input_chunks.jsonl"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
# SSLO
DATASET="${DATASET:-combine}"
# SSLO
MAX_DIALOGUES="${MAX_DIALOGUES:-2048}"
# SSLO
CHUNK_UNIT_FOR_TTS="${CHUNK_UNIT_FOR_TTS:-sentence}"
# SSLO
PREPARE_OUTPUT_ROOT="${EXP_ROOT}/output/prepared/${DATASET}"
# SSLO
SKIP_PREPARE="${SKIP_PREPARE:-0}"
# SSLO
HF_TOKEN="${HF_TOKEN:-}"
# Optional: cap chunks for a smoke run by exporting MAX_ROWS=N
MAX_ROWS_ARG=""
if [[ -n "${MAX_ROWS:-}" ]]; then
  MAX_ROWS_ARG="--max-rows ${MAX_ROWS}"
fi

# Per-backend output dirs (slug = repo path with / -> __).
KOKORO_DIR="${OUTPUT_ROOT}/hexgrad__Kokoro-82M"
# SSLO
SUPERTONIC_DIR="${OUTPUT_ROOT}/Supertone__supertonic-3"

require_container() {
  if ! docker inspect "$1" >/dev/null 2>&1; then
    echo "Required container is not available: $1" >&2
    exit 1
  fi
}

run_in_container() {
  docker exec -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HUB_TOKEN="${HF_TOKEN}" \
    "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    cd ${CONTAINER_REPO}
    export HF_HOME=/cache
    $*
  "
}

ensure_dependencies() {
  docker exec "${CONTAINER_NAME}" bash -lc '
    set -euo pipefail
    export HF_HOME=/cache
    python3 -m pip install -q "kokoro==0.9.4" "supertonic" soundfile huggingface_hub
    if ! command -v espeak-ng >/dev/null 2>&1; then
      apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng
    fi
  '
}

require_container "${CONTAINER_NAME}"

# SSLO
if [[ -z "${HF_TOKEN}" && "${SKIP_PREPARE}" != "1" ]]; then
  echo "HF_TOKEN is required when preparing dialogue chunks. Export HF_TOKEN or set SKIP_PREPARE=1." >&2
  exit 1
fi

# SSLO
if [[ "${SKIP_PREPARE}" == "1" ]]; then
  # Sanity: input file must be present
  if ! docker exec "${CONTAINER_NAME}" test -f "${INPUT_JSONL}"; then
    echo "Missing ${INPUT_JSONL}. Copy a chunks.jsonl from one run_sslo cell." >&2
    exit 1
  fi
else
  # SSLO: prepare can abort during HF datasets cleanup (PyGILState race)
  # even after writing chunks.jsonl successfully. Tolerate the non-zero
  # exit code, then assert the output exists before continuing.
  run_in_container "python3 ${EXP_ROOT}/prepare_dataset_chunks.py \
    --dataset-name ${DATASET} \
    --max-dialogues ${MAX_DIALOGUES} \
    --output-root ${PREPARE_OUTPUT_ROOT} || true"
  run_in_container "test -s \
    ${PREPARE_OUTPUT_ROOT}/${CHUNK_UNIT_FOR_TTS}/text_chunks/chunks.jsonl"
  run_in_container "cp \
    ${PREPARE_OUTPUT_ROOT}/${CHUNK_UNIT_FOR_TTS}/text_chunks/chunks.jsonl \
    ${INPUT_JSONL}"
fi

ensure_dependencies

# SSLO: parallel shard config — Kokoro fans out across 4 GPUs (one process
# per GPU). Supertonic fans out into 4 CPU processes with ONNX intra_op=auto.
# The Kokoro phase finishes before the Supertonic phase begins.
SHARDS=4

# SSLO: helper — run measure_audio_duration in the background inside the
# container with a given CUDA_VISIBLE_DEVICES and shard id. Per-shard logs
# go to ${out_dir}_shard${shard_id}.log INSIDE the container (mounted
# under the same /workspace/mlsys path).
run_shard_async() {
  local model="$1" out_dir="$2" shard_id="$3" cuda_devs="$4"
  # SSLO: optional extra CLI args (e.g. "--supertonic-intra-op-num-threads 8")
  local extra_args="${5:-}"
  local log_path="${out_dir}_shard${shard_id}.log"
  docker exec -e CUDA_VISIBLE_DEVICES="${cuda_devs}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HF_HUB_TOKEN="${HF_TOKEN}" \
    "${CONTAINER_NAME}" bash -lc "
      set -euo pipefail
      cd ${CONTAINER_REPO}
      export HF_HOME=/cache
      mkdir -p \$(dirname '${log_path}')
      python3 ${EXP_ROOT}/measure_audio_duration.py \
        --input-jsonl ${INPUT_JSONL} \
        --output-dir ${out_dir}/shard_${shard_id} \
        --tts-model ${model} \
        --shard-id ${shard_id} --shard-count ${SHARDS} \
        ${MAX_ROWS_ARG} ${extra_args} > ${log_path} 2>&1
    " &
}

# SSLO: concat per-shard durations.{jsonl,csv} into a single set under out_dir.
concat_shards() {
  local out_dir="$1"
  run_in_container "
    cd ${out_dir}
    # JSONL: cat all shard outputs.
    : > durations.jsonl
    for i in \$(seq 0 $((SHARDS-1))); do
      cat shard_\${i}/durations.jsonl >> durations.jsonl
    done
    # CSV: header from shard_0 + tail from each.
    head -n 1 shard_0/durations.csv > durations.csv
    for i in \$(seq 0 $((SHARDS-1))); do
      tail -n +2 shard_\${i}/durations.csv >> durations.csv
    done
    wc -l durations.csv durations.jsonl
  "
}

# SSLO: Kokoro phase — 4 GPU, fan out then wait.
echo "[kokoro] launching ${SHARDS} shards across GPU 0..$((SHARDS-1))"
for s in $(seq 0 $((SHARDS-1))); do
  run_shard_async "hexgrad/Kokoro-82M" "${KOKORO_DIR}" "${s}" "${s}"
done
wait
concat_shards "${KOKORO_DIR}"

# SSLO: Supertonic phase — 4 CPU processes, intra_op capped to avoid
# CPU oversubscription. Without cap, 4 procs × default(~all cores) threads
# thrash the scheduler. 8 threads/proc × 4 procs = 32 threads fits a
# typical 32-core box.
SUPERTONIC_INTRA_OP="${SUPERTONIC_INTRA_OP:-8}"
echo "[supertonic] launching ${SHARDS} CPU shards (intra_op=${SUPERTONIC_INTRA_OP})"
for s in $(seq 0 $((SHARDS-1))); do
  # CUDA_VISIBLE_DEVICES="" hides GPUs from the ONNX CPU runtime.
  run_shard_async "Supertone/supertonic-3" "${SUPERTONIC_DIR}" "${s}" "" \
    "--supertonic-intra-op-num-threads ${SUPERTONIC_INTRA_OP}"
done
wait
concat_shards "${SUPERTONIC_DIR}"

# Concat both backends' csvs into one (header from Kokoro, append Supertonic data rows)
run_in_container "
  head -n 1 ${KOKORO_DIR}/durations.csv > ${OUTPUT_ROOT}/combined_durations.csv
  tail -n +2 ${KOKORO_DIR}/durations.csv >> ${OUTPUT_ROOT}/combined_durations.csv
  tail -n +2 ${SUPERTONIC_DIR}/durations.csv >> ${OUTPUT_ROOT}/combined_durations.csv
  wc -l ${OUTPUT_ROOT}/combined_durations.csv
"

echo "Done. Per-model CSVs in:"
echo "  ${KOKORO_DIR}/durations.csv"
echo "  ${SUPERTONIC_DIR}/durations.csv"
echo "Combined: ${OUTPUT_ROOT}/combined_durations.csv"
