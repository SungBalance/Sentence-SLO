#!/usr/bin/env bash
# Run baseline (9B) once per (lang, task) category, feeding each category's
# prompt cache via DATASET_CACHE_DIR + DATASET_NAME=combine.
#
# Calls run_test.py DIRECTLY (not run_test.sh) because this study needs small,
# pool-sized --warmup-target / --measurement-target so the whole pool is the
# measurement window. run_test.sh doesn't forward those flags and its defaults
# (warmup=cap*2, measurement=1024) would hang on a 512-prompt pool. No existing
# script is modified.
#
# Prereq: build_category_pools.sh populated output/pools/<cat>/.
# Run inside the sk-sslo-vllm container from /workspace/mlsys.
set -euo pipefail
cd /workspace/mlsys

STUDY_DIR="exp/chunk_length_study"
POOLS_DIR="${STUDY_DIR}/output/pools"
RUNS_DIR="${STUDY_DIR}/output/runs"
mkdir -p "$RUNS_DIR"

export HF_HOME=/cache HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CAP="${CAP:-128}"
RATE="${RATE:-8}"
GPU="${GPU:-0}"
MIN_POOL="${MIN_POOL:-64}"
# Warmup gate small so most of the pool lands in the measurement window.
WARMUP="${WARMUP:-16}"

if [[ ! -d "$POOLS_DIR" ]]; then
  echo "ERROR: $POOLS_DIR not found — run build_category_pools.sh first." >&2
  exit 1
fi

for pool in "$POOLS_DIR"/*/; do
  cat=$(basename "$pool")
  n=$(wc -l < "${pool}/processed_dataset.jsonl" 2>/dev/null || echo 0)
  if (( n < MIN_POOL )); then
    echo "===== category=${cat}: SKIP (pool ${n} < ${MIN_POOL}) ====="
    continue
  fi
  # run_test.py appends rate_<r>/ to --output-dir, so pass the run_1 base.
  base_outdir="${RUNS_DIR}/${cat}/baseline/run_1"
  outdir="${base_outdir}/rate_${RATE}"
  mkdir -p "$outdir"
  # Measure every prompt after the warmup gate (cap below pool size).
  meas=$(( n - WARMUP ))
  (( meas < 1 )) && meas=1
  echo "===== category=${cat} → baseline ${MODEL} cap=${CAP} "\
"(pool=${n}, warmup=${WARMUP}, measure=${meas}) ====="

  export SSLO_STATS_LOG_PATH="${outdir}/scheduler_stats.jsonl"
  env \
    DATASET_CACHE_DIR="$(realpath "$pool")" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    python3 exp/run_sslo/run_test.py \
      --run-kind baseline \
      --model "$MODEL" \
      --max-num-seqs "$CAP" \
      --num-prompts "$n" \
      --warmup-target "$WARMUP" \
      --measurement-target "$meas" \
      --generation-max-tokens 2048 \
      --max-model-len 0 \
      --output-dir "$base_outdir" \
      --request-rates "$RATE" \
      --request-rate-seed 44 \
      --repeat 1 \
      --seconds-per-word 0.28 \
      --dataset-name combine \
      --dataset-seed 42 \
      --conversation-only \
      --max-response-chunk-chars 0 \
      --consume-mode read \
      > "${outdir}/run.log" 2>&1
  echo "  done → ${outdir}"
done

echo "===== all categories complete ====="
