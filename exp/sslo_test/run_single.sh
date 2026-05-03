#!/usr/bin/env bash
# Run one SSLO test config (baseline + SSLO + adaptive SSLO subprocesses).
#
# Usage:
#   run_single.sh <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output] [request_rate=0] [request_rate_seed=42]
#
# request_rate: Poisson arrival rate in reqs/sec. 0 (default) submits all at once.
# Run inside the sk-sslo container. The repo is mounted at /workspace/mlsys.
# Output goes to ${output_root}/seqs_${max_num_seqs}/
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output] [request_rate=0] [request_rate_seed=42]" >&2
  exit 2
fi

# === vLLM args ===
MAX_NUM_SEQS="$1"
MODEL="$2"
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
# GPU_MEMORY_UTILIZATION lives in run_test.py argparse default (0.95).

# === Common (experiment) args ===
NUM_PROMPTS="${3:-256}"
GENERATION_MAX_TOKENS="${4:-512}"
OUTPUT_ROOT="${5:-exp/sslo_test/output}"
REQUEST_RATE="${6:-0}"
REQUEST_RATE_SEED="${7:-42}"
DATASET_NAME="koala"

# === SSLO args ===
CHUNK_UNIT="sentence"
SECONDS_PER_WORD=0.28

# ---------------------------------------------------------------
CONFIG_OUTPUT_DIR="${OUTPUT_ROOT}/seqs_${MAX_NUM_SEQS}"

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
# Pin to GPU 1 to avoid contention with other containers on GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "[run_single] max_num_seqs=${MAX_NUM_SEQS} model=${MODEL} num_prompts=${NUM_PROMPTS} max_tokens=${GENERATION_MAX_TOKENS} request_rate=${REQUEST_RATE} (seed=${REQUEST_RATE_SEED})"
echo "[run_single] output -> ${CONFIG_OUTPUT_DIR}"

python3 exp/sslo_test/run_test.py \
  --model "${MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --chunk-unit "${CHUNK_UNIT}" \
  --seconds-per-word "${SECONDS_PER_WORD}" \
  --output-dir "${CONFIG_OUTPUT_DIR}" \
  --request-rate "${REQUEST_RATE}" \
  --request-rate-seed "${REQUEST_RATE_SEED}"
