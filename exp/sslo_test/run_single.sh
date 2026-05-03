#!/usr/bin/env bash
# Run one SSLO test config (baseline + SSLO subprocesses, back-to-back).
#
# Usage:
#   run_single.sh <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output]
#
# Run inside the sk-sslo container. The repo is mounted at /workspace/mlsys.
# Output goes to ${output_root}/seqs_${max_num_seqs}/
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output]" >&2
  exit 2
fi

# === Common (experiment) args ===
NUM_PROMPTS="${3:-256}"
GENERATION_MAX_TOKENS="${4:-512}"
OUTPUT_ROOT="${5:-exp/sslo_test/output}"
DATASET_NAME="koala"

# === vLLM args ===
MODEL="$2"
MAX_NUM_SEQS="$1"
MAX_MODEL_LEN=8192
TENSOR_PARALLEL_SIZE=1
# GPU_MEMORY_UTILIZATION lives in run_test.py argparse default (0.95).

# === SSLO args ===
CHUNK_UNIT="sentence"
SECONDS_PER_WORD=0.28

# ---------------------------------------------------------------
CONFIG_OUTPUT_DIR="${OUTPUT_ROOT}/seqs_${MAX_NUM_SEQS}"

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

echo "[run_single] max_num_seqs=${MAX_NUM_SEQS} model=${MODEL} num_prompts=${NUM_PROMPTS} max_tokens=${GENERATION_MAX_TOKENS}"
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
  --output-dir "${CONFIG_OUTPUT_DIR}"
