#!/usr/bin/env bash
# Run one SSLO test config (5-way ablation: baseline / sslo / sslo_offload /
# sslo_adaptive / sslo_adaptive_offload, each in its own subprocess).
#
# OffloadingConnector (KVTransferConfig) is the always-on KV transfer substrate
# across all 5 conditions; the SSLO ablation toggles only sslo_config.offloading
# and sslo_config.adaptive_batch_size.
#
# Usage:
#   run_single.sh <max_num_seqs> <model> [num_prompts=256] [generation_max_tokens=512] [output_root=exp/sslo_test/output] [request_rate=0] [request_rate_seed=42]
#
# request_rate: Poisson arrival rate in reqs/sec. 0 (default) submits all at once.
#
# Optional env vars:
#   SSLO_KV_OFFLOAD_EXTRA  JSON dict for OffloadingConnector tuning.
#                          Default: '{}'. Example: '{"block_size": 64}'.
#
# Run inside the sk-sslo-vllm container. The repo is mounted at /workspace/mlsys.
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
# GPU memory utilization — overridable via env var since shared GPUs may be busy.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

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
# OffloadingConnector tuning passthrough (consumed by run_test.py).
# cpu_bytes_to_use is REQUIRED by CPUOffloadingSpec; default 16 GiB.
export SSLO_KV_OFFLOAD_EXTRA="${SSLO_KV_OFFLOAD_EXTRA:-{\"cpu_bytes_to_use\": 17179869184}}"

echo "[run_single] max_num_seqs=${MAX_NUM_SEQS} model=${MODEL} num_prompts=${NUM_PROMPTS} max_tokens=${GENERATION_MAX_TOKENS} request_rate=${REQUEST_RATE} (seed=${REQUEST_RATE_SEED})"
echo "[run_single] modes=baseline,sslo,sslo_offload,sslo_adaptive,sslo_adaptive_offload  kv_offload_extra=${SSLO_KV_OFFLOAD_EXTRA}"
echo "[run_single] output -> ${CONFIG_OUTPUT_DIR}"

python3 exp/sslo_test/run_test.py \
  --model "${MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --generation-max-tokens "${GENERATION_MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --chunk-unit "${CHUNK_UNIT}" \
  --seconds-per-word "${SECONDS_PER_WORD}" \
  --output-dir "${CONFIG_OUTPUT_DIR}" \
  --request-rate "${REQUEST_RATE}" \
  --request-rate-seed "${REQUEST_RATE_SEED}"
