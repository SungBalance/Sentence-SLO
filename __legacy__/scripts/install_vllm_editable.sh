#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/Sentence-SLO}"
VLLM_DIR="${REPO_ROOT}/vllm"

if [[ ! -d "${VLLM_DIR}" ]]; then
  echo "vLLM submodule path not found: ${VLLM_DIR}"
  echo "Run from repo root or pass repo path as first argument."
  exit 1
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "${VLLM_DIR}/requirements/build.txt"
python3 -m pip install -r "${VLLM_DIR}/requirements/cuda.txt"
python3 -m pip install -e "${VLLM_DIR}"

echo "vLLM editable install complete: ${VLLM_DIR}"