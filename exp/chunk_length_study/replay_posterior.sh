#!/usr/bin/env bash
# Oracle vs online-posterior replay over the per-category baseline runs.
# Pure offline replay of chunks.jsonl (no GPU). Run inside sk-sslo-vllm
# from /workspace/mlsys.
set -euo pipefail
cd /workspace/mlsys

python3 exp/chunk_length_study/replay_posterior.py "$@"
