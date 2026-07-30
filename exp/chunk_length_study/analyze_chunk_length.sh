#!/usr/bin/env bash
# Analyze chunk-length vs request-output-length variability across the
# per-category baseline runs. Run inside sk-sslo-vllm from /workspace/mlsys.
set -euo pipefail
cd /workspace/mlsys

python3 exp/chunk_length_study/analyze_chunk_length.py "$@"
