#!/usr/bin/env bash
# Global consumable-unit (chunk) length distribution per category + the
# requests whose chunk-length distribution diverges most from the global
# pool (two-sample KS). Pure offline analysis (no GPU). Run inside
# sk-sslo-vllm from /workspace/mlsys.
set -euo pipefail
cd /workspace/mlsys

python3 exp/chunk_length_study/analyze_unit_distribution.py "$@"
