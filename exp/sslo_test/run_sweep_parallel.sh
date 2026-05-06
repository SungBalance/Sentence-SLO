#!/usr/bin/env bash
# Thin wrapper: run_sweep.sh in 4-GPU parallel mode.
# Equivalent to: PARALLEL=4 bash exp/sslo_test/run_sweep.sh "$@"
#
# Usage:
#   run_sweep_parallel.sh [num_runs=3] [--label NAME]
#
# Run inside the sk-sslo container from /workspace/mlsys.
set -euo pipefail

PARALLEL=4 exec bash "$(dirname "$0")/run_sweep.sh" "$@"
