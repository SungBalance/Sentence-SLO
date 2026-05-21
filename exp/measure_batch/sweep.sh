#!/usr/bin/env bash
# Sweep max_num_seqs for one model. Stops when throughput growth < 10%
# between consecutive batch sizes.
#
# Usage:
#   sweep.sh <model> <tp_size> <gpus> [start_batch]
#
# Example:
#   sweep.sh Qwen/Qwen3.5-9B 1 0           # 1 GPU on cuda:0
#   sweep.sh Qwen/Qwen3.5-27B 2 0,1        # 2 GPUs on cuda:0,1
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <model> <tp_size> <gpus> [start_batch]" >&2
  exit 1
fi

MODEL="$1"
TP_SIZE="$2"
GPUS="$3"
START_BATCH="${4:-8}"

cd /workspace/mlsys
# HF_TOKEN must be set in the environment by the caller (gated lmsys dataset).

slug() {
  echo "$1" | sed -e 's|/|__|g' -e 's|[^A-Za-z0-9._-]|_|g'
}
MODEL_SLUG=$(slug "$MODEL")
ROOT="exp/measure_batch/output/${MODEL_SLUG}"
mkdir -p "$ROOT"

# Batch sizes to try, powers of 2.
BATCHES=(8 16 32 64 128 256 512 1024)
PREV_TPUT=0

for batch in "${BATCHES[@]}"; do
  if (( batch < START_BATCH )); then continue; fi
  out="$ROOT/bsz_${batch}"
  mkdir -p "$out"
  echo "=== ${MODEL_SLUG} bsz=${batch} ==="
  CUDA_VISIBLE_DEVICES="$GPUS" \
    HF_HOME=/cache HF_HUB_CACHE=/cache/hub \
    python3 exp/measure_batch/run_batch.py \
      --model "$MODEL" \
      --tensor-parallel-size "$TP_SIZE" \
      --max-num-seqs "$batch" \
      --output-dir "$out" \
      > "$out/run.log" 2>&1 || { echo "  ERROR (exit $?); check $out/run.log"; break; }

  TPUT=$(python3 -c "import json; print(json.load(open('$out/result.json'))['throughput_tokens_per_second'])")
  echo "  throughput=${TPUT} tok/s"

  if (( $(python3 -c "print(1 if $PREV_TPUT > 0 else 0)") == 1 )); then
    RATIO=$(python3 -c "print($TPUT / $PREV_TPUT)")
    echo "  ratio vs prev=${RATIO}"
    if (( $(python3 -c "print(1 if $TPUT < $PREV_TPUT * 1.10 else 0)") == 1 )); then
      echo "  STOP: growth ${RATIO} < 1.10"
      break
    fi
  fi
  PREV_TPUT="$TPUT"
done

echo "Done: ${MODEL_SLUG}"
