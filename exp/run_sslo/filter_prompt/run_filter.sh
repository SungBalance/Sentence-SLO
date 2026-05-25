#!/usr/bin/env bash
# Prompt-filter pipeline launcher.
#
# Runs Qwen3.5-9B baseline on the ENTIRE current cache (combine_2.jsonl)
# at the same sampling parameters used by the regular sweep so that the
# outputs reflect what real sweep runs would produce. After the run,
# classify.py reads the outputs and identifies prompts whose responses
# contain garbage / non-English / repetition / markdown table / code
# block, and apply_filter.py removes them from the cache.
#
# Output: exp/run_sslo/filter_prompt/output/.../rate_32/chunks.jsonl + requests.jsonl
#
# Defaults (override via env):
#   MODEL=Qwen/Qwen3.5-9B            (per user — 9B most prone to noise)
#   CAP=128                          max_num_seqs
#   RATE=32                          single high rate to drain the pool
#   GPU=0
set -euo pipefail
cd /workspace/mlsys

OUTPUT_ROOT="${OUTPUT_ROOT:-exp/run_sslo/filter_prompt/output}"
ROOT="${OUTPUT_ROOT}/sentence"
mkdir -p "$ROOT"

MODEL_SLUG="${MODEL_SLUG:-Qwen3.5-9B}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CAP="${CAP:-128}"
RATE="${RATE:-32}"
GPU="${GPU:-0}"

# Use the entire cache as the prompt pool. combine_2.jsonl has ~3303
# unique prompts; we want to classify each once.
CACHE_PATH="${CACHE_PATH:-exp/tools/dataset_cache/combine_2.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-$(wc -l < "${CACHE_PATH}")}"

GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
DATASET_SEED="${DATASET_SEED:-42}"
MAX_RESPONSE_CHUNK_CHARS="${MAX_RESPONSE_CHUNK_CHARS:-1000}"

# read mode is enough to drive generation; CONSUME_MODE doesn't affect
# output text. Pin SECONDS_PER_WORD to default so deadline math is sane.
SECONDS_PER_WORD="${SECONDS_PER_WORD:-0.28}"

OUTDIR="${ROOT}/${MODEL_SLUG}/classification/run_1"
mkdir -p "$OUTDIR"

echo "[filter_prompt] model=${MODEL} cap=${CAP} rate=${RATE} num_prompts=${NUM_PROMPTS} gpu=${GPU}"
echo "  outdir=${OUTDIR}"

env CONSUME_MODE=read \
  TTS_MODEL= \
  TTS_PROFILE_PATH= \
  OUTPUT_DIR="$OUTDIR" \
  REQUEST_RATES="$RATE" \
  REQUEST_RATE_SEED="42" \
  REPEAT="1" \
  SUMMARY_CSV= \
  NUM_PROMPTS="$NUM_PROMPTS" \
  GENERATION_MAX_TOKENS="$GENERATION_MAX_TOKENS" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  CHUNK_UNIT=sentence \
  CUDA_VISIBLE_DEVICES="$GPU" \
  DATASET_NAME=koala \
  DATASET_SEED="$DATASET_SEED" \
  EXCLUDE_CODE=1 \
  CONVERSATION_ONLY=1 \
  ENGLISH_ONLY=1 \
  MAX_RESPONSE_CHUNK_CHARS="$MAX_RESPONSE_CHUNK_CHARS" \
  SECONDS_PER_WORD="$SECONDS_PER_WORD" \
  bash exp/run_sslo/filter_prompt/run_test.sh baseline "$CAP" "$MODEL" \
  2>&1 | tee "$OUTDIR/run.log"

echo
echo "[filter_prompt] run complete. Next:"
echo "  python3 exp/run_sslo/filter_prompt/classify.py --input ${OUTDIR}/rate_${RATE}"
echo "  python3 exp/run_sslo/filter_prompt/apply_filter.py --classification ${OUTDIR}/rate_${RATE}/classification.jsonl"
