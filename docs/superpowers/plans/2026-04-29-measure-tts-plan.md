# measure_tts — Implementation Plan

**Date:** 2026-04-29  
**Spec:** `docs/superpowers/specs/2026-04-29-measure-tts-design.md`

## Task 1: Core experiment script

**File:** `exp/measure_tts/measure_tts_duration.py`

Implement the full measurement script:

1. CLI args: `--model` (required), `--chunk-unit` (`sentence`|`paragraph`, default `sentence`), `--output-dir`, `--dataset-split` (default `test`)
2. Dataset loading: call `eval_datasets.py` loader for `HuggingFaceH4/Koala-test-set`; filter prompts containing ``` or with non-alpha ratio > 0.4
3. Chunking: split each prompt text into chunks by sentence (split on `.!?` boundaries) or paragraph (`\n\n`); skip empty/whitespace-only chunks
4. TTS backends (if/elif on `--model`):
   - `hexgrad/Kokoro-82M`: use `kokoro` package, measure WAV duration
   - `microsoft/VibeVoice-Realtime-0.5B`: use `transformers` pipeline, measure output duration
   - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`: use `qwen-tts` package, measure WAV duration
5. Duration cache: load `cache.jsonl` at start; key = `md5(model_name + text)`; skip synthesis for cached chunks; append new entries immediately (resumable)
6. Output writing:
   - `durations.jsonl` + `durations.csv`: one row per chunk with `text`, `word_count`, `duration_sec`
   - `stats_raw.csv`: group by `word_count`, compute `n`, `mean_sec`, `var_sec`
   - `stats_binned.csv`: bins `1-5`, `6-10`, `11-20`, `21-50`, `51+`; compute `n`, `mean_sec`, `var_sec`
   - `summary.json`: model, chunk_unit, total_chunks, overall_mean_sec, overall_var_sec
7. Output path: `exp/measure_tts/output/{model_slug}/{chunk_unit}/` where slug = model name with `/` → `__`

**Verify:** `python3 -m py_compile exp/measure_tts/measure_tts_duration.py` inside `sk-sslo-omni`

---

## Task 2: Launcher script + README

**Files:** `exp/measure_tts/run_experiment.sh`, `exp/measure_tts/README.md`

1. `run_experiment.sh`:
   - Constants at top: `MODELS=("hexgrad/Kokoro-82M" "microsoft/VibeVoice-Realtime-0.5B" "Qwen/Qwen3-TTS-12Hz-1.7B-Base")`, `CHUNK_UNITS=("sentence" "paragraph")`
   - Set `HF_HOME=/cache HF_HUB_CACHE=/cache/hub`
   - Nested `for` loop over models × chunk units, calling `python3 exp/measure_tts/measure_tts_duration.py --model $MODEL --chunk-unit $CHUNK_UNIT --output-dir exp/measure_tts/output`
   - Run inside `sk-sslo-omni` container via `docker exec`

2. `README.md`: brief description of experiment, how to run, output layout (matching spec)

**Verify:** `bash -n exp/measure_tts/run_experiment.sh` inside `sk-sslo-omni`
