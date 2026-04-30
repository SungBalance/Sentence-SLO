# measure_tts — Design Spec

**Date:** 2026-04-29

## Goal

Measure TTS audio duration per text chunk across 3 TTS models and produce a word-count × duration statistics table (mean, variance) for both raw word counts and binned word counts.

## Dataset

- **Source:** `HuggingFaceH4/Koala-test-set` (loaded via existing `exp/slack_dist/eval_datasets.py`)
- **Code filtering:** chunks containing ``` backticks or with non-alphabetic character ratio > 0.4 are skipped

## Chunking

Both `sentence` and `paragraph` units are collected, each in its own output subdirectory. Sentence boundaries use the same punctuation-based detection as `exp/slack_dist/benchmark.py`. Paragraph boundaries split on `\n\n`.

## TTS Models

| Model | Backend |
|---|---|
| `hexgrad/Kokoro-82M` | `kokoro` package |
| `microsoft/VibeVoice-Realtime-0.5B` | HuggingFace `transformers` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `qwen-tts` package |

All three run in the same container (`sk-sslo-omni`). The `--model` CLI flag selects the backend via if/elif dispatch inside the script.

## Architecture

Single script: `exp/measure_tts/measure_tts_duration.py`

Flow:
1. Load dataset, apply code filter
2. Chunk text (sentence + paragraph)
3. For each chunk: synthesize via selected TTS backend → measure audio duration in seconds
4. Cache results to `cache.jsonl` (keyed by `md5(model + text)`) — resumable
5. Write `durations.jsonl`, `durations.csv`
6. Aggregate `stats_raw.csv` and `stats_binned.csv`

## Output Layout

```
exp/measure_tts/output/{tts_model_slug}/{chunk_unit}/
  durations.jsonl        # per-chunk: text, word_count, duration_sec
  durations.csv
  cache.jsonl            # resumable TTS cache
  stats_raw.csv          # word_count | n | mean_sec | var_sec
  stats_binned.csv       # bin_label | n | mean_sec | var_sec
  summary.json           # model, chunk_unit, total_chunks, overall_mean, overall_var
```

Bins: `1-5`, `6-10`, `11-20`, `21-50`, `51+`

## Launcher

`exp/measure_tts/run_experiment.sh` iterates over the 3 TTS models and both chunk units as shell variables with a `for` loop. No extra arguments needed to run.

## Files Created

- `exp/measure_tts/measure_tts_duration.py`
- `exp/measure_tts/run_experiment.sh`
- `exp/measure_tts/README.md`
