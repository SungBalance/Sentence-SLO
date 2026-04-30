# Measure TTS Duration

This experiment measures how long TTS output becomes for dialogue text chunks.
It prepares a dialogue dataset without code-like turns, chunks each utterance at
sentence and paragraph boundaries, synthesizes audio, and then aggregates
duration statistics by word count.

Run from the host:

```bash
bash exp/measure_tts_duration/run_experiment.sh
```

All execution happens inside the `sk-sslo-omni` container.

## Pipeline

1. `prepare_dataset_chunks.py`
   - Loads the dialogue dataset.
   - Drops code-like turns.
   - Flattens dialogue turns and writes sentence/paragraph chunk rows.
2. `measure_audio_duration.py`
   - Loads one TTS model.
   - Synthesizes each chunk and records audio duration metadata.
3. `summarize_word_stats.py`
   - Groups duration rows by `word_count`.
   - Writes mean/variance tables.
4. `fit_duration_regression.py`
   - Fits `word_count -> duration_seconds` regression functions.
   - Writes per-model summaries, scatter-plus-fit figures, and reviewer-facing
     quantile regression figures (`p10`, `p50`, `p90`, `p95` by default).

## Default Models

- `hexgrad/Kokoro-82M`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

`microsoft/VibeVoice-Realtime-0.5B` is intentionally not included in this
experiment folder for now.

## Default Dataset

- `HuggingFaceH4/ultrachat_200k`
- split: `train_sft`

This experiment currently supports one dialogue dataset. The loader keeps
non-system turns and skips code-like text because the TTS models are being
measured on conversational text rather than code reading.

## Output Layout

```text
exp/measure_tts_duration/output/{dataset_slug}/
  sentence/
    text_chunks/
      chunks.jsonl
      chunks.csv
      summary.json
    hexgrad__Kokoro-82M/
      audio_durations/
        durations.jsonl
        durations.csv
        duration_cache.jsonl
      results/
        word_count_duration_stats.jsonl
        word_count_duration_stats.csv
        summary.json
    Qwen__Qwen3-TTS-12Hz-1.7B-Base/
      ...
    combined_results/
      word_count_duration_stats.jsonl
      word_count_duration_stats.csv
      summary.json
  paragraph/
    ...
```

`text_chunks/` is shared across models for the same chunk unit. Each model gets
its own duration rows and per-model summary. `combined_results/` contains the
cross-model table for that chunk unit.
