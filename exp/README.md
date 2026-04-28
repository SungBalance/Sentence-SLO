# SSLO Experiment Code

Run from the appropriate Docker container, where this repository is mounted at
`/workspace/mlsys`.

```bash
bash exp/run_experiment.sh
```

`run_experiment.sh` is the only launcher. Options live near the top of the file:
models, dataset, slack modes, benchmark settings, and TTS settings.

## Pipeline

1. `benchmark.py` runs LLM inference on the dataset and records request/chunk
   timestamps.
2. `audio_duration.py` reads chunk output, runs vLLM-Omni TTS, and records
   per-chunk audio durations.
3. `analyze_results.py` reads text chunks and audio durations, computes one
   slack mode, writes result rows, and draws the figure.

The launcher runs stage 1 and stage 2 once per model/dataset, then runs stage 3
for both slack modes: `previous_chunk` and `cumulative`.

## Output Layout

```text
exp/output/{model_slug}/{dataset_slug}/{slack_mode}/
  text_outputs/
    requests.jsonl
    chunks.jsonl
    chunks.csv
    summary.json
  audio_durations/
    durations.jsonl
    durations.csv
    duration_cache.jsonl
  results/
    slack_rows.jsonl
    slack_rows.csv
    summary.json
    slack_distribution.png
```

`dataset_slug` is derived from `DATASET_PATH` when set, otherwise from
`DATASET_NAME`. Slashes are converted to `__` and other unsafe characters become
`_`.
