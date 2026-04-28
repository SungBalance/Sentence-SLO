# SSLO Experiment Code

Run from the appropriate Docker container, where this repository is mounted at
`/workspace/mlsys`.

```bash
bash exp/run_experiment.sh
```

`run_experiment.sh` is the only launcher. Options live near the top of the file:
models, dataset, slack modes, benchmark settings, and TTS settings.

`benchmark.py` fixes the benchmark backend to vLLM async engine mode and always
applies the model chat template, so the launcher does not pass those flags.

For human-reading-only runs, use `run_experiment_read.sh`. It skips TTS and
attaches `CHUNK_UNITS=("sentence" "paragraph")` collectors to the same inference
pass, so the same model/dataset output can be compared with sentence chunks and
paragraph chunks.

Stage 1 runs a short GPU warmup before measured inference. `REQUEST_RATE="inf"`
keeps the previous all-at-once engine request injection behavior; set it to a
positive requests/sec value to pace request arrivals. `REQUEST_BURSTINESS=1.0`
uses Poisson/exponential arrivals.
`MAX_CHUNKS_PER_REQUEST=48` keeps at most 48 streamed sentence chunks for each
measured request before writing text outputs for TTS and analysis.

## Pipeline

1. `benchmark.py` runs LLM inference on the dataset and records request metrics
   from standard vLLM `RequestOutput.metrics` plus chunk timestamps from the
   benchmark's streamed `RequestOutput` receive path.
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

`requests.jsonl` includes `request_submit_ts`, `scheduled_ts`,
`first_token_ts`, and `last_token_ts` relationships. Chunk `start_time_ts` and
`end_time_ts` are captured when `benchmark.py` receives streamed outputs, so the
pipeline does not depend on SSLO-specific vLLM fields such as `chunk_timings`.
Each request row also records whether chunk output was truncated by
`MAX_CHUNKS_PER_REQUEST`.

The read-only launcher writes chunk comparison outputs under:

```text
exp/output/{model_slug}/{dataset_slug}/{chunk_unit}/{slack_mode}/
```

`dataset_slug` is derived from `DATASET_PATH` when set, otherwise from
`DATASET_NAME`. Slashes are converted to `__` and other unsafe characters become
`_`.
