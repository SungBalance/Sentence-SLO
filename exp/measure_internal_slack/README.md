# measure_internal_slack

Measures the Cumulative Slack Time distribution for human reading across three
dimensions: model size, chunk index, and chunk type.

## Purpose

This experiment asks: *does the LLM generate text fast enough for a human reader
to consume it chunk-by-chunk without waiting?* It uses the cumulative slack
metric — the time by which each chunk arrives ahead of (or behind) the moment
the reader would have finished all previous chunks.

The three plots produced by Stage 3 answer:

1. **`slack_by_model.png`** — Does model size (35B vs 27B) affect slack?
2. **`slack_by_chunk_index.png`** — Does slack improve as generation warms up
   (chunk 1 vs 2 vs 3 vs 4 vs 5+)? Chunk index 0 is excluded because no prior
   chunk exists to define a deadline.
3. **`slack_by_chunk_type.png`** — Does chunking at sentence vs paragraph
   boundaries change the distribution shape?

Positive slack means the chunk arrived before the reader finished the previous
one (good). Negative slack means the reader had to wait (bad).

## Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `exp/slack_dist/benchmark.py` | vLLM inference; records sentence and paragraph chunk timelines in one pass |
| 2 | `exp/slack_dist/analyze_results.py` | Computes cumulative human slack rows (`slack_rows.jsonl`) |
| 3 | `exp/measure_internal_slack/analyze_slack_dist.py` | Joins all runs; produces three plots and `summary.csv` |

## How to run

```bash
bash exp/measure_internal_slack/run_experiment.sh
```

Both `sk-sslo` containers must be running. The script handles all three stages
in sequence across both models (`Qwen3.5-35B-A3B`, `Qwen3.5-27B`) and both
chunk types (`sentence`, `paragraph`).

## Output layout

```
exp/output/measure_internal_slack/
  {model_slug}/{dataset_slug}/{chunk_type}/
    text_outputs/          <- Stage 1: chunks.jsonl, requests.jsonl, summary.json
    cumulative/
      results/             <- Stage 2: slack_rows.jsonl, summary.json, slack_distribution.png
  analysis/                <- Stage 3: three plots + summary.csv
```

## Interpreting the plots

All three plots show kernel-density-style histograms of `human_slack_seconds`
with a vertical dashed line at x=0.

- **Area left of zero**: fraction of chunks where the reader had to wait.
- **Distribution center**: typical slack; a rightward shift means more headroom.
- **Distribution width**: variability — a narrow peak is more predictable.

The `summary.csv` table gives `p50`, `p95`, `min`, `max`, and `negative_fraction`
per (model, chunk_type) pair for quick numerical comparison.
