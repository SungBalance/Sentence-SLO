# SSLO Scheduler End-to-End Sweep

This experiment compares baseline vLLM scheduling against SSLO and adaptive
SSLO scheduling for `Qwen/Qwen3-8B` on the Koala prompt workload. The launcher
sweeps `max_num_seqs`, `request_rate`, and `chunk_unit` (sentence / paragraph).

## Scripts

- `run_test.py`: inference-only runner. Runs ONE mode for ONE config and writes
  its JSONLs. No subprocess spawning, no GPU memory polling.
- `run_test.sh <run_kind> <max_num_seqs> <model>`: thin shell wrapper around
  `run_test.py`. Sets HF cache env vars, `SSLO_STATS_LOG_PATH` (for sslo*
  modes), and `SSLO_OFFLOAD_LOG_PATH` (for offload modes). All other settings
  come from env vars (see script header for defaults).
- `run_sweep.sh [num_runs=3]`: unified full-sweep launcher over
  `CHUNK_UNITS × MAX_NUM_SEQS_VALUES × REQUEST_RATES × N runs × modes`.
  Handles per-mode subprocess invocation, GPU memory drain polling between
  modes, `run_status.json` writes, per-cell `analyze.py` calls, and
  per-chunk-unit `aggregate_sweep.py` at the end.
  Set `PARALLEL=2` or `PARALLEL=4` to split rates across multiple GPUs.
- `analyze.py`: reads per-request latency rows, chunk slack rows, and scheduler
  stats; writes per-config `summary.json`.
- `metrics_utils.py`: shared constants (`MODES_DEFAULT`, `METRICS`) and helpers
  (`fmt_pair`, `parse_modes_arg`) used by the aggregators and `run_test.py`.
- `analysis/aggregate_repeats.py`: aggregates N repeated runs for one config.
  Accepts `--modes` (default: all 5 modes).
- `analysis/aggregate_sweep.py`: aggregates a full rate×seqs grid.
  Accepts `--modes` (default: all 5 modes).
- `analysis/README.md`: notes on analysis scripts.

> `run_single.sh`, `run_full_sweep.sh`, and `run_repeat.sh` have been removed.
> Their functionality is now split between `run_test.sh` (single-mode wrapper)
> and `run_sweep.sh` (orchestration, GPU polling, repeat loop).

## Modes

Five scheduling modes are supported:
`baseline`, `sslo`, `sslo_offload`, `sslo_adaptive`, `sslo_adaptive_offload`.

All aggregators default to all 5 modes. Pass `--modes baseline,sslo` to restrict.

## Chunk Units

`CHUNK_UNIT` controls sentence vs. paragraph boundary detection:

- `sentence` (default): chunk boundary at sentence end.
- `paragraph`: chunk boundary at paragraph end (fewer, larger chunks per request).

`run_sweep.sh` sweeps both units and places results in separate subtrees.

## Run

Inside the `sk-sslo` container, from `/workspace/mlsys`:

Single mode run:

```bash
OUTPUT_DIR=exp/sslo_test/output/test \
bash exp/sslo_test/run_test.sh sslo 64 Qwen/Qwen3-8B
```

Full sweep, sequential (both chunk units, 5 modes, N=3 runs):

```bash
bash exp/sslo_test/run_sweep.sh 3
```

Full sweep, parallel (4-GPU):

```bash
PARALLEL=4 bash exp/sslo_test/run_sweep.sh 3
```

Host-side invocation:

```bash
docker exec sk-sslo bash -lc 'cd /workspace/mlsys && bash exp/sslo_test/run_sweep.sh 3 2>&1 | tail -40'
```

## Output Layout

Full-sweep output under `exp/sslo_test/output_sweep/{chunk_unit}/`:

```
seqs_${seqs}/
  rate_${rate}/
    run_{i}/
      requests.jsonl          (all modes, mode column prepended)
      chunks.jsonl            (all modes, mode column prepended)
      scheduler_stats.jsonl   (sslo* modes only)
      offload_log.jsonl       (offload modes only)
      summary.json
      run_status.json
```

Six files per cell (down from ~22). Per-mode tmp files (`requests_${mode}.jsonl`, etc.) are written by `run_test.py` and merged with a `mode` column by `run_sweep.sh` after each mode completes.

## Metrics

`analyze.py` reads `requests.jsonl`, `chunks.jsonl`, and `scheduler_stats.jsonl`, splits rows by `mode`, and writes `summary.json`:

- `config`: max_num_seqs, chunk_unit, request_rate, is_control, modes_run
- `metrics.ttft.<mode>.{all, post_cap}`: Distribution + cohort string
- `metrics.tpot.<mode>`: Distribution
- `metrics.queue_stall.<mode>`: Distribution
- `metrics.slack.<mode>`: SlackDistribution + `magnitude` Distribution
- `metrics.slo_compliance.<mode>`: Compliance
- `metrics.scheduler.<sslo_mode>.{running, combined}`: Distribution
- `metrics.pending.<sslo_mode>.{time, intervals}`: Distribution
- `metrics.inter_chunk_delay.<mode>`: Distribution
- `queue_stall_available`: bool
- `scheduler_saturation.<sslo_mode>`: {max_combined, iterations_above_cap, max_pending}
- `passes`: {pending_used, ttft_not_worse, neg_slack_not_worse} — each keyed by sslo mode

Distribution shape: `{count, mean, p50, p90, p99, max}`.
SlackDistribution: `{count, mean, neg_ratio, p5, p50, p90, p95, p99, max}`.
Compliance: `{rate, count, total_requests}`.

The aggregators (`aggregate_repeats.py`, `aggregate_sweep.py`) iterate `DISPLAY_GROUPS` (7 categories) from `metrics_utils.py` and use `lookup()` to read nested values from `summary.json`.

## Pass Criteria

Three boolean verdicts under `passes` in `summary.json`:

- `pending_used.<mode>`: True if any pending time was recorded for that mode.
- `ttft_not_worse.<mode>`: True if post-cap p90 TTFT is within 10% of baseline.
- `neg_slack_not_worse.<mode>`: True if negative-slack ratio is within 10% of baseline.