# SSLO Scheduler End-to-End Sweep

This experiment compares baseline vLLM scheduling against SSLO and adaptive
SSLO scheduling for `Qwen/Qwen3-8B` on the Koala prompt workload. The launcher
sweeps `max_num_seqs`, `request_rate`, and `chunk_unit` (sentence / paragraph).

## Scripts

- `run_test.py`: inference-only runner. Runs ONE mode for ONE config and writes
  its JSONLs. No subprocess spawning, no GPU memory polling.
- `run_test.sh <run_kind> <max_num_seqs> <model>`: thin shell wrapper around
  `run_test.py`. Sets HF cache env vars and `SSLO_STATS_LOG_PATH` (for sslo*
  modes). All other settings come from env vars (see script header for
  defaults). For `progress_serve_offload`, KV-offload knobs (`CPU_OFFLOAD_GB`,
  `SSLO_KV_ONLOAD_LEAD_ITERS`, ...) are read directly by `run_test.py`.
- `run_sweep.sh [num_runs=3]`: unified full-sweep launcher over
  `CHUNK_UNITS × MAX_NUM_SEQS_VALUES × REQUEST_RATES × N runs × modes`.
  Handles per-mode subprocess invocation, GPU memory drain polling between
  modes, `run_status.json` writes, per-cell `analyze.py` calls, and a
  refresh of the sweep-wide `summary.csv` after every cell.
  `PARALLEL=4` (default) splits rates across 4 GPUs; `PARALLEL=0` runs
  sequentially.
- `analyze.py`: reads per-request latency rows, chunk slack rows, and scheduler
  stats; writes per-config `summary.json`.
- `metrics_utils.py`: shared constants (`MODES_DEFAULT`, `METRICS`) and helpers
  (`fmt_pair`, `parse_modes_arg`) used by the aggregators and `run_test.py`.
- `analysis/sweep_analysis.py`: subcommands `csv` (flat summary.csv from
  every summary.json), `tables` (compact seqs×rate tables for headline
  metrics), `agg-sweep` (per-chunk-unit seqs×rate aggregate over runs),
  and `agg-repeat` (mean ± stddev across N repeats of one cell).
- `analysis/README.md`: notes on analysis scripts.

> `run_single.sh`, `run_full_sweep.sh`, and `run_repeat.sh` have been removed.
> Their functionality is now split between `run_test.sh` (single-mode wrapper)
> and `run_sweep.sh` (orchestration, GPU polling, repeat loop).

## Modes

Seven scheduling modes are supported (see `MODES_DEFAULT` in
`metrics_utils.py`): `baseline`, `progress_serve`, `progress_serve_adaptive`,
`progress_serve_offload`, `progress_serve_offload_adaptive`,
`progress_serve_token_budget`, `progress_serve_offload_token_budget`.

The last two were named `progress_serve[_offload]_prefill_budget` before the
P* → TB* rename. Those names live on in `MODES_DEPRECATED` so the aggregators
keep reading the phaseP output directories written under them; they are not
selectable as a `--run-kind`.

`progress_serve_offload` runs `progress_serve` with the KV offload tier
(`kv_offload=True`). It requires the CPU-offload connector — `run_test.py`
wires `KVTransferConfig(kv_connector="SimpleCPUOffloadConnector", ...)` in eager
mode and forces `enable_prefix_caching=True` (the connector self-disables
without it). CPU capacity comes from `CPU_OFFLOAD_GB` (default 16). Offload
events are recorded per-step in `scheduler_stats.jsonl` (`num_offloads` /
`num_onloads` = ops this step, `num_offloaded` = current CPU-resident
population, `kv_capped`, `k_star_unconstrained`) and per-event in
`decisions.jsonl` (`kind="offload"` / `kind="onload"`); there is no separate
offload log file. The tier counts CPU-parked requests in the service share
(`kv_offload_share_includes_parked=True`, the adopted default since ablation A1
— WORKLOG 2026-08-07); `SSLO_KV_OFFLOAD_SHARE_INCLUDES_PARKED=0` only reproduces
the pre-A1 semantics. `progress_serve_offload_adaptive` is the same mode with
adaptive batching also enabled.

`progress_serve_token_budget` runs `progress_serve` with the deadline-aware
Token Budget (`token_budget_control=True`), which doses the per-step prefill
allowance on the expected-violation ledger `E_viol` (since 2026-08-11; the
worst-case `Δ_dec + κ_p·P ≤ γ·t_min` form let one near-deadline survivor pin
the step):

- **P axis** — the **total** prefill tokens of a step (chunked-prefill
  carry-over in the running loop plus new admits in the waiting loop, sharing
  one counter) are capped at the largest `P ∈ [floor, max_num_batched_tokens]`
  whose `E_viol(Δ_dec + κ_p·P) − E_viol(Δ_dec)` stays within
  `token_budget_risk_eps`, with κ_p the online per-prefill-token step-time cost.

Decode tokens are never capped: the decode set is settled by the run/defer
split alone (the D axis that used to shrink it was dropped 2026-08-13 — see
`paper/scheduling.md` §6). The mode composes with the offload tier —
`progress_serve_offload_token_budget` is both.
`SSLO_TOKEN_BUDGET_PREFILL_FLOOR` (default 512 tokens) and
`SSLO_TOKEN_BUDGET_RISK_EPS` (default 0.01) override the knobs
(`SSLO_TOKEN_BUDGET_GAMMA` is deprecated with the rejected rule). Per step,
`scheduler_stats.jsonl` records `token_budget_prefill` (TB*_pre)
and `prefill_kappa_ms_per_tok` (κ_p).

All aggregators default to all modes. Pass `--modes baseline,progress_serve` to restrict.

## Dialogue (multi-turn) Workload

`DIALOGUE_PROMPTS=1` switches the prompt pool from single-turn prompts to
multi-turn dialogue prefixes: each conversation is truncated to its last user
turn and rendered with the model's chat template, so prefill sees the whole
history. It requires `DATASET_NAME=wildchat|lmsys|combine` — the default
`koala` is a single-turn instruction set and raises immediately.
`MAX_PROMPT_TOKENS=N` drops prompts longer than N tokens (0 = off).
`MIN_RESPONSE_CHARS=N` keeps only dialogues whose reference response — the
assistant turn right after the last user turn, i.e. the message dropped when
the prompt is built — is at least N chars (0 = off). The reference length
correlates with the served output length, so the filter yields a long-output
workload with a larger per-request KV footprint. All three env vars are
honored by `run_test.sh` and passed through by `run_sweep.sh`.

Filtered dialogues are cached raw (before chat-template application, so the
file is reusable across models) at
`exp/tools/dataset_cache/dialogues_{dataset}_{filters}.jsonl`, keyed by source
dataset and the `CONVERSATION_ONLY` / `ENGLISH_ONLY` / `EXCLUDE_CODE` /
`MIN_RESPONSE_CHARS` filter combination (e.g.
`dialogues_wildchat_conv-en-nocode.jsonl`, and
`dialogues_wildchat_conv-en-nocode-minresp3000.jsonl` with the length gate);
only the first cold run pays the HF streaming cost. If the cache
holds fewer dialogues than `NUM_PROMPTS`, the run logs the shortfall and uses
what is there — delete the file to rebuild it larger.

## Chunk Units

`CHUNK_UNIT` controls sentence vs. paragraph boundary detection:

- `sentence` (default): chunk boundary at sentence end.
- `paragraph`: chunk boundary at paragraph end (fewer, larger chunks per request).

`run_sweep.sh` sweeps both units and places results in separate subtrees.

## Run

Inside the `sk-sslo` container, from `/workspace/mlsys`:

Single mode run:

```bash
OUTPUT_DIR=exp/run_sslo/output/test \
bash exp/run_sslo/run_test.sh progress_serve 64 Qwen/Qwen3-8B
```

Full sweep, sequential (2 default modes `baseline,progress_serve`, N=3 runs):

```bash
bash exp/run_sslo/run_sweep.sh 3
```

Full sweep, parallel (4-GPU):

```bash
PARALLEL=4 bash exp/run_sslo/run_sweep.sh 3
```

Host-side invocation:

```bash
docker exec sk-sslo bash -lc 'cd /workspace/mlsys && bash exp/run_sslo/run_sweep.sh 3 2>&1 | tail -40'
```

## Output Layout

Full-sweep output under `exp/run_sslo/output_sweep/{chunk_unit}/`:

```
seqs_${seqs}/
  rate_${rate}/
    run_{i}/
      requests.jsonl          (all modes, mode column prepended)
      chunks.jsonl            (all modes, mode column prepended)
      scheduler_stats.jsonl   (sslo* modes only; carries per-step offload counters)
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
- `metrics.offload.<offload_mode>.{total_offloaded_time_s, num_onloads}`: Distribution (only when the requests carry KV-offload fields; omitted otherwise)
- `metrics.inter_chunk_delay.<mode>`: Distribution
- `queue_stall_available`: bool
- `scheduler_saturation.<sslo_mode>`: {max_combined, iterations_above_cap, max_pending}
- `passes`: {pending_used, ttft_not_worse, neg_slack_not_worse} — each keyed by sslo mode

Distribution shape: `{count, mean, p50, p90, p99, max}`.
SlackDistribution: `{count, mean, neg_ratio, p5, p50, p90, p95, p99, max}`.
Compliance: `{rate, count, total_requests}`.

The aggregator subcommands in `analysis/sweep_analysis.py` (`agg-sweep`, `agg-repeat`) iterate `DISPLAY_GROUPS` (7 categories) from `metrics_utils.py` and use `lookup()` to read nested values from `summary.json`.

## Pass Criteria

Three boolean verdicts under `passes` in `summary.json`:

- `pending_used.<mode>`: True if any pending time was recorded for that mode.
- `ttft_not_worse.<mode>`: True if post-cap p90 TTFT is within 10% of baseline.
- `neg_slack_not_worse.<mode>`: True if negative-slack ratio is within 10% of baseline.