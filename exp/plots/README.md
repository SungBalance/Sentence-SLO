# Figure 5/6 Plotting

This folder contains the data, preprocessing scripts, plotting scripts, and
rendered PNGs for the Figure 5/6 experiment plots.

## Layout

- `figures/data/`: source data used by the figure preprocessors.
- `figures/data/output_sweep/`: main sweep outputs used by the sweep-based
  figures. Treat this as the only source for SSLO sweep analysis; files under
  `figures/processed/` are derived caches.
- `figures/processed/`: per-figure CSVs emitted by preprocessing scripts.
- `figures/scripts/preprocess/`: `process_fig_*.py` scripts that transform
  `figures/data/` inputs into `figures/processed/` CSVs.
- `figures/scripts/figs/`: plotting scripts and shared plotting helpers.
- `figures/*.png`: rendered figure images.

## SSLO Analysis Data Contract

Use only `figures/data/output_sweep/` as the source for SSLO sweep figures.
Do not source sweep metrics from backup directories or ad hoc analysis output.
`figures/processed/*.csv` files are acceptable plotting inputs only when they
were regenerated from `output_sweep`.

Exclude warmup and drain data. All request-, chunk-, scheduler-, and
decision-level preprocessing should keep only the measurement window:

- `summary.json`: use directly. The summary metrics are already computed over
  `in_window=True` requests, and include `in_window_count`,
  `out_of_window_count`, and `measurement_window_seconds` for audit.
- `requests.jsonl`: keep `request.get("in_window")`.
- `chunks.jsonl`: join through request id; keep chunks whose parent request is
  in-window.
- `scheduler_stats.jsonl`: keep `kind == "step"` rows with monotonic
  timestamps inside `[measurement_window_start_mono_ts,
  measurement_window_end_mono_ts]`.
- `decisions.jsonl`: keep rows with monotonic `ts` inside
  `[measurement_window_start_mono_ts, measurement_window_end_mono_ts]`.

Read window metadata from `run_meta.json`:

```python
mw0_wall = meta["measurement_window_start_ts"]       # wall clock: time.time()
mw1_wall = meta["measurement_window_end_ts"]
mw0_mono = meta["measurement_window_start_mono_ts"]  # monotonic: time.monotonic()
mw1_mono = meta["measurement_window_end_mono_ts"]
```

Do not mix clock domains:

- Compare `requests.completion_wall_ts` with wall-clock `mw*_ts`.
- Compare `chunks.*_time`, `scheduler_stats.ts`, and `decisions.ts` with
  monotonic `mw*_mono_ts`.
- Because `chunks.jsonl` belongs to parent requests whose `in_window` flag was
  already determined from request completion wall time, chunk filtering should
  be by request-id join, not by chunk timestamp.

Preprocess at run granularity first, then average across runs. The reporting
key should be `model x consume_profile_label x policy x max_num_seqs x
lambda_req_s` unless a figure has a narrower documented selection. Keep
`run_count` when practical so missing or partial run sets are visible.

Current implementation review:

- Sweep figures default to `figures/data/output_sweep/` through
  `DEFAULT_RAW_DATA`; Figure 5.0 is the exception because it uses separate
  word-count duration profile CSVs.
- Summary-based Figures 5.3, 5.4, 5.8, and 6.1-6.7 aggregate run rows with
  `mean()` and/or retain `run_count`; the summary source is already in-window.
- Figures 5.6, 5.8, 5.9, 6.4, 6.6, and 6.7d filter request-derived rows with
  `requests.jsonl` `in_window`.
- Figures 5.1, 5.2, 5.5, and 5.7 read chunk or decision rows directly and
  should be rechecked before relying on them for final analysis. Figure 5.5
  currently uses `decision["phase"] == "MEASURED"` but does not explicitly
  cross-check `run_meta.json` monotonic bounds, and Figure 5.6 scheduler-step
  samples currently filter by `kind == "step"` but not by explicit measurement
  timestamps.

## Commands

Run from `/workspace/mlsys/exp/plots/figures` inside the Docker container:

```bash
for script in scripts/preprocess/process_fig_5_*.py; do python3 "$script"; done
for script in scripts/figs/fig5_*.py; do python3 "$script"; done
```
