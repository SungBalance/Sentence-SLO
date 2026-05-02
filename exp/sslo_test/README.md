# SSLO Scheduler End-to-End Test

This experiment validates two SSLO scheduler hypotheses with the same Koala prompt workload and Qwen/Qwen3-8B generation settings.

- H1: with `max_num_seqs=64` and SSLO enabled, the scheduler can hold more than 64 requests across `running + sslo_pending`.
- H2: with SSLO enabled, median TTFT for newly arriving requests is lower than the baseline with SSLO disabled.

## Scripts

- `run_test.py`: parent runner that launches baseline and SSLO subprocesses, then runs analysis.
- `run_test.sh`: no-argument launcher with the full-run constants and Hugging Face cache environment.
- `analyze.py`: reads per-request TTFT rows and scheduler stats, prints hypothesis verdicts, and writes `summary.json`.

## Run

Run from the repository root inside the `sk-sslo` container:

```bash
HF_HOME=/cache HF_HUB_CACHE=/cache/hub FLASHINFER_DISABLE_VERSION_CHECK=1 bash exp/sslo_test/run_test.sh
```

For a small harness check, call `run_test.py` directly with reduced constants:

```bash
HF_HOME=/cache HF_HUB_CACHE=/cache/hub FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 exp/sslo_test/run_test.py \
  --num-prompts 16 \
  --generation-max-tokens 128 \
  --output-dir exp/sslo_test/output_smoke
```

## Output Layout

Outputs are written under `exp/sslo_test/output/`:

- `baseline_ttft.jsonl`: per-request TTFT rows from the SSLO-disabled run.
- `sslo_ttft.jsonl`: per-request TTFT rows from the SSLO-enabled run.
- `sslo_stats.jsonl`: scheduler iteration stats from `SSLO_STATS_LOG_PATH`.
- `summary.json`: machine-readable H1/H2 summary.

## Interpretation

H1 passes when `summary.json` has `h1_pending_plus_running_exceeds_cap: true`.
H2 passes when `h2_ttft_p50_pct_change` is negative, meaning SSLO p50 TTFT is lower than baseline p50 TTFT.
