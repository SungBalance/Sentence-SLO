# SSLO Scheduler End-to-End Sweep

This experiment compares baseline vLLM scheduling against SSLO scheduling for
`Qwen/Qwen3-8B` on the Koala prompt workload. The launcher sweeps
`max_num_seqs` over `32 64 128 256`; each config runs baseline and SSLO
back-to-back in separate subprocesses.

## Scripts

- `run_test.py`: parent runner that launches one config's baseline and SSLO
  subprocesses, waits for GPU memory cleanup, then writes a per-config summary.
- `run_test.sh`: no-argument full sweep launcher with the model, workload, cache,
  and GPU memory constants.
- `analyze.py`: reads per-request latency rows, chunk slack rows, and scheduler
  stats; writes per-config `summary.json` or top-level `sweep_summary.json`.

## Run

Run from the repository root inside the `sk-sslo` container:

```bash
HF_HOME=/cache HF_HUB_CACHE=/cache/hub FLASHINFER_DISABLE_VERSION_CHECK=1 \
bash exp/sslo_test/run_test.sh
```

The requested host-side invocation is:

```bash
docker exec sk-sslo bash -lc \
  'cd /workspace/mlsys && timeout 3000 bash exp/sslo_test/run_test.sh 2>&1 | tail -80'
```

For a smaller harness check, call `run_test.py` directly with reduced constants:

```bash
HF_HOME=/cache HF_HUB_CACHE=/cache/hub FLASHINFER_DISABLE_VERSION_CHECK=1 \
python3 exp/sslo_test/run_test.py \
  --num-prompts 16 \
  --max-num-seqs 8 \
  --generation-max-tokens 128 \
  --output-dir exp/sslo_test/output_smoke
```

## Output Layout

Each config writes under `exp/sslo_test/output/seqs_{N}/`:

- `baseline_ttft.jsonl`: per-request baseline rows with TTFT, TPOT, token count,
  timestamps, and compact `slo_chunk_records`.
- `sslo_ttft.jsonl`: matching per-request SSLO rows.
- `baseline_chunks.jsonl`: one baseline chunk row per completed chunk:
  `{request_id, chunk_idx, cumulative_slack, gen_time, pending_time, word_count}`.
- `sslo_chunks.jsonl`: matching SSLO chunk rows.
- `sslo_stats.jsonl`: scheduler iteration stats from `SSLO_STATS_LOG_PATH`.
- `summary.json`: machine-readable H1/H2/H3/H4 results for that config.
- `run_status.json`: baseline and SSLO subprocess exit codes.

The sweep also writes `exp/sslo_test/output/sweep_summary.json` and prints a
side-by-side table.

## Metrics

- TTFT: `t_first_token - t_submit`.
- TPOT: `(t_finish - t_first_token) / max(1, num_output_tokens - 1)`.
- Chunk slack: `cumulative_slack` from `output.slo_chunk_records`.
- `neg_slack_ratio`: `count(cumulative_slack < 0) / total_chunks`.

## Pass Criteria

- H1: `pending + running > max_num_seqs` at some SSLO scheduler iteration.
- H2-TTFT: SSLO p50 TTFT is no worse than baseline p50 TTFT for the
  post-cap-arrival cohort, `request_idx >= max_num_seqs`.
- H3-Slack: SSLO `neg_slack_ratio` is no worse than baseline across all chunks.
- H4-TPOT: informational only; reported side-by-side.

When `max_num_seqs=256` and `num_prompts=256`, the run is a control case with no
expected waiting queue pressure. H1 should fail explicitly, while H2/H3/H4
should be roughly equal.
