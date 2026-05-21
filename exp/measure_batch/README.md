# exp/measure_batch

Throughput vs `max_num_seqs` for baseline vLLM (no SSLO). Finds the
batch-size knee where doubling `max_num_seqs` no longer yields ≥10%
extra tokens/sec.

## Workload

- Pool = wildchat 2000 + lmsys 2000 (`conversation_only=True`,
  `exclude_code=True`, `dataset_seed=42`) — same as `exp/run_sslo`.
- `num_prompts = max_num_seqs × 4`. Randomly drawn (deterministic via
  `--sampling-seed`).
- **Burst** submit: all `num_prompts` reqs created at `t0`; no Poisson.

## Measurement window

`base_n = max_num_seqs`. Sort completions by wall-clock:

- `window_start = base_n`-th finisher's `complete_ts` (skips the warmup
  cohort).
- `window_end   = (3·base_n)`-th finisher's `complete_ts`.
- Window covers the `2·base_n` requests in between; the final `base_n`
  reqs are the cooldown tail and not counted.

`throughput = sum(num_output_tokens of window reqs) / window_duration_s`.

## Models / GPU layout

| Model | GPUs |
|---|---|
| `Qwen/Qwen3.5-9B` | 1 GPU |
| `Qwen/Qwen3.5-35B-A3B` | 1 GPU |
| `Qwen/Qwen3.5-27B` | 2 GPU |
| `Qwen/Qwen3.5-122B-A10B` | 2 GPU |

## Sweep rule

Try `max_num_seqs ∈ {8, 16, 32, 64, 128, 256, 512, ...}`. Stop when
`throughput[k+1] < throughput[k] × 1.10`. Tracked per-model.

## Usage

```bash
# Inside sk-sslo-vllm container.
HF_TOKEN=... bash exp/measure_batch/sweep.sh Qwen/Qwen3.5-9B  1 0
HF_TOKEN=... bash exp/measure_batch/sweep.sh Qwen/Qwen3.5-35B-A3B 1 1
HF_TOKEN=... bash exp/measure_batch/sweep.sh Qwen/Qwen3.5-27B 2 0,1
HF_TOKEN=... bash exp/measure_batch/sweep.sh Qwen/Qwen3.5-122B-A10B 2 2,3
```

## Output

```
exp/measure_batch/output/
  Qwen__Qwen3.5-9B/
    bsz_8/
      result.json
      requests.jsonl
      run.log
    bsz_16/...
    bsz_32/...     ← last batch before 10% stop
```

`result.json` keys: `max_num_seqs`, `num_prompts`, `window_duration_s`,
`window_request_count`, `window_total_output_tokens`,
`throughput_tokens_per_second`, `throughput_req_per_second`.
