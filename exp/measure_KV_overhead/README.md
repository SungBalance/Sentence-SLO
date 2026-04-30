# measure_KV_overhead

Profiles GPU↔CPU transfer overhead for vLLM KV cache blocks as a function of block count.

## What it measures

Given a HuggingFace model name, the experiment:
1. Reads `num_hidden_layers`, `num_key_value_heads`, and `head_dim` from the model config (no weights downloaded).
2. Allocates a GPU tensor and a pinned-memory CPU tensor shaped `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`.
3. Sweeps over a configurable list of `num_blocks` values.
4. For each count, measures average GPU→CPU (offload) and CPU→GPU (onload) transfer time using CUDA events, then computes effective bandwidth.

## Key scripts

| Script | Purpose |
|--------|---------|
| `measure_kv_overhead.py` | Main profiler — accepts `--model`, `--block-size`, `--num-blocks`, etc. |
| `run_experiment.sh` | Host launcher — runs the profiler inside `sk-sslo` with default parameters |

## Running

From the host (container `sk-sslo` must be running):

```bash
bash exp/measure_KV_overhead/run_experiment.sh
```

To profile a different model or custom sweep:

```bash
docker exec -e HF_HOME=/cache -e HF_HUB_CACHE=/cache/hub sk-sslo \
    python3 /workspace/mlsys/exp/measure_KV_overhead/measure_kv_overhead.py \
    --model meta-llama/Llama-3.1-8B \
    --block-size 32 \
    --num-blocks 1 4 16 64 256 1024
```

## Output layout

```
measure_KV_overhead/outputs/
  kv_overhead.csv   — columns: num_blocks, total_bytes, offload_ms, onload_ms,
                                offload_bandwidth_GBs, onload_bandwidth_GBs
  kv_overhead.png   — two-panel plot: transfer time and bandwidth vs block count (log2 x-axis)
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model name or local path |
| `--block-size` | `16` | Tokens per KV cache block |
| `--dtype` | `bfloat16` | KV tensor dtype (`bfloat16`, `float16`, `float32`) |
| `--num-blocks` | `1 2 4 … 2048` | Block counts to sweep |
| `--warmup-runs` | `3` | Warmup transfers (excluded from timing) |
| `--timed-runs` | `10` | Timed repetitions to average |
| `--output-dir` | `outputs` | Output directory for CSV and plot |
