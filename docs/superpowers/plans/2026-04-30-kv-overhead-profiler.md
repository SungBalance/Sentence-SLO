# KV Cache Offload/Onload Timing Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure GPU→CPU offload and CPU→GPU onload time for vLLM KV cache blocks as a function of block count, given a model's KV cache tensor shape derived from its HuggingFace config.

**Architecture:** A standalone Python profiling script reads a model's HF config to compute KV cache shape (`num_layers`, `num_kv_heads`, `head_dim`), allocates a GPU tensor and a pinned-memory CPU tensor shaped `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`, and measures transfer times using CUDA events across a configurable sweep of block counts. Results are written as CSV and a matplotlib two-panel plot. All execution happens inside the `sk-sslo` Docker container; the host-side run script wraps `docker exec`.

**Tech Stack:** Python 3, PyTorch (CUDA events + `pin_memory`), `transformers.AutoConfig`, `matplotlib`, `argparse`. Runs inside `sk-sslo` container at `/workspace/mlsys`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `exp/measure_KV_overhead/measure_kv_overhead.py` | Create | Main profiler: shape extraction, timing loop, CSV + plot output |
| `exp/measure_KV_overhead/run_experiment.sh` | Create | Host launcher: `docker exec` wrapper with default model and sweep params |
| `exp/measure_KV_overhead/README.md` | Create | Experiment documentation |

All output lands in `exp/measure_KV_overhead/outputs/` (created at runtime).

---

### Task 1: Main profiling script (`measure_kv_overhead.py`)

**Files:**
- Create: `exp/measure_KV_overhead/measure_kv_overhead.py`

**Context:** KV cache tensor shape per block is `[num_layers, 2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]`. The script allocates both a GPU tensor and a pinned-memory CPU tensor of this shape, then uses `torch.Tensor.copy_()` (the fastest PyTorch path to pinned memory) with CUDA events for nanosecond-level timing.

- [ ] **Step 1: Create file with arg parsing skeleton**

Create `exp/measure_KV_overhead/measure_kv_overhead.py`:

```python
#!/usr/bin/env python3
"""Profile KV cache block offload/onload time as a function of block count.

Usage:
    python measure_kv_overhead.py --model Qwen/Qwen3-8B
    python measure_kv_overhead.py --model Qwen/Qwen3-8B --block-size 32 --num-blocks 1 4 16 64 256
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile KV cache block offload/onload overhead."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument(
        "--block-size", type=int, default=16, help="KV cache block size (tokens per block)"
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        help="Block counts to sweep",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Warmup transfers before timing"
    )
    parser.add_argument(
        "--timed-runs", type=int, default=10, help="Timed repetitions to average"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for CSV and plot"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[placeholder] model={args.model}, block_size={args.block_size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add KV shape extraction function**

Add before `main()`:

```python
def get_kv_shape(model_name: str) -> tuple[int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim) from HF model config."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(f"Cannot find num_hidden_layers in config for {model_name}")

    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "num_attention_heads", None)
    if num_kv_heads is None:
        raise ValueError(f"Cannot find num_key_value_heads or num_attention_heads for {model_name}")

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_attn_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is None or num_attn_heads is None:
            raise ValueError(f"Cannot derive head_dim for {model_name}")
        head_dim = hidden_size // num_attn_heads

    return num_layers, num_kv_heads, head_dim
```

- [ ] **Step 3: Add timing function**

Add after `get_kv_shape()`:

```python
def measure_transfer(
    gpu_tensor: torch.Tensor,
    cpu_tensor: torch.Tensor,
    warmup_runs: int,
    timed_runs: int,
) -> tuple[float, float]:
    """Return (avg_offload_ms, avg_onload_ms) measured with CUDA events."""
    for _ in range(warmup_runs):
        cpu_tensor.copy_(gpu_tensor)
        gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_runs):
        cpu_tensor.copy_(gpu_tensor)
    end.record()
    torch.cuda.synchronize()
    offload_ms = start.elapsed_time(end) / timed_runs

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(timed_runs):
        gpu_tensor.copy_(cpu_tensor)
    end2.record()
    torch.cuda.synchronize()
    onload_ms = start2.elapsed_time(end2) / timed_runs

    return offload_ms, onload_ms
```

- [ ] **Step 4: Implement full main() — replace the placeholder**

```python
def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    bytes_per_element = torch.finfo(dtype).bits // 8

    print(f"Loading config for {args.model} ...")
    num_layers, num_kv_heads, head_dim = get_kv_shape(args.model)
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={args.block_size}, dtype={args.dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for num_blocks in sorted(set(args.num_blocks)):
        shape = (num_layers, 2, num_blocks, args.block_size, num_kv_heads, head_dim)
        total_bytes = (
            num_layers * 2 * num_blocks * args.block_size * num_kv_heads * head_dim * bytes_per_element
        )

        gpu_tensor = torch.empty(shape, dtype=dtype, device="cuda")
        cpu_tensor = torch.empty(shape, dtype=dtype).pin_memory()

        print(f"  num_blocks={num_blocks:6d}  size={total_bytes / 1e6:.1f} MB ...", end=" ", flush=True)
        offload_ms, onload_ms = measure_transfer(
            gpu_tensor, cpu_tensor, args.warmup_runs, args.timed_runs
        )

        offload_bw = total_bytes / (offload_ms * 1e-3) / 1e9
        onload_bw = total_bytes / (onload_ms * 1e-3) / 1e9

        print(
            f"offload={offload_ms:.2f} ms ({offload_bw:.1f} GB/s)  "
            f"onload={onload_ms:.2f} ms ({onload_bw:.1f} GB/s)"
        )
        rows.append({
            "num_blocks": num_blocks,
            "total_bytes": total_bytes,
            "offload_ms": round(offload_ms, 3),
            "onload_ms": round(onload_ms, 3),
            "offload_bandwidth_GBs": round(offload_bw, 2),
            "onload_bandwidth_GBs": round(onload_bw, 2),
        })

        del gpu_tensor, cpu_tensor
        torch.cuda.empty_cache()

    csv_path = output_dir / "kv_overhead.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to {csv_path}")

    block_counts = [r["num_blocks"] for r in rows]
    offload_times = [r["offload_ms"] for r in rows]
    onload_times = [r["onload_ms"] for r in rows]
    offload_bws = [r["offload_bandwidth_GBs"] for r in rows]
    onload_bws = [r["onload_bandwidth_GBs"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"KV Cache Transfer Overhead — {args.model}")

    ax1.plot(block_counts, offload_times, "o-", label="Offload (GPU→CPU)")
    ax1.plot(block_counts, onload_times, "s-", label="Onload (CPU→GPU)")
    ax1.set_xlabel("Number of KV Cache Blocks")
    ax1.set_ylabel("Transfer Time (ms)")
    ax1.set_title("Transfer Time vs Block Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    ax2.plot(block_counts, offload_bws, "o-", label="Offload (GPU→CPU)")
    ax2.plot(block_counts, onload_bws, "s-", label="Onload (CPU→GPU)")
    ax2.set_xlabel("Number of KV Cache Blocks")
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_title("Effective Bandwidth vs Block Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    plt.tight_layout()
    plot_path = output_dir / "kv_overhead.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
```

- [ ] **Step 5: Syntax check in container**

```bash
docker exec sk-sslo python3 -m compileall /workspace/mlsys/exp/measure_KV_overhead/measure_kv_overhead.py
```

Expected: `Compiling '...measure_kv_overhead.py'...` with no errors.

- [ ] **Step 6: Commit**

```bash
git add exp/measure_KV_overhead/measure_kv_overhead.py
git commit -m "feat: add KV cache offload/onload timing profiler"
```

---

### Task 2: Launch script (`run_experiment.sh`)

**Files:**
- Create: `exp/measure_KV_overhead/run_experiment.sh`

**Context:** Follows the same pattern as `exp/measure_internal_slack/run_experiment.sh` — runs on the host, delegates execution to the container via `docker exec`. Model and sweep params are shell constants at the top.

- [ ] **Step 1: Write run script**

Create `exp/measure_KV_overhead/run_experiment.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Host launcher — runs profiler inside sk-sslo container.
# Edit constants below to change model or sweep.

export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub

CONTAINER="sk-sslo"
CONTAINER_REPO="/workspace/mlsys"
SCRIPT="${CONTAINER_REPO}/exp/measure_KV_overhead/measure_kv_overhead.py"

MODEL="Qwen/Qwen3-8B"
BLOCK_SIZE=16
DTYPE="bfloat16"
NUM_BLOCKS="1 2 4 8 16 32 64 128 256 512 1024 2048"
WARMUP_RUNS=3
TIMED_RUNS=10
OUTPUT_DIR="${CONTAINER_REPO}/exp/measure_KV_overhead/outputs"

docker exec \
    -e HF_HOME="${HF_HOME}" \
    -e HF_HUB_CACHE="${HF_HUB_CACHE}" \
    "${CONTAINER}" \
    python3 "${SCRIPT}" \
        --model "${MODEL}" \
        --block-size "${BLOCK_SIZE}" \
        --dtype "${DTYPE}" \
        --num-blocks ${NUM_BLOCKS} \
        --warmup-runs "${WARMUP_RUNS}" \
        --timed-runs "${TIMED_RUNS}" \
        --output-dir "${OUTPUT_DIR}"
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x exp/measure_KV_overhead/run_experiment.sh
bash -n exp/measure_KV_overhead/run_experiment.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add exp/measure_KV_overhead/run_experiment.sh
git commit -m "feat: add KV overhead experiment launch script"
```

---

### Task 3: README

**Files:**
- Create: `exp/measure_KV_overhead/README.md`

- [ ] **Step 1: Write README**

Create `exp/measure_KV_overhead/README.md` with this exact content:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add exp/measure_KV_overhead/README.md
git commit -m "docs: add README for measure_KV_overhead experiment"
```

---

### Task 4: End-to-end verification

**Files:** None created.

- [ ] **Step 1: Check transformers is available in container**

```bash
docker exec sk-sslo python3 -c "import transformers; print(transformers.__version__)"
```

If missing:
```bash
docker exec sk-sslo pip install transformers --quiet
```

- [ ] **Step 2: Smoke test with minimal sweep (3 points)**

```bash
docker exec \
    -e HF_HOME=/cache \
    -e HF_HUB_CACHE=/cache/hub \
    sk-sslo \
    python3 /workspace/mlsys/exp/measure_KV_overhead/measure_kv_overhead.py \
        --model Qwen/Qwen3-8B \
        --num-blocks 1 2 4 \
        --warmup-runs 1 \
        --timed-runs 2 \
        --output-dir /workspace/mlsys/exp/measure_KV_overhead/outputs
```

Expected: prints shape info (e.g. `num_layers=36, num_kv_heads=8, head_dim=128` — exact values depend on model config), then one timing line per block count. Creates `outputs/kv_overhead.csv` and `outputs/kv_overhead.png`.

- [ ] **Step 3: Verify CSV has correct columns**

```bash
head -2 exp/measure_KV_overhead/outputs/kv_overhead.csv
```

Expected output (first two lines):
```
num_blocks,total_bytes,offload_ms,onload_ms,offload_bandwidth_GBs,onload_bandwidth_GBs
1,...
```

- [ ] **Step 4: Run full experiment via launch script**

```bash
bash exp/measure_KV_overhead/run_experiment.sh
```

Expected: 12 block counts swept (1–2048), CSV and plot written to `outputs/`, script exits 0.
