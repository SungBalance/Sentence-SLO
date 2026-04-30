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


if __name__ == "__main__":
    main()
