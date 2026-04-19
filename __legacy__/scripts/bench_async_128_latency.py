#!/usr/bin/env python3
"""Run async vLLM latency benchmark with concurrent requests."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass
class RequestMetrics:
    request_id: str
    success: bool
    output_tokens: int
    start_ts: float
    first_token_ts: float | None
    end_ts: float
    ttft_ms: float | None
    latency_ms: float
    tpot_ms: float | None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM AsyncLLM with concurrent requests."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override AsyncEngineArgs.max_num_seqs.",
    )
    return parser.parse_args()


def build_prompts(num_requests: int, prompt_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    words = [
        "system",
        "benchmark",
        "latency",
        "decode",
        "token",
        "throughput",
        "request",
        "response",
        "scheduler",
        "engine",
        "gpu",
        "memory",
    ]
    prompts: list[str] = []
    for i in range(num_requests):
        seq = [rng.choice(words) for _ in range(max(1, prompt_len))]
        prompts.append(f"[req-{i}] " + " ".join(seq))
    return prompts


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


async def run_one_request(
    engine: Any,
    request_id: str,
    prompt: str,
    sampling_params: Any,
) -> RequestMetrics:
    start_ts = time.perf_counter()
    first_token_ts: float | None = None
    output_tokens = 0
    error = ""
    success = True

    try:
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            now = time.perf_counter()
            delta_tokens = sum(len(comp.token_ids) for comp in output.outputs)
            if delta_tokens > 0 and first_token_ts is None:
                first_token_ts = now
            output_tokens += delta_tokens
            if output.finished:
                break
    except Exception as exc:  # pragma: no cover - runtime path
        success = False
        error = str(exc)

    end_ts = time.perf_counter()
    latency_ms = (end_ts - start_ts) * 1000.0
    ttft_ms = (first_token_ts - start_ts) * 1000.0 if first_token_ts else None
    tpot_ms = None
    if first_token_ts is not None and output_tokens > 1:
        tpot_ms = ((end_ts - first_token_ts) / (output_tokens - 1)) * 1000.0

    return RequestMetrics(
        request_id=request_id,
        success=success,
        output_tokens=output_tokens,
        start_ts=start_ts,
        first_token_ts=first_token_ts,
        end_ts=end_ts,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        tpot_ms=tpot_ms,
        error=error,
    )


def print_summary(results: list[RequestMetrics]) -> None:
    success_results = [r for r in results if r.success]
    ttft_values = [r.ttft_ms for r in success_results if r.ttft_ms is not None]
    latency_values = [r.latency_ms for r in success_results]
    tpot_values = [r.tpot_ms for r in success_results if r.tpot_ms is not None]
    failure_count = len(results) - len(success_results)

    print("=== Async 128 Request Latency Benchmark Summary ===")
    print(f"total_requests={len(results)}")
    print(f"success={len(success_results)}")
    print(f"failed={failure_count}")

    def print_metric(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"{name}: no_data")
            return
        print(
            f"{name}: mean={statistics.mean(vals):.3f} "
            f"p50={percentile(vals, 50):.3f} "
            f"p90={percentile(vals, 90):.3f} "
            f"p95={percentile(vals, 95):.3f} "
            f"p99={percentile(vals, 99):.3f}"
        )

    print_metric("ttft_ms", ttft_values)
    print_metric("latency_ms", latency_values)
    print_metric("tpot_ms", tpot_values)


def write_csv(path: str, results: list[RequestMetrics]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "success",
                "output_tokens",
                "ttft_ms",
                "latency_ms",
                "tpot_ms",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.request_id,
                    int(r.success),
                    r.output_tokens,
                    f"{r.ttft_ms:.6f}" if r.ttft_ms is not None else "",
                    f"{r.latency_ms:.6f}",
                    f"{r.tpot_ms:.6f}" if r.tpot_ms is not None else "",
                    r.error,
                ]
            )
    print(f"wrote_csv={out_path}")


async def run_benchmark(args: argparse.Namespace) -> int:
    # Delayed import so --help works even when vllm is not installed.
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    prompts = build_prompts(args.num_requests, args.prompt_len, args.seed)

    engine_args_kwargs = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
    }
    if args.max_num_seqs is not None:
        engine_args_kwargs["max_num_seqs"] = args.max_num_seqs
    engine_args = AsyncEngineArgs(**engine_args_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_kind=RequestOutputKind.DELTA,
    )

    try:
        tasks = [
            run_one_request(
                engine=engine,
                request_id=f"req-{i}",
                prompt=prompt,
                sampling_params=sampling_params,
            )
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        engine.shutdown()

    print_summary(results)
    if args.output_csv:
        write_csv(args.output_csv, results)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
