#!/usr/bin/env python3
"""Measure global iteration TPOT by decode batch size using vLLM AsyncLLM."""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep decode batch sizes and write iteration-level TPOT CSV."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--decode-batch-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default="results/tpot_by_decode_batch.csv")
    return parser.parse_args()


def parse_decode_batch_sizes(raw: str) -> list[int]:
    sizes = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        val = int(token)
        if val <= 0:
            raise ValueError("decode batch sizes must be positive")
        sizes.append(val)
    if not sizes:
        raise ValueError("decode batch sizes cannot be empty")
    return sizes


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


async def _request_runner(
    engine: Any,
    request_id: str,
    prompt: str,
    sampling_params: Any,
    queue: "asyncio.Queue[tuple[float, int, bool]]",
) -> None:
    finished = False
    try:
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            now = time.perf_counter()
            delta_tokens = sum(len(comp.token_ids) for comp in output.outputs)
            await queue.put((now, delta_tokens, output.finished))
            if output.finished:
                finished = True
                break
    except Exception:  # pragma: no cover - runtime path
        pass
    finally:
        if not finished:
            await queue.put((time.perf_counter(), 0, True))


async def run_single_measurement(
    *,
    model: str,
    decode_batch_size: int,
    num_requests: int,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    seed: int,
    record_rows: bool,
    run_id: str,
) -> list[dict[str, Any]]:
    # Delayed import so --help works even when vllm is not installed.
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=model,
        max_num_seqs=decode_batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        output_kind=RequestOutputKind.DELTA,
    )

    queue: asyncio.Queue[tuple[float, int, bool]] = asyncio.Queue()
    tasks = [
        asyncio.create_task(
            _request_runner(
                engine=engine,
                request_id=f"req-{i}",
                prompt=prompts[i],
                sampling_params=sampling_params,
                queue=queue,
            )
        )
        for i in range(num_requests)
    ]

    rows: list[dict[str, Any]] = []
    finished_requests = 0
    start_ts = time.perf_counter()
    last_token_event_ts: float | None = None
    iteration_idx = 0

    try:
        while finished_requests < num_requests:
            ts, delta_tokens, is_finished = await queue.get()
            if is_finished:
                finished_requests += 1

            if delta_tokens <= 0:
                continue

            if last_token_event_ts is None:
                last_token_event_ts = ts
                continue

            delta_time = ts - last_token_event_ts
            last_token_event_ts = ts
            if delta_time <= 0:
                continue

            iteration_idx += 1
            tpot_ms = (delta_time / delta_tokens) * 1000.0
            active_requests = max(0, num_requests - finished_requests)
            if record_rows:
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "decode_batch_size": decode_batch_size,
                        "iteration_idx": iteration_idx,
                        "elapsed_ms": (ts - start_ts) * 1000.0,
                        "delta_tokens": delta_tokens,
                        "tpot_ms": tpot_ms,
                        "active_requests": active_requests,
                        "finished_requests": finished_requests,
                    }
                )
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        engine.shutdown()

    return rows


def write_rows_to_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "model",
        "decode_batch_size",
        "iteration_idx",
        "elapsed_ms",
        "delta_tokens",
        "tpot_ms",
        "active_requests",
        "finished_requests",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote_csv={out_path}")


async def run_sweep(args: argparse.Namespace) -> int:
    decode_batch_sizes = parse_decode_batch_sizes(args.decode_batch_sizes)
    prompts = build_prompts(args.num_requests, args.prompt_len, args.seed)
    all_rows: list[dict[str, Any]] = []

    for decode_batch_size in decode_batch_sizes:
        for warmup_idx in range(args.warmup_runs):
            print(
                f"[warmup] decode_batch_size={decode_batch_size} "
                f"run={warmup_idx + 1}/{args.warmup_runs}"
            )
            await run_single_measurement(
                model=args.model,
                decode_batch_size=decode_batch_size,
                num_requests=args.num_requests,
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=args.seed + warmup_idx,
                record_rows=False,
                run_id=f"warmup_bs{decode_batch_size}_{warmup_idx + 1}",
            )

        print(f"[measure] decode_batch_size={decode_batch_size}")
        run_rows = await run_single_measurement(
            model=args.model,
            decode_batch_size=decode_batch_size,
            num_requests=args.num_requests,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed + 10_000 + decode_batch_size,
            record_rows=True,
            run_id=f"bs{decode_batch_size}",
        )
        all_rows.extend(run_rows)
        print(
            f"[done] decode_batch_size={decode_batch_size} "
            f"iterations={len(run_rows)}"
        )

    write_rows_to_csv(args.output_csv, all_rows)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_sweep(args))


if __name__ == "__main__":
    raise SystemExit(main())
