#!/usr/bin/env python3
"""Internal-slack benchmark for measure_internal_slack experiment.

Runs vLLM inference and collects per-chunk timing records from RequestSLOState
(engine-side timestamps), NOT from the async consumer. Engine sslo_params select
the chunk detector before the engine is created.

Output per run:
  {output_dir}/chunks.jsonl    -- one row per chunk (includes cumulative_slack)
  {output_dir}/requests.jsonl  -- one row per request
  {output_dir}/summary.json    -- run metadata
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_datasets import load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark internal slack via RequestSLOState engine-side timing."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default="koala")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--generation-max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--chunk-unit", choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--seconds-per-word", type=float, default=0.28)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def slugify(value: str) -> str:
    import re
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


async def collect_request(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
) -> Any:
    """Stream one request to completion; return the final RequestOutput."""
    request_id = str(request_idx)
    gen = engine.generate(prompt, sampling_params, request_id=request_id)
    last_output = None
    async for output in gen:
        last_output = output
    return last_output


async def run_warmup(engine: Any, prompts: list[str], sampling_params: Any, n: int) -> None:
    tasks = [
        asyncio.create_task(collect_request(engine, i, prompts[i % len(prompts)], sampling_params))
        for i in range(n)
    ]
    await asyncio.gather(*tasks)


async def run_benchmark(
    engine: Any,
    prompts: list[str],
    sampling_params: Any,
    model_slug: str,
    chunk_unit: str,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    tasks = [
        asyncio.create_task(collect_request(engine, i, p, sampling_params))
        for i, p in enumerate(prompts)
    ]
    outputs = await asyncio.gather(*tasks)

    chunk_rows: list[dict] = []
    request_rows: list[dict] = []

    for request_idx, output in enumerate(outputs):
        if output is None:
            continue
        request_id = output.request_id
        slo_records = output.slo_chunk_records or []
        total_tokens = sum(len(o.token_ids) for o in output.outputs) if output.outputs else 0
        prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0

        request_rows.append({
            "request_id": request_id,
            "request_idx": request_idx,
            "prompt_tokens": prompt_tokens,
            "num_chunks": len(slo_records),
            "total_output_tokens": total_tokens,
        })

        for record in slo_records:
            chunk_rows.append({
                "model": model_slug,
                "chunk_unit": chunk_unit,
                "batch_size": batch_size,
                "request_id": request_id,
                "request_idx": request_idx,
                **record,
            })

    return chunk_rows, request_rows


async def main_async(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slugify(args.model)
    chunk_unit = args.chunk_unit

    prompts = load_prompts(args.dataset_name, num_prompts=args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from {args.dataset_name}")

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_log_requests=False,
        sslo_params={
            "chunk_unit": args.chunk_unit,
            "seconds_per_word": args.seconds_per_word,
        },
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.generation_max_tokens,
        temperature=0.0,
    )

    if args.warmup_requests > 0:
        print(f"Warming up with {args.warmup_requests} request(s)...")
        await run_warmup(engine, prompts, sampling_params, args.warmup_requests)

    print(f"Running {len(prompts)} requests (chunk_unit={chunk_unit}, max_num_seqs={args.max_num_seqs})...")
    t0 = time.monotonic()
    chunk_rows, request_rows = await run_benchmark(
        engine, prompts, sampling_params, model_slug, chunk_unit, args.max_num_seqs
    )
    elapsed = time.monotonic() - t0

    total_chunks = len(chunk_rows)
    total_requests = len(request_rows)
    print(f"Done: {total_requests} requests, {total_chunks} chunks in {elapsed:.1f}s")

    write_jsonl(output_dir / "chunks.jsonl", chunk_rows)
    write_jsonl(output_dir / "requests.jsonl", request_rows)

    summary = {
        "model": args.model,
        "model_slug": model_slug,
        "chunk_unit": chunk_unit,
        "batch_size": args.max_num_seqs,
        "dataset_name": args.dataset_name,
        "num_prompts": args.num_prompts,
        "num_requests_completed": total_requests,
        "total_chunks": total_chunks,
        "elapsed_seconds": elapsed,
        "seconds_per_word": args.seconds_per_word,
        "generation_max_tokens": args.generation_max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote output to {output_dir}")

    engine.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
