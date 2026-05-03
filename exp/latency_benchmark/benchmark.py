#!/usr/bin/env python3
"""Latency benchmark: engine-side TTFT, TPOT, E2E, and chunk-level slack.

Request-level metrics come from vLLM's internal RequestStateStats
(output.metrics), giving engine-side timestamps for TTFT, E2E, and a
breakdown into queued / prefill / decode time.

Chunk-level slack comes from output.slo_chunk_records, which is computed
inside the vLLM EngineCore using the SSLO extension.  SSLO_CHUNK_UNIT and
SSLO_SECONDS_PER_WORD env vars select the detector and reading speed before
engine creation.

The only project dependency outside of vLLM is exp/tools/lm_datasets.py.

Output per run:
  {output_dir}/requests.jsonl  -- per-request: TTFT, TPOT, E2E,
                                  queued/prefill/decode breakdown
  {output_dir}/chunks.jsonl   -- per-chunk: cumulative_slack, word_count,
                                  cumulative_consume (engine-side timestamps)
  {output_dir}/summary.json   -- run metadata + aggregate stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_datasets import load_prompts


# ---------------------------------------------------------------------------
# Per-request collection
# ---------------------------------------------------------------------------

async def collect_request(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
) -> Any:
    """Stream one request to completion; return the final RequestOutput."""
    request_id = str(request_idx)
    last_output = None
    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        last_output = output
    return last_output


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_warmup(
    engine: Any,
    prompts: list[str],
    sampling_params: Any,
    n: int,
) -> None:
    tasks = [
        asyncio.create_task(
            collect_request(engine, i, prompts[i % len(prompts)], sampling_params)
        )
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

    request_rows: list[dict] = []
    chunk_rows: list[dict] = []

    for request_idx, output in enumerate(outputs):
        if output is None:
            continue

        m = output.metrics  # RequestStateStats | None
        num_output_tokens = (
            sum(len(o.token_ids) for o in output.outputs) if output.outputs else 0
        )
        prompt_tokens = (
            len(output.prompt_token_ids) if output.prompt_token_ids else 0
        )
        slo_records = output.slo_chunk_records or []

        # Engine-side latency breakdown (all in seconds unless noted)
        ttft_s       = m.first_token_latency if m else None
        e2e_s        = (m.last_token_ts  - m.arrival_time)   if m else None
        queued_s     = (m.scheduled_ts   - m.queued_ts)      if m else None
        prefill_s    = (m.first_token_ts - m.scheduled_ts)   if m else None
        decode_s     = (m.last_token_ts  - m.first_token_ts) if m else None
        # TPOT: time per output token in ms (decode phase only, per generation token)
        tpot_ms = (
            decode_s / max(num_output_tokens - 1, 1) * 1000
            if decode_s is not None and num_output_tokens > 1
            else None
        )

        request_rows.append({
            "request_id":       output.request_id,
            "request_idx":      request_idx,
            "prompt_tokens":    prompt_tokens,
            "num_output_tokens": num_output_tokens,
            "num_chunks":       len(slo_records),
            "ttft_s":           ttft_s,
            "tpot_ms":          tpot_ms,
            "e2e_latency_s":    e2e_s,
            "queued_time_s":    queued_s,
            "prefill_time_s":   prefill_s,
            "decode_time_s":    decode_s,
        })

        for record in slo_records:
            chunk_rows.append({
                "model":       model_slug,
                "chunk_unit":  chunk_unit,
                "batch_size":  batch_size,
                "request_id":  output.request_id,
                "request_idx": request_idx,
                **record,
            })

    return request_rows, chunk_rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    sv = sorted(vals)
    n  = len(sv)

    def pct(q: float) -> float:
        idx = q * (n - 1)
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        return sv[lo] * (1 - idx + lo) + sv[hi] * (idx - lo)

    return {
        "mean": sum(vals) / n,
        "p50":  pct(0.50),
        "p95":  pct(0.95),
        "p99":  pct(0.99),
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Engine-side latency benchmark: TTFT, TPOT, E2E, chunk slack."
    )
    p.add_argument("--model",                  required=True)
    p.add_argument("--dataset-name",           default="koala")
    p.add_argument("--num-prompts",            type=int,   default=256)
    p.add_argument("--max-model-len",          type=int,   default=8192)
    p.add_argument("--max-num-seqs",           type=int,   default=64)
    p.add_argument("--generation-max-tokens",  type=int,   default=8192)
    p.add_argument("--tensor-parallel-size",   type=int,   default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--warmup-requests",        type=int,   default=1)
    p.add_argument("--chunk-unit",             choices=["sentence", "paragraph"],
                   default="sentence")
    p.add_argument("--seconds-per-word",       type=float, default=0.28)
    p.add_argument("--output-dir",             required=True)
    return p.parse_args()


def slugify(value: str) -> str:
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    # Configure SSLO detector before engine creation so EngineCore picks it up.
    os.environ["SSLO_CHUNK_UNIT"]      = args.chunk_unit
    os.environ["SSLO_SECONDS_PER_WORD"] = str(args.seconds_per_word)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slugify(args.model)

    prompts = load_prompts(args.dataset_name, num_prompts=args.num_prompts)
    print(f"Loaded {len(prompts)} prompts from {args.dataset_name}")

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_log_requests=False,
        # disable_log_stats defaults to False → metrics are collected
    ))

    sampling_params = SamplingParams(max_tokens=args.generation_max_tokens, temperature=0.0)

    if args.warmup_requests > 0:
        print(f"Warming up with {args.warmup_requests} request(s)...")
        await run_warmup(engine, prompts, sampling_params, args.warmup_requests)

    print(f"Running {len(prompts)} requests "
          f"(chunk_unit={args.chunk_unit}, max_num_seqs={args.max_num_seqs})...")
    t0 = time.monotonic()
    request_rows, chunk_rows = await run_benchmark(
        engine, prompts, sampling_params,
        model_slug, args.chunk_unit, args.max_num_seqs,
    )
    elapsed = time.monotonic() - t0
    print(f"Done: {len(request_rows)} requests, {len(chunk_rows)} chunks in {elapsed:.1f}s")

    write_jsonl(output_dir / "requests.jsonl", request_rows)
    write_jsonl(output_dir / "chunks.jsonl",   chunk_rows)

    # Aggregate stats
    def _col(rows: list[dict], key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    slacks = [c["cumulative_slack"] for c in chunk_rows if c.get("chunk_idx", 0) > 0]

    summary: dict[str, Any] = {
        "model":                  args.model,
        "model_slug":             model_slug,
        "chunk_unit":             args.chunk_unit,
        "batch_size":             args.max_num_seqs,
        "dataset_name":           args.dataset_name,
        "num_prompts":            args.num_prompts,
        "num_requests_completed": len(request_rows),
        "total_chunks":           len(chunk_rows),
        "elapsed_seconds":        elapsed,
        "seconds_per_word":       args.seconds_per_word,
        "generation_max_tokens":  args.generation_max_tokens,
        "max_model_len":          args.max_model_len,
        "tensor_parallel_size":   args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # Engine-side latency stats
        "ttft_s":        _stats(_col(request_rows, "ttft_s")),
        "tpot_ms":       _stats(_col(request_rows, "tpot_ms")),
        "e2e_latency_s": _stats(_col(request_rows, "e2e_latency_s")),
        "queued_time_s": _stats(_col(request_rows, "queued_time_s")),
        "prefill_time_s": _stats(_col(request_rows, "prefill_time_s")),
        "decode_time_s": _stats(_col(request_rows, "decode_time_s")),
        # Chunk-level slack stats (chunk_idx > 0 only)
        "cumulative_slack_s": {
            **_stats(slacks),
            "neg_fraction": (
                sum(1 for s in slacks if s < 0) / len(slacks) if slacks else None
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def fmt(d: dict) -> str:
        if d.get("mean") is None:
            return "n/a"
        return f"mean={d['mean']:.3f}  p50={d['p50']:.3f}  p95={d['p95']:.3f}"

    print(f"  TTFT (s):          {fmt(summary['ttft_s'])}")
    print(f"  TPOT (ms):         {fmt(summary['tpot_ms'])}")
    print(f"  E2E latency (s):   {fmt(summary['e2e_latency_s'])}")
    print(f"  Queued (s):        {fmt(summary['queued_time_s'])}")
    print(f"  Prefill (s):       {fmt(summary['prefill_time_s'])}")
    print(f"  Decode (s):        {fmt(summary['decode_time_s'])}")
    print(f"  Cumul. slack (s):  {fmt(summary['cumulative_slack_s'])}")
    print(f"Wrote output to {output_dir}")

    engine.shutdown()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
