#!/usr/bin/env python3
"""Run one SSLO test mode for one config and write its JSONLs."""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import random as _random
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_datasets import load_prompts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import MODES_DEFAULT


DEFAULT_OUTPUT_DIR = "exp/sslo_test/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-kind",
        required=True,
        choices=list(MODES_DEFAULT),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-name", default="koala")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument(
        "--max-model-len", type=int, default=0,
        help="Max model context length. 0 (default) = auto, "
             "let vLLM derive from the model's HF config.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--chunk-unit", choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--seconds-per-word", type=float, default=0.28)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.0,
        help="Poisson arrival rate in reqs/sec. 0 (default) submits all prompts at once.",
    )
    parser.add_argument(
        "--request-rate-seed",
        type=int,
        default=42,
        help="Seed for the Poisson inter-arrival sampler (reproducibility).",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_workload(dataset_name: str, num_prompts: int) -> list[str]:
    prompts = load_prompts(dataset_name, num_prompts=num_prompts)
    if len(prompts) >= num_prompts:
        return prompts[:num_prompts]
    repeated: list[str] = []
    while len(repeated) < num_prompts:
        repeated.extend(prompts)
    return repeated[:num_prompts]


def positive_number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def extract_chunk_records(request_output: Any) -> list[dict[str, Any]]:
    records = getattr(request_output, "slo_chunk_records", None)
    if not records:
        return []
    # records is a list of dicts (RequestSLOState.chunk_records returns
    # asdict() results); use dict.get, not getattr-which-returns-None.
    def _val(r, k):
        return r.get(k) if isinstance(r, dict) else getattr(r, k, None)

    # Map Phase A v2 ChunkRecord fields (slack_s, pending_time_s,
    # gen_finish_ts, ...) to the analyze.py-stable schema below. Fall back
    # to v1 field names when present for backward compatibility.
    normalized = []
    for record in records:
        slack = _val(record, "slack_s")
        if slack is None:
            slack = _val(record, "cumulative_slack")
        pending = _val(record, "pending_time_s")
        if pending is None:
            pending = _val(record, "pending_time")
        end_ts = _val(record, "gen_finish_ts")
        if end_ts is None:
            end_ts = _val(record, "end_time_ts")
        normalized.append({
            "chunk_idx": _val(record, "chunk_idx"),
            "cumulative_slack": slack,
            "gen_time": _val(record, "gen_time"),
            "pending_time": pending,
            "word_count": _val(record, "word_count"),
            "end_time_ts": end_ts,
        })
    return normalized


async def collect_request_with_delay(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
    arrival_offset_s: float,
) -> dict[str, Any]:
    """Sleep until the prompt's scheduled arrival, then collect."""
    if arrival_offset_s > 0:
        await asyncio.sleep(arrival_offset_s)
    return await collect_request(engine, request_idx, prompt, sampling_params)


async def collect_request(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
) -> dict[str, Any]:
    request_id = str(request_idx)
    last_output = None

    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        last_output = output

    if last_output is None:
        return {
            "request_id": request_id,
            "request_idx": request_idx,
            "ttft": None,
            "tpot": None,
            "queue_stall": None,
            "num_output_tokens": 0,
            "slo_chunk_records": [],
            "total_pending_time_s": None,
            "num_pending_intervals": 0,
            "max_consecutive_pending": 0,
        }

    metrics = getattr(last_output, "metrics", None)
    ttft = getattr(metrics, "first_token_latency", None) if metrics else None
    if ttft == 0.0:
        ttft = None

    num_gen = getattr(metrics, "num_generation_tokens", 0) if metrics else 0
    num_gen = int(num_gen or 0)
    first_ts = positive_number(getattr(metrics, "first_token_ts", None)) if metrics else None
    last_ts = positive_number(getattr(metrics, "last_token_ts", None)) if metrics else None
    tpot = None
    if metrics and num_gen > 1 and last_ts is not None and first_ts is not None:
        tpot = (last_ts - first_ts) / (num_gen - 1)

    queued = positive_number(getattr(metrics, "queued_ts", None)) if metrics else None
    scheduled = positive_number(getattr(metrics, "scheduled_ts", None)) if metrics else None
    queue_stall = (
        scheduled - queued if (queued and scheduled and scheduled >= queued) else None
    )

    slo_chunk_records = extract_chunk_records(last_output)
    sslo_metrics = getattr(last_output, "sslo_metrics", None)
    total_pending_time_s = (
        getattr(sslo_metrics, "total_pending_time_s", None) if sslo_metrics else None
    )
    num_pending_intervals = (
        getattr(sslo_metrics, "num_pending_intervals", 0) if sslo_metrics else 0
    )
    max_consecutive_pending = (
        getattr(sslo_metrics, "max_consecutive_pending", 0) if sslo_metrics else 0
    )

    return {
        "request_id": request_id,
        "request_idx": request_idx,
        "num_output_tokens": num_gen,
        "ttft": ttft,
        "tpot": tpot,
        "queue_stall": queue_stall,
        "decoding_start_ts": first_ts,
        "slo_chunk_records": slo_chunk_records,
        "total_pending_time_s": total_pending_time_s,
        "num_pending_intervals": num_pending_intervals,
        "max_consecutive_pending": max_consecutive_pending,
    }


async def run_one(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_workload(args.dataset_name, args.num_prompts)
    print(f"{args.run_kind}: loaded {len(prompts)} prompts from {args.dataset_name}")

    # Build sslo_params incrementally: start with the base SSLO config used by
    # all sslo* modes, then add flags per mode suffix. Field names match the
    # Phase A v2 SsloConfig (vllm/vllm/sslo/config.py).
    if args.run_kind == "baseline":
        sslo_params = {"enabled": False}
    else:
        sslo_params = {
            "enabled": True,
            "chunk_unit": args.chunk_unit,
            "seconds_per_word": args.seconds_per_word,
        }
        if "adaptive" in args.run_kind:
            sslo_params["adaptive_batching"] = True
        if "offload" in args.run_kind:
            sslo_params["offloading"] = True

    # KV transfer config: only enable the CPU-offload connector for the two
    # offload SSLO modes. Non-offload modes (baseline, sslo, sslo_adaptive)
    # run with the engine's default — no kv_transfer plumbing — so the
    # baseline truly is vanilla vLLM and the sslo/sslo_adaptive comparisons
    # don't carry connector overhead. Extra config overridable via
    # SSLO_KV_OFFLOAD_EXTRA env (JSON).
    needs_kv_offload = args.run_kind in ("sslo_offload", "sslo_adaptive_offload")
    kv_transfer_config = None
    if needs_kv_offload:
        kv_transfer_config = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config=json.loads(
                os.environ.get("SSLO_KV_OFFLOAD_EXTRA", "{}")
            ),
        )

    engine_kwargs: dict[str, Any] = dict(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_log_requests=False,
        sslo_params=sslo_params,
        # Auto-load HF model's generation_config so SamplingParams defaults
        # (temperature, top_p, top_k, etc.) come from the model.
        generation_config="auto",
    )
    if args.max_model_len > 0:
        engine_kwargs["max_model_len"] = args.max_model_len
    # else: omit so vLLM picks the model's config max.
    if kv_transfer_config is not None:
        engine_kwargs["kv_transfer_config"] = kv_transfer_config
        engine_kwargs["disable_hybrid_kv_cache_manager"] = False
        engine_kwargs["enable_prefix_caching"] = True
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # Use the model's HF generation_config defaults; only override max_tokens
    # so all runs produce the same response budget.
    sampling_params = SamplingParams(max_tokens=args.generation_max_tokens)

    # Generate Poisson inter-arrival offsets relative to t0.
    rate = args.request_rate
    rng = _random.Random(args.request_rate_seed)
    arrival_offsets: list[float] = []
    cur = 0.0
    for _ in prompts:
        arrival_offsets.append(cur)
        if rate > 0:
            cur += rng.expovariate(rate)
    if rate > 0:
        print(
            f"{args.run_kind}: Poisson arrivals at {rate} req/s, "
            f"last_offset={arrival_offsets[-1]:.2f}s, seed={args.request_rate_seed}"
        )

    try:
        t0 = time.monotonic()
        tasks = [
            asyncio.create_task(collect_request_with_delay(
                engine, i, prompt, sampling_params, arrival_offsets[i]))
            for i, prompt in enumerate(prompts)
        ]
        rows = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0
        request_rows = [{k: v for k, v in row.items() if k != "slo_chunk_records"} for row in rows]
        write_jsonl(output_dir / f"requests_{args.run_kind}.jsonl", request_rows)
        chunk_rows = [
            {"request_id": str(row["request_id"]), **chunk}
            for row in rows
            for chunk in (row.get("slo_chunk_records") or [])
        ]
        write_jsonl(output_dir / f"chunks_{args.run_kind}.jsonl", chunk_rows)
        print(
            f"{args.run_kind}: completed {len(rows)} requests in {elapsed:.1f}s; "
            f"wrote {output_dir / f'requests_{args.run_kind}.jsonl'}"
        )
        print(
            f"{args.run_kind}: wrote {len(chunk_rows)} chunks to "
            f"{output_dir / f'chunks_{args.run_kind}.jsonl'}"
        )
    finally:
        engine.shutdown()
        del engine
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(2)


def main() -> None:
    args = parse_args()
    asyncio.run(run_one(args))


if __name__ == "__main__":
    main()
