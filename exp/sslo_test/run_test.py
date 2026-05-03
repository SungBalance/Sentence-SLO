#!/usr/bin/env python3
"""Run baseline and SSLO TTFT experiments in isolated subprocesses."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from lm_datasets import load_prompts


DEFAULT_OUTPUT_DIR = "exp/sslo_test/output"
GPU_READY_FREE_MEMORY_MIB = 92160
GPU_READY_TIMEOUT_S = 60.0
GPU_READY_BASE_SLEEP_S = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-kind", choices=["both", "baseline", "sslo"], default="both")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-name", default="koala")
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--chunk-unit", choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--seconds-per-word", type=float, default=0.28)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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


def output_token_count(request_output: Any) -> int:
    return sum(len(output.token_ids or []) for output in request_output.outputs or [])


def record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def extract_chunk_records(request_output: Any) -> list[dict[str, Any]]:
    records = getattr(request_output, "slo_chunk_records", None)
    if not records:
        return []
    normalized = []
    for record in records:
        normalized.append({
            "chunk_idx": record_value(record, "chunk_idx"),
            "cumulative_slack": record_value(record, "cumulative_slack"),
            "gen_time": record_value(record, "gen_time"),
            "pending_time": record_value(record, "pending_time"),
            "word_count": record_value(record, "word_count"),
        })
    return normalized


def request_chunk_rows(
    request_id: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "request_id": request_id,
            "chunk_idx": record.get("chunk_idx"),
            "cumulative_slack": record.get("cumulative_slack"),
            "gen_time": record.get("gen_time"),
            "pending_time": record.get("pending_time"),
            "word_count": record.get("word_count"),
        }
        for record in records
    ]


async def collect_request(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
) -> dict[str, Any]:
    request_id = str(request_idx)
    t_submit = time.monotonic()
    t_first_token: float | None = None
    last_output = None

    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        last_output = output
        if t_first_token is None and output_token_count(output) > 0:
            t_first_token = time.monotonic()

    t_finish = time.monotonic()
    num_tokens = output_token_count(last_output) if last_output is not None else 0
    metrics = getattr(last_output, 'metrics', None) if last_output is not None else None
    queue_stall = None
    first_token_ts = None
    if metrics is not None:
        queued = getattr(metrics, 'queued_ts', None)
        scheduled = getattr(metrics, 'scheduled_ts', None)
        first_token_ts = getattr(metrics, 'first_token_ts', None)
        if queued and scheduled and scheduled >= queued:
            queue_stall = scheduled - queued
    slo_chunk_records = (
        extract_chunk_records(last_output) if last_output is not None else []
    )
    tpot = None
    if t_first_token is not None:
        tpot = (t_finish - t_first_token) / max(1, num_tokens - 1)
    return {
        "request_id": request_id,
        "request_idx": request_idx,
        "t_submit": t_submit,
        "t_first_token": t_first_token,
        "ttft": (t_first_token - t_submit) if t_first_token is not None else None,
        "t_finish": t_finish,
        "num_tokens": num_tokens,
        "num_output_tokens": num_tokens,
        "tpot": tpot,
        "queue_stall": queue_stall,
        "decoding_start_ts": first_token_ts,
        "slo_chunk_records": slo_chunk_records,
    }


async def run_one(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_workload(args.dataset_name, args.num_prompts)
    print(f"{args.run_kind}: loaded {len(prompts)} prompts from {args.dataset_name}")

    if args.run_kind == "sslo":
        sslo_params = {
            "enabled": True,
            "chunk_unit": args.chunk_unit,
            "seconds_per_word": args.seconds_per_word,
            "chunk_gen_estimator": "p99",
            "chunk_gen_p99_window": 100,
        }
    else:
        sslo_params = {"enabled": False}

    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_log_requests=False,
        sslo_params=sslo_params,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        max_tokens=args.generation_max_tokens,
        temperature=0.0,
    )

    try:
        t0 = time.monotonic()
        tasks = [
            asyncio.create_task(collect_request(engine, i, prompt, sampling_params))
            for i, prompt in enumerate(prompts)
        ]
        rows = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0
        write_jsonl(output_dir / f"{args.run_kind}_ttft.jsonl", rows)
        chunk_rows = [
            chunk_row
            for row in rows
            for chunk_row in request_chunk_rows(
                str(row["request_id"]), row.get("slo_chunk_records") or []
            )
        ]
        write_jsonl(output_dir / f"{args.run_kind}_chunks.jsonl", chunk_rows)
        print(
            f"{args.run_kind}: completed {len(rows)} requests in {elapsed:.1f}s; "
            f"wrote {output_dir / f'{args.run_kind}_ttft.jsonl'}"
        )
        print(
            f"{args.run_kind}: wrote {len(chunk_rows)} chunks to "
            f"{output_dir / f'{args.run_kind}_chunks.jsonl'}"
        )
    finally:
        engine.shutdown()
        del engine
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(2)


def child_command(args: argparse.Namespace, run_kind: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-kind",
        run_kind,
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--generation-max-tokens",
        str(args.generation_max_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--chunk-unit",
        args.chunk_unit,
        "--seconds-per-word",
        str(args.seconds_per_word),
        "--output-dir",
        args.output_dir,
    ]


def run_child(args: argparse.Namespace, run_kind: str) -> int:
    output_dir = Path(args.output_dir)
    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env.pop("SSLO_STATS_LOG_PATH", None)
    if run_kind == "sslo":
        env["SSLO_STATS_LOG_PATH"] = str(output_dir / "sslo_stats.jsonl")
    print(f"Starting {run_kind} subprocess...")
    return subprocess.run(child_command(args, run_kind), check=False, env=env).returncode


def wait_for_gpu_memory_ready() -> None:
    print(
        "Waiting for GPU cleanup: "
        f"sleep {GPU_READY_BASE_SLEEP_S:.0f}s, then require "
        f">{GPU_READY_FREE_MEMORY_MIB} MiB free on GPU 0..."
    )
    time.sleep(GPU_READY_BASE_SLEEP_S)
    deadline = time.monotonic() + GPU_READY_TIMEOUT_S
    last_error: str | None = None
    last_free_mib: int | None = None
    while time.monotonic() < deadline:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            last_free_mib = int(output.splitlines()[0].strip())
            if last_free_mib > GPU_READY_FREE_MEMORY_MIB:
                print(f"GPU cleanup ready: free_memory={last_free_mib} MiB")
                return
            print(
                "GPU cleanup pending: "
                f"free_memory={last_free_mib} MiB "
                f"(need >{GPU_READY_FREE_MEMORY_MIB} MiB)"
            )
        except Exception as exc:
            last_error = str(exc)
            print(f"GPU cleanup poll failed: {last_error}")
        time.sleep(2)
    print(
        "WARNING: GPU cleanup wait timed out; proceeding anyway. "
        f"last_free_memory={last_free_mib} last_error={last_error}"
    )


def run_both(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "baseline_ttft.jsonl",
        "baseline_chunks.jsonl",
        "sslo_ttft.jsonl",
        "sslo_chunks.jsonl",
        "sslo_stats.jsonl",
        "summary.json",
        "run_status.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    statuses = {"baseline": run_child(args, "baseline")}
    wait_for_gpu_memory_ready()
    if statuses["baseline"] == 0:
        statuses["sslo"] = run_child(args, "sslo")
        wait_for_gpu_memory_ready()
    else:
        statuses["sslo"] = None
    (output_dir / "run_status.json").write_text(json.dumps(statuses, indent=2) + "\n")

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "analyze.py"),
            "--output-dir",
            args.output_dir,
            "--max-num-seqs",
            str(args.max_num_seqs),
        ],
        check=True,
    )
    failed = {name: rc for name, rc in statuses.items() if rc not in (0, None)}
    if failed:
        raise SystemExit(f"Experiment subprocess failed: {failed}")


def main() -> None:
    args = parse_args()
    if args.run_kind == "both":
        run_both(args)
    else:
        asyncio.run(run_one(args))


if __name__ == "__main__":
    main()
