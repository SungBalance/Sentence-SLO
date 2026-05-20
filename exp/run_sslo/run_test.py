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
from lm_datasets import load_prompts, _load_wildchat, _load_lmsys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import MODES_DEFAULT
from analysis.cpslo_names import classify_request


DEFAULT_OUTPUT_DIR = "exp/run_sslo/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-kind",
        required=True,
        choices=list(MODES_DEFAULT),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-name", default="koala",
                        choices=["koala", "wildchat", "lmsys", "combine"])
    parser.add_argument(
        "--exclude-code", action="store_true",
        help="Filter out code-generation prompts (prompt regex + first "
             "assistant-response check). Reduces long markdown/code chunks "
             "that strain the chunk-length predictor.",
    )
    parser.add_argument(
        "--conversation-only", action="store_true",
        help="Keep only multi-turn conversation rows (≥2 user, ≥2 "
             "assistant messages). Koala is single-turn — passing this "
             "flag with --dataset-name koala raises; combine drops koala "
             "regardless.",
    )
    parser.add_argument(
        "--dataset-seed", type=int, default=42,
        help="Seed used by the pool builder shuffle.",
    )
    parser.add_argument("--num-prompts", type=int, default=4000,
                        help="Pool size (prompts loaded into the sampling pool).")
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
        default=4.0,
        help="Poisson arrival rate in reqs/sec. Must be > 0.",
    )
    parser.add_argument(
        "--request-rate-seed",
        type=int,
        default=42,
        help="Seed for the Poisson inter-arrival sampler (reproducibility).",
    )
    parser.add_argument(
        "--measurement-window-s",
        type=float,
        default=180.0,
        help="Measurement window duration in seconds, starting from the first completed request.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed for pool sampling order. Defaults to --request-rate-seed.",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Apply chat template with enable_thinking=True (unified "
             "Instruct+Thinking models like Qwen3.5). Default: False.",
    )
    parser.add_argument(
        "--no-chat-template", dest="apply_chat_template", action="store_false",
        help="Skip chat template application — send raw user text as completion.",
    )
    parser.set_defaults(apply_chat_template=True)
    # Sampling overrides — when set, supersede model HF generation_config
    # defaults. Use to pin sampling across heterogeneous model families
    # (e.g. Qwen3.5-9B has no generation_config.json while 27B does).
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()
    if args.request_rate <= 0:
        raise ValueError(
            f"--request-rate must be > 0 (got {args.request_rate}). "
            "The zero-rate batch mode is no longer supported."
        )
    return args


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_workload(
    dataset_name: str, num_prompts: int, *,
    exclude_code: bool = False, dataset_seed: int = 42,
    conversation_only: bool = False,
) -> list[str]:
    prompts = load_prompts(
        dataset_name, num_prompts=num_prompts,
        exclude_code=exclude_code, seed=dataset_seed,
        conversation_only=conversation_only)
    if len(prompts) >= num_prompts:
        return prompts[:num_prompts]
    repeated: list[str] = []
    while len(repeated) < num_prompts:
        repeated.extend(prompts)
    return repeated[:num_prompts]


def _build_pool(args: argparse.Namespace) -> list[str]:
    """Build the 4000-prompt sampling pool: wildchat 2000 + lmsys 2000."""
    wildchat_raw = _load_wildchat(
        split="train", num_prompts=2500,
        exclude_code=True, conversation_only=True)
    lmsys_raw = _load_lmsys(
        split="train", num_prompts=2500,
        exclude_code=True, conversation_only=True)
    wildchat_trimmed = wildchat_raw[:2000]
    lmsys_trimmed = lmsys_raw[:2000]
    combined = wildchat_trimmed + lmsys_trimmed
    rng = _random.Random(args.dataset_seed)
    rng.shuffle(combined)
    if args.apply_chat_template:
        combined = apply_chat_template_to_prompts(
            combined, args.model, enable_thinking=args.enable_thinking)
    assert len(combined) == 4000, f"Expected 4000 prompts, got {len(combined)}"
    return combined


def apply_chat_template_to_prompts(
    prompts: list[str],
    model: str,
    *,
    enable_thinking: bool,
) -> list[str]:
    """Wrap each user prompt with the model's chat template.

    For unified Instruct+Thinking models (Qwen3.5), enable_thinking=False
    emits the empty `<think>\\n\\n</think>\\n\\n` block in the template so
    the assistant starts directly with the response.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    out = []
    for p in prompts:
        out.append(tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ))
    return out


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

    # Map vLLM ChunkRecord fields onto the chunks.jsonl schema.
    normalized = []
    for record in records:
        slack = _val(record, "slack_s")
        pending = _val(record, "pending_time_s")
        end_ts = _val(record, "gen_finish_ts")
        start_ts = _val(record, "start_time_ts")
        deadline = _val(record, "deadline_ts")
        normalized.append({
            "chunk_idx": _val(record, "chunk_idx"),
            "chunk_slack": slack,
            "deadline_ts": deadline,
            "start_time_ts": start_ts,
            "end_time_ts": end_ts,
            "pending_time": pending,
            "num_words": _val(record, "word_count"),
            "num_token": _val(record, "num_token"),
            "num_iters": _val(record, "num_iters"),
            "num_running_iters": _val(record, "num_running_iters"),
            "num_pending_iters": _val(record, "num_pending_iters"),
            "expected_len": _val(record, "expected_len"),
            # CP-SLO canonical names (dual-write alongside legacy keys above).
            "chunk_deadline_ts": deadline,
            "chunk_deadline_margin_s": slack,
            "chunk_generation_start_ts": start_ts,
            "chunk_generation_end_ts": end_ts,
            "token_start_idx": _val(record, "token_start_idx"),
            "token_end_idx": _val(record, "token_end_idx"),
            "cumulative_tokens_at_end": _val(record, "cumulative_tokens_at_end"),
            "chunk_consume_time_s": _val(record, "chunk_consume_time_s"),
            "demand_window_start_ts": _val(record, "demand_window_start_ts"),
            "demand_window_end_ts": _val(record, "demand_window_end_ts"),
            "expected_chunk_len_high": _val(record, "expected_chunk_len_high"),
            "predictor_source": _val(record, "predictor_source"),
            "stall_start_ts": _val(record, "stall_start_ts"),
            "stall_end_ts": _val(record, "stall_end_ts"),
            "stall_duration_s": _val(record, "stall_duration_s"),
            "text": _val(record, "text"),
        })
    return normalized


async def collect_one(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
    injection_ts: float,
    warmup_event: asyncio.Event,
    warmup_counter: list[int],
    warmup_target: int,
) -> dict[str, Any]:
    """Stream one request; set warmup_event once warmup_target completions reached."""
    request_id = str(request_idx)
    last_output = None

    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        last_output = output

    if last_output is None:
        return {
            "request_id": request_id,
            "request_idx": request_idx,
            "injection_ts": injection_ts,
            "injection_idx": request_idx,
            "ttft": None,
            "tpot": None,
            "queue_stall": None,
            "num_output_tokens": 0,
            "num_prompt_tokens": None,
            "slo_chunk_records": [],
            "total_pending_time_s": None,
            "num_pending_intervals": 0,
            "terminal_outcome": "in_progress",
        }

    metrics = getattr(last_output, "metrics", None)
    num_prompt_tokens = getattr(metrics, "num_prompt_tokens", None) if metrics else None
    if num_prompt_tokens is None:
        num_prompt_tokens = len(getattr(last_output, "prompt_token_ids", []) or []) or None
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
    # SSLO Phase 6 F3: surface vLLM's per-request preemption counter so the
    # run-level sidecar can sum it. Missing attr defaults to 0 for safety
    # under older vLLM builds.
    num_preemptions = int(getattr(metrics, "num_preemptions", 0) or 0) if metrics else 0

    slo_chunk_records = extract_chunk_records(last_output)
    # ttfc — time to first chunk, from queue entry to first chunk completion.
    ttfc = None
    if queued and slo_chunk_records:
        first_chunk_end = slo_chunk_records[0].get("end_time_ts")
        if first_chunk_end is not None and first_chunk_end >= queued:
            ttfc = float(first_chunk_end) - float(queued)
    sslo_metrics = getattr(last_output, "sslo_metrics", None)
    total_pending_time_s = (
        getattr(sslo_metrics, "total_pending_time_s", None) if sslo_metrics else None
    )
    num_pending_iters_per_request = (
        getattr(sslo_metrics, "num_pending_intervals", 0) if sslo_metrics else 0
    )

    admitted_ts_val = (
        getattr(sslo_metrics, "admitted_ts", None) if sslo_metrics else None
    )
    consume_start_ts_val = (
        getattr(sslo_metrics, "consume_start_ts", None) if sslo_metrics else None
    )
    terminal_outcome_val = (
        getattr(sslo_metrics, "terminal_outcome", "in_progress") if sslo_metrics else "in_progress"
    )

    # Tally completions and open the warmup gate at the threshold.
    if terminal_outcome_val == "completed":
        warmup_counter[0] += 1
        if (warmup_counter[0] >= warmup_target
                and not warmup_event.is_set()):
            warmup_event.set()

    queue_stall_s = (
        float(admitted_ts_val) - float(queued)
        if (admitted_ts_val is not None and queued is not None and admitted_ts_val >= queued)
        else None
    )
    reference_output_tokens = num_gen if num_gen > 0 else None
    request_class = classify_request(reference_output_tokens)
    return {
        "request_id": request_id,
        "request_idx": request_idx,
        "injection_ts": injection_ts,
        "injection_idx": request_idx,
        "num_output_tokens": num_gen,
        "num_prompt_tokens": num_prompt_tokens,
        "num_chunks": len(slo_chunk_records),
        "ttft": ttft,
        "ttfc": ttfc,
        "tpot": tpot,
        "queue_stall": queue_stall,
        "decoding_start_ts": first_ts,
        "slo_chunk_records": slo_chunk_records,
        "total_pending_time_s": total_pending_time_s,
        "num_pending_iters_per_request": num_pending_iters_per_request,
        # CP-SLO lifecycle fields.
        "admitted_ts": admitted_ts_val,
        "consume_start_ts": consume_start_ts_val,
        "terminal_outcome": terminal_outcome_val,
        "queue_stall_s": queue_stall_s,
        # CP-SLO canonical aliases (same values, new names).
        "first_token_ts": first_ts,
        "TTFC": ttfc,
        "num_pending_intervals": num_pending_iters_per_request,
        # F2: request classification by observed generation length.
        "reference_output_tokens": reference_output_tokens,
        "request_class": request_class,
        # SSLO Phase 6 F3: per-request preemption count (vLLM v1).
        "num_preemptions": num_preemptions,
    }


def _make_run_id(args: argparse.Namespace, ts: float) -> str:
    return (
        f"{args.run_kind}_{args.max_num_seqs}_{args.request_rate}_"
        f"{args.request_rate_seed}_{int(ts)}"
    )


def _policy_label(args: argparse.Namespace) -> str:
    # run_kind identifies the scheduling mode (baseline / sslo / sslo_adaptive / ...).
    return args.run_kind


def _make_variant_label(args: argparse.Namespace) -> str:
    parts = [args.run_kind]
    if "adaptive" in args.run_kind:
        parts.append("abatch=on")
    else:
        parts.append("abatch=off")
    return "/".join(parts)


def _gpu_peak_bytes() -> int | None:
    # Best-effort: CPU-only or torch-less environments return None.
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def _sum_preemptions(rows: list[dict[str, Any]]) -> int:
    return sum(int(r.get("num_preemptions", 0) or 0) for r in rows)


def write_run_meta(
    args: argparse.Namespace,
    output_dir: Path,
    run_started_ts: float,
    run_ended_ts: float,
    requests_rows: list[dict[str, Any]],
    *,
    window_start_ts: float | None,
    window_end_ts: float | None,
    pool_size: int,
    sampling_seed: int,
    pool_pass_count: int,
    injected_count: int,
    in_window_count: int,
) -> None:
    # Sidecar consumed by analyze.py / _consolidate_mode_outputs.py so the
    # validity gate can see run-level identifiers and F3 counters even if
    # the per-row JSONL doesn't.
    meta = {
        "run_id": _make_run_id(args, run_started_ts),
        "policy": _policy_label(args),
        "variant": _make_variant_label(args),
        "seed": args.request_rate_seed,
        "trace_id": (
            f"poisson_rate{args.request_rate}_seed{args.request_rate_seed}"
        ),
        "workload_id": "wildchat2k_lmsys2k",
        "N": len(requests_rows),
        "M": args.max_num_seqs,
        "measurement_window_start_ts": window_start_ts,
        "measurement_window_end_ts": window_end_ts,
        "measurement_window_seconds": args.measurement_window_s,
        "pool_size": pool_size,
        "pool_source": "wildchat2k_lmsys2k",
        "sampling_seed": sampling_seed,
        "pool_pass_count": pool_pass_count,
        "injected_count": injected_count,
        "in_window_count": in_window_count,
        "gpu_memory_peak_bytes": _gpu_peak_bytes(),
        "num_preemptions_total": _sum_preemptions(requests_rows),
        "num_requests_completed": sum(
            1 for r in requests_rows
            if r.get("terminal_outcome") == "completed"
        ),
        "num_requests_total": len(requests_rows),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


async def run_one(args: argparse.Namespace) -> None:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = _build_pool(args)
    print(f"{args.run_kind}: built pool of {len(pool)} prompts (wildchat2k+lmsys2k)")

    # Build sslo_params. Both baseline and sslo modes run through
    # schedule_sslo() so chunk records / scheduler_stats / tpot EMA are
    # collected uniformly; method="baseline" skips the SSLO placement
    # logic so admission is equivalent to vanilla vLLM.
    sslo_params = {
        "chunk_unit": args.chunk_unit,
        "seconds_per_word": args.seconds_per_word,
        "method": "baseline" if args.run_kind == "baseline" else "sslo",
    }
    if args.run_kind != "baseline":
        # SSLO_POLICY env selects placement algorithm:
        #   threshold (default) | pressure | buffer | combined.
        sslo_params["policy"] = os.environ.get(
            "SSLO_POLICY", "threshold")
        if "adaptive" in args.run_kind:
            sslo_params["adaptive_batching"] = True
        if "offload" in args.run_kind:
            sslo_params["offloading"] = True
        # SSLO: sslo_mlp pins method=sslo with policy=multi_level_pressure
        # and forces adaptive_batching=True (MLP's critical branch already
        # shrinks the cap; KV offload connector is wired by needs_kv_offload).
        if args.run_kind == "sslo_mlp":
            sslo_params["policy"] = "multi_level_pressure"
            sslo_params["adaptive_batching"] = (
                os.environ.get("SSLO_ADAPTIVE_BATCHING", "1") != "0")
            # SSLO: optional waiting-admission floor under critical mode.
            if os.environ.get("MLP_CRITICAL_WAITING_FLOOR", "0") == "1":
                sslo_params["mlp_critical_waiting_floor"] = True

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
        language_model_only=True,
    )
    if args.max_model_len > 0:
        engine_kwargs["max_model_len"] = args.max_model_len
    # else: omit so vLLM picks the model's config max.
    if kv_transfer_config is not None:
        engine_kwargs["kv_transfer_config"] = kv_transfer_config
        engine_kwargs["disable_hybrid_kv_cache_manager"] = False
        engine_kwargs["enable_prefix_caching"] = True
    if args.enable_thinking:
        engine_kwargs["reasoning_parser"] = "qwen3"
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    if args.enable_thinking:
        sampling_kwargs = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                           "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    else:
        sampling_kwargs = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                           "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    # CLI overrides win over the defaults above.
    for name in ("temperature", "top_p", "top_k", "min_p",
                 "presence_penalty", "repetition_penalty"):
        v = getattr(args, name)
        if v is not None:
            sampling_kwargs[name] = v
    sampling_kwargs["max_tokens"] = args.generation_max_tokens
    sampling_params = SamplingParams.from_optional(**sampling_kwargs)
    print(f"{args.run_kind}: sampling_params={sampling_params}")

    def _per_req_sampling_params(seed: int):
        p = sampling_params.clone()
        p.seed = seed
        return p

    # Set up pool sampling state.
    seed = args.sampling_seed if args.sampling_seed is not None else args.request_rate_seed
    rng = _random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    cursor = 0
    pool_pass_count = 0

    def next_prompt() -> str:
        nonlocal cursor, pool_pass_count
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
            pool_pass_count += 1
        p = pool[order[cursor]]
        cursor += 1
        return p

    warmup_event = asyncio.Event()
    warmup_counter: list[int] = [0]
    warmup_target = args.max_num_seqs
    window_start_ts: float | None = None
    window_end_ts: float | None = None
    injected: list[tuple[int, asyncio.Task, float]] = []
    injection_seq = 0

    async def watcher() -> None:
        nonlocal window_start_ts, window_end_ts
        await warmup_event.wait()
        window_start_ts = time.time()
        window_end_ts = window_start_ts + args.measurement_window_s

    async def injector() -> None:
        nonlocal injection_seq
        while window_end_ts is None or time.time() < window_end_ts:
            inj_ts = time.time()
            idx = injection_seq
            injection_seq += 1
            prompt = next_prompt()
            task = asyncio.create_task(collect_one(
                engine, idx, prompt,
                _per_req_sampling_params(idx),
                inj_ts, warmup_event, warmup_counter, warmup_target))
            injected.append((idx, task, inj_ts))
            await asyncio.sleep(rng.expovariate(args.request_rate))

    try:
        run_started_ts = time.time()
        t0 = time.monotonic()
        await asyncio.gather(watcher(), injector())
        rows = list(await asyncio.gather(*[t for _, t, _ in injected]))
        elapsed = time.monotonic() - t0
        run_ended_ts = time.time()

        # Compute in_window for each row now that window bounds are known.
        for row in rows:
            inj_ts = row.get("injection_ts")
            if (inj_ts is not None and window_start_ts is not None
                    and window_end_ts is not None):
                row["in_window"] = bool(window_start_ts <= inj_ts < window_end_ts)
            else:
                row["in_window"] = False

        in_window_count = sum(1 for r in rows if r.get("in_window"))

        request_rows = [
            {"mode": args.run_kind,
             **{k: v for k, v in row.items() if k != "slo_chunk_records"}}
            for row in rows
        ]
        write_jsonl(output_dir / "requests.jsonl", request_rows)
        chunk_rows = [
            {
                "mode": args.run_kind,
                "request_id": str(row["request_id"]),
                "request_idx": row.get("request_idx"),
                **chunk,
            }
            for row in rows
            for chunk in (row.get("slo_chunk_records") or [])
        ]
        write_jsonl(output_dir / "chunks.jsonl", chunk_rows)
        # Sidecar for analyze.py: SSLO config options (per-mode) to embed in
        # summary.config. Only emitted for sslo* modes; baseline gets {}.
        sslo_config_path = output_dir / "sslo_config.json"
        sslo_config_path.write_text(json.dumps(sslo_params) + "\n")
        # SSLO Phase 6: run-level sidecar for the validity gate (run_id,
        # GPU peak, preemption totals, completion counts).
        write_run_meta(
            args, output_dir, run_started_ts, run_ended_ts, request_rows,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            pool_size=len(pool),
            sampling_seed=seed,
            pool_pass_count=pool_pass_count,
            injected_count=len(rows),
            in_window_count=in_window_count,
        )
        print(
            f"{args.run_kind}: completed {len(rows)} requests in {elapsed:.1f}s; "
            f"wrote {output_dir / 'requests.jsonl'}"
        )
        print(
            f"{args.run_kind}: wrote {len(chunk_rows)} chunks to "
            f"{output_dir / 'chunks.jsonl'}"
        )
        print(
            f"{args.run_kind}: window={window_start_ts:.1f}..{window_end_ts:.1f} "
            f"injected={len(rows)} in_window={in_window_count}"
        )
        prompt_counts = [r["num_prompt_tokens"] for r in rows if r and r.get("num_prompt_tokens") is not None]
        output_counts = [r["num_output_tokens"] for r in rows if r and r.get("num_output_tokens") is not None]
        prompt_mean = f"{sum(prompt_counts)/len(prompt_counts):.1f}" if prompt_counts else "n/a"
        output_mean = f"{sum(output_counts)/len(output_counts):.1f}" if output_counts else "n/a"
        print(f"{args.run_kind}: workload mean num_prompt_tokens={prompt_mean}, num_output_tokens={output_mean}")
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
