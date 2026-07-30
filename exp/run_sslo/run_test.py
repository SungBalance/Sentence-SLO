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
        "--english-only", action="store_true",
        help="Strictly keep only rows whose row-level language field "
             "equals 'English' (WildChat / LMSYS). Removes Chinese, "
             "Persian, multilingual hashtag spam, and 'Nolang' rows that "
             "confuse the sentence boundary detector.",
    )
    parser.add_argument(
        "--max-response-chunk-chars", type=int, default=0,
        help="Drop rows whose first assistant response produces a chunk "
             "longer than this many chars under SSLO's ChunkSeparator "
             "(sentence mode). Filters out dump-style responses (long "
             "comma lists, ASCII output, repeated tokens) that lack "
             "sentence-end punctuation. 0 = disabled.",
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
    # SSLO
    parser.add_argument(
        "--warmup-target",
        type=int,
        default=0,
        help="In-window gate opens after N completed requests (steady-state "
             "primer). 0 = auto: max_num_seqs * 2.",
    )
    # SSLO
    parser.add_argument(
        "--measurement-target",
        type=int,
        default=1024,
        help="Number of completed requests measured per rate after the "
             "warmup gate. Default 1024 gives statistical confidence at "
             "small caps too.",
    )
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--chunk-unit", choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--seconds-per-word", type=float, default=0.28)
    # SSLO
    parser.add_argument("--consume-mode", choices=["read", "tts"], default="read")
    # SSLO
    parser.add_argument("--tts-profile-path", default=None)
    # SSLO
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--request-rates",
        type=str,
        default="4",
        help="Comma- or space-separated list of Poisson arrival rates "
             "(reqs/sec). All rates are swept under a single engine; "
             "between rates the SSLO state is reset. Each rate writes "
             "to OUTPUT_DIR/rate_<r>/.",
    )
    parser.add_argument(
        "--request-rate-seed",
        type=int,
        default=42,
        help="Seed for the Poisson inter-arrival sampler (reproducibility).",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="If set, after each rate finishes run analyze.py on its "
             "output dir and append a one-line summary row to this CSV "
             "(file-locked, safe for parallel jobs).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat index (1-based) recorded in the summary CSV row.",
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
    # Parse rates list (accept "0.5,1,2" or "0.5 1 2" or single "4").
    raw = args.request_rates.replace(",", " ").split()
    try:
        rates = [float(x) for x in raw]
    except ValueError:
        raise ValueError(
            f"--request-rates must be numeric (got {args.request_rates!r})")
    if not rates or any(r <= 0 for r in rates):
        raise ValueError(
            f"--request-rates must be positive (got {rates}). "
            "The zero-rate batch mode is no longer supported."
        )
    args.rates = rates
    return args


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_workload(
    dataset_name: str, num_prompts: int, *,
    exclude_code: bool = False, dataset_seed: int = 42,
    conversation_only: bool = False, english_only: bool = False,
) -> list[str]:
    prompts = load_prompts(
        dataset_name, num_prompts=num_prompts,
        exclude_code=exclude_code, seed=dataset_seed,
        conversation_only=conversation_only,
        english_only=english_only)
    if len(prompts) >= num_prompts:
        return prompts[:num_prompts]
    repeated: list[str] = []
    while len(repeated) < num_prompts:
        repeated.extend(prompts)
    return repeated[:num_prompts]


def _build_pool(args: argparse.Namespace) -> list[str]:
    """Build the sampling pool from the seed-agnostic combined cache.

    The shared curated pool lives at
    exp/tools/dataset_cache/processed_dataset.jsonl and is reused across
    seeds. Seed only controls the load-time shuffle in
    `load_or_build_combine_pool`. Chat template is then applied
    per-model.
    """
    from dataset_cache import load_or_build_combine_pool

    def _build_uncached() -> list[str]:
        max_resp = args.max_response_chunk_chars or None
        wildchat_raw = _load_wildchat(
            split="train", num_prompts=2500,
            exclude_code=True, conversation_only=True,
            english_only=args.english_only,
            max_response_chunk_chars=max_resp)
        lmsys_raw = _load_lmsys(
            split="train", num_prompts=2500,
            exclude_code=True, conversation_only=True,
            english_only=args.english_only,
            max_response_chunk_chars=max_resp)
        return wildchat_raw[:2048] + lmsys_raw[:2048]

    combined = load_or_build_combine_pool(
        num_datasets=2, seed=args.dataset_seed,
        build_fn=_build_uncached)
    if args.apply_chat_template:
        combined = apply_chat_template_to_prompts(
            combined, args.model, enable_thinking=args.enable_thinking)
    if not combined:
        raise RuntimeError("combined pool is empty")
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
    # records may contain ChunkRecord objects or asdict() results.
    def _val(r, k):
        return r.get(k) if isinstance(r, dict) else getattr(r, k, None)

    # Map vLLM ChunkRecord fields onto the chunks.jsonl schema.
    normalized = []
    for record in records:
        normalized.append({
            "unit_index": _val(record, "unit_index"),
            "token_start": _val(record, "token_start"),
            "token_end": _val(record, "token_end"),
            "consumer_ready_time": _val(record, "consumer_ready_time"),
            "consume_duration": _val(record, "consume_duration"),
            "deadline": _val(record, "deadline"),
            "unit_deadline_miss_s": _val(record, "unit_deadline_miss_s"),
            "unit_deadline_missed": _val(record, "unit_deadline_missed"),
            "text_generation_end_time": _val(record, "text_generation_end_time"),
            "conversion_time": _val(record, "conversion_time"),
            "token_boundary": _val(record, "token_boundary"),
            "word_count": _val(record, "word_count"),
            "num_token": _val(record, "num_token"),
            "text": _val(record, "text"),
        })
    return normalized


async def collect_one(
    engine: Any,
    request_idx: int,
    prompt: str,
    sampling_params: Any,
    arrival_ts: float,
    warmup_event: asyncio.Event,
    warmup_counter: list[int],
    warmup_target: int,
    measurement_done_event: asyncio.Event,
    measurement_counter: list[int],
    measurement_target: int,
    window_ts_holder: dict[str, float | None],
) -> dict[str, Any]:
    """Stream one request. Tracks two completion gates:
      - warmup_event fires when `warmup_target` completions reached
        (collection window starts)
      - measurement_done_event fires when `measurement_target` more
        completions accrue after warmup (collection window ends)."""
    request_id = str(request_idx)
    last_output = None

    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        last_output = output
    completion_wall_ts = time.time()

    if last_output is None:
        return {
            "request_id": request_id,
            "request_idx": request_idx,
            "arrival_ts": arrival_ts,
            "completion_wall_ts": completion_wall_ts,
            "ttft": None,
            "ttfc": None,
            "tpot": None,
            "queue_stall_s": None,
            "consume_start_time": None,
            "num_consumable_units": 0,
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
    # SSLO Phase 6 F3: surface vLLM's per-request preemption counter so the
    # run-level sidecar can sum it. Missing attr defaults to 0 for safety
    # under older vLLM builds.
    num_preemptions = int(getattr(metrics, "num_preemptions", 0) or 0) if metrics else 0

    slo_chunk_records = extract_chunk_records(last_output)
    # ttfc — time to first chunk, from queue entry to first chunk completion.
    ttfc = None
    if queued and slo_chunk_records:
        first_chunk_end = slo_chunk_records[0].get("text_generation_end_time")
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
    consume_start_time_val = (
        getattr(sslo_metrics, "consume_start_time", None) if sslo_metrics else None
    )
    terminal_outcome_val = (
        getattr(sslo_metrics, "terminal_outcome", "in_progress") if sslo_metrics else "in_progress"
    )

    # Tally completions: open warmup gate, then close measurement gate.
    # Critical: window_start_ts / window_end_ts are recorded INLINE here
    # (same task, same callback) — capturing them in the watcher after
    # an `await event.wait()` would race with other tasks' completions
    # firing in between, producing a window that's narrower than the
    # gate semantics imply (in_window_count → near zero).
    # The boundary task (one that flips warmup_event) is NOT counted in
    # measurement — `elif` skips it; its completion_wall_ts equals
    # window_start_ts so in_window comparison is well-defined.
    completion_mono_ts = time.monotonic()
    if terminal_outcome_val == "completed":
        warmup_counter[0] += 1
        if (warmup_counter[0] >= warmup_target
                and not warmup_event.is_set()):
            window_ts_holder["start_ts"] = completion_wall_ts
            window_ts_holder["start_mono_ts"] = completion_mono_ts
            warmup_event.set()
        elif warmup_event.is_set():
            measurement_counter[0] += 1
            if (measurement_counter[0] >= measurement_target
                    and not measurement_done_event.is_set()):
                window_ts_holder["end_ts"] = completion_wall_ts
                window_ts_holder["end_mono_ts"] = completion_mono_ts
                measurement_done_event.set()

    queue_stall_s = (
        float(admitted_ts_val) - float(queued)
        if (admitted_ts_val is not None and queued is not None and admitted_ts_val >= queued)
        else None
    )
    reference_output_tokens = num_gen if num_gen > 0 else None
    request_class = classify_request(reference_output_tokens)

    # Per-request CU-SLO aggregates from slo_chunk_records (spec section 6/12).
    request_max_stall_s = 0.0
    request_total_stall_s = 0.0
    request_demand_duration_s = 0.0
    if slo_chunk_records:
        misses = [float(c.get("unit_deadline_miss_s", 0.0) or 0.0)
                  for c in slo_chunk_records]
        request_max_stall_s = max(misses) if misses else 0.0
        request_total_stall_s = sum(misses)
        last = slo_chunk_records[-1]
        last_deadline = last.get("deadline")
        last_consume = last.get("consume_duration") or 0.0
        if (last_deadline is not None and consume_start_time_val is not None):
            request_demand_duration_s = max(
                0.0, float(last_deadline) + float(last_consume)
                - float(consume_start_time_val))
    request_stall_fraction = (
        request_total_stall_s / request_demand_duration_s
        if request_demand_duration_s > 0 else 0.0
    )
    cu_violations = {
        f"request_cu_slo_violated_tau_{tau}":
            int(request_max_stall_s > float(tau))
        for tau in ("0.5", "1", "2", "5")
    }

    return {
        "request_id": request_id,
        "request_idx": request_idx,
        # SSLO: store prompt for offline dataset-filter analysis. Truncated
        # to 2000 chars to avoid jsonl bloat; full text not needed.
        "prompt": (prompt[:2000] if isinstance(prompt, str) else ""),
        "arrival_ts": arrival_ts,
        "completion_wall_ts": completion_wall_ts,
        "num_output_tokens": num_gen,
        "num_prompt_tokens": num_prompt_tokens,
        "num_consumable_units": len(slo_chunk_records),
        "ttft": ttft,
        "ttfc": ttfc,
        "tpot": tpot,
        "decoding_start_ts": first_ts,
        "slo_chunk_records": slo_chunk_records,
        "total_pending_time_s": total_pending_time_s,
        "num_pending_iters_per_request": num_pending_iters_per_request,
        # CP-SLO lifecycle fields.
        "admitted_ts": admitted_ts_val,
        "consume_start_time": consume_start_time_val,
        "terminal_outcome": terminal_outcome_val,
        "queue_stall_s": queue_stall_s,
        "first_token_ts": first_ts,
        "num_pending_intervals": num_pending_iters_per_request,
        # F2: request classification by observed generation length.
        "reference_output_tokens": reference_output_tokens,
        "request_class": request_class,
        # SSLO Phase 6 F3: per-request preemption count (vLLM v1).
        "num_preemptions": num_preemptions,
        # Spec section 6 + 12 per-request CU-SLO aggregates.
        "request_max_stall_s": request_max_stall_s,
        "request_total_stall_s": request_total_stall_s,
        "request_demand_duration_s": request_demand_duration_s,
        "request_stall_fraction": request_stall_fraction,
        **cu_violations,
    }


def _make_run_id(args: argparse.Namespace, ts: float, rate: float) -> str:
    return (
        f"{args.run_kind}_{args.max_num_seqs}_{rate}_"
        f"{args.request_rate_seed}_{int(ts)}"
    )


def _format_rate(r: float) -> str:
    """Compact directory-safe rate string. 0.5 -> '0.5', 1.0 -> '1'."""
    return f"{r:g}"


def _policy_label(args: argparse.Namespace) -> str:
    # run_kind identifies the scheduling mode (baseline / progress_serve).
    return args.run_kind


def _make_variant_label(args: argparse.Namespace) -> str:
    # Adaptive batching was removed; ProgressServe runs at a fixed cap.
    return f"{args.run_kind}/abatch=off"


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
    rate: float,
    window_start_ts: float | None,
    window_end_ts: float | None,
    window_start_mono_ts: float | None,
    window_end_mono_ts: float | None,
    pool_size: int,
    sampling_seed: int,
    pool_pass_count: int,
    injected_count: int,
    in_window_count: int,
    warmup_target: int,
    measurement_target: int,
) -> None:
    # Sidecar consumed by analyze.py / _consolidate_mode_outputs.py so the
    # validity gate can see run-level identifiers and F3 counters even if
    # the per-row JSONL doesn't.
    meta = {
        "run_id": _make_run_id(args, run_started_ts, rate),
        "policy": _policy_label(args),
        "variant": _make_variant_label(args),
        "seed": args.request_rate_seed,
        "request_rate": rate,
        "trace_id": (
            f"poisson_rate{rate}_seed{args.request_rate_seed}"
        ),
        "workload_id": "wildchat2048_lmsys2048",
        "N": len(requests_rows),
        "M": args.max_num_seqs,
        "measurement_window_start_ts": window_start_ts,
        "measurement_window_end_ts": window_end_ts,
        # Monotonic-clock counterparts so downstream code can match the
        # clock used by scheduler_stats.jsonl / chunks slo_chunk_records.
        "measurement_window_start_mono_ts": window_start_mono_ts,
        "measurement_window_end_mono_ts": window_end_mono_ts,
        # Completion-count gates (new workflow). No fixed time window.
        "warmup_target_completions": warmup_target,
        "measurement_target_completions": measurement_target,
        "pool_size": pool_size,
        "pool_source": "wildchat2048_lmsys2048",
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
    from vllm.engine.arg_utils import AsyncEngineArgs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = _build_pool(args)
    print(f"{args.run_kind}: built pool of {len(pool)} prompts (wildchat2048+lmsys2048)")

    # Build sslo_params. Both baseline and progress_serve modes run through
    # schedule_sslo() so chunk records / scheduler_stats / wall-step EMA are
    # collected uniformly; method="baseline" skips the SSLO placement logic
    # so admission is equivalent to vanilla vLLM.
    sslo_params = {
        "chunk_unit": args.chunk_unit,
        "seconds_per_word": args.seconds_per_word,
        "method": (
            "baseline" if args.run_kind == "baseline" else "progress_serve"),
    }
    # SSLO: adaptive batching on for the dedicated mode, or via env toggle on
    # the plain progress_serve mode. Never for baseline.
    if args.run_kind != "baseline":
        sslo_params["adaptive_batching"] = (
            args.run_kind == "progress_serve_adaptive"
            or os.environ.get("SSLO_ADAPTIVE_BATCHING", "0") != "0")
    # SSLO
    if args.consume_mode == "tts":
        # SSLO
        sslo_params["consume_mode"] = "tts"
        # SSLO
        sslo_params["tts_profile_path"] = args.tts_profile_path
        # SSLO
        sslo_params["tts_model"] = args.tts_model

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

    base_output_dir = output_dir
    request_idx_offset = 0
    # vLLM EngineCore runs in a separate ZMQ subprocess. Its env was
    # captured at fork/spawn time, so changing os.environ here does
    # NOT propagate. All rates write to the single SSLO_STATS_LOG_PATH
    # set by the .sh wrapper; we post-trim each rate's segment using
    # the monotonic window bounds.
    shared_stats_path = os.environ.get("SSLO_STATS_LOG_PATH")
    shared_stats_path = Path(shared_stats_path) if shared_stats_path else None
    shared_decisions_path = (
        shared_stats_path.parent / "decisions.jsonl"
        if shared_stats_path is not None else None)
    try:
        for rate_i, rate in enumerate(args.rates):
            rate_dir = base_output_dir / f"rate_{_format_rate(rate)}"
            rate_dir.mkdir(parents=True, exist_ok=True)
            print(f"{args.run_kind}: ===== rate={rate} "
                  f"({rate_i+1}/{len(args.rates)}) =====")
            injected_count = await _run_one_rate(
                engine=engine,
                pool=pool,
                sampling_params=sampling_params,
                rate=rate,
                output_dir=rate_dir,
                args=args,
                sslo_params=sslo_params,
                request_idx_offset=request_idx_offset,
                shared_stats_path=shared_stats_path,
                shared_decisions_path=shared_decisions_path)
            request_idx_offset += injected_count
            # Drain + reset SSLO state for next rate (skip after last).
            if rate_i + 1 < len(args.rates):
                # abort_requests_async (called above via engine.abort)
                # is fire-and-forget — the ABORT messages race with the
                # reset RPC in the EngineCore's ZMQ input queue. Poll
                # the post-reset state, retry if queues are non-empty
                # (one EngineCore step is ~few ms; 1s grace is usually
                # enough but not guaranteed under heavy load).
                post = None
                for attempt in range(5):
                    await asyncio.sleep(1.0)
                    try:
                        post = await engine.reset_sslo_state()
                    except Exception as e:  # noqa: BLE001
                        print(f"{args.run_kind}: reset_sslo_state failed: "
                              f"{type(e).__name__}: {e}")
                        post = None
                        break
                    if (post.get("waiting", 0) == 0
                            and post.get("running", 0) == 0):
                        break
                    print(f"{args.run_kind}: reset_sslo_state attempt "
                          f"{attempt+1}: queues still busy {post}, retrying")
                if post is not None:
                    print(f"{args.run_kind}: reset_sslo_state -> {post}")
                    if (post.get("waiting", 0) != 0
                            or post.get("running", 0) != 0):
                        print(f"{args.run_kind}: WARNING — queues not "
                              f"empty after 5 retries; rate {args.rates[rate_i+1]} "
                              f"will start with stale state")
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
        # Force-kill any descendants of this process (vLLM EngineCore
        # subprocs, multiprocessing.resource_tracker children) that
        # survived shutdown. Scoped to our own subtree — peer processes
        # on other GPUs (separate PID trees) are untouched. Prevents
        # orphan EngineCore from blocking the GPU when a subsequent
        # job's engine init tries to allocate KV cache.
        try:
            import psutil
            me = psutil.Process(os.getpid())
            kids = me.children(recursive=True)
            for p in kids:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if kids:
                psutil.wait_procs(kids, timeout=5.0)
        except Exception as e:  # noqa: BLE001
            print(f"{args.run_kind}: descendant cleanup raised: "
                  f"{type(e).__name__}: {e}")
        time.sleep(2)


_SUMMARY_CSV_HEADER = [
    # identifier
    "model", "max_num_seqs", "lambda_req_s", "policy", "seed",
    # window / counts
    "num_requests_in_measurement_window", "num_arrivals_total",
    # throughput / urgent-mode share
    # Request-side throughput (output_tokens_per_second is legacy alias
    # for tokens_per_second) — selection-biased by which reqs completed
    # inside the cap*4 measurement window.
    # Scheduler-side throughput (decode_tps / prefill_tps) is computed
    # from per-step num_decode_tokens and is bias-free.
    "tokens_per_second", "output_tokens_per_second",
    "input_tokens_per_second", "total_tokens_per_second",
    "decode_tokens_per_second", "prefill_tokens_per_second",
    "scheduled_tokens_per_second",
    "urgent_mode_fraction_pct",
    # queue occupancy (time-weighted means from scheduler_stats)
    "mean_running", "mean_pending", "mean_waiting", "mean_handling_users",
    # per-request latencies — q stall / pending / completion
    "queue_stall_p95_s", "queue_stall_p99_s", "queue_stall_max_s",
    "pending_time_p95_s", "pending_time_p99_s", "pending_time_max_s",
    "max_continuous_pending_s", "starvation_count",
    # first-token / first-chunk / per-token / completion
    "ttft_p95_s", "ttfc_p95_s", "tpot_p95_s",
    "completion_latency_p95_s", "completion_latency_p99_s",
    # per-chunk stall summary
    "max_stall_interval_s_mean", "max_stall_interval_s_p50",
    "max_stall_interval_s_p90", "max_stall_interval_s_p99",
    "total_stall_time_s_mean", "total_stall_time_s_p50",
    "total_stall_time_s_p90", "total_stall_time_s_p99",
    # violations — explicit naming (chunk = per-chunk, request = per-req)
    "chunk_slo_violation_rate_tau_0_5s",
    "chunk_slo_violation_rate_tau_1s",
    "chunk_slo_violation_rate_tau_2s",
    "request_slo_violation_rate_tau_0_5s",
    "request_slo_violation_rate_tau_1s",
    "request_slo_violation_rate_tau_2s",
    # other request-level stall
    "stall_fraction_mean", "stall_event_count_mean",
    "unit_deadline_miss_rate",
    # workload — token / unit / consume time distributions
    "input_tokens_mean", "input_tokens_p50",
    "input_tokens_p90", "input_tokens_p99",
    "output_tokens_mean", "output_tokens_p50",
    "output_tokens_p90", "output_tokens_p99",
    "units_per_request_mean", "units_per_request_p50",
    "units_per_request_p90", "units_per_request_p99",
    "consume_time_per_unit_mean", "consume_time_per_unit_p50",
    "consume_time_per_unit_p90", "consume_time_per_unit_p99",
    # SSLO: KV cache block occupancy (mean used / total capacity)
    "kv_blocks_used_mean", "kv_blocks_total",
]


def _fmt(v: Any, ndigits: int = 4) -> str:
    """Format a numeric metric for CSV. None/missing -> '' (empty cell)."""
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return ""


def _summary_row(
    summary: dict[str, Any],
    *,
    model: str, cap: int, mode: str, repeat: int, rate: float,
) -> list[str]:
    m = summary.get("metrics") or {}
    mode_metrics = m.get(mode) or {}
    sched = (m.get("scheduler") or {}).get(mode) or {}
    wl = (m.get("workload") or {}).get(mode) or {}
    tp = (m.get("throughput") or {}).get(mode) or {}
    stall_t = (m.get("stall_time") or {}).get(mode) or {}
    req_viol = (m.get("request_cu_slo_violation") or {}).get(mode) or {}
    chunk_viol = {}

    def dist_from_flat(prefix: str) -> dict[str, Any]:
        return {
            stat: mode_metrics.get(f"{prefix}_{stat}_s")
            for stat in ("mean", "p50", "p90", "p95", "p99", "max")
        }

    max_int_dist = mode_metrics.get("request_max_stall_s") or {}
    total_stall_dist = mode_metrics.get("request_total_stall_s") or {}
    stall_frac_dist = mode_metrics.get("request_stall_fraction") or {}
    num_stall_dist: dict[str, Any] = {}
    completion_dist = dist_from_flat("completion_latency")
    ttft_all = dist_from_flat("ttft")
    ttfc = dist_from_flat("ttfc")
    tpot = dist_from_flat("tpot")
    qs = dist_from_flat("queue_stall")
    pending_dist = dist_from_flat("pending_time")
    input_dist = wl.get("num_prompt_tokens") or {}
    output_dist = wl.get("num_output_tokens") or {}
    units_dist = wl.get("num_consumable_units") or {}
    consume_dist = wl.get("consume_duration") or {}
    starv_dist = wl.get("num_pending_intervals") or {}

    def viol_pct(d, tau):
        suffix = tau.removeprefix("tau_").replace("_", ".")
        key = f"tau_{suffix}"
        node = d.get(tau) or d.get(key) or {}
        rate_value = node.get("rate")
        return (float(rate_value) * 100) if rate_value is not None else None

    def fmt_viol_pct(d, tau):
        return _fmt(viol_pct(d, tau))

    def first_present(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    # unit_deadline_miss_rate = units with stall > 0 / total units >= 1.
    # If units have no field, fall back to ratio of violated/total from
    # stall_time_stats output (analyze.py line ~85).
    udmr = None
    if stall_t and stall_t.get("count"):
        violated = stall_t.get("violated_count") or 0
        total = stall_t.get("count") or 1
        udmr = (violated / total) * 100

    # max_continuous_pending_s = max request-level stall interval.
    max_cont_pending = max_int_dist.get("max")

    # starvation_count = sum of num_pending_intervals across in-window reqs.
    # Approximate via dist count × mean (count = N reqs, mean = avg events).
    starv_count = None
    if starv_dist.get("count") and starv_dist.get("mean") is not None:
        starv_count = float(starv_dist["count"]) * float(starv_dist["mean"])
    if mode_metrics.get("starvation_count") is not None:
        starv_count = mode_metrics["starvation_count"]
    corrected_tps = first_present(
        mode_metrics, "corrected_processed_tokens_per_s")
    if corrected_tps is None:
        corrected_tps = first_present(wl, "tokens_per_second")

    return [
        model, str(cap), f"{rate:g}", mode, str(repeat),
        str(summary.get("in_window_count") or 0),
        str(summary.get("injected_count") or 0),
        _fmt(corrected_tps, 2),
        _fmt(tp.get("output_tokens_per_second"), 2),
        _fmt(tp.get("input_tokens_per_second"), 2),
        _fmt(tp.get("total_tokens_per_second"), 2),
        _fmt(sched.get("decode_tokens_per_second"), 2),
        _fmt(sched.get("prefill_tokens_per_second"), 2),
        _fmt(sched.get("scheduled_tokens_per_second"), 2),
        _fmt((sched.get("urgent_mode_fraction") or 0) * 100, 4),
        _fmt(mode_metrics.get("mean_running"), 4),
        _fmt(mode_metrics.get("mean_pending"), 4),
        _fmt(mode_metrics.get("mean_waiting"), 4),
        _fmt(mode_metrics.get("mean_handling_users"), 4),
        _fmt(qs.get("p95")), _fmt(qs.get("p99")), _fmt(qs.get("max")),
        _fmt(pending_dist.get("p95")), _fmt(pending_dist.get("p99")),
        _fmt(pending_dist.get("max")),
        _fmt(max_cont_pending),
        _fmt(starv_count, 0),
        _fmt(ttft_all.get("p95")), _fmt(ttfc.get("p95")), _fmt(tpot.get("p95")),
        _fmt(completion_dist.get("p95")), _fmt(completion_dist.get("p99")),
        _fmt(max_int_dist.get("mean")), _fmt(max_int_dist.get("p50")),
        _fmt(max_int_dist.get("p90")), _fmt(max_int_dist.get("p99")),
        _fmt(total_stall_dist.get("mean")), _fmt(total_stall_dist.get("p50")),
        _fmt(total_stall_dist.get("p90")), _fmt(total_stall_dist.get("p99")),
        fmt_viol_pct(chunk_viol, "tau_0_5"),
        fmt_viol_pct(chunk_viol, "tau_1"),
        fmt_viol_pct(chunk_viol, "tau_2"),
        fmt_viol_pct(req_viol, "tau_0_5"),
        fmt_viol_pct(req_viol, "tau_1"),
        fmt_viol_pct(req_viol, "tau_2"),
        _fmt(stall_frac_dist.get("mean")), _fmt(num_stall_dist.get("mean")),
        _fmt(udmr),
        _fmt(input_dist.get("mean"), 1), _fmt(input_dist.get("p50"), 1),
        _fmt(input_dist.get("p90"), 1), _fmt(input_dist.get("p99"), 1),
        _fmt(output_dist.get("mean"), 1), _fmt(output_dist.get("p50"), 1),
        _fmt(output_dist.get("p90"), 1), _fmt(output_dist.get("p99"), 1),
        _fmt(units_dist.get("mean"), 1), _fmt(units_dist.get("p50"), 1),
        _fmt(units_dist.get("p90"), 1), _fmt(units_dist.get("p99"), 1),
        _fmt(consume_dist.get("mean")), _fmt(consume_dist.get("p50")),
        _fmt(consume_dist.get("p90")), _fmt(consume_dist.get("p99")),
        _fmt(sched.get("kv_blocks_used_mean"), 1),
        _fmt(sched.get("kv_blocks_total"), 0),
    ]


async def _analyze_and_append_summary(
    *,
    rate_dir: Path,
    summary_csv: Path,
    args: argparse.Namespace,
    rate: float,
) -> None:
    """Run analyze.py on `rate_dir`, then append one row to summary_csv
    under an fcntl lock so parallel jobs don't race on append."""
    import fcntl
    cmd = [
        "python3", "exp/run_sslo/analyze.py",
        "--output-dir", str(rate_dir),
        "--max-num-seqs", str(args.max_num_seqs),
        "--chunk-unit", args.chunk_unit,
        "--request-rate", f"{rate:g}",
        "--model", args.model,
        "--generation-max-tokens", str(args.generation_max_tokens),
        "--max-model-len", str(args.max_model_len),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        print(f"{args.run_kind}: analyze failed for rate={rate}: "
              f"rc={proc.returncode}; stderr tail: "
              f"{err.decode(errors='replace')[-400:]}")
        return
    sum_json_path = rate_dir / "summary.json"
    if not sum_json_path.exists():
        print(f"{args.run_kind}: summary.json missing at {sum_json_path}")
        return
    try:
        summary = json.loads(sum_json_path.read_text())
    except json.JSONDecodeError as e:
        print(f"{args.run_kind}: summary.json parse failed: {e}")
        return
    row = _summary_row(
        summary, model=args.model, cap=args.max_num_seqs,
        mode=args.run_kind, repeat=args.repeat, rate=rate)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    # Append under an exclusive lock. We open the file in 'a+' mode and
    # lock it, then check size: if empty, write the header first.
    with summary_csv.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)  # SEEK_END
            if f.tell() == 0:
                f.write(",".join(_SUMMARY_CSV_HEADER) + "\n")
            f.write(",".join(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    print(f"{args.run_kind}: summary row appended to {summary_csv}")


async def _run_one_rate(
    *,
    engine,
    pool: list[str],
    sampling_params,
    rate: float,
    output_dir: Path,
    args: argparse.Namespace,
    sslo_params: dict,
    request_idx_offset: int,
    shared_stats_path: Path | None,
    shared_decisions_path: Path | None,
) -> int:
    """Drive one rate's measurement on an already-loaded engine.

    Caller is responsible for ensuring SSLO state is reset BEFORE this
    call (so each rate starts cold). Writes per-rate output under
    `output_dir`. Returns the number of requests injected (for the
    caller to advance request_idx_offset).
    """
    # SSLO: same prompt order across all rates within a cell. Each rate
    # starts from prompt 0 of the SAME shuffled pool, so rate is the only
    # variable in the experiment. Seed depends ONLY on base_seed, not on
    # rate. Cursor is reset to 0 below so each rate iterates from the
    # beginning.
    base_seed = (
        args.sampling_seed if args.sampling_seed is not None
        else args.request_rate_seed)
    seed = int(base_seed)
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

    def _per_req_sampling_params(seed_value: int):
        p = sampling_params.clone()
        p.seed = seed_value
        return p

    warmup_event = asyncio.Event()
    measurement_done_event = asyncio.Event()
    warmup_counter: list[int] = [0]
    measurement_counter: list[int] = [0]
    warmup_target = (
        args.warmup_target if args.warmup_target > 0
        else args.max_num_seqs * 2)
    measurement_target = args.measurement_target
    max_total_requests = len(pool)
    # Window timestamps are written inline by the gate-flipping task in
    # collect_one (see comment there) and read here after the watcher
    # observes the events. Two clocks for downstream consumers:
    #   *_ts        — time.time()       wall clock (request side)
    #   *_mono_ts   — time.monotonic()  scheduler_stats / chunk ts side
    window_ts_holder: dict[str, float | None] = {
        "start_ts": None, "start_mono_ts": None,
        "end_ts": None, "end_mono_ts": None,
    }
    injected: list[tuple[int, asyncio.Task, float]] = []
    injection_seq = 0

    async def watcher() -> None:
        # No safety timeout — caller is responsible for picking
        # (cap, rate) cells whose warmup_target / measurement_target
        # are reachable.
        await warmup_event.wait()
        await measurement_done_event.wait()

    async def injector() -> None:
        nonlocal injection_seq
        while (injection_seq < max_total_requests
               and not measurement_done_event.is_set()):
            inj_ts = time.time()
            local_idx = injection_seq
            injection_seq += 1
            # Globally unique across rates within a single engine — vLLM
            # rejects duplicate request_ids.
            global_idx = request_idx_offset + local_idx
            prompt = next_prompt()
            task = asyncio.create_task(collect_one(
                engine, global_idx, prompt,
                _per_req_sampling_params(global_idx),
                inj_ts, warmup_event, warmup_counter, warmup_target,
                measurement_done_event, measurement_counter,
                measurement_target, window_ts_holder))
            injected.append((global_idx, task, inj_ts))
            await asyncio.sleep(rng.expovariate(rate))

    run_started_ts = time.time()
    t0 = time.monotonic()
    try:
        await asyncio.gather(watcher(), injector())
        # No cooldown — abort all still-in-flight requests at the engine
        # level (releases engine_core resources and unblocks any waiting
        # async generators), then cancel local asyncio tasks. Tolerate
        # API differences across vLLM versions via broad except.
        in_flight_ids = [
            str(idx) for idx, task, _ in injected if not task.done()
        ]
        if in_flight_ids:
            try:
                await engine.abort(in_flight_ids)
            except Exception as e:  # noqa: BLE001
                print(f"{args.run_kind}: engine.abort failed: "
                      f"{type(e).__name__}: {e}")
        for _, task, _ in injected:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(
            *[t for _, t, _ in injected], return_exceptions=True)
        rows = [r for r in results if isinstance(r, dict)]
        elapsed = time.monotonic() - t0
        run_ended_ts = time.time()

        window_start_ts = window_ts_holder["start_ts"]
        window_end_ts = window_ts_holder["end_ts"]
        window_start_mono_ts = window_ts_holder["start_mono_ts"]
        window_end_mono_ts = window_ts_holder["end_mono_ts"]

        # in_window = request COMPLETED inside [window_start, window_end].
        # Uses completion_wall_ts (time.time() at the moment the async
        # generator finished) so it shares the clock with window bounds.
        # Chunk record timestamps use time.monotonic() — DON'T compare
        # them to window bounds directly.
        for row in rows:
            comp_ts = row.get("completion_wall_ts")
            if (comp_ts is not None and window_start_ts is not None
                    and window_end_ts is not None
                    and row.get("terminal_outcome") == "completed"):
                row["in_window"] = bool(
                    window_start_ts <= comp_ts <= window_end_ts)
            else:
                row["in_window"] = False

        in_window_count = sum(1 for r in rows if r.get("in_window"))

        request_fields = (
            "request_id",
            "prompt",
            "arrival_ts",
            "consume_start_time",
            "num_consumable_units",
            "ttft",
            "ttfc",
            "tpot",
            "queue_stall_s",
            "completion_wall_ts",
            "num_output_tokens",
            "num_prompt_tokens",
            # Spec section 6 + 12 per-request CU-SLO aggregates.
            "request_max_stall_s",
            "request_total_stall_s",
            "request_demand_duration_s",
            "request_stall_fraction",
            "request_cu_slo_violated_tau_0.5",
            "request_cu_slo_violated_tau_1",
            "request_cu_slo_violated_tau_2",
            "request_cu_slo_violated_tau_5",
        )
        request_rows = [
            {field: row.get(field) for field in request_fields}
            for row in rows
        ]
        write_jsonl(output_dir / "requests.jsonl", request_rows)
        # chunks.jsonl carries only chunks from in-window (completed in
        # window) requests so downstream analysis doesn't see warmup or
        # post-window-end partial chunks.
        in_window_rids = {
            str(row["request_id"]) for row in rows if row.get("in_window")
        }
        chunk_rows = [
            {
                "request_id": str(row["request_id"]),
                **chunk,
            }
            for row in rows
            if str(row["request_id"]) in in_window_rids
            for chunk in (row.get("slo_chunk_records") or [])
        ]
        write_jsonl(output_dir / "chunks.jsonl", chunk_rows)
        # Trim scheduler_stats / decisions from the shared engine
        # output to this rate's monotonic window, writing per-rate
        # copies. vLLM stamps each step with time.monotonic(), so we
        # must use the monotonic-clock window bounds. The shared file
        # accumulates across rates (since EngineCore is a fixed-env
        # subprocess) — we slice it here, but we do NOT truncate it
        # since later rates still need to append.
        def _trim_jsonl_to_window(src: Path | None, dst: Path) -> int:
            if (src is None or not src.exists()
                    or window_start_mono_ts is None
                    or window_end_mono_ts is None):
                return 0
            n = 0
            with src.open() as fi, dst.open("w") as fo:
                for line in fi:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = r.get("ts")
                    if ts is None:
                        continue
                    if window_start_mono_ts <= ts <= window_end_mono_ts:
                        fo.write(line)
                        n += 1
            return n
        n_stats = _trim_jsonl_to_window(
            shared_stats_path, output_dir / "scheduler_stats.jsonl")
        n_dec = _trim_jsonl_to_window(
            shared_decisions_path, output_dir / "decisions.jsonl")
        print(f"{args.run_kind}: trimmed shared logs -> "
              f"scheduler_stats={n_stats}, decisions={n_dec}")
        # Sidecar for analyze.py: SSLO config options (per-mode) to embed in
        # summary.config. Only emitted for sslo* modes; baseline gets {}.
        sslo_config_path = output_dir / "sslo_config.json"
        sslo_config_path.write_text(json.dumps(sslo_params) + "\n")
        # SSLO Phase 6: run-level sidecar for the validity gate (run_id,
        # GPU peak, preemption totals, completion counts).
        write_run_meta(
            args, output_dir, run_started_ts, run_ended_ts, rows,
            rate=rate,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            window_start_mono_ts=window_start_mono_ts,
            window_end_mono_ts=window_end_mono_ts,
            pool_size=len(pool),
            sampling_seed=seed,
            pool_pass_count=pool_pass_count,
            injected_count=len(rows),
            in_window_count=in_window_count,
            warmup_target=warmup_target,
            measurement_target=measurement_target,
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
            f"{args.run_kind}: window="
            f"{window_start_ts if window_start_ts is None else f'{window_start_ts:.1f}'}.."
            f"{window_end_ts if window_end_ts is None else f'{window_end_ts:.1f}'} "
            f"injected={len(rows)} in_window={in_window_count} "
            f"warmup_completed={warmup_counter[0]} "
            f"measurement_completed={measurement_counter[0]}"
        )
        prompt_counts = [r["num_prompt_tokens"] for r in rows if r and r.get("num_prompt_tokens") is not None]
        output_counts = [r["num_output_tokens"] for r in rows if r and r.get("num_output_tokens") is not None]
        prompt_mean = f"{sum(prompt_counts)/len(prompt_counts):.1f}" if prompt_counts else "n/a"
        output_mean = f"{sum(output_counts)/len(output_counts):.1f}" if output_counts else "n/a"
        print(f"{args.run_kind}: workload mean num_prompt_tokens={prompt_mean}, num_output_tokens={output_mean}")
        # Per-rate analyze + append a summary row. Skip if --summary-csv
        # unset. Run inline (not in background) so the row lands before
        # the next rate starts and so the caller can observe success.
        if args.summary_csv:
            await _analyze_and_append_summary(
                rate_dir=output_dir,
                summary_csv=Path(args.summary_csv),
                args=args, rate=rate)
        return len(injected)
    finally:
        # Per-rate task cleanup only — engine lifecycle is managed by the
        # caller (run_one) so it can be shared across rate iterations.
        pass


def main() -> None:
    args = parse_args()
    asyncio.run(run_one(args))


if __name__ == "__main__":
    main()
