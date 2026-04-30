# SPDX-License-Identifier: Apache-2.0
"""Stage 1: run vLLM inference and record request/chunk timelines."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HF_HUB_CACHE", "/cache/hub")

try:
    import uvloop
except ImportError:
    uvloop = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.utils.async_utils import merge_async_iterators

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.slack_utils import (
    SLACK_COLUMNS,
    build_slack_rows,
    mean_or_zero,
    text_output_paths,
    write_csv,
    write_json,
    write_jsonl,
)
from eval_datasets import (
    KOALA_DATASET_ID,
    EvalDatasetItem,
    load_eval_dataset,
    normalize_dataset_name,
)


TimelineRecord = dict[str, Any]
SENTENCE_ENDINGS = frozenset((".", "!", "?", "。", "！", "？", "…"))
PARAGRAPH_BOUNDARY_RE = re.compile(r"(?:\r?\n[ \t]*){2,}$")
FIXED_BACKEND = "vllm"
FIXED_ASYNC_ENGINE = True
FIXED_APPLY_CHAT_TEMPLATE = True


@dataclass
class ChunkRecord:
    output_index: int
    chunk_index: int
    text: str
    token_count: int
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass(frozen=True)
class BenchmarkOutputGroup:
    chunk_unit: str
    summary_json: Path | None
    requests_jsonl: Path
    chunks_jsonl: Path | None
    chunks_csv: Path | None


class StreamingChunkCollector:
    """Collect chunk timings from standard streamed RequestOutputs."""

    def __init__(self, chunk_unit: str) -> None:
        self.chunk_unit = chunk_unit
        self.chunks: list[ChunkRecord] = []
        self._prev_text_by_output: dict[int, str] = {}
        self._prev_token_count_by_output: dict[int, int] = {}
        self._chunk_text_by_output: dict[int, str] = {}
        self._chunk_token_count_by_output: dict[int, int] = {}
        self._chunk_start_by_output: dict[int, float] = {}

    def on_request_output(self, request_output: RequestOutput) -> None:
        now = time.monotonic()
        for output in request_output.outputs:
            output_index = output.index
            text = output.text or ""
            token_count = len(output.token_ids or [])
            prev_text = self._prev_text_by_output.get(output_index, "")
            prev_token_count = self._prev_token_count_by_output.get(output_index, 0)

            if text.startswith(prev_text):
                text_delta = text[len(prev_text) :]
            else:
                # Keep the collector robust if a future engine returns deltas.
                text_delta = text

            token_delta = max(0, token_count - prev_token_count)
            if not text_delta and token_delta == 0:
                continue

            self._chunk_start_by_output.setdefault(output_index, now)
            self._chunk_text_by_output[output_index] = (
                self._chunk_text_by_output.get(output_index, "") + text_delta
            )
            self._chunk_token_count_by_output[output_index] = (
                self._chunk_token_count_by_output.get(output_index, 0) + token_delta
            )
            self._prev_text_by_output[output_index] = text
            self._prev_token_count_by_output[output_index] = token_count

            if self._should_flush(text_delta, self._chunk_text_by_output[output_index]):
                self._flush(output_index, now)

    def finish(self) -> None:
        now = time.monotonic()
        for output_index in list(self._chunk_text_by_output):
            self._flush(output_index, now)

    def _flush(self, output_index: int, end_time: float) -> None:
        text = self._chunk_text_by_output.get(output_index, "")
        token_count = self._chunk_token_count_by_output.get(output_index, 0)
        if not text.strip():
            self._chunk_text_by_output.pop(output_index, None)
            self._chunk_token_count_by_output.pop(output_index, None)
            self._chunk_start_by_output.pop(output_index, None)
            return

        start_time = self._chunk_start_by_output.get(output_index, end_time)
        self.chunks.append(
            ChunkRecord(
                output_index=output_index,
                chunk_index=sum(
                    1 for chunk in self.chunks if chunk.output_index == output_index
                ),
                text=text,
                token_count=token_count,
                start_time=start_time,
                end_time=max(end_time, start_time),
            )
        )
        self._chunk_text_by_output.pop(output_index, None)
        self._chunk_token_count_by_output.pop(output_index, None)
        self._chunk_start_by_output.pop(output_index, None)

    def _is_chunk_boundary(self, text: str) -> bool:
        if self.chunk_unit == "sentence":
            return self._is_sentence_boundary(text)
        if self.chunk_unit == "paragraph":
            return self._is_paragraph_boundary(text)
        raise ValueError(f"Unknown chunk unit: {self.chunk_unit}")

    def _should_flush(self, text_delta: str, current_chunk_text: str) -> bool:
        if self.chunk_unit == "sentence" and not text_delta.strip():
            return False
        return self._is_chunk_boundary(current_chunk_text)

    @staticmethod
    def _is_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in SENTENCE_ENDINGS

    @staticmethod
    def _is_paragraph_boundary(text: str) -> bool:
        return bool(PARAGRAPH_BOUNDARY_RE.search(text))


def run_async(coro):
    if uvloop is not None:
        return uvloop.run(coro)
    return asyncio.run(coro)


def request_arrival_delays(
    *,
    num_requests: int,
    request_rate: float,
    burstiness: float,
) -> list[float]:
    if request_rate == float("inf"):
        return [0.0] * num_requests

    delays = [0.0]
    current_delay = 0.0
    for _ in range(1, num_requests):
        # burstiness=1 is Poisson/exponential arrival; other values use gamma.
        interval = random.gammavariate(
            alpha=burstiness,
            beta=1.0 / (request_rate * burstiness),
        )
        current_delay += interval
        delays.append(current_delay)
    return delays


def request_rate_for_json(request_rate: float) -> float | str:
    return "inf" if request_rate == float("inf") else request_rate


def get_slack_output_jsonl_path(args: argparse.Namespace) -> str:
    if args.slack_output_jsonl:
        return args.slack_output_jsonl
    if args.output_json:
        root, _ = os.path.splitext(args.output_json)
        return f"{root}.slack.jsonl"
    return os.path.join("exp", "slack_timeline.jsonl")


def apply_output_dir(args: argparse.Namespace) -> None:
    args.uses_canonical_output_dir = bool(args.output_dir)
    if args.chunk_output_group or not args.output_dir:
        return

    # Stage 1 owns the canonical names in text_outputs/.
    paths = text_output_paths(args.output_dir)
    args.output_json = str(paths.summary_json)
    args.slack_output_jsonl = str(paths.requests_jsonl)
    args.chunk_output_jsonl = str(paths.chunks_jsonl)
    args.chunk_output_csv = str(paths.chunks_csv)


def apply_fixed_benchmark_options(args: argparse.Namespace) -> None:
    # These are fixed for this experiment pipeline, not launcher knobs.
    args.backend = FIXED_BACKEND
    args.async_engine = FIXED_ASYNC_ENGINE
    args.apply_chat_template = FIXED_APPLY_CHAT_TEMPLATE


def parse_chunk_output_groups(
    args: argparse.Namespace,
) -> dict[str, BenchmarkOutputGroup]:
    if args.chunk_output_group:
        if any(
            [
                args.output_dir,
                args.output_json,
                args.slack_output_jsonl,
                args.chunk_output_jsonl,
                args.chunk_output_csv,
            ]
        ):
            raise ValueError(
                "--chunk-output-group cannot be combined with single-output paths."
            )

        groups: dict[str, BenchmarkOutputGroup] = {}
        for spec in args.chunk_output_group:
            chunk_unit, separator, output_dir = spec.partition("=")
            if not separator or not chunk_unit or not output_dir:
                raise ValueError(
                    "--chunk-output-group must use UNIT=OUTPUT_DIR, "
                    "for example sentence=/tmp/text_outputs."
                )
            if chunk_unit not in {"sentence", "paragraph"}:
                raise ValueError(
                    "--chunk-output-group UNIT must be sentence or paragraph."
                )
            if chunk_unit in groups:
                raise ValueError(f"Duplicate chunk output group: {chunk_unit}")

            paths = text_output_paths(output_dir)
            groups[chunk_unit] = BenchmarkOutputGroup(
                chunk_unit=chunk_unit,
                summary_json=paths.summary_json,
                requests_jsonl=paths.requests_jsonl,
                chunks_jsonl=paths.chunks_jsonl,
                chunks_csv=paths.chunks_csv,
            )
        return groups

    return {
        args.chunk_unit: BenchmarkOutputGroup(
            chunk_unit=args.chunk_unit,
            summary_json=Path(args.output_json) if args.output_json else None,
            requests_jsonl=Path(get_slack_output_jsonl_path(args)),
            chunks_jsonl=Path(args.chunk_output_jsonl)
            if args.chunk_output_jsonl
            else None,
            chunks_csv=Path(args.chunk_output_csv) if args.chunk_output_csv else None,
        )
    }


def _get_metric(metrics: Any, name: str) -> float:
    value = getattr(metrics, name, None)
    if value is None:
        raise ValueError(f"RequestOutput.metrics is missing {name!r}.")
    return float(value)


def limit_chunks_per_request(
    chunks: list[ChunkRecord],
    max_chunks_per_request: int | None,
) -> list[ChunkRecord]:
    # Optional guard for downstream analysis size.
    if max_chunks_per_request is None:
        return chunks
    return chunks[:max_chunks_per_request]


def build_timeline_record(
    request_output: RequestOutput,
    eval_item: EvalDatasetItem,
    chunks: list[ChunkRecord],
    *,
    request_submit_ts: float,
    max_chunks_per_request: int | None,
    chunk_unit: str,
) -> TimelineRecord:
    metrics = request_output.metrics
    if metrics is None:
        raise ValueError(
            f"Request {request_output.request_id!r} has no metrics. "
            "Slack timeline measurement requires log stats to be enabled."
        )

    prefill_start_ts = _get_metric(metrics, "scheduled_ts")
    decoding_start_ts = _get_metric(metrics, "first_token_ts")
    request_end_ts = _get_metric(metrics, "last_token_ts")
    if prefill_start_ts == 0.0 or decoding_start_ts == 0.0 or request_end_ts == 0.0:
        raise ValueError(
            f"Request {request_output.request_id!r} has incomplete metrics: "
            f"scheduled_ts={prefill_start_ts}, "
            f"first_token_ts={decoding_start_ts}, "
            f"last_token_ts={request_end_ts}."
        )

    prompt_tokens = (
        len(request_output.prompt_token_ids) if request_output.prompt_token_ids else 0
    )
    output_tokens = sum(len(output.token_ids or []) for output in request_output.outputs)
    original_chunk_count = len(chunks)
    chunks = limit_chunks_per_request(chunks, max_chunks_per_request)
    chunk_records = []
    for chunk in chunks:
        chunk_records.append(
            {
                "output_index": chunk.output_index,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "word_count": len(chunk.text.split()),
                "start_time_ts": chunk.start_time,
                "end_time_ts": chunk.end_time,
                "duration_seconds": chunk.duration_seconds,
                "start_relative_to_request_submit": chunk.start_time
                - request_submit_ts,
                "start_relative_to_prefill_start": chunk.start_time
                - prefill_start_ts,
                "start_relative_to_decoding_start": chunk.start_time
                - decoding_start_ts,
                "end_relative_to_request_submit": chunk.end_time
                - request_submit_ts,
                "end_relative_to_prefill_start": chunk.end_time - prefill_start_ts,
                "end_relative_to_decoding_start": chunk.end_time
                - decoding_start_ts,
            }
        )

    return {
        "request_id": request_output.request_id,
        "dataset_item_id": eval_item.item_id,
        "chunk_unit": chunk_unit,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "request_submit_ts": request_submit_ts,
        "prefill_start_ts": prefill_start_ts,
        "decoding_start_ts": decoding_start_ts,
        "request_end_ts": request_end_ts,
        "scheduled_relative_to_request_submit": prefill_start_ts
        - request_submit_ts,
        "first_token_relative_to_request_submit": decoding_start_ts
        - request_submit_ts,
        "request_end_relative_to_request_submit": request_end_ts
        - request_submit_ts,
        "prefill_time": decoding_start_ts - prefill_start_ts,
        "decode_time": request_end_ts - decoding_start_ts,
        "request_end_relative_to_prefill_start": request_end_ts - prefill_start_ts,
        "request_end_relative_to_decoding_start": request_end_ts
        - decoding_start_ts,
        "num_chunks_before_cap": original_chunk_count,
        "max_chunks_per_request": max_chunks_per_request,
        "chunks_truncated": len(chunks) < original_chunk_count,
        "chunks": chunk_records,
    }


def summarize_timeline_records(records: list[TimelineRecord]) -> dict[str, Any]:
    chunks = [chunk for record in records for chunk in record["chunks"]]
    return {
        "num_timeline_requests": len(records),
        "num_timeline_chunks": len(chunks),
        "num_requests_with_chunk_cap": sum(
            1 for record in records if record["chunks_truncated"]
        ),
        "mean_prefill_time": mean_or_zero(
            [record["prefill_time"] for record in records]
        ),
        "mean_decode_time": mean_or_zero([record["decode_time"] for record in records]),
        "mean_chunks_per_request": (len(chunks) / len(records) if records else 0.0),
        "mean_chunk_duration": mean_or_zero(
            [chunk["duration_seconds"] for chunk in chunks]
        ),
        "mean_chunk_tokens": mean_or_zero(
            [float(chunk["token_count"]) for chunk in chunks]
        ),
        "mean_chunk_words": mean_or_zero(
            [float(chunk["word_count"]) for chunk in chunks]
        ),
    }


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k] for k in ["elapsed_time", "num_requests", "total_num_tokens"]
        },
    )
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def filter_items_for_dp(
    items: list[EvalDatasetItem],
    data_parallel_size: int,
) -> list[EvalDatasetItem]:
    if data_parallel_size == 1:
        return items

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    data_parallel_rank = global_rank // (world_size // data_parallel_size)
    return [
        item
        for i, item in enumerate(items)
        if i % data_parallel_size == data_parallel_rank
    ]


def get_eval_items(
    args: argparse.Namespace,
) -> list[EvalDatasetItem]:
    items = load_eval_dataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        num_prompts=args.num_prompts,
    )
    return filter_items_for_dp(items, args.data_parallel_size)


def validate_args(args: argparse.Namespace) -> None:
    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model
    normalize_dataset_name(args.dataset_name)

    if getattr(args, "disable_log_stats", False):
        raise ValueError(
            "Slack timeline measurement requires RequestOutput.metrics. "
            "Do not pass --disable-log-stats."
        )
    if args.disable_detokenize:
        warnings.warn(
            "--disable-detokenize is ignored because chunking requires text.",
            stacklevel=2,
        )
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative.")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be positive or inf.")
    if args.request_burstiness <= 0:
        raise ValueError("--request-burstiness must be positive.")
    if args.max_chunks_per_request is not None and args.max_chunks_per_request <= 0:
        raise ValueError("--max-chunks-per-request must be positive when set.")

    if args.data_parallel_size > 1 and (
        args.distributed_executor_backend != "external_launcher" or args.async_engine
    ):
        raise ValueError(
            "Data parallel is only supported with external_launcher and sync engine "
            "in this offline benchmark."
        )


async def run_vllm_async(
    eval_items: list[EvalDatasetItem],
    n: int,
    engine_args: AsyncEngineArgs,
    do_profile: bool,
    tokenizer: Any,
    generation_max_tokens: int | None,
    request_rate: float,
    request_burstiness: float,
    warmup_requests: int,
    chunk_units: list[str],
) -> tuple[
    float,
    float,
    list[RequestOutput],
    dict[str, dict[str, list[ChunkRecord]]],
    dict[str, float],
]:
    from vllm import SamplingParams
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        model_config = llm.model_config
        default_sampling_kwargs = dict(model_config.get_diff_sampling_param())

        prompts: list[TextPrompt] = []
        sampling_params = []
        for item in eval_items:
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = TextPrompt(prompt=templated_prompt)
            prompt_len = len(tokenizer(templated_prompt).input_ids)

            max_tokens = model_config.max_model_len - prompt_len
            if generation_max_tokens is not None:
                max_tokens = min(generation_max_tokens, max_tokens)
            if max_tokens <= 0:
                raise ValueError(
                    "Please ensure that max_model_len is greater than prompt_len "
                    "for all requests. "
                    f"Got prompt_len={prompt_len}, "
                    f"max_model_len={model_config.max_model_len}."
                )

            prompts.append(prompt)
            sampling_kwargs = dict(default_sampling_kwargs)
            sampling_kwargs["max_tokens"] = max_tokens
            sampling_params.append(
                SamplingParams.from_optional(
                    **sampling_kwargs,
                    n=n,
                    detokenize=True,
                )
            )

        warmup_elapsed = 0.0
        if warmup_requests:
            # Warmup requests are fully consumed but excluded from measured outputs.
            warmup_started = time.perf_counter()
            warmup_bar = (
                tqdm(
                    total=warmup_requests,
                    desc="Warmup requests",
                    unit="req",
                    dynamic_ncols=True,
                )
                if tqdm is not None
                else nullcontext()
            )
            with warmup_bar as progress:
                for i in range(warmup_requests):
                    prompt = prompts[i % len(prompts)]
                    sampling_param = sampling_params[i % len(sampling_params)]
                    async for _ in llm.generate(
                        prompt,
                        sampling_param,
                        request_id=f"warmup{i}",
                    ):
                        pass
                    if progress is not None:
                        progress.update(1)
            warmup_elapsed = time.perf_counter() - warmup_started

        generators = []
        collectors: dict[str, dict[str, StreamingChunkCollector]] = {}
        final_outputs_by_request_id: dict[str, RequestOutput] = {}
        request_submit_ts_by_request_id: dict[str, float] = {}
        arrival_delays = request_arrival_delays(
            num_requests=len(prompts),
            request_rate=request_rate,
            burstiness=request_burstiness,
        )

        async def delayed_generate(
            *,
            delay: float,
            prompt: TextPrompt,
            sampling_param: SamplingParams,
            request_id: str,
        ):
            if delay > 0:
                await asyncio.sleep(delay)
            # Submit time marks when the benchmark hands the request to vLLM.
            request_submit_ts_by_request_id[request_id] = time.monotonic()
            async for request_output in llm.generate(
                prompt,
                sampling_param,
                request_id=request_id,
            ):
                yield request_output

        # Launch requests with optional arrival pacing and collect stream chunks.
        start = time.perf_counter()
        if do_profile:
            await llm.start_profile()
        for i, (prompt, sampling_param, delay) in enumerate(
            zip(prompts, sampling_params, arrival_delays)
        ):
            request_id = f"test{i}"
            collectors[request_id] = {
                chunk_unit: StreamingChunkCollector(chunk_unit)
                for chunk_unit in chunk_units
            }
            generators.append(
                delayed_generate(
                    delay=delay,
                    prompt=prompt,
                    sampling_param=sampling_param,
                    request_id=request_id,
                )
            )

        completed_request_ids: set[str] = set()
        measured_bar = (
            tqdm(
                total=len(prompts),
                desc="Measured requests",
                unit="req",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else nullcontext()
        )
        with measured_bar as progress:
            async for _, request_output in merge_async_iterators(*generators):
                request_collectors = collectors[request_output.request_id]
                for collector in request_collectors.values():
                    collector.on_request_output(request_output)
                if request_output.finished:
                    for collector in request_collectors.values():
                        collector.finish()
                    final_outputs_by_request_id[request_output.request_id] = (
                        request_output
                    )
                    if request_output.request_id not in completed_request_ids:
                        completed_request_ids.add(request_output.request_id)
                        if progress is not None:
                            output_tokens = sum(
                                len(output.token_ids or [])
                                for output in request_output.outputs
                            )
                            progress.update(1)
                            progress.set_postfix(
                                {
                                    "last": request_output.request_id,
                                    "tokens": output_tokens,
                                    "chunks": ",".join(
                                        f"{unit}:{len(collector.chunks)}"
                                        for unit, collector in (
                                            request_collectors.items()
                                        )
                                    ),
                                },
                                refresh=False,
                            )

        if do_profile:
            await llm.stop_profile()
        end = time.perf_counter()

        final_outputs = [
            final_outputs_by_request_id[f"test{i}"] for i in range(len(eval_items))
        ]
        chunk_records_by_unit = {
            chunk_unit: {
                request_id: request_collectors[chunk_unit].chunks
                for request_id, request_collectors in collectors.items()
            }
            for chunk_unit in chunk_units
        }
        return (
            end - start,
            warmup_elapsed,
            final_outputs,
            chunk_records_by_unit,
            request_submit_ts_by_request_id,
        )


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=KOALA_DATASET_ID,
        help="Evaluation dataset name. Currently supports HuggingFaceH4/Koala-test-set.",
    )
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help=(
            "Number of prompts to use. Values larger than the dataset are "
            "clamped to the dataset size."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--slack-output-jsonl", type=str, default=None)
    parser.add_argument("--chunk-output-jsonl", type=str, default=None)
    parser.add_argument("--chunk-output-csv", type=str, default=None)
    parser.add_argument("--seconds-per-word", type=float, default=0.28)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of requests to run before measured inference.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request arrival rate in requests/sec. Use inf to launch all at once.",
    )
    parser.add_argument(
        "--request-burstiness",
        type=float,
        default=1.0,
        help="Gamma arrival burstiness. 1.0 is Poisson/exponential arrival.",
    )
    parser.add_argument(
        "--generation-max-tokens",
        type=int,
        default=None,
        help=(
            "Optional safety cap for generation length. When omitted, each "
            "request can use the remaining model context and stops by the "
            "model/vLLM default stop conditions such as EOS."
        ),
    )
    parser.add_argument(
        "--max-chunks-per-request",
        type=int,
        default=None,
        help="Maximum number of chunks to keep for each measured request.",
    )
    parser.add_argument(
        "--chunk-unit",
        choices=["sentence", "paragraph"],
        default="sentence",
        help="Boundary used when grouping streamed output into chunks.",
    )
    parser.add_argument(
        "--chunk-output-group",
        action="append",
        default=[],
        metavar="UNIT=OUTPUT_DIR",
        help=(
            "Write an additional chunk unit output group from the same run. "
            "Can be repeated, for example sentence=/out/sentence and "
            "paragraph=/out/paragraph."
        ),
    )
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument("--profile", action="store_true", default=False)
    AsyncEngineArgs.add_cli_args(parser)


def main(args: argparse.Namespace) -> None:
    apply_output_dir(args)
    apply_fixed_benchmark_options(args)
    validate_args(args)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)
    output_groups = parse_chunk_output_groups(args)
    chunk_units = list(output_groups)

    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    eval_items = get_eval_items(args)
    # Stage 1: run LLM inference and keep per-request streaming timelines.
    (
        elapsed_time,
        warmup_elapsed_time,
        request_outputs,
        chunk_records_by_unit,
        request_submit_ts_by_request_id,
    ) = run_async(
        run_vllm_async(
            eval_items,
            args.n,
            AsyncEngineArgs.from_cli_args(args),
            do_profile=args.profile,
            tokenizer=tokenizer,
            generation_max_tokens=args.generation_max_tokens,
            request_rate=args.request_rate,
            request_burstiness=args.request_burstiness,
            warmup_requests=args.warmup_requests,
            chunk_units=chunk_units,
        )
    )

    total_prompt_tokens = 0
    total_output_tokens = 0
    for request_output in request_outputs:
        total_prompt_tokens += (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids
            else 0
        )
        total_output_tokens += sum(
            len(output.token_ids or []) for output in request_output.outputs
        )
    total_num_tokens = total_prompt_tokens + total_output_tokens

    print(
        f"Throughput: {len(eval_items) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(
        f"Warmup: {args.warmup_requests} requests in "
        f"{warmup_elapsed_time:.2f}s"
    )
    print(
        f"Request rate: {args.request_rate} requests/s, "
        f"burstiness={args.request_burstiness}"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")

    base_results = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "dataset_id": normalize_dataset_name(args.dataset_name),
        "dataset_split": args.dataset_split,
        "elapsed_time": elapsed_time,
        "warmup_elapsed_time": warmup_elapsed_time,
        "warmup_requests": args.warmup_requests,
        "num_requests": len(eval_items),
        "total_num_tokens": total_num_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "requests_per_second": len(eval_items) / elapsed_time,
        "tokens_per_second": total_num_tokens / elapsed_time,
        "output_tokens_per_second": total_output_tokens / elapsed_time,
        "request_rate": request_rate_for_json(args.request_rate),
        "request_burstiness": args.request_burstiness,
        "request_timing_source": "vllm_request_output_metrics",
        "chunk_timing_source": "benchmark_stream_receive",
        "apply_chat_template": FIXED_APPLY_CHAT_TEMPLATE,
        "backend": FIXED_BACKEND,
        "async_engine": FIXED_ASYNC_ENGINE,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "generation_max_tokens": args.generation_max_tokens,
        "max_chunks_per_request": args.max_chunks_per_request,
    }

    for chunk_unit, output_group in output_groups.items():
        chunk_records = chunk_records_by_unit[chunk_unit]
        records = [
            build_timeline_record(
                request_output,
                eval_item,
                chunk_records.get(request_output.request_id, []),
                request_submit_ts=request_submit_ts_by_request_id[
                    request_output.request_id
                ],
                max_chunks_per_request=args.max_chunks_per_request,
                chunk_unit=chunk_unit,
            )
            for request_output, eval_item in zip(request_outputs, eval_items)
        ]
        timeline_summary = summarize_timeline_records(records)
        output_group.requests_jsonl.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_group.requests_jsonl, records)

        # Convert request timelines into per-chunk human reading slack rows.
        chunk_rows = build_slack_rows(
            records,
            model=args.model,
            seconds_per_word=args.seconds_per_word,
        )
        if output_group.chunks_jsonl:
            write_jsonl(output_group.chunks_jsonl, chunk_rows)
        if output_group.chunks_csv:
            write_csv(output_group.chunks_csv, chunk_rows, columns=SLACK_COLUMNS)

        print(
            f"[{chunk_unit}] Slack timeline: "
            f"{timeline_summary['num_timeline_requests']} requests, "
            f"{timeline_summary['num_timeline_chunks']} chunks -> "
            f"{output_group.requests_jsonl}"
        )
        if output_group.chunks_jsonl or output_group.chunks_csv:
            print(
                f"[{chunk_unit}] Human slack chunks: {len(chunk_rows)} rows -> "
                f"{output_group.chunks_jsonl or output_group.chunks_csv}"
            )

        if output_group.summary_json:
            results = {
                **base_results,
                "slack_output_jsonl": str(output_group.requests_jsonl),
                "chunk_output_jsonl": str(output_group.chunks_jsonl)
                if output_group.chunks_jsonl
                else None,
                "chunk_output_csv": str(output_group.chunks_csv)
                if output_group.chunks_csv
                else None,
                "chunk_unit": chunk_unit,
                **timeline_summary,
            }
            write_json(output_group.summary_json, results)
            if not args.uses_canonical_output_dir and len(output_groups) == 1:
                save_to_pytorch_benchmark_format(args, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark async vLLM slack timeline with streamed chunks."
    )
    add_cli_args(parser)
    main(parser.parse_args())
