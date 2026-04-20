# SPDX-License-Identifier: Apache-2.0
"""Slack timeline benchmark using the prebuilt vLLM package.

This script intentionally does not rely on SSLO-specific vLLM internals. It
collects sentence chunk timings from the async streaming path so it can run with
the prebuilt vLLM installed in the container.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HF_HUB_CACHE", "/cache/hub")

try:
    import uvloop
except ImportError:
    uvloop = None

from vllm.benchmarks.datasets import (
    AIMODataset,
    BurstGPTDataset,
    ConversationDataset,
    InstructCoderDataset,
    MultiModalConversationDataset,
    PrefixRepetitionRandomDataset,
    RandomDataset,
    RandomDatasetForReranking,
    RandomMultiModalDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
    add_random_dataset_base_args,
    add_random_multimodal_dataset_args,
)
from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.tokenizers import get_tokenizer
from vllm.utils.async_utils import merge_async_iterators


TimelineRecord = dict[str, Any]
SENTENCE_ENDINGS = frozenset((".", "!", "?", "。", "！", "？", "…"))


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


class StreamingChunkCollector:
    """Collect sentence chunk timings from cumulative streaming outputs."""

    def __init__(self) -> None:
        self.chunks: list[ChunkRecord] = []
        self._prev_text_by_output: dict[int, str] = {}
        self._prev_token_count_by_output: dict[int, int] = {}
        self._chunk_text_by_output: dict[int, str] = {}
        self._chunk_token_count_by_output: dict[int, int] = {}
        self._chunk_start_by_output: dict[int, float] = {}

    def on_request_output(self, request_output: RequestOutput) -> None:
        now = time.perf_counter()
        for output in request_output.outputs:
            output_index = output.index
            text = output.text or ""
            token_count = len(output.token_ids or [])
            prev_text = self._prev_text_by_output.get(output_index, "")
            prev_token_count = self._prev_token_count_by_output.get(output_index, 0)

            if text.startswith(prev_text):
                text_delta = text[len(prev_text) :]
            else:
                # DELTA-like or reset-like output. Keep the collector robust even
                # if the engine is configured differently in the future.
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

            if self._is_sentence_boundary(text):
                self._flush(output_index, now)

    def finish(self) -> None:
        now = time.perf_counter()
        for output_index in list(self._chunk_text_by_output):
            self._flush(output_index, now)

    def _flush(self, output_index: int, end_time: float) -> None:
        text = self._chunk_text_by_output.get(output_index, "")
        token_count = self._chunk_token_count_by_output.get(output_index, 0)
        if not text and token_count == 0:
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

    @staticmethod
    def _is_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in SENTENCE_ENDINGS


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_async(coro):
    if uvloop is not None:
        return uvloop.run(coro)
    return asyncio.run(coro)


def get_slack_output_jsonl_path(args: argparse.Namespace) -> str:
    if args.slack_output_jsonl:
        return args.slack_output_jsonl
    if args.output_json:
        root, _ = os.path.splitext(args.output_json)
        return f"{root}.slack.jsonl"
    return os.path.join("exp", "slack_timeline.jsonl")


def _get_metric(metrics: Any, name: str) -> float:
    value = getattr(metrics, name, None)
    if value is None:
        raise ValueError(f"RequestOutput.metrics is missing {name!r}.")
    return float(value)


def build_timeline_record(
    request_output: RequestOutput,
    chunks: list[ChunkRecord],
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
                "start_relative_to_prefill_start": chunk.start_time
                - prefill_start_ts,
                "start_relative_to_decoding_start": chunk.start_time
                - decoding_start_ts,
                "end_relative_to_prefill_start": chunk.end_time - prefill_start_ts,
                "end_relative_to_decoding_start": chunk.end_time
                - decoding_start_ts,
            }
        )

    return {
        "request_id": request_output.request_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_start_ts": prefill_start_ts,
        "decoding_start_ts": decoding_start_ts,
        "request_end_ts": request_end_ts,
        "prefill_time": decoding_start_ts - prefill_start_ts,
        "decode_time": request_end_ts - decoding_start_ts,
        "request_end_relative_to_prefill_start": request_end_ts - prefill_start_ts,
        "request_end_relative_to_decoding_start": request_end_ts
        - decoding_start_ts,
        "chunks": chunk_records,
    }


def summarize_timeline_records(records: list[TimelineRecord]) -> dict[str, Any]:
    chunks = [chunk for record in records for chunk in record["chunks"]]
    return {
        "num_timeline_requests": len(records),
        "num_timeline_chunks": len(chunks),
        "mean_prefill_time": _mean([record["prefill_time"] for record in records]),
        "mean_decode_time": _mean([record["decode_time"] for record in records]),
        "mean_chunks_per_request": (len(chunks) / len(records) if records else 0.0),
        "mean_chunk_duration": _mean([chunk["duration_seconds"] for chunk in chunks]),
        "mean_chunk_tokens": _mean([float(chunk["token_count"]) for chunk in chunks]),
        "mean_chunk_words": _mean([float(chunk["word_count"]) for chunk in chunks]),
    }


def write_timeline_jsonl(path: str, records: list[TimelineRecord]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def filter_requests_for_dp(
    requests: list[SampleRequest],
    data_parallel_size: int,
) -> list[SampleRequest]:
    if data_parallel_size == 1:
        return requests

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    data_parallel_rank = global_rank // (world_size // data_parallel_size)
    return [
        request
        for i, request in enumerate(requests)
        if i % data_parallel_size == data_parallel_rank
    ]


def get_requests(
    args: argparse.Namespace,
    tokenizer: Any,
) -> list[SampleRequest]:
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs: dict[str, Any] = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "lora_assignment": getattr(args, "lora_assignment", "random"),
        "num_requests": args.num_prompts,
    }

    if args.dataset_name == "random" or (
        args.dataset_path is None
        and args.dataset_name not in {"prefix_repetition", "random-mm", "random-rerank"}
    ):
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = (
            args.random_prefix_len
            if getattr(args, "random_prefix_len", None) is not None
            else args.prefix_len
        )
        sample_kwargs["input_len"] = (
            args.random_input_len
            if getattr(args, "random_input_len", None) is not None
            else args.input_len
        )
        sample_kwargs["output_len"] = (
            args.random_output_len
            if getattr(args, "random_output_len", None) is not None
            else args.output_len
        )
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
        if args.output_len is not None:
            sample_kwargs["output_len"] = args.output_len
    elif args.dataset_name == "sonnet":
        assert tokenizer.chat_template or tokenizer.default_chat_template, (
            "Tokenizer/model must have chat template for sonnet dataset."
        )
        dataset_cls = SonnetDataset
        sample_kwargs["prefix_len"] = args.prefix_len
        sample_kwargs["return_prompt_formatted"] = True
        if args.input_len is not None:
            sample_kwargs["input_len"] = args.input_len
        if args.output_len is not None:
            sample_kwargs["output_len"] = args.output_len
    elif args.dataset_name == "burstgpt":
        dataset_cls = BurstGPTDataset
    elif args.dataset_name == "hf":
        if args.output_len is not None:
            sample_kwargs["output_len"] = args.output_len
        common_kwargs["dataset_split"] = args.hf_split or "train"
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = VisionArenaDataset
            common_kwargs["dataset_subset"] = None
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = InstructCoderDataset
        elif args.dataset_path in MultiModalConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = MultiModalConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
            sample_kwargs["enable_multimodal_chat"] = True
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = ConversationDataset
            common_kwargs["dataset_subset"] = args.hf_subset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_cls = AIMODataset
            common_kwargs["dataset_subset"] = None
            common_kwargs["dataset_split"] = "train"
        else:
            raise ValueError(f"{args.dataset_path} is not supported by hf dataset.")
    elif args.dataset_name == "prefix_repetition":
        dataset_cls = PrefixRepetitionRandomDataset
        sample_kwargs["prefix_len"] = args.prefix_repetition_prefix_len
        sample_kwargs["suffix_len"] = args.prefix_repetition_suffix_len
        sample_kwargs["num_prefixes"] = args.prefix_repetition_num_prefixes
        sample_kwargs["output_len"] = args.prefix_repetition_output_len
    elif args.dataset_name == "random-mm":
        dataset_cls = RandomMultiModalDataset
        sample_kwargs["input_len"] = (
            args.random_input_len
            if getattr(args, "random_input_len", None) is not None
            else getattr(args, "input_len", None)
        )
        sample_kwargs["output_len"] = (
            args.random_output_len
            if getattr(args, "random_output_len", None) is not None
            else getattr(args, "output_len", None)
        )
        sample_kwargs["base_items_per_request"] = getattr(
            args, "random_mm_base_items_per_request", None
        )
        sample_kwargs["num_mm_items_range_ratio"] = getattr(
            args, "random_mm_num_mm_items_range_ratio", None
        )
        sample_kwargs["limit_mm_per_prompt"] = getattr(
            args, "random_mm_limit_mm_per_prompt", None
        )
        sample_kwargs["bucket_config"] = getattr(args, "random_mm_bucket_config", None)
        sample_kwargs["enable_multimodal_chat"] = True
        sample_kwargs["prefix_len"] = (
            args.random_prefix_len
            if getattr(args, "random_prefix_len", None) is not None
            else getattr(args, "prefix_len", None)
        )
        sample_kwargs["range_ratio"] = args.random_range_ratio
    elif args.dataset_name == "random-rerank":
        dataset_cls = RandomDatasetForReranking
        sample_kwargs["input_len"] = (
            args.random_input_len
            if getattr(args, "random_input_len", None) is not None
            else getattr(args, "input_len", None)
        )
        sample_kwargs["output_len"] = (
            args.random_output_len
            if getattr(args, "random_output_len", None) is not None
            else getattr(args, "output_len", None)
        )
        sample_kwargs["batchsize"] = getattr(args, "random_batch_size", 1)
        sample_kwargs["is_reranker"] = not getattr(args, "no_reranker", False)
        sample_kwargs["range_ratio"] = args.random_range_ratio
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    sample_kwargs = {key: value for key, value in sample_kwargs.items() if value is not None}
    requests = dataset_cls(**common_kwargs).sample(**sample_kwargs)
    return filter_requests_for_dp(requests, args.data_parallel_size)


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated. "
            "Please use '--dataset-name' and '--dataset-path' instead.",
            stacklevel=2,
        )
        args.dataset_path = args.dataset

    if not getattr(args, "tokenizer", None):
        args.tokenizer = args.model

    if args.backend != "vllm" or not args.async_engine:
        raise ValueError(
            "Slack timeline measurement supports only --backend vllm --async-engine."
        )
    if getattr(args, "disable_log_stats", False):
        raise ValueError(
            "Slack timeline measurement requires RequestOutput.metrics. "
            "Do not pass --disable-log-stats."
        )
    if args.disable_detokenize:
        warnings.warn(
            "--disable-detokenize is ignored because sentence chunking requires text.",
            stacklevel=2,
        )

    if (
        not args.dataset
        and not args.dataset_path
        and args.dataset_name not in {"prefix_repetition"}
    ):
        print("When dataset path is not set, it will default to random dataset")
        args.dataset_name = "random"
        if args.input_len is None and getattr(args, "random_input_len", None) is None:
            raise ValueError(
                "Either --input-len or --random-input-len must be provided "
                "for a random dataset."
            )

    if args.data_parallel_size > 1 and (
        args.distributed_executor_backend != "external_launcher" or args.async_engine
    ):
        raise ValueError(
            "Data parallel is only supported with external_launcher and sync engine "
            "in this offline benchmark."
        )


async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    do_profile: bool,
    apply_chat_template: bool,
    tokenizer: Any,
) -> tuple[float, list[RequestOutput], dict[str, list[ChunkRecord]]]:
    from vllm import SamplingParams
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        model_config = llm.model_config

        prompts: list[TextPrompt | TokensPrompt] = []
        sampling_params = []
        lora_requests: list[LoRARequest | None] = []
        for request in requests:
            expected_output_len = request.expected_output_len
            if apply_chat_template:
                if not isinstance(request.prompt, str):
                    raise ValueError(
                        "--apply-chat-template supports text prompts only."
                    )
                templated_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": request.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt = TextPrompt(prompt=templated_prompt)
                prompt_len = len(tokenizer(templated_prompt).input_ids)
            elif "prompt_token_ids" in request.prompt:
                prompt_token_ids = request.prompt["prompt_token_ids"]
                prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
                prompt_len = len(prompt_token_ids)
            else:
                prompt = TextPrompt(prompt=request.prompt)
                prompt_len = request.prompt_len

            if model_config.max_model_len < prompt_len + expected_output_len:
                raise ValueError(
                    "Please ensure that max_model_len is greater than "
                    "prompt_len + expected_output_len for all requests. "
                    f"Got prompt_len={prompt_len}, "
                    f"expected_output_len={expected_output_len}, "
                    f"max_model_len={model_config.max_model_len}."
                )
            if request.multi_modal_data:
                assert isinstance(request.multi_modal_data, dict)
                prompt["multi_modal_data"] = request.multi_modal_data

            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=expected_output_len,
                    detokenize=True,
                )
            )
            lora_requests.append(request.lora_request)

        generators = []
        collectors: dict[str, StreamingChunkCollector] = {}
        final_outputs_by_request_id: dict[str, RequestOutput] = {}

        start = time.perf_counter()
        if do_profile:
            await llm.start_profile()
        for i, (prompt, sampling_param, lora_request) in enumerate(
            zip(prompts, sampling_params, lora_requests)
        ):
            request_id = f"test{i}"
            collectors[request_id] = StreamingChunkCollector()
            generators.append(
                llm.generate(
                    prompt,
                    sampling_param,
                    lora_request=lora_request,
                    request_id=request_id,
                )
            )

        async for _, request_output in merge_async_iterators(*generators):
            collector = collectors[request_output.request_id]
            collector.on_request_output(request_output)
            if request_output.finished:
                collector.finish()
                final_outputs_by_request_id[request_output.request_id] = request_output

        if do_profile:
            await llm.stop_profile()
        end = time.perf_counter()

        final_outputs = [
            final_outputs_by_request_id[f"test{i}"] for i in range(len(requests))
        ]
        chunk_records = {
            request_id: collector.chunks for request_id, collector in collectors.items()
        }
        return end - start, final_outputs, chunk_records


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm"],
        default="vllm",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[
            "sharegpt",
            "random",
            "sonnet",
            "burstgpt",
            "hf",
            "prefix_repetition",
            "random-mm",
            "random-rerank",
        ],
        default="sharegpt",
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--slack-output-jsonl", type=str, default=None)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help=(
            "Wrap each sampled text prompt as a single user message using the "
            "model tokenizer's chat template. Multi-turn support is not included."
        ),
    )
    parser.add_argument("--async-engine", action="store_true", default=False)
    parser.add_argument("--disable-detokenize", action="store_true")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument(
        "--lora-assignment",
        type=str,
        default="random",
        choices=["random", "round-robin"],
    )
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--hf-subset", type=str, default=None)
    parser.add_argument("--hf-split", type=str, default=None)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--prefix-repetition-prefix-len", type=int, default=None)
    parser.add_argument("--prefix-repetition-suffix-len", type=int, default=None)
    parser.add_argument("--prefix-repetition-num-prefixes", type=int, default=None)
    parser.add_argument("--prefix-repetition-output-len", type=int, default=None)
    add_random_dataset_base_args(parser)
    add_random_multimodal_dataset_args(parser)
    AsyncEngineArgs.add_cli_args(parser)


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)

    tokenizer = get_tokenizer(
        args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    requests = get_requests(args, tokenizer)
    elapsed_time, request_outputs, chunk_records = run_async(
        run_vllm_async(
            requests,
            args.n,
            AsyncEngineArgs.from_cli_args(args),
            do_profile=args.profile,
            apply_chat_template=args.apply_chat_template,
            tokenizer=tokenizer,
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

    records = [
        build_timeline_record(
            request_output,
            chunk_records.get(request_output.request_id, []),
        )
        for request_output in request_outputs
    ]
    timeline_summary = summarize_timeline_records(records)
    slack_output_jsonl = get_slack_output_jsonl_path(args)
    write_timeline_jsonl(slack_output_jsonl, records)

    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed_time:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")
    print(
        "Slack timeline: "
        f"{timeline_summary['num_timeline_requests']} requests, "
        f"{timeline_summary['num_timeline_chunks']} chunks -> {slack_output_jsonl}"
    )

    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
            "output_tokens_per_second": total_output_tokens / elapsed_time,
            "slack_output_jsonl": slack_output_jsonl,
            "apply_chat_template": args.apply_chat_template,
            **timeline_summary,
        }
        parent = os.path.dirname(args.output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark async vLLM slack timeline with prebuilt vLLM."
    )
    add_cli_args(parser)
    main(parser.parse_args())
