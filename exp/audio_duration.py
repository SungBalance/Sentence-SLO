#!/usr/bin/env python3
"""Generate per-chunk audio durations using Qwen3-TTS through vLLM-Omni."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.slack_utils import (
    DURATION_COLUMNS,
    append_jsonl,
    audio_duration_paths,
    build_duration_rows,
    load_cache,
    patch_qwen_tts_runtime,
    read_jsonl,
    row_key,
    split_text_for_tts,
    text_output_paths,
    write_csv,
    write_jsonl,
)

# Apply container-specific TTS import patches before vLLM-Omni is loaded.
patch_qwen_tts_runtime()

import torch
from vllm_omni import Omni


LOGGER = logging.getLogger(__name__)
DEFAULT_PLACEHOLDER_PROMPT_LEN = 2048


@dataclass(frozen=True)
class TTSRequestOptions:
    language: str
    speaker: str
    instruct: str
    max_new_tokens: int
    task_type: str = "CustomVoice"


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class PromptLengthEstimator:
    """Estimate the Stage-0 placeholder prompt length expected by vLLM-Omni."""

    def __init__(
        self,
        model_name: str,
        minimum_prompt_len: int = DEFAULT_PLACEHOLDER_PROMPT_LEN,
    ) -> None:
        self.model_name = model_name
        self.minimum_prompt_len = minimum_prompt_len
        self._tokenizer: Any | None = None
        self._talker_config: Any | None = None
        self._talker_cls: Any | None = None
        self._load_local_estimator()

    @staticmethod
    def _first(value: Any, default: Any) -> Any:
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    def _load_local_estimator(self) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
                Qwen3TTSConfig,
            )
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
            )
            config = Qwen3TTSConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            LOGGER.warning(
                "Prompt length estimator unavailable, using %d: %s",
                self.minimum_prompt_len,
                exc,
            )
            return

        self._tokenizer = tokenizer
        self._talker_config = getattr(config, "talker_config", None)
        self._talker_cls = Qwen3TTSTalkerForConditionalGeneration

    def estimate(self, additional_information: dict[str, Any]) -> int:
        # Qwen3-TTS builds real inputs from additional_information; vLLM only needs length.
        if self._tokenizer is None or self._talker_cls is None:
            return self.minimum_prompt_len

        try:
            task_type = self._first(
                additional_information.get("task_type"),
                "CustomVoice",
            )
            prompt_len = self._talker_cls.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=str(task_type),
                tokenize_prompt=self._tokenize_prompt,
                codec_language_id=getattr(
                    self._talker_config,
                    "codec_language_id",
                    None,
                ),
                spk_is_dialect=getattr(
                    self._talker_config,
                    "spk_is_dialect",
                    None,
                ),
                estimate_ref_code_len=lambda _: None,
            )
            # Longer placeholders are padded by the Talker with tts_pad_embed.
            return max(int(prompt_len), self.minimum_prompt_len)
        except Exception as exc:
            LOGGER.warning(
                "Prompt length estimation failed, using %d: %s",
                self.minimum_prompt_len,
                exc,
            )
            return self.minimum_prompt_len

    def _tokenize_prompt(self, text: str) -> list[int]:
        return self._tokenizer(text, padding=False)["input_ids"]


def build_tts_additional_information(
    text: str,
    options: TTSRequestOptions,
) -> dict[str, Any]:
    return {
        "task_type": [options.task_type],
        "text": [text],
        "language": [options.language],
        "speaker": [options.speaker],
        "instruct": [options.instruct],
        "max_new_tokens": [options.max_new_tokens],
    }


def build_omni_tts_input(
    text: str,
    options: TTSRequestOptions,
    prompt_len_estimator: PromptLengthEstimator,
) -> dict[str, Any]:
    # vLLM-Omni needs prompt_token_ids only as a length mirror of Stage-0 input.
    additional_information = build_tts_additional_information(text, options)
    return {
        "prompt_token_ids": [0] * prompt_len_estimator.estimate(additional_information),
        "additional_information": additional_information,
    }


def build_omni_tts_inputs(
    texts: list[str],
    options: TTSRequestOptions,
    prompt_len_estimator: PromptLengthEstimator,
) -> list[dict[str, Any]]:
    return [
        build_omni_tts_input(text, options, prompt_len_estimator) for text in texts
    ]


def output_duration_seconds(multimodal_output: dict[str, Any]) -> tuple[float, int]:
    audio_data = multimodal_output["audio"]
    sr_raw = multimodal_output["sr"]
    sr_value = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sample_rate = sr_value.item() if hasattr(sr_value, "item") else int(sr_value)
    audio_tensor = (
        torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    )
    return float(audio_tensor.numel() / sample_rate), int(sample_rate)


def generate_missing_audio_omni(
    *,
    rows: list[dict[str, Any]],
    duration_cache: dict[str, dict[str, Any]],
    cache_jsonl: str | Path,
    tts_model_name: str,
    stage_configs_path: str,
    batch_size: int,
    language: str,
    speaker: str,
    instruct: str,
    max_new_tokens: int,
    max_chunks: int | None,
    sort_by_text_length: bool,
    max_segment_chars: int,
    max_retries: int,
) -> dict[str, dict[str, Any]]:
    from tqdm import tqdm

    if not is_power_of_two(batch_size):
        raise ValueError("vLLM-Omni Qwen3-TTS batch_size should be a power of two.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")

    # Only synthesize chunks that are not already present in the resumable cache.
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        if row_key(row) not in duration_cache:
            missing_rows.append(row)

    if max_chunks is not None:
        missing_rows = missing_rows[:max_chunks]

    if sort_by_text_length:
        missing_rows.sort(key=lambda row: len(str(row.get("text", "")).strip()))

    if not missing_rows:
        return duration_cache

    segment_items: list[tuple[dict[str, Any], str]] = []
    row_states: dict[str, dict[str, Any]] = {}
    for row in missing_rows:
        # Long chunks are split for TTS, then merged back into one duration row.
        segments = split_text_for_tts(str(row.get("text", "")), max_segment_chars)
        key = row_key(row)
        row_states[key] = {
            "row": row,
            "remaining": len(segments),
            "duration": 0.0,
            "segment_count": len(segments),
        }
        for segment in segments:
            segment_items.append((row, segment))

    if sort_by_text_length:
        segment_items.sort(key=lambda item: len(item[1]))

    def create_omni() -> Omni:
        return Omni(
            model=tts_model_name,
            stage_configs_path=stage_configs_path,
            init_timeout=600,
            stage_init_timeout=600,
            log_stats=False,
            async_chunk=True,
        )

    progress = tqdm(total=len(segment_items), desc="Generating vLLM-Omni TTS")
    sample_rate = 24000
    started = time.perf_counter()
    tts_options = TTSRequestOptions(
        language=language,
        speaker=speaker,
        instruct=instruct,
        max_new_tokens=max_new_tokens,
    )
    prompt_len_estimator = PromptLengthEstimator(tts_model_name)
    omni = create_omni()
    try:
        pending_segments = segment_items
        for attempt in range(max_retries + 1):
            if not pending_segments:
                break
            current_batch_size = max(1, batch_size // (2**attempt))
            if attempt:
                LOGGER.warning(
                    "Retrying %d missing TTS segments with batch_size=%d "
                    "(attempt %d/%d)",
                    len(pending_segments),
                    current_batch_size,
                    attempt,
                    max_retries,
                )

            next_pending: list[tuple[dict[str, Any], str]] = []
            for start in range(0, len(pending_segments), current_batch_size):
                # Each window is submitted as a list so vLLM-Omni owns scheduling.
                window_items = pending_segments[start : start + current_batch_size]
                inputs = build_omni_tts_inputs(
                    [segment for _, segment in window_items],
                    tts_options,
                    prompt_len_estimator=prompt_len_estimator,
                )
                window_durations: dict[int, tuple[float, int]] = {}
                completed_records: list[dict[str, Any]] = []
                try:
                    for stage_outputs in omni.generate(inputs):
                        request_output = stage_outputs.request_output
                        try:
                            local_index = int(
                                str(request_output.request_id).split("_", 1)[0]
                            )
                            if local_index < 0 or local_index >= len(window_items):
                                raise IndexError(
                                    f"request index {local_index} outside window"
                                )
                            if local_index in window_durations:
                                LOGGER.warning(
                                    "Skipping duplicate TTS output for local index %d",
                                    local_index,
                                )
                                continue
                            duration, output_sample_rate = output_duration_seconds(
                                request_output.outputs[0].multimodal_output
                            )
                        except Exception as exc:
                            LOGGER.warning("Skipping malformed TTS output: %s", exc)
                            continue
                        window_durations[local_index] = (
                            duration,
                            output_sample_rate,
                        )
                except Exception as exc:
                    LOGGER.warning(
                        "TTS window failed; lowering batch size for %d pending "
                        "segments: %s",
                        len(pending_segments) - start,
                        exc,
                    )
                    next_pending.extend(window_items)
                    next_pending.extend(
                        pending_segments[start + current_batch_size :]
                    )
                    omni.close()
                    omni = create_omni()
                    break

                for local_index, (duration, output_sample_rate) in window_durations.items():
                    row, _ = window_items[local_index]
                    sample_rate = output_sample_rate
                    key = row_key(row)
                    state = row_states[key]
                    state["duration"] += duration
                    state["remaining"] -= 1
                    if state["remaining"] == 0:
                        record = {
                            "key": key,
                            "model": str(row["model"]),
                            "request_idx": int(row["request_idx"]),
                            "chunk_idx": int(row["chunk_idx"]),
                            "audio_path": "",
                            "audio_sample_rate": int(sample_rate),
                            "audio_chunk_consume": float(state["duration"]),
                            "duration_mode": "vllm_omni_wav",
                            "tts_segment_count": int(state["segment_count"]),
                            "tts_backend": "vllm_omni",
                        }
                        duration_cache[key] = record
                        completed_records.append(record)
                append_jsonl(cache_jsonl, completed_records)
                progress.update(len(window_durations))

                for local_index, segment_item in enumerate(window_items):
                    if local_index not in window_durations:
                        next_pending.append(segment_item)
            pending_segments = next_pending

        if pending_segments:
            raise RuntimeError(
                f"TTS failed to return {len(pending_segments)} segment outputs "
                f"after {max_retries + 1} attempts."
            )
    finally:
        progress.close()
        omni.close()
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Generated %d segment durations in %.2fs (%.3f segments/s)",
        len(segment_items),
        elapsed,
        len(segment_items) / elapsed if elapsed else 0.0,
    )
    return duration_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize chunk audio durations with vLLM-Omni."
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--stage-configs-path", required=True)
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--tts-max-segment-chars", type=int, default=240)
    parser.add_argument("--tts-max-retries", type=int, default=4)
    parser.add_argument("--no-sort-by-text-length", action="store_true")
    parser.add_argument("--allow-partial-output", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.input_dir:
        input_jsonl = text_output_paths(args.input_dir).chunks_jsonl
    elif args.input_jsonl:
        input_jsonl = Path(args.input_jsonl)
    else:
        raise ValueError("Provide --input-dir or --input-jsonl.")

    if args.output_dir:
        paths = audio_duration_paths(args.output_dir)
        output_jsonl = paths.durations_jsonl
        output_csv = paths.durations_csv
        cache_jsonl = paths.duration_cache_jsonl
    elif args.output_jsonl and args.output_csv and args.cache_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_csv = Path(args.output_csv)
        cache_jsonl = Path(args.cache_jsonl)
    else:
        raise ValueError(
            "Provide --output-dir or all of --output-jsonl, --output-csv, "
            "and --cache-jsonl."
        )

    return input_jsonl, output_jsonl, output_csv, cache_jsonl


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    input_jsonl, output_jsonl, output_csv, cache_jsonl = resolve_paths(args)
    rows = read_jsonl(input_jsonl)
    duration_cache = load_cache(cache_jsonl)
    duration_cache = generate_missing_audio_omni(
        rows=rows,
        duration_cache=duration_cache,
        cache_jsonl=cache_jsonl,
        tts_model_name=args.tts_model,
        stage_configs_path=args.stage_configs_path,
        batch_size=args.batch_size,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
        max_chunks=args.max_chunks,
        sort_by_text_length=not args.no_sort_by_text_length,
        max_segment_chars=args.tts_max_segment_chars,
        max_retries=args.tts_max_retries,
    )
    output_rows = build_duration_rows(
        rows,
        duration_cache,
        require_complete=not args.allow_partial_output,
    )
    # Stage 2 output is only duration metadata; analysis adds slack columns later.
    write_jsonl(output_jsonl, output_rows)
    write_csv(output_csv, output_rows, columns=DURATION_COLUMNS)
    print(
        f"Wrote {len(output_rows)} vLLM-Omni audio duration rows to "
        f"{output_jsonl} and {output_csv}"
    )


if __name__ == "__main__":
    main()
