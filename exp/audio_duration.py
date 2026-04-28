#!/usr/bin/env python3
"""Generate per-chunk audio durations using Qwen3-TTS through vLLM-Omni."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.slack_utils import (
    DURATION_COLUMNS,
    append_jsonl,
    build_duration_rows,
    load_cache,
    patch_qwen_tts_runtime,
    read_jsonl,
    row_key,
    split_text_for_tts,
    write_csv,
    write_jsonl,
)

# Apply container-specific TTS import patches before vLLM-Omni is loaded.
patch_qwen_tts_runtime()

import torch
from vllm_omni import Omni


LOGGER = logging.getLogger(__name__)


def estimate_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
    cache: dict[str, Any] | None = None,
) -> int:
    # vLLM-Omni expects a placeholder prompt length for the TTS side channel.
    if cache is None:
        cache = {}
    try:
        from transformers import AutoTokenizer
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        if model_name not in cache:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left"
            )
            config = Qwen3TTSConfig.from_pretrained(
                model_name, trust_remote_code=True
            )
            cache[model_name] = (tokenizer, getattr(config, "talker_config", None))

        tokenizer, talker_config = cache[model_name]
        task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
        return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda text: tokenizer(text, padding=False)["input_ids"],
            codec_language_id=getattr(talker_config, "codec_language_id", None),
            spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
            estimate_ref_code_len=lambda _: None,
        )
    except Exception as exc:
        LOGGER.warning("Prompt length estimation failed, using 2048: %s", exc)
        return 2048


def build_omni_tts_inputs(
    *,
    texts: list[str],
    model_name: str,
    language: str,
    speaker: str,
    instruct: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    # Build one Omni input object per TTS segment.
    inputs: list[dict[str, Any]] = []
    for text in texts:
        additional_information = {
            "task_type": ["CustomVoice"],
            "text": [text],
            "language": [language],
            "speaker": [speaker],
            "instruct": [instruct],
            "max_new_tokens": [max_new_tokens],
        }
        inputs.append(
            {
                "prompt_token_ids": [0]
                * estimate_prompt_len(additional_information, model_name),
                "additional_information": additional_information,
            }
        )
    return inputs


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
    cache: dict[str, dict[str, Any]],
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
) -> dict[str, dict[str, Any]]:
    from tqdm import tqdm

    # Only synthesize chunks that are not already present in the resumable cache.
    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        if row_key(row) not in cache:
            missing_rows.append(row)

    if max_chunks is not None:
        missing_rows = missing_rows[:max_chunks]

    if sort_by_text_length:
        missing_rows.sort(key=lambda row: len(str(row.get("text", "")).strip()))

    if not missing_rows:
        return cache

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

    omni = Omni(
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
    for start in range(0, len(segment_items), batch_size):
        # Batch segment synthesis, but cache only completed chunk-level rows.
        batch_items = segment_items[start : start + batch_size]
        inputs = build_omni_tts_inputs(
            texts=[segment for _, segment in batch_items],
            model_name=tts_model_name,
            language=language,
            speaker=speaker,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )
        completed_records: list[dict[str, Any]] = []
        for stage_outputs in omni.generate(inputs):
            request_output = stage_outputs.request_output
            local_index = int(str(request_output.request_id).split("_", 1)[0])
            row, _ = batch_items[local_index]
            duration, sample_rate = output_duration_seconds(
                request_output.outputs[0].multimodal_output
            )
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
                cache[key] = record
                completed_records.append(record)
        append_jsonl(cache_jsonl, completed_records)
        progress.update(len(batch_items))
    progress.close()
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Generated %d segment durations in %.2fs (%.3f segments/s)",
        len(segment_items),
        elapsed,
        len(segment_items) / elapsed if elapsed else 0.0,
    )
    omni.close()
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize chunk audio durations with vLLM-Omni."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--cache-jsonl", required=True)
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
    parser.add_argument("--no-sort-by-text-length", action="store_true")
    parser.add_argument("--allow-partial-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    cache = load_cache(args.cache_jsonl)
    cache = generate_missing_audio_omni(
        rows=rows,
        cache=cache,
        cache_jsonl=args.cache_jsonl,
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
    )
    output_rows = build_duration_rows(
        rows,
        cache,
        require_complete=not args.allow_partial_output,
    )
    # Stage 2 output is only duration metadata; analysis adds slack columns later.
    write_jsonl(args.output_jsonl, output_rows)
    write_csv(args.output_csv, output_rows, columns=DURATION_COLUMNS)
    print(
        f"Wrote {len(output_rows)} vLLM-Omni audio duration rows to "
        f"{args.output_jsonl} and {args.output_csv}"
    )


if __name__ == "__main__":
    main()
