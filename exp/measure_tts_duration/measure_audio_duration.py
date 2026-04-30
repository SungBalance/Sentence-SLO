#!/usr/bin/env python3
"""Measure per-chunk TTS audio duration for one model."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HF_HUB_CACHE", "/cache/hub")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DURATION_COLUMNS,
    append_jsonl,
    duration_output_paths,
    load_cache,
    read_jsonl,
    write_csv,
    write_jsonl,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_QWEN_REF_AUDIO = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
)
DEFAULT_QWEN_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TTS model on prepared text chunks and record durations."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--qwen-language", default="English")
    parser.add_argument("--qwen-ref-audio", default=DEFAULT_QWEN_REF_AUDIO)
    parser.add_argument("--qwen-ref-text", default=DEFAULT_QWEN_REF_TEXT)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=2048)
    parser.add_argument("--kokoro-lang-code", default="a")
    parser.add_argument("--kokoro-voice", default="af_heart")
    return parser.parse_args()


def model_backend(model_name: str) -> str:
    if model_name == "hexgrad/Kokoro-82M":
        return "kokoro"
    if model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base":
        return "qwen_tts"
    raise ValueError(f"Unsupported TTS model: {model_name}")


def chunk_duration_record(
    row: dict[str, Any],
    *,
    model_name: str,
    duration_seconds: float,
    audio_sample_rate: int,
    tts_segment_count: int,
    tts_backend: str,
) -> dict[str, Any]:
    return {
        "chunk_id": str(row["chunk_id"]),
        "model": model_name,
        "dataset_name": str(row["dataset_name"]),
        "dataset_item_id": str(row["dataset_item_id"]),
        "turn_idx": int(row["turn_idx"]),
        "role": str(row["role"]),
        "chunk_unit": str(row["chunk_unit"]),
        "chunk_idx": int(row["chunk_idx"]),
        "text": str(row["text"]),
        "word_count": int(row["word_count"]),
        "char_count": int(row["char_count"]),
        "duration_seconds": float(duration_seconds),
        "audio_sample_rate": int(audio_sample_rate),
        "tts_segment_count": int(tts_segment_count),
        "tts_backend": tts_backend,
    }


def synthesize_kokoro(
    *,
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_jsonl: Path,
    model_name: str,
    lang_code: str,
    voice: str,
) -> dict[str, dict[str, Any]]:
    import logging

    from kokoro import KPipeline
    from tqdm.auto import tqdm

    # Kokoro is lightweight, so we keep one pipeline alive for the full run.
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    pipeline = KPipeline(lang_code=lang_code)
    progress = tqdm(rows, desc="Generating Kokoro durations")
    for row in progress:
        chunk_id = str(row["chunk_id"])
        if chunk_id in cache:
            continue

        total_duration = 0.0
        segment_count = 0
        sample_rate = 24000
        for _, _, audio in pipeline(str(row["text"]), voice=voice):
            total_duration += float(len(audio) / sample_rate)
            segment_count += 1

        record = chunk_duration_record(
            row,
            model_name=model_name,
            duration_seconds=total_duration,
            audio_sample_rate=sample_rate,
            tts_segment_count=max(segment_count, 1),
            tts_backend="kokoro",
        )
        cache[chunk_id] = record
        append_jsonl(cache_jsonl, [record])
    return cache


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def synthesize_qwen(
    *,
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_jsonl: Path,
    model_name: str,
    batch_size: int,
    language: str,
    ref_audio: str,
    ref_text: str,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
    import torch
    from qwen_tts import Qwen3TTSModel

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    # Qwen Base needs one reusable voice-clone prompt for the whole sweep.
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    completed: list[dict[str, Any]] = []
    missing_rows = [row for row in rows if str(row["chunk_id"]) not in cache]
    for batch_rows in batched(missing_rows, batch_size):
        texts = [str(row["text"]) for row in batch_rows]
        languages = [language] * len(batch_rows)
        wavs, sample_rate = model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
            max_new_tokens=max_new_tokens,
        )
        for row, wav in zip(batch_rows, wavs):
            record = chunk_duration_record(
                row,
                model_name=model_name,
                duration_seconds=float(len(wav) / sample_rate),
                audio_sample_rate=int(sample_rate),
                tts_segment_count=1,
                tts_backend="qwen_tts",
            )
            cache[str(row["chunk_id"])] = record
            completed.append(record)

    append_jsonl(cache_jsonl, completed)
    return cache


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    input_rows = read_jsonl(Path(args.input_dir) / "chunks.jsonl")
    if args.max_rows is not None:
        input_rows = input_rows[: args.max_rows]

    paths = duration_output_paths(args.output_dir)
    cache = load_cache(paths.duration_cache_jsonl)
    backend = model_backend(args.tts_model)

    if backend == "kokoro":
        cache = synthesize_kokoro(
            rows=input_rows,
            cache=cache,
            cache_jsonl=paths.duration_cache_jsonl,
            model_name=args.tts_model,
            lang_code=args.kokoro_lang_code,
            voice=args.kokoro_voice,
        )
    elif backend == "qwen_tts":
        cache = synthesize_qwen(
            rows=input_rows,
            cache=cache,
            cache_jsonl=paths.duration_cache_jsonl,
            model_name=args.tts_model,
            batch_size=args.batch_size,
            language=args.qwen_language,
            ref_audio=args.qwen_ref_audio,
            ref_text=args.qwen_ref_text,
            max_new_tokens=args.qwen_max_new_tokens,
        )
    else:
        raise AssertionError(f"Unhandled backend: {backend}")

    output_rows = [cache[str(row["chunk_id"])] for row in input_rows if str(row["chunk_id"]) in cache]
    write_jsonl(paths.durations_jsonl, output_rows)
    write_csv(paths.durations_csv, output_rows, columns=DURATION_COLUMNS)
    LOGGER.info(
        "Wrote %d duration rows for %s under %s",
        len(output_rows),
        args.tts_model,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
