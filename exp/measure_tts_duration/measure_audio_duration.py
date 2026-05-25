#!/usr/bin/env python3
"""Profile per-chunk TTS conversion_time, audio_duration, and RTF.

Input  : an SSLO-style chunks file (default `input_chunks.jsonl` in this
         folder, mapped via load_sslo_chunks.transform_row).
Output : <output_dir>/durations.{jsonl,csv} — one row per chunk with
         conversion_time_s, audio_duration_s, real_time_factor.

Backends:
  - hexgrad/Kokoro-82M    (PyTorch, GPU)
  - Supertone/supertonic-3 (ONNX runtime, CPU; default voice "M1", lang "en")

Resumable: a per-output `duration_cache.jsonl` lets you re-run after
a crash without re-synthesising completed chunks.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
    write_csv,
    write_jsonl,
)
from load_sslo_chunks import load as load_input_chunks

LOGGER = logging.getLogger(__name__)


# SSLO: Supertonic-3 ONNX TTS. `auto_download=True` fetches the model
# files into HF_HUB_CACHE on first run.
SUPERTONIC_MODEL_ID = "Supertone/supertonic-3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl", default="input_chunks.jsonl",
        help="run_sslo chunks.jsonl copied into this folder.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tts-model", required=True,
        choices=["hexgrad/Kokoro-82M", "Supertone/supertonic-3"],
        help="Backend selection.")
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap on number of chunks (for smoke tests).")
    parser.add_argument("--kokoro-lang-code", default="a")
    parser.add_argument("--kokoro-voice", default="af_heart")
    # SSLO
    parser.add_argument("--supertonic-voice", default="M1")
    # SSLO
    parser.add_argument("--supertonic-lang", default="en")
    # SSLO: cap ONNX intra_op threads (None = ONNX default which spawns
    # ~one thread per core). Critical when running multiple Supertonic
    # processes in parallel — without a cap, N processes × cores threads
    # oversubscribe the CPU and throughput collapses.
    parser.add_argument(
        "--supertonic-intra-op-num-threads", type=int, default=None)
    # SSLO: shard for multi-process parallelism. Each process picks rows
    # where (row_index % shard_count == shard_id). Output dirs must be
    # per-shard so caches / cache_jsonl don't collide.
    parser.add_argument("--shard-id", type=int, default=0)
    # SSLO
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def model_backend(model_name: str) -> str:
    if model_name == "hexgrad/Kokoro-82M":
        return "kokoro"
    # SSLO
    if model_name == SUPERTONIC_MODEL_ID:
        return "supertonic"
    raise ValueError(f"Unsupported TTS model: {model_name}")


def make_record(
    row: dict[str, Any],
    *,
    model_name: str,
    tts_backend: str,
    conversion_time_s: float,
    audio_duration_s: float,
    audio_sample_rate: int,
    tts_segment_count: int,
) -> dict[str, Any]:
    rtf = (audio_duration_s / conversion_time_s
            if conversion_time_s > 0 else 0.0)
    return {
        "chunk_id": row["chunk_id"],
        "request_id": row["request_id"],
        "chunk_idx": int(row["chunk_idx"]),
        "word_count": int(row["word_count"]),
        "num_token": int(row["num_token"]),
        "model": model_name,
        "tts_backend": tts_backend,
        "conversion_time_s": float(conversion_time_s),
        "audio_duration_s": float(audio_duration_s),
        "real_time_factor": float(rtf),
        "audio_sample_rate": int(audio_sample_rate),
        "tts_segment_count": int(tts_segment_count),
        "text": row["text"],
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
    from kokoro import KPipeline
    from tqdm.auto import tqdm

    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    pipeline = KPipeline(lang_code=lang_code)
    sample_rate = 24000  # Kokoro hardcoded

    for row in tqdm(rows, desc="Kokoro"):
        chunk_id = row["chunk_id"]
        if chunk_id in cache:
            continue

        t0 = time.monotonic()
        total_samples = 0
        segment_count = 0
        for _, _, audio in pipeline(row["text"], voice=voice):
            total_samples += len(audio)
            segment_count += 1
        conversion_time_s = time.monotonic() - t0
        audio_duration_s = total_samples / sample_rate

        record = make_record(
            row, model_name=model_name, tts_backend="kokoro",
            conversion_time_s=conversion_time_s,
            audio_duration_s=audio_duration_s,
            audio_sample_rate=sample_rate,
            tts_segment_count=max(segment_count, 1))
        cache[chunk_id] = record
        append_jsonl(cache_jsonl, [record])
    return cache


# SSLO
def synthesize_supertonic(
    *,
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_jsonl: Path,
    model_name: str,
    voice_name: str,
    lang: str,
    intra_op_num_threads: int | None,
) -> dict[str, dict[str, Any]]:
    from supertonic import TTS
    from tqdm.auto import tqdm

    LOGGER.info("Loading Supertonic TTS (%s, voice=%s, lang=%s, intra_op=%s)",
                model_name, voice_name, lang, intra_op_num_threads)
    tts = TTS(
        auto_download=True,
        intra_op_num_threads=intra_op_num_threads,
    )
    style = tts.get_voice_style(voice_name=voice_name)
    sample_rate = int(tts.sample_rate)

    n_skipped = 0
    for row in tqdm(rows, desc="Supertonic"):
        chunk_id = row["chunk_id"]
        if chunk_id in cache:
            continue

        try:
            t0 = time.monotonic()
            wav, dur_arr = tts.synthesize(
                row["text"], voice_style=style, lang=lang)
            conversion_time_s = time.monotonic() - t0
        except Exception as e:  # noqa: BLE001 — robustness over precision
            # Supertonic can fail on (a) ValueError for unsupported
            # characters (∑, ∫, etc.) or (b) onnxruntime RuntimeException
            # for sequences past the model's attention size. Skip the
            # chunk and keep the shard alive — partial coverage is
            # better than a dead shard.
            LOGGER.warning("Supertonic skip chunk_id=%s: %s: %s",
                           chunk_id, type(e).__name__, e)
            n_skipped += 1
            continue
        # wav: (1, N) float32; dur_arr: (1,) float32 — duration in seconds.
        audio_duration_s = float(dur_arr.reshape(-1)[0])

        record = make_record(
            row, model_name=model_name, tts_backend="supertonic",
            conversion_time_s=conversion_time_s,
            audio_duration_s=audio_duration_s,
            audio_sample_rate=sample_rate,
            tts_segment_count=1)
        cache[chunk_id] = record
        append_jsonl(cache_jsonl, [record])
    if n_skipped:
        LOGGER.warning("Supertonic skipped %d chunks due to unsupported "
                       "characters.", n_skipped)
    return cache


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    input_rows = load_input_chunks(args.input_jsonl, max_rows=args.max_rows)
    # SSLO: shard filter — keeps rows whose index falls into this shard.
    if args.shard_count > 1:
        if not (0 <= args.shard_id < args.shard_count):
            raise SystemExit(
                f"--shard-id ({args.shard_id}) must be in "
                f"[0, --shard-count={args.shard_count})")
        input_rows = [r for i, r in enumerate(input_rows)
                      if i % args.shard_count == args.shard_id]
        LOGGER.info("Shard %d/%d → %d rows after sharding",
                    args.shard_id, args.shard_count, len(input_rows))
    LOGGER.info("Loaded %d input chunks from %s", len(input_rows), args.input_jsonl)
    if not input_rows:
        LOGGER.error("No input rows — copy a run_sslo chunks.jsonl to %s",
                     args.input_jsonl)
        sys.exit(1)

    paths = duration_output_paths(args.output_dir)
    cache = load_cache(paths.duration_cache_jsonl)
    LOGGER.info("Loaded %d cached records from %s",
                len(cache), paths.duration_cache_jsonl)

    backend = model_backend(args.tts_model)
    if backend == "kokoro":
        cache = synthesize_kokoro(
            rows=input_rows, cache=cache,
            cache_jsonl=paths.duration_cache_jsonl,
            model_name=args.tts_model,
            lang_code=args.kokoro_lang_code,
            voice=args.kokoro_voice)
    # SSLO
    elif backend == "supertonic":
        cache = synthesize_supertonic(
            rows=input_rows, cache=cache,
            cache_jsonl=paths.duration_cache_jsonl,
            model_name=args.tts_model,
            voice_name=args.supertonic_voice,
            lang=args.supertonic_lang,
            intra_op_num_threads=args.supertonic_intra_op_num_threads)
    else:
        raise AssertionError(f"Unhandled backend: {backend}")

    output_rows = [cache[r["chunk_id"]] for r in input_rows
                    if r["chunk_id"] in cache]
    write_jsonl(paths.durations_jsonl, output_rows)
    write_csv(paths.durations_csv, output_rows, columns=DURATION_COLUMNS)
    LOGGER.info("Wrote %d rows -> %s", len(output_rows), paths.durations_csv)


if __name__ == "__main__":
    main()
