#!/usr/bin/env python3
"""Shared helpers for SSLO experiment post-processing scripts."""

from __future__ import annotations

import csv
import hashlib
import importlib.machinery
import json
import statistics
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# Shared output schemas for the three experiment stages.
SLACK_COLUMNS = [
    "model",
    "request_idx",
    "chunk_idx",
    "token_count",
    "word_count",
    "text",
    "start_time_ts",
    "end_time_ts",
    "duration_seconds",
    "cur_chunk_consume",
    "cumulative_chunk_consume",
    "cur_chunk_deadline",
    "cur_chunk_slack",
    "cumulative_chunk_deadline",
    "cumulative_chunk_slack",
]

DURATION_COLUMNS = [
    "key",
    "model",
    "request_idx",
    "chunk_idx",
    "audio_path",
    "audio_sample_rate",
    "audio_chunk_consume",
    "duration_mode",
    "tts_segment_count",
    "tts_backend",
]

RESULT_COLUMNS = [
    "model",
    "request_idx",
    "chunk_idx",
    "token_count",
    "word_count",
    "text",
    "start_time_ts",
    "end_time_ts",
    "duration_seconds",
    "slack_mode",
    "human_consume_seconds",
    "audio_consume_seconds",
    "human_deadline_ts",
    "audio_deadline_ts",
    "human_slack_seconds",
    "audio_slack_seconds",
    "audio_sample_rate",
    "tts_segment_count",
    "tts_backend",
]


@dataclass(frozen=True)
class TimelineColumns:
    current_consume: str
    cumulative_consume: str
    current_deadline: str
    current_slack: str
    cumulative_deadline: str
    cumulative_slack: str


HUMAN_TIMELINE_COLUMNS = TimelineColumns(
    current_consume="cur_chunk_consume",
    cumulative_consume="cumulative_chunk_consume",
    current_deadline="cur_chunk_deadline",
    current_slack="cur_chunk_slack",
    cumulative_deadline="cumulative_chunk_deadline",
    cumulative_slack="cumulative_chunk_slack",
)

AUDIO_TIMELINE_COLUMNS = TimelineColumns(
    current_consume="audio_chunk_consume",
    cumulative_consume="audio_cumulative_chunk_consume",
    current_deadline="audio_cur_chunk_deadline",
    current_slack="audio_cur_chunk_slack",
    cumulative_deadline="audio_cumulative_chunk_deadline",
    cumulative_slack="audio_cumulative_chunk_slack",
)


def request_idx(record: dict[str, Any], fallback: int) -> int:
    request_id = str(record.get("request_id", ""))
    if request_id.startswith("test") and request_id[4:].isdigit():
        return int(request_id[4:])
    return fallback


def mean_or_zero(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean_or_zero(values),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def add_deadline_slack_columns(
    rows: Sequence[dict[str, Any]],
    *,
    initial_time: float,
    consume_fn: Callable[[dict[str, Any]], float],
    columns: TimelineColumns,
    end_time_column: str = "end_time_ts",
) -> list[dict[str, Any]]:
    """Apply previous-chunk and cumulative slack math to one request timeline."""

    output_rows: list[dict[str, Any]] = []
    prev_end_time = initial_time
    prev_consume = 0.0
    cumulative_consume = 0.0

    for row in rows:
        current_consume = consume_fn(row)
        next_cumulative_consume = cumulative_consume + current_consume
        end_time = float(row[end_time_column])
        current_deadline = prev_end_time + prev_consume
        cumulative_deadline = prev_end_time + cumulative_consume

        new_row = dict(row)
        new_row.update(
            {
                columns.current_consume: current_consume,
                columns.cumulative_consume: next_cumulative_consume,
                columns.current_deadline: current_deadline,
                columns.current_slack: current_deadline - end_time,
                columns.cumulative_deadline: cumulative_deadline,
                columns.cumulative_slack: cumulative_deadline - end_time,
            }
        )
        output_rows.append(new_row)

        prev_end_time = end_time
        prev_consume = current_consume
        cumulative_consume = next_cumulative_consume

    return output_rows


def build_slack_rows(
    records: list[dict[str, Any]],
    *,
    model: str,
    seconds_per_word: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fallback_request_idx, record in enumerate(records):
        # Convert one request timeline into per-chunk rows first, then add slack.
        chunks = sorted(
            record.get("chunks", []),
            key=lambda chunk: (
                chunk.get("output_index", 0),
                chunk.get("chunk_index", 0),
                chunk.get("end_time_ts", 0.0),
            ),
        )

        request_rows: list[dict[str, Any]] = []
        for chunk_idx, chunk in enumerate(chunks):
            word_count = int(chunk.get("word_count", 0))
            request_rows.append(
                {
                    "model": model,
                    "request_idx": request_idx(record, fallback_request_idx),
                    "chunk_idx": chunk_idx,
                    "token_count": int(chunk.get("token_count", 0)),
                    "word_count": word_count,
                    "text": chunk.get("text", ""),
                    "start_time_ts": float(chunk["start_time_ts"]),
                    "end_time_ts": float(chunk["end_time_ts"]),
                    "duration_seconds": float(chunk["duration_seconds"]),
                }
            )

        rows.extend(
            add_deadline_slack_columns(
                request_rows,
                initial_time=float(record["decoding_start_ts"]),
                consume_fn=lambda row: float(row["word_count"]) * seconds_per_word,
                columns=HUMAN_TIMELINE_COLUMNS,
            )
        )
    return rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_many(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)
        return

    fieldnames = list(columns) if columns is not None else list(rows[0]) if rows else []
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def row_key(row: dict[str, Any]) -> str:
    text_hash = hashlib.sha1(str(row.get("text", "")).encode()).hexdigest()[:10]
    return f"{row['model']}::{row['request_idx']}::{row['chunk_idx']}::{text_hash}"


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text or " "]

    segments: list[str] = []
    current_words: list[str] = []
    current_len = 0
    for word in text.split():
        word_len = len(word)
        extra = word_len if not current_words else word_len + 1
        if current_words and current_len + extra > max_chars:
            segments.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        else:
            current_words.append(word)
            current_len += extra
    if current_words:
        segments.append(" ".join(current_words))
    return segments or [" "]


def stub_torchaudio_for_ngc() -> None:
    """Avoid importing the incompatible public torchaudio wheel in NGC vLLM."""

    if "torchaudio" in sys.modules:
        return

    def stub_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    torchaudio = stub_module("torchaudio")
    compliance = stub_module("torchaudio.compliance")
    kaldi = stub_module("torchaudio.compliance.kaldi")
    compliance.kaldi = kaldi
    torchaudio.compliance = compliance
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi


def patch_qwen_tts_runtime() -> None:
    """Apply small import/runtime patches needed by Qwen3-TTS in these containers."""

    stub_torchaudio_for_ngc()
    try:
        import torch

        torch.backends.cudnn.enabled = False
    except Exception:
        pass

    try:
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
        from vllm_omni.worker.gpu_generation_model_runner import (
            GPUGenerationModelRunner,
        )

        GPUARModelRunner.routed_experts_initialized = False
        GPUGenerationModelRunner.routed_experts_initialized = False
    except Exception:
        pass


def load_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    return {str(record["key"]): record for record in read_jsonl(cache_path)}


def build_duration_rows(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    *,
    require_complete: bool,
) -> list[dict[str, Any]]:
    duration_rows: list[dict[str, Any]] = []
    for row in rows:
        key = row_key(row)
        if key not in cache:
            if require_complete:
                raise RuntimeError(
                    "Missing audio duration for "
                    f"model={row['model']} request={row['request_idx']} "
                    f"chunk={row['chunk_idx']}. Re-run without --max-chunks "
                    "or let the audio generation finish."
                )
            continue

        audio_record = cache[key]
        duration_rows.append(
            {
                "key": key,
                "model": str(row["model"]),
                "request_idx": int(row["request_idx"]),
                "chunk_idx": int(row["chunk_idx"]),
                "audio_path": audio_record.get("audio_path", ""),
                "audio_sample_rate": int(audio_record["audio_sample_rate"]),
                "audio_chunk_consume": float(audio_record["audio_chunk_consume"]),
                "duration_mode": audio_record.get("duration_mode", ""),
                "tts_segment_count": int(audio_record.get("tts_segment_count", 1)),
                "tts_backend": audio_record.get("tts_backend", ""),
            }
        )
    return duration_rows


def attach_audio_timeline(
    chunk_rows: list[dict[str, Any]],
    duration_rows: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    duration_by_key = {str(row["key"]): row for row in duration_rows}
    output_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in chunk_rows:
        grouped.setdefault((str(row["model"]), int(row["request_idx"])), []).append(row)

    for _, request_rows in sorted(grouped.items(), key=lambda item: item[0]):
        request_rows.sort(key=lambda row: int(row["chunk_idx"]))

        request_audio_rows: list[dict[str, Any]] = []
        for row in request_rows:
            key = row_key(row)
            if key not in duration_by_key:
                if require_complete:
                    raise RuntimeError(
                        "Missing audio duration for "
                        f"model={row['model']} request={row['request_idx']} "
                        f"chunk={row['chunk_idx']}."
                    )
                break

            duration = duration_by_key[key]
            new_row = dict(row)
            new_row.update(
                {
                    "audio_path": duration.get("audio_path", ""),
                    "audio_sample_rate": int(duration["audio_sample_rate"]),
                    "audio_chunk_consume": float(duration["audio_chunk_consume"]),
                    "tts_segment_count": int(duration.get("tts_segment_count", 1)),
                    "tts_backend": duration.get("tts_backend", ""),
                }
            )
            request_audio_rows.append(new_row)

        output_rows.extend(
            add_deadline_slack_columns(
                request_audio_rows,
                initial_time=float(request_rows[0]["cur_chunk_deadline"]),
                consume_fn=lambda row: float(row["audio_chunk_consume"]),
                columns=AUDIO_TIMELINE_COLUMNS,
            )
        )
    return output_rows
