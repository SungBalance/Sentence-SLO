#!/usr/bin/env python3
"""Shared helpers for the measure_tts_duration experiment."""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CHUNK_COLUMNS = [
    "chunk_id",
    "dataset_name",
    "dataset_item_id",
    "turn_idx",
    "role",
    "chunk_unit",
    "chunk_idx",
    "text",
    "word_count",
    "char_count",
]

DURATION_COLUMNS = [
    "chunk_id",
    "model",
    "dataset_name",
    "dataset_item_id",
    "turn_idx",
    "role",
    "chunk_unit",
    "chunk_idx",
    "text",
    "word_count",
    "char_count",
    "duration_seconds",
    "audio_sample_rate",
    "tts_segment_count",
    "tts_backend",
]

SUMMARY_COLUMNS = [
    "model",
    "chunk_unit",
    "word_count",
    "row_count",
    "mean_duration_seconds",
    "variance_duration_seconds",
    "std_duration_seconds",
    "min_duration_seconds",
    "max_duration_seconds",
]

SENTENCE_ENDINGS = frozenset((".", "!", "?", "。", "！", "？", "…"))
CODE_FENCE_RE = re.compile(r"```")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from |return |SELECT |INSERT |UPDATE |DELETE |<[^>]+>)",
    re.IGNORECASE | re.MULTILINE,
)
PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])(?:\s+|\n+)")


@dataclass(frozen=True)
class ChunkOutputPaths:
    chunks_jsonl: Path
    chunks_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class DurationOutputPaths:
    durations_jsonl: Path
    durations_csv: Path
    duration_cache_jsonl: Path


@dataclass(frozen=True)
class SummaryOutputPaths:
    stats_jsonl: Path
    stats_csv: Path
    summary_json: Path


def slugify(value: str) -> str:
    slug = value.replace("/", "__").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", slug)


def chunk_output_paths(root: str | Path) -> ChunkOutputPaths:
    root = Path(root)
    return ChunkOutputPaths(
        chunks_jsonl=root / "chunks.jsonl",
        chunks_csv=root / "chunks.csv",
        summary_json=root / "summary.json",
    )


def duration_output_paths(root: str | Path) -> DurationOutputPaths:
    root = Path(root)
    return DurationOutputPaths(
        durations_jsonl=root / "durations.jsonl",
        durations_csv=root / "durations.csv",
        duration_cache_jsonl=root / "duration_cache.jsonl",
    )


def summary_output_paths(root: str | Path) -> SummaryOutputPaths:
    root = Path(root)
    return SummaryOutputPaths(
        stats_jsonl=root / "word_count_duration_stats.jsonl",
        stats_csv=root / "word_count_duration_stats.csv",
        summary_json=root / "summary.json",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    columns: Sequence[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    return {str(row["chunk_id"]): row for row in read_jsonl(cache_path)}


def word_count(text: str) -> int:
    return len([part for part in text.split() if part])


def has_code_like_content(text: str) -> bool:
    if CODE_FENCE_RE.search(text) or INLINE_CODE_RE.search(text):
        return True
    if CODE_LINE_RE.search(text):
        return True

    suspicious_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith((";", "{", "}", "</code>")):
            suspicious_lines += 1
        if any(token in stripped for token in ("::", "=>", "==", "!=")):
            suspicious_lines += 1
    return suspicious_lines >= 2


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    return paragraphs or [text.strip()]


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    pieces = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not pieces:
        return [text]

    merged: list[str] = []
    for piece in pieces:
        if merged and piece and piece[0].islower() and merged[-1][-1] not in SENTENCE_ENDINGS:
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)
    return merged


def chunk_text(text: str, chunk_unit: str) -> list[str]:
    if chunk_unit == "sentence":
        return split_sentences(text)
    if chunk_unit == "paragraph":
        return split_paragraphs(text)
    raise ValueError(f"Unsupported chunk unit: {chunk_unit}")


def summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "row_count": 0,
            "mean_duration_seconds": 0.0,
            "variance_duration_seconds": 0.0,
            "std_duration_seconds": 0.0,
            "min_duration_seconds": 0.0,
            "max_duration_seconds": 0.0,
        }

    variance = statistics.pvariance(values) if len(values) > 1 else 0.0
    return {
        "row_count": len(values),
        "mean_duration_seconds": statistics.fmean(values),
        "variance_duration_seconds": variance,
        "std_duration_seconds": variance**0.5,
        "min_duration_seconds": min(values),
        "max_duration_seconds": max(values),
    }
