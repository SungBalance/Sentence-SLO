#!/usr/bin/env python3
"""Aggregate duration rows by word count."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SUMMARY_COLUMNS,
    read_jsonl,
    summary_output_paths,
    summary_stats,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TTS duration rows into word-count tables."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Repeat to merge multiple model outputs into one table.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def build_stats(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["model"]), str(row["chunk_unit"]), int(row["word_count"]))
        grouped[key].append(float(row["duration_seconds"]))

    stats_rows: list[dict] = []
    for (model, chunk_unit, word_count), durations in sorted(grouped.items()):
        summary = summary_stats(durations)
        stats_rows.append(
            {
                "model": model,
                "chunk_unit": chunk_unit,
                "word_count": word_count,
                **summary,
            }
        )
    return stats_rows


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []
    for input_path in args.input_jsonl:
        all_rows.extend(read_jsonl(input_path))

    stats_rows = build_stats(all_rows)
    paths = summary_output_paths(args.output_dir)
    write_jsonl(paths.stats_jsonl, stats_rows)
    write_csv(paths.stats_csv, stats_rows, columns=SUMMARY_COLUMNS)
    write_json(
        paths.summary_json,
        {
            "num_duration_rows": len(all_rows),
            "num_summary_rows": len(stats_rows),
            "models": sorted({str(row["model"]) for row in all_rows}),
            "chunk_units": sorted({str(row["chunk_unit"]) for row in all_rows}),
        },
    )
    print(f"Wrote {len(stats_rows)} word-count summary rows under {args.output_dir}")


if __name__ == "__main__":
    main()
