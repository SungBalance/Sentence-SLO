#!/usr/bin/env python3
"""Post-process slack timeline JSONL into per-chunk JSONL and CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


COLUMNS = [
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


def _load_summary_model(path: str | None) -> str:
    if path is None:
        return ""
    summary_path = Path(path)
    if not summary_path.exists():
        return ""
    with summary_path.open() as f:
        summary = json.load(f)
    return str(summary.get("model", ""))


def _request_idx(record: dict[str, Any], fallback: int) -> int:
    request_id = str(record.get("request_id", ""))
    if request_id.startswith("test") and request_id[4:].isdigit():
        return int(request_id[4:])
    return fallback


def build_rows(
    records: list[dict[str, Any]],
    *,
    model: str,
    seconds_per_word: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fallback_request_idx, record in enumerate(records):
        request_idx = _request_idx(record, fallback_request_idx)
        chunks = sorted(
            record.get("chunks", []),
            key=lambda chunk: (
                chunk.get("output_index", 0),
                chunk.get("chunk_index", 0),
                chunk.get("end_time_ts", 0.0),
            ),
        )

        prev_end_time = float(record["decoding_start_ts"])
        prev_chunk_consume = 0.0
        previous_cumulative_chunk_consume = 0.0
        for chunk_idx, chunk in enumerate(chunks):
            word_count = int(chunk.get("word_count", 0))
            cur_chunk_consume = word_count * seconds_per_word
            cumulative_chunk_consume = (
                previous_cumulative_chunk_consume + cur_chunk_consume
            )

            end_time_ts = float(chunk["end_time_ts"])
            cur_chunk_deadline = prev_end_time + prev_chunk_consume
            cumulative_chunk_deadline = (
                prev_end_time + previous_cumulative_chunk_consume
            )

            rows.append(
                {
                    "model": model,
                    "request_idx": request_idx,
                    "chunk_idx": chunk_idx,
                    "token_count": int(chunk.get("token_count", 0)),
                    "word_count": word_count,
                    "text": chunk.get("text", ""),
                    "start_time_ts": float(chunk["start_time_ts"]),
                    "end_time_ts": end_time_ts,
                    "duration_seconds": float(chunk["duration_seconds"]),
                    "cur_chunk_consume": cur_chunk_consume,
                    "cumulative_chunk_consume": cumulative_chunk_consume,
                    "cur_chunk_deadline": cur_chunk_deadline,
                    "cur_chunk_slack": end_time_ts - cur_chunk_deadline,
                    "cumulative_chunk_deadline": cumulative_chunk_deadline,
                    "cumulative_chunk_slack": end_time_ts - cumulative_chunk_deadline,
                }
            )
            prev_end_time = end_time_ts
            prev_chunk_consume = cur_chunk_consume
            previous_cumulative_chunk_consume = cumulative_chunk_consume
    return rows


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process slack timeline JSONL into per-chunk JSONL and CSV."
    )
    parser.add_argument(
        "--input-jsonl",
        default="exp/slack_results.jsonl",
        help="Input slack timeline JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        default="exp/slack_chunks_postprocessed.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="exp/slack_chunks_postprocessed.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--summary-json",
        default="exp/slack_results.json",
        help="Optional summary JSON used to infer model metadata.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model label for the output rows. Overrides --summary-json metadata.",
    )
    parser.add_argument(
        "--seconds-per-word",
        type=float,
        default=0.28,
        help="Consumer reading time per generated word.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model if args.model is not None else _load_summary_model(args.summary_json)
    records = read_jsonl(args.input_jsonl)
    rows = build_rows(records, model=model, seconds_per_word=args.seconds_per_word)
    write_jsonl(args.output_jsonl, rows)
    write_csv(args.output_csv, rows)
    print(
        f"Wrote {len(rows)} chunk rows from {len(records)} requests "
        f"to {args.output_jsonl} and {args.output_csv}"
    )


if __name__ == "__main__":
    main()
