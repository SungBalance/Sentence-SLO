#!/usr/bin/env python3
"""Compute human/audio slack rows and plots for one slack mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.slack_utils import (
    RESULT_COLUMNS,
    attach_audio_timeline,
    read_jsonl,
    summary_stats,
    write_csv,
    write_json,
    write_jsonl,
)


# This mapping is the only branch between the two slack calculation modes.
SLACK_MODE_COLUMNS = {
    "previous_chunk": {
        "human_consume": "cur_chunk_consume",
        "human_deadline": "cur_chunk_deadline",
        "human_slack": "cur_chunk_slack",
        "audio_consume": "audio_chunk_consume",
        "audio_deadline": "audio_cur_chunk_deadline",
        "audio_slack": "audio_cur_chunk_slack",
    },
    "cumulative": {
        "human_consume": "cumulative_chunk_consume",
        "human_deadline": "cumulative_chunk_deadline",
        "human_slack": "cumulative_chunk_slack",
        "audio_consume": "audio_cumulative_chunk_consume",
        "audio_deadline": "audio_cumulative_chunk_deadline",
        "audio_slack": "audio_cumulative_chunk_slack",
    },
}


def build_result_rows(
    *,
    chunk_rows: list[dict[str, Any]],
    duration_rows: list[dict[str, Any]],
    slack_mode: str,
    require_complete: bool,
) -> list[dict[str, Any]]:
    columns = SLACK_MODE_COLUMNS[slack_mode]
    # Add audio deadlines once, then normalize the selected mode into common names.
    combined_rows = attach_audio_timeline(
        chunk_rows,
        duration_rows,
        require_complete=require_complete,
    )
    result_rows: list[dict[str, Any]] = []

    for row in combined_rows:
        result_rows.append(
            {
                "model": str(row["model"]),
                "request_idx": int(row["request_idx"]),
                "chunk_idx": int(row["chunk_idx"]),
                "token_count": int(row["token_count"]),
                "word_count": int(row["word_count"]),
                "text": row.get("text", ""),
                "start_time_ts": float(row["start_time_ts"]),
                "end_time_ts": float(row["end_time_ts"]),
                "duration_seconds": float(row["duration_seconds"]),
                "slack_mode": slack_mode,
                "human_consume_seconds": float(row[columns["human_consume"]]),
                "audio_consume_seconds": float(row[columns["audio_consume"]]),
                "human_deadline_ts": float(row[columns["human_deadline"]]),
                "audio_deadline_ts": float(row[columns["audio_deadline"]]),
                "human_slack_seconds": float(row[columns["human_slack"]]),
                "audio_slack_seconds": float(row[columns["audio_slack"]]),
                "audio_sample_rate": int(row["audio_sample_rate"]),
                "tts_segment_count": int(row.get("tts_segment_count", 1)),
                "tts_backend": row.get("tts_backend", ""),
            }
        )
    return result_rows


def summarize(rows: list[dict[str, Any]], *, slack_mode: str) -> dict[str, Any]:
    human_values = [float(row["human_slack_seconds"]) for row in rows]
    audio_values = [float(row["audio_slack_seconds"]) for row in rows]
    return {
        "slack_mode": slack_mode,
        "row_count": len(rows),
        "human": summary_stats(human_values),
        "audio": summary_stats(audio_values),
    }


def plot_distribution(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
    slack_mode: str,
    trim_quantile: float,
) -> None:
    import matplotlib.pyplot as plt

    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        # Plot human and TTS audio slack on the same axis for quick comparison.
        plot_rows.append(
            {
                "consume_type": "human",
                "slack_seconds": float(row["human_slack_seconds"]),
            }
        )
        plot_rows.append(
            {
                "consume_type": "audio",
                "slack_seconds": float(row["audio_slack_seconds"]),
            }
        )

    frame = pd.DataFrame(plot_rows)
    if frame.empty:
        raise ValueError("No rows available for plotting.")
    if trim_quantile > 0:
        lo = frame["slack_seconds"].quantile(trim_quantile)
        hi = frame["slack_seconds"].quantile(1.0 - trim_quantile)
        if lo != hi:
            frame = frame[
                (frame["slack_seconds"] >= lo) & (frame["slack_seconds"] <= hi)
            ]

    sns.set_theme(style="whitegrid", context="talk")
    axis = sns.histplot(
        data=frame,
        x="slack_seconds",
        hue="consume_type",
        bins=90,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        palette={"human": "#2F6F9F", "audio": "#D9822B"},
    )
    axis.axvline(0.0, color="#1F2933", linewidth=1.2, linestyle="--")
    axis.set_xlabel("slack seconds (deadline - actual chunk end)")
    axis.set_ylabel("density")
    axis.set_title(f"{slack_mode} slack distribution")
    axis.grid(alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot human/audio slack for one calculation mode."
    )
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--durations-jsonl", required=True)
    parser.add_argument(
        "--slack-mode",
        choices=sorted(SLACK_MODE_COLUMNS),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trim-quantile", type=float, default=0.01)
    parser.add_argument("--allow-missing-durations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = build_result_rows(
        chunk_rows=read_jsonl(args.chunks_jsonl),
        duration_rows=read_jsonl(args.durations_jsonl),
        slack_mode=args.slack_mode,
        require_complete=not args.allow_missing_durations,
    )
    write_jsonl(output_dir / "slack_rows.jsonl", result_rows)
    write_csv(output_dir / "slack_rows.csv", result_rows, columns=RESULT_COLUMNS)
    write_json(
        output_dir / "summary.json",
        summarize(result_rows, slack_mode=args.slack_mode),
    )
    plot_distribution(
        rows=result_rows,
        output_path=output_dir / "slack_distribution.png",
        slack_mode=args.slack_mode,
        trim_quantile=args.trim_quantile,
    )
    print(f"Wrote {args.slack_mode} slack results under {output_dir}")


if __name__ == "__main__":
    main()
