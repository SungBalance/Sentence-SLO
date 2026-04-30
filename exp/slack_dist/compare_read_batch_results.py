#!/usr/bin/env python3
"""Compare human slack distributions across read-only batch sweep runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.slack_utils import percentile, read_jsonl, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human slack results across max_num_seqs batch runs."
    )
    parser.add_argument(
        "--batch-result",
        action="append",
        default=[],
        metavar="MAX_NUM_SEQS=RESULTS_DIR",
        help=(
            "Batch result directory mapping, for example "
            "16=/tmp/batch_16/previous_chunk/results."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slack-mode", required=True)
    parser.add_argument("--trim-quantile", type=float, default=0.01)
    return parser.parse_args()


def parse_batch_results(
    specs: list[str],
) -> list[tuple[int, Path]]:
    if not specs:
        raise ValueError("Provide at least one --batch-result entry.")

    pairs: list[tuple[int, Path]] = []
    for spec in specs:
        batch_size, separator, result_dir = spec.partition("=")
        if not separator or not batch_size or not result_dir:
            raise ValueError(
                "--batch-result must use MAX_NUM_SEQS=RESULTS_DIR."
            )
        pairs.append((int(batch_size), Path(result_dir)))
    return sorted(pairs, key=lambda item: item[0])


def build_summary_rows(batch_results: list[tuple[int, Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []

    for max_num_seqs, result_dir in batch_results:
        slack_rows = read_jsonl(result_dir / "slack_rows.jsonl")
        summary = json.loads((result_dir / "summary.json").read_text())
        summary_rows.append(
            {
                "max_num_seqs": max_num_seqs,
                "row_count": int(summary.get("row_count", len(slack_rows))),
                "mean": float(summary["human"]["mean"]),
                "p05": float(summary["human"]["p05"]),
                "p50": float(summary["human"]["p50"]),
                "p95": float(summary["human"]["p95"]),
                "min": float(summary["human"]["min"]),
                "max": float(summary["human"]["max"]),
                "tensor_parallel_size": summary.get("tensor_parallel_size"),
                "gpu_memory_utilization": summary.get("gpu_memory_utilization"),
            }
        )
        for row in slack_rows:
            plot_rows.append(
                {
                    "max_num_seqs": str(max_num_seqs),
                    "slack_seconds": float(row["human_slack_seconds"]),
                }
            )

    return summary_rows, plot_rows


def plot_distribution(
    *,
    plot_rows: list[dict[str, Any]],
    output_path: Path,
    slack_mode: str,
    trim_quantile: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import pandas as pd

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

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6))
    legend_handles: list[Patch] = []
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for color_index, (max_num_seqs, group) in enumerate(
        frame.groupby("max_num_seqs", sort=True)
    ):
        color = colors[color_index % len(colors)] if colors else None
        axis.hist(
            group["slack_seconds"],
            bins=90,
            density=True,
            histtype="stepfilled",
            alpha=0.25,
            color=color,
        )
        legend_handles.append(
            Patch(facecolor=color, alpha=0.25, label=f"max_num_seqs={max_num_seqs}")
        )
    axis.axvline(0.0, color="#1F2933", linewidth=1.2, linestyle="--")
    axis.set_xlabel("human slack seconds (deadline - actual chunk end)")
    axis.set_ylabel("density")
    axis.set_title(f"{slack_mode} human slack by max_num_seqs")
    axis.grid(alpha=0.25)
    if legend_handles:
        axis.legend(handles=legend_handles, title="batch")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    batch_results = parse_batch_results(args.batch_result)
    summary_rows, plot_rows = build_summary_rows(batch_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "batch_summary.json", summary_rows)
    write_csv(
        output_dir / "batch_summary.csv",
        summary_rows,
        columns=[
            "max_num_seqs",
            "row_count",
            "mean",
            "p05",
            "p50",
            "p95",
            "min",
            "max",
            "tensor_parallel_size",
            "gpu_memory_utilization",
        ],
    )
    plot_distribution(
        plot_rows=plot_rows,
        output_path=output_dir / "slack_distribution_by_batch.png",
        slack_mode=args.slack_mode,
        trim_quantile=args.trim_quantile,
    )


if __name__ == "__main__":
    main()
