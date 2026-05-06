#!/usr/bin/env python3
"""Build a flat summary.csv from every summary.json under a sweep root.

Walks `<sweep_root>/<label>/<chunk_unit>/seqs_<n>/rate_<r>/run_<i>/summary.json`
and writes one CSV row per (cell, run, mode) — i.e. one Python invocation.

For each distribution metric (TTFT, TPOT, queue stall, slack, scheduler
occupancy, pending time, inter-chunk delay) the CSV carries mean / p50 /
p90 / p99 columns. Re-running the script overwrites the CSV with the
current contents on disk.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DistMetric:
    """A distribution metric — mean/p50/p90/p99 will be emitted per mode."""
    name: str
    path: tuple[str, ...]
    scale: float = 1.0


# Distribution metrics: each yields name_mean, name_p50, name_p90, name_p99.
DIST_METRICS: tuple[DistMetric, ...] = (
    DistMetric("ttft_s",                ("ttft", "all"),           1.0),
    DistMetric("tpot_ms",               ("tpot",),                 1000.0),
    DistMetric("queue_stall_ms",        ("queue_stall",),          1000.0),
    DistMetric("slack_s",               ("slack",),                1.0),
    DistMetric("running",               ("scheduler", "running"),  1.0),
    DistMetric("combined",              ("scheduler", "combined"), 1.0),
    DistMetric("pending_time_s",        ("pending", "time"),       1.0),
    DistMetric("inter_chunk_delay_ms",  ("inter_chunk_delay",),    1000.0),
)
STATS = ("mean", "p50", "p90", "p99")

# Scalar (non-distribution) metrics — emit one column each.
SCALAR_METRICS: tuple[tuple[str, tuple[str, ...], str, float], ...] = (
    ("slack_neg_ratio",      ("slack",),          "neg_ratio", 1.0),
    ("slo_compliance_rate",  ("slo_compliance",), "rate",      1.0),
    ("slo_compliance_count", ("slo_compliance",), "count",     1.0),
    ("slo_total_requests",   ("slo_compliance",), "total_requests", 1.0),
)

CONTEXT_COLUMNS = (
    "Model", "Label", "mode", "chunk_unit", "req/s", "max_num_seqs",
    "generation_max_tokens", "max_model_len", "run_idx", "num_requests",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-root", default="exp/sslo_test/output_sweep",
        help="Parent directory containing one subdir per label.")
    p.add_argument(
        "--output", default=None,
        help="CSV output path (default: <sweep_root>/summary.csv).")
    return p.parse_args()


def parse_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def discover_runs(sweep_root: Path) -> list[Path]:
    """Every <root>/<label>/<unit>/seqs_*/rate_*/run_*/summary.json."""
    if not sweep_root.exists():
        return []
    return sorted(sweep_root.glob("*/*/seqs_*/rate_*/run_*/summary.json"))


def metric_node(summary: dict, path: tuple[str, ...], mode: str):
    """Return summary['metrics'][path[0]][mode][path[1]]... or None."""
    node = summary.get("metrics", {}).get(path[0], {}).get(mode)
    for segment in path[1:]:
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node


def emit_rows(summary_path: Path) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config", {})
    # Path layout: <root>/<label>/<unit>/seqs_<n>/rate_<r>/run_<i>/summary.json
    path_label = summary_path.parents[4].name
    run_idx = parse_int_suffix(summary_path.parent.name, "run_")
    rows: list[dict] = []
    for mode in cfg.get("modes_run", []):
        ttft_node = metric_node(summary, ("ttft", "all"), mode) or {}
        row: dict = {
            "Model": cfg.get("model"),
            "Label": cfg.get("label") or path_label,
            "mode": mode,
            "chunk_unit": cfg.get("chunk_unit"),
            "req/s": cfg.get("request_rate"),
            "max_num_seqs": cfg.get("max_num_seqs"),
            "generation_max_tokens": cfg.get("generation_max_tokens"),
            "max_model_len": cfg.get("max_model_len"),
            "run_idx": run_idx,
            "num_requests": ttft_node.get("count"),
        }
        for dm in DIST_METRICS:
            node = metric_node(summary, dm.path, mode) or {}
            for stat in STATS:
                value = node.get(stat)
                row[f"{dm.name}_{stat}"] = (
                    None if value is None else float(value) * dm.scale)
        for name, path, field, scale in SCALAR_METRICS:
            node = metric_node(summary, path, mode) or {}
            value = node.get(field)
            row[name] = None if value is None else float(value) * scale
        rows.append(row)
    return rows


def build_dataframe(sweep_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in discover_runs(sweep_root):
        rows.extend(emit_rows(path))
    columns = list(CONTEXT_COLUMNS)
    for dm in DIST_METRICS:
        columns.extend(f"{dm.name}_{stat}" for stat in STATS)
    columns.extend(name for name, *_ in SCALAR_METRICS)
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    output_path = Path(args.output) if args.output else (
        sweep_root / "summary.csv")
    df = build_dataframe(sweep_root)
    if df.empty:
        print(f"  no summary.json found under {sweep_root}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
