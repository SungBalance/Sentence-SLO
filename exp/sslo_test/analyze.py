#!/usr/bin/env python3
"""Summarize the SSLO scheduler end-to-end validation run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MAX_NUM_SEQS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exp/sslo_test/output",
        help="Directory containing baseline_ttft.jsonl, sslo_ttft.jsonl, and sslo_stats.jsonl.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def ttft_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = [
        float(row["ttft"])
        for row in rows
        if row.get("ttft") is not None
    ]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def h2_rows(
    baseline_rows: list[dict[str, Any]],
    sslo_rows: list[dict[str, Any]],
    max_num_seqs: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_new = [
        row for row in baseline_rows if int(row.get("request_idx", -1)) >= max_num_seqs
    ]
    sslo_new = [
        row for row in sslo_rows if int(row.get("request_idx", -1)) >= max_num_seqs
    ]
    if baseline_new and sslo_new:
        return "post_cap_arrivals", baseline_new, sslo_new
    if baseline_new and not sslo_new:
        return "post_cap_arrivals_missing_sslo", baseline_new, sslo_new
    return "all_requests_smoke_fallback", baseline_rows, sslo_rows


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old * 100.0


def format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_stats(label: str, stats: dict[str, float | int | None]) -> None:
    print(
        f"{label} TTFT: count={stats['count']} "
        f"mean={format_value(stats['mean'])} "
        f"p50={format_value(stats['p50'])} "
        f"p90={format_value(stats['p90'])} "
        f"p99={format_value(stats['p99'])}"
    )


def analyze(output_dir: Path, max_num_seqs: int) -> dict[str, Any]:
    baseline_rows = read_jsonl(output_dir / "baseline_ttft.jsonl")
    sslo_rows = read_jsonl(output_dir / "sslo_ttft.jsonl")
    stats_rows = read_jsonl(output_dir / "sslo_stats.jsonl")
    run_status = read_json(output_dir / "run_status.json")
    missing_files = [
        name
        for name in ("baseline_ttft.jsonl", "sslo_ttft.jsonl", "sslo_stats.jsonl")
        if not (output_dir / name).exists()
    ]

    baseline_all_stats = ttft_stats(baseline_rows)
    sslo_all_stats = ttft_stats(sslo_rows)
    h2_cohort, baseline_h2_rows, sslo_h2_rows = h2_rows(
        baseline_rows, sslo_rows, max_num_seqs
    )
    baseline_h2_stats = ttft_stats(baseline_h2_rows)
    sslo_h2_stats = ttft_stats(sslo_h2_rows)
    p50_change = pct_change(sslo_h2_stats["p50"], baseline_h2_stats["p50"])

    combined_values = [
        int(row.get("combined", int(row.get("running", 0)) + int(row.get("pending", 0))))
        for row in stats_rows
    ]
    pending_values = [int(row.get("pending", 0)) for row in stats_rows]
    h1_max_combined = max(combined_values, default=0)
    h1_iterations_above_cap = sum(
        1
        for row in stats_rows
        if int(row.get("combined", 0)) > max_num_seqs
        and int(row.get("pending", 0)) > 0
        and int(row.get("running", 0)) <= max_num_seqs
    )
    h1_pass = h1_iterations_above_cap > 0
    h2_pass = (
        baseline_h2_stats["p50"] is not None
        and sslo_h2_stats["p50"] is not None
        and sslo_h2_stats["p50"] < baseline_h2_stats["p50"]
    )

    summary = {
        "max_num_seqs": max_num_seqs,
        "h1_pending_plus_running_exceeds_cap": h1_pass,
        "h1_max_combined": h1_max_combined,
        "h1_iterations_above_cap": h1_iterations_above_cap,
        "h2_ttft_p50_baseline": baseline_h2_stats["p50"],
        "h2_ttft_p50_sslo": sslo_h2_stats["p50"],
        "h2_ttft_p50_pct_change": p50_change,
        "h2_ttft_p90_baseline": baseline_h2_stats["p90"],
        "h2_ttft_p90_sslo": sslo_h2_stats["p90"],
        "h2_ttft_cohort": h2_cohort,
        "h2_ttft_count_baseline": baseline_h2_stats["count"],
        "h2_ttft_count_sslo": sslo_h2_stats["count"],
        "h2_ttft_p50_improved": h2_pass,
        "run_complete": not missing_files
        and bool(baseline_rows)
        and bool(sslo_rows)
        and all(value == 0 for value in run_status.values()),
        "missing_files": missing_files,
        "run_status": run_status,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print_stats("Baseline", baseline_all_stats)
    print_stats("SSLO", sslo_all_stats)
    if h2_cohort != "all_requests_smoke_fallback":
        print(f"H2 TTFT cohort: {h2_cohort}")
        print_stats("Baseline H2 cohort", baseline_h2_stats)
        print_stats("SSLO H2 cohort", sslo_h2_stats)
    else:
        print("H2 TTFT cohort: all_requests_smoke_fallback")
    if not summary["run_complete"]:
        print(
            "Run completeness: INCOMPLETE "
            f"missing_files={missing_files} run_status={run_status}"
        )
    print(f"p50 TTFT percent change: {format_value(p50_change)}%")
    print(
        "SSLO scheduler stats: "
        f"max_running_plus_pending={h1_max_combined} "
        f"iterations_above_cap={h1_iterations_above_cap} "
        f"max_pending={max(pending_values, default=0)}"
    )
    print(f"H1 verdict: {'PASS' if h1_pass else 'FAIL'}")
    print(f"H2 verdict: {'PASS' if h2_pass else 'FAIL'}")
    return summary


def main() -> None:
    args = parse_args()
    analyze(Path(args.output_dir), args.max_num_seqs)


if __name__ == "__main__":
    main()
