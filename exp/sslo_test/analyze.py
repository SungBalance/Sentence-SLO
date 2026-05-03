#!/usr/bin/env python3
"""Summarize SSLO scheduler end-to-end validation runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from jsonl_utils import read_jsonl

MAX_NUM_SEQS = 64
DEFAULT_OUTPUT_DIR = "exp/sslo_test/output"
SWEEP_OUTPUT = "sweep_summary.json"
BASELINE_TTFT = "baseline_ttft.jsonl"
BASELINE_CHUNKS = "baseline_chunks.jsonl"
SSLO_TTFT = "sslo_ttft.jsonl"
SSLO_CHUNKS = "sslo_chunks.jsonl"
SSLO_STATS = "sslo_stats.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing per-run JSONL files for one config.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument(
        "--sweep-root",
        help="Aggregate output/seqs_*/summary.json files under this directory.",
    )
    return parser.parse_args()


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


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]


def latency_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = numeric_values(rows, key)
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def tpot_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = numeric_values(rows, "tpot")
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def queue_stall_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = numeric_values(rows, "queue_stall")
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def slack_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = numeric_values(rows, "cumulative_slack")
    neg_count = sum(1 for value in values if value < 0)
    return {
        "count": len(values),
        "neg_slack_ratio": (neg_count / len(values)) if values else None,
        "p5": percentile(values, 5),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
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
    return "all_requests_control_fallback", baseline_rows, sslo_rows


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old * 100.0


def leq_or_equal(new: float | None, old: float | None) -> bool:
    return new is not None and old is not None and new <= old


def format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def print_stats(label: str, metric: str, stats: dict[str, float | int | None]) -> None:
    keys = [key for key in ("count", "mean", "p5", "p50", "p90", "p95", "p99")
            if key in stats]
    rendered = " ".join(f"{key}={format_value(stats[key])}" for key in keys)
    print(f"{label} {metric}: {rendered}")


def control_case(max_num_seqs: int, baseline_rows: list[dict[str, Any]]) -> bool:
    return bool(baseline_rows) and len(baseline_rows) <= max_num_seqs


def analyze(output_dir: Path, max_num_seqs: int) -> dict[str, Any]:
    baseline_rows = read_jsonl(output_dir / BASELINE_TTFT)
    sslo_rows = read_jsonl(output_dir / SSLO_TTFT)
    baseline_chunks = read_jsonl(output_dir / BASELINE_CHUNKS)
    sslo_chunks = read_jsonl(output_dir / SSLO_CHUNKS)
    stats_rows = read_jsonl(output_dir / SSLO_STATS)
    run_status = read_json(output_dir / "run_status.json")
    missing_files = [
        name
        for name in (
            BASELINE_TTFT,
            BASELINE_CHUNKS,
            SSLO_TTFT,
            SSLO_CHUNKS,
            SSLO_STATS,
        )
        if not (output_dir / name).exists()
    ]

    baseline_all_ttft = latency_stats(baseline_rows, "ttft")
    sslo_all_ttft = latency_stats(sslo_rows, "ttft")
    baseline_tpot = tpot_stats(baseline_rows)
    sslo_tpot = tpot_stats(sslo_rows)
    baseline_queue_stall = queue_stall_stats(baseline_rows)
    sslo_queue_stall = queue_stall_stats(sslo_rows)
    baseline_slack = slack_stats(baseline_chunks)
    sslo_slack = slack_stats(sslo_chunks)
    queue_stall_available = (
        baseline_queue_stall["count"] > 0 or sslo_queue_stall["count"] > 0
    )
    if (baseline_rows or sslo_rows) and not queue_stall_available:
        print(
            "Warning: RequestOutput.metrics queue stall timestamps were not "
            "available; queue_stall summary columns will be n/a."
        )

    h2_cohort, baseline_h2_rows, sslo_h2_rows = h2_rows(
        baseline_rows, sslo_rows, max_num_seqs
    )
    baseline_h2_ttft = latency_stats(baseline_h2_rows, "ttft")
    sslo_h2_ttft = latency_stats(sslo_h2_rows, "ttft")
    p50_change = pct_change(sslo_h2_ttft["p50"], baseline_h2_ttft["p50"])

    combined_values = [
        int(row.get("combined", int(row.get("running", 0)) + int(row.get("pending", 0))))
        for row in stats_rows
    ]
    pending_values = [int(row.get("pending", 0)) for row in stats_rows]
    h1_max_combined = max(combined_values, default=0)
    h1_iterations_above_cap = sum(
        1
        for row in stats_rows
        if int(
            row.get(
                "combined",
                int(row.get("running", 0)) + int(row.get("pending", 0)),
            )
        ) > max_num_seqs
        and int(row.get("pending", 0)) > 0
        and int(row.get("running", 0)) <= max_num_seqs
    )
    is_control = control_case(max_num_seqs, baseline_rows)
    h1_pass = h1_iterations_above_cap > 0
    h2_pass = leq_or_equal(sslo_h2_ttft["p50"], baseline_h2_ttft["p50"])
    h3_pass = leq_or_equal(
        sslo_slack["neg_slack_ratio"], baseline_slack["neg_slack_ratio"]
    )
    run_complete = (
        not missing_files
        and bool(baseline_rows)
        and bool(sslo_rows)
        and all(value == 0 for value in run_status.values())
    )

    summary = {
        "max_num_seqs": max_num_seqs,
        "control_case": is_control,
        "h1_pending_plus_running_exceeds_cap": h1_pass,
        "h1_note": (
            "control_case_no_waiting_queue_pressure" if is_control else None
        ),
        "h1_max_combined": h1_max_combined,
        "h1_iterations_above_cap": h1_iterations_above_cap,
        "h2_ttft_p50_baseline": baseline_h2_ttft["p50"],
        "h2_ttft_p50_sslo": sslo_h2_ttft["p50"],
        "h2_ttft_p50_pct_change": p50_change,
        "h2_ttft_p90_baseline": baseline_h2_ttft["p90"],
        "h2_ttft_p90_sslo": sslo_h2_ttft["p90"],
        "h2_ttft_cohort": h2_cohort,
        "h2_ttft_count_baseline": baseline_h2_ttft["count"],
        "h2_ttft_count_sslo": sslo_h2_ttft["count"],
        "h2_ttft_p50_not_worse": h2_pass,
        "h3_slack_neg_ratio_baseline": baseline_slack["neg_slack_ratio"],
        "h3_slack_neg_ratio_sslo": sslo_slack["neg_slack_ratio"],
        "h3_slack_neg_ratio_not_worse": h3_pass,
        "h4_tpot_p50_baseline": baseline_tpot["p50"],
        "h4_tpot_p50_sslo": sslo_tpot["p50"],
        "h4_tpot_p50_pct_change": pct_change(sslo_tpot["p50"], baseline_tpot["p50"]),
        "queue_stall_p50_baseline": baseline_queue_stall["p50"],
        "queue_stall_p50_sslo": sslo_queue_stall["p50"],
        "queue_stall_p90_baseline": baseline_queue_stall["p90"],
        "queue_stall_p90_sslo": sslo_queue_stall["p90"],
        "queue_stall_p50_pct_change": pct_change(
            sslo_queue_stall["p50"], baseline_queue_stall["p50"]
        ),
        "ttft": {
            "baseline_all": baseline_all_ttft,
            "sslo_all": sslo_all_ttft,
            "baseline_h2": baseline_h2_ttft,
            "sslo_h2": sslo_h2_ttft,
        },
        "tpot": {
            "baseline": baseline_tpot,
            "sslo": sslo_tpot,
        },
        "queue_stall": {
            "baseline": baseline_queue_stall,
            "sslo": sslo_queue_stall,
            "available": queue_stall_available,
        },
        "slack": {
            "baseline": baseline_slack,
            "sslo": sslo_slack,
        },
        "run_complete": run_complete,
        "missing_files": missing_files,
        "run_status": run_status,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print_stats("Baseline", "TTFT", baseline_all_ttft)
    print_stats("SSLO", "TTFT", sslo_all_ttft)
    print(f"H2 TTFT cohort: {h2_cohort}")
    print_stats("Baseline H2 cohort", "TTFT", baseline_h2_ttft)
    print_stats("SSLO H2 cohort", "TTFT", sslo_h2_ttft)
    print_stats("Baseline", "TPOT", baseline_tpot)
    print_stats("SSLO", "TPOT", sslo_tpot)
    print_stats("Baseline", "Queue stall", baseline_queue_stall)
    print_stats("SSLO", "Queue stall", sslo_queue_stall)
    print(
        "Baseline Slack: "
        f"count={baseline_slack['count']} "
        f"neg_slack_ratio={format_value(baseline_slack['neg_slack_ratio'])} "
        f"p5={format_value(baseline_slack['p5'])} "
        f"p50={format_value(baseline_slack['p50'])} "
        f"p95={format_value(baseline_slack['p95'])}"
    )
    print(
        "SSLO Slack: "
        f"count={sslo_slack['count']} "
        f"neg_slack_ratio={format_value(sslo_slack['neg_slack_ratio'])} "
        f"p5={format_value(sslo_slack['p5'])} "
        f"p50={format_value(sslo_slack['p50'])} "
        f"p95={format_value(sslo_slack['p95'])}"
    )
    if is_control:
        print("Control case: num_prompts <= max_num_seqs, so H1 is expected to fail.")
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
    print(f"H1 verdict: {format_bool(h1_pass)}")
    print(f"H2-TTFT verdict: {format_bool(h2_pass)}")
    print(f"H3-Slack verdict: {format_bool(h3_pass)}")
    print("H4-TPOT verdict: informational")
    return summary


def summary_sort_key(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("seqs_"))
    except ValueError:
        return 0


def collect_sweep(root: Path) -> dict[str, Any]:
    summaries = []
    for path in sorted(root.glob("seqs_*/summary.json"), key=summary_sort_key):
        summary = read_json(path)
        if summary:
            summary["output_dir"] = str(path.parent)
            summaries.append(summary)
    return {
        "configs": summaries,
        "run_complete": bool(summaries)
        and all(summary.get("run_complete") for summary in summaries),
    }


def print_sweep_table(sweep: dict[str, Any]) -> None:
    rows = sweep.get("configs", [])
    if not rows:
        print("No per-config summary.json files found.")
        return
    headers = [
        "seqs",
        "ctrl",
        "H1",
        "H2",
        "H3",
        "base_ttft_p50",
        "sslo_ttft_p50",
        "base_neg_slack",
        "sslo_neg_slack",
        "base_tpot_p50",
        "sslo_tpot_p50",
        "base_qstall_p50",
        "sslo_qstall_p50",
        "base_qstall_p90",
        "sslo_qstall_p90",
        "qstall_p50_pct",
    ]
    data = []
    for row in rows:
        data.append([
            str(row.get("max_num_seqs", "n/a")),
            "yes" if row.get("control_case") else "no",
            format_bool(bool(row.get("h1_pending_plus_running_exceeds_cap"))),
            format_bool(bool(row.get("h2_ttft_p50_not_worse"))),
            format_bool(bool(row.get("h3_slack_neg_ratio_not_worse"))),
            format_value(row.get("h2_ttft_p50_baseline")),
            format_value(row.get("h2_ttft_p50_sslo")),
            format_value(row.get("h3_slack_neg_ratio_baseline")),
            format_value(row.get("h3_slack_neg_ratio_sslo")),
            format_value(row.get("h4_tpot_p50_baseline")),
            format_value(row.get("h4_tpot_p50_sslo")),
            format_value(row.get("queue_stall_p50_baseline")),
            format_value(row.get("queue_stall_p50_sslo")),
            format_value(row.get("queue_stall_p90_baseline")),
            format_value(row.get("queue_stall_p90_sslo")),
            format_value(row.get("queue_stall_p50_pct_change")),
        ])
    widths = [
        max(len(str(item)) for item in [header] + [row[i] for row in data])
        for i, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in data:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def analyze_sweep(root: Path) -> dict[str, Any]:
    sweep = collect_sweep(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / SWEEP_OUTPUT).write_text(json.dumps(sweep, indent=2) + "\n")
    print_sweep_table(sweep)
    print(f"Wrote {root / SWEEP_OUTPUT}")
    return sweep


def main() -> None:
    args = parse_args()
    if args.sweep_root:
        analyze_sweep(Path(args.sweep_root))
    else:
        analyze(Path(args.output_dir), args.max_num_seqs)


if __name__ == "__main__":
    main()
