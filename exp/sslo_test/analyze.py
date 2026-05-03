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
SSLO_VARIANTS = ("sslo", "sslo_adaptive")
EXTRA_METRIC_PREFIXES = (
    "h2_ttft_p99",
    "h2_ttft_max",
    "queue_stall_p99",
    "queue_stall_max",
    "slo_compliance_rate",
    "slo_compliance_count",
    "slo_total_requests",
    "neg_slack_mag_p50",
    "neg_slack_mag_p90",
    "neg_slack_mag_p99",
    "neg_slack_mag_max",
    "running_mean",
    "running_p50",
    "running_p99",
    "combined_mean",
    "combined_p50",
    "combined_p99",
    "pending_time_p50",
    "pending_time_p90",
    "pending_time_p99",
    "pending_intervals_p50",
    "pending_intervals_p90",
    "inter_chunk_delay_p50",
    "inter_chunk_delay_p90",
    "inter_chunk_delay_p99",
    "inter_chunk_delay_max",
)


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
        "max": max(values, default=None),
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
        "p99": percentile(values, 99),
        "max": max(values, default=None),
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


def distribution_stats(
    values: list[float],
    percentiles: tuple[int, ...],
    *,
    include_mean: bool = False,
    include_max: bool = False,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {"count": len(values)}
    if include_mean:
        stats["mean"] = statistics.fmean(values) if values else None
    for pct in percentiles:
        stats[f"p{pct}"] = percentile(values, pct)
    if include_max:
        stats["max"] = max(values, default=None)
    return stats


def request_compliance_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    request_has_chunk: dict[str, bool] = {}
    request_has_neg_slack: dict[str, bool] = {}
    for row in rows:
        request_id = row.get("request_id")
        if request_id is None:
            continue
        request_id = str(request_id)
        request_has_chunk[request_id] = True
        slack = row.get("cumulative_slack")
        if slack is not None and float(slack) < 0:
            request_has_neg_slack[request_id] = True

    total = len(request_has_chunk)
    compliant = sum(
        1
        for request_id in request_has_chunk
        if not request_has_neg_slack.get(request_id, False)
    )
    return {
        "rate": (compliant / total) if total else None,
        "count": compliant,
        "total_requests": total,
    }


def neg_slack_magnitude_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = [
        -float(row["cumulative_slack"])
        for row in rows
        if row.get("cumulative_slack") is not None
        and float(row["cumulative_slack"]) < 0
    ]
    return distribution_stats(values, (50, 90, 99), include_max=True)


def scheduler_queue_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = numeric_values(rows, key)
    return distribution_stats(values, (50, 99), include_mean=True)


def pending_request_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "time": distribution_stats(
            numeric_values(rows, "total_pending_time_s"), (50, 90, 99)
        ),
        "intervals": distribution_stats(
            numeric_values(rows, "num_pending_intervals"), (50, 90)
        ),
    }


def inter_chunk_delay_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    by_request: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if request_id is None:
            continue
        if row.get("chunk_idx") is None or row.get("end_time_ts") is None:
            continue
        by_request.setdefault(str(request_id), []).append(row)

    delays: list[float] = []
    for request_rows in by_request.values():
        ordered = sorted(request_rows, key=lambda row: int(row["chunk_idx"]))
        for prev, cur in zip(ordered, ordered[1:]):
            cur_end = float(cur["end_time_ts"])
            prev_end = float(prev["end_time_ts"])
            if cur_end >= prev_end:
                delays.append(cur_end - prev_end)
    return distribution_stats(delays, (50, 90, 99), include_max=True)


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
    keys = [key for key in ("count", "mean", "p5", "p50", "p90", "p95", "p99", "max")
            if key in stats]
    rendered = " ".join(f"{key}={format_value(stats[key])}" for key in keys)
    print(f"{label} {metric}: {rendered}")


def control_case(max_num_seqs: int, baseline_rows: list[dict[str, Any]]) -> bool:
    return bool(baseline_rows) and len(baseline_rows) <= max_num_seqs


def variant_file(run_kind: str, suffix: str) -> str:
    return f"{run_kind}_{suffix}.jsonl"


def expected_files(run_status: dict[str, Any]) -> tuple[str, ...]:
    files = [BASELINE_TTFT, BASELINE_CHUNKS]
    expected_variants = [
        variant for variant in SSLO_VARIANTS if variant == "sslo" or variant in run_status
    ]
    for variant in expected_variants:
        files.extend(
            (
                variant_file(variant, "ttft"),
                variant_file(variant, "chunks"),
                variant_file(variant, "stats"),
            )
        )
    return tuple(files)


def extra_metric_fields(output_dir: Path, max_num_seqs: int) -> dict[str, Any]:
    baseline_rows = read_jsonl(output_dir / BASELINE_TTFT)
    baseline_chunks = read_jsonl(output_dir / BASELINE_CHUNKS)
    variant_rows = {
        variant: read_jsonl(output_dir / variant_file(variant, "ttft"))
        for variant in SSLO_VARIANTS
    }
    chunks_by_mode = {
        "baseline": baseline_chunks,
        **{
            variant: read_jsonl(output_dir / variant_file(variant, "chunks"))
            for variant in SSLO_VARIANTS
        },
    }
    stats_by_mode = {
        variant: read_jsonl(output_dir / variant_file(variant, "stats"))
        for variant in SSLO_VARIANTS
    }
    rows_by_mode = {
        "baseline": baseline_rows,
        **variant_rows,
    }

    fields: dict[str, Any] = {}
    for mode, rows in rows_by_mode.items():
        mode_present = (
            (output_dir / BASELINE_TTFT).exists()
            if mode == "baseline"
            else any(
                (output_dir / variant_file(mode, suffix)).exists()
                for suffix in ("ttft", "chunks", "stats")
            )
        )
        if not mode_present:
            for prefix in EXTRA_METRIC_PREFIXES:
                fields[f"{prefix}_{mode}"] = None
            continue
        if mode == "baseline":
            _cohort, h2_mode_rows, _variant_rows = h2_rows(
                baseline_rows, variant_rows["sslo"], max_num_seqs
            )
        else:
            _cohort, _baseline_h2_rows, h2_mode_rows = h2_rows(
                baseline_rows, rows, max_num_seqs
            )
        h2_ttft = latency_stats(h2_mode_rows, "ttft")
        queue_stall = queue_stall_stats(rows)
        chunks = chunks_by_mode[mode]
        compliance = request_compliance_stats(chunks)
        neg_slack_mag = neg_slack_magnitude_stats(chunks)
        pending = pending_request_stats(rows)
        inter_chunk = inter_chunk_delay_stats(chunks)

        fields[f"h2_ttft_p99_{mode}"] = h2_ttft["p99"]
        fields[f"h2_ttft_max_{mode}"] = h2_ttft["max"]
        fields[f"queue_stall_p99_{mode}"] = queue_stall["p99"]
        fields[f"queue_stall_max_{mode}"] = queue_stall["max"]
        fields[f"slo_compliance_rate_{mode}"] = compliance["rate"]
        fields[f"slo_compliance_count_{mode}"] = compliance["count"]
        fields[f"slo_total_requests_{mode}"] = compliance["total_requests"]
        fields[f"neg_slack_mag_p50_{mode}"] = neg_slack_mag["p50"]
        fields[f"neg_slack_mag_p90_{mode}"] = neg_slack_mag["p90"]
        fields[f"neg_slack_mag_p99_{mode}"] = neg_slack_mag["p99"]
        fields[f"neg_slack_mag_max_{mode}"] = neg_slack_mag["max"]
        fields[f"pending_time_p50_{mode}"] = pending["time"]["p50"]
        fields[f"pending_time_p90_{mode}"] = pending["time"]["p90"]
        fields[f"pending_time_p99_{mode}"] = pending["time"]["p99"]
        fields[f"pending_intervals_p50_{mode}"] = pending["intervals"]["p50"]
        fields[f"pending_intervals_p90_{mode}"] = pending["intervals"]["p90"]
        fields[f"inter_chunk_delay_p50_{mode}"] = inter_chunk["p50"]
        fields[f"inter_chunk_delay_p90_{mode}"] = inter_chunk["p90"]
        fields[f"inter_chunk_delay_p99_{mode}"] = inter_chunk["p99"]
        fields[f"inter_chunk_delay_max_{mode}"] = inter_chunk["max"]

        if mode == "baseline":
            running = {"mean": None, "p50": None, "p99": None}
            combined = {"mean": None, "p50": None, "p99": None}
        else:
            running = scheduler_queue_stats(stats_by_mode[mode], "running")
            combined = scheduler_queue_stats(stats_by_mode[mode], "combined")
        fields[f"running_mean_{mode}"] = running["mean"]
        fields[f"running_p50_{mode}"] = running["p50"]
        fields[f"running_p99_{mode}"] = running["p99"]
        fields[f"combined_mean_{mode}"] = combined["mean"]
        fields[f"combined_p50_{mode}"] = combined["p50"]
        fields[f"combined_p99_{mode}"] = combined["p99"]

    return fields


def clear_mode_fields(summary: dict[str, Any], mode: str) -> None:
    for prefix in EXTRA_METRIC_PREFIXES:
        summary[f"{prefix}_{mode}"] = None


def h1_stats(
    rows: list[dict[str, Any]],
    max_num_seqs: int,
) -> dict[str, int | bool]:
    def int_field(row: dict[str, Any], key: str, default: int = 0) -> int:
        value = row.get(key)
        if value is None:
            return default
        return int(value)

    def combined_field(row: dict[str, Any]) -> int:
        value = row.get("combined")
        if value is not None:
            return int(value)
        return int_field(row, "running") + int_field(row, "pending")

    combined_values = [
        combined_field(row)
        for row in rows
    ]
    pending_values = [int_field(row, "pending") for row in rows]
    iterations_above_cap = sum(
        1
        for row in rows
        if combined_field(row) > max_num_seqs
        and int_field(row, "pending") > 0
        and int_field(row, "running") <= max_num_seqs
    )
    return {
        "pending_plus_running_exceeds_cap": iterations_above_cap > 0,
        "max_combined": max(combined_values, default=0),
        "iterations_above_cap": iterations_above_cap,
        "max_pending": max(pending_values, default=0),
    }


def print_slack(label: str, slack: dict[str, float | int | None]) -> None:
    print(
        f"{label} Slack: "
        f"count={slack['count']} "
        f"neg_slack_ratio={format_value(slack['neg_slack_ratio'])} "
        f"p5={format_value(slack['p5'])} "
        f"p50={format_value(slack['p50'])} "
        f"p95={format_value(slack['p95'])}"
    )


def analyze(output_dir: Path, max_num_seqs: int) -> dict[str, Any]:
    baseline_rows = read_jsonl(output_dir / BASELINE_TTFT)
    baseline_chunks = read_jsonl(output_dir / BASELINE_CHUNKS)
    run_status = read_json(output_dir / "run_status.json")
    missing_files = [
        name
        for name in expected_files(run_status)
        if not (output_dir / name).exists()
    ]

    baseline_all_ttft = latency_stats(baseline_rows, "ttft")
    baseline_tpot = tpot_stats(baseline_rows)
    baseline_queue_stall = queue_stall_stats(baseline_rows)
    baseline_slack = slack_stats(baseline_chunks)
    baseline_compliance = request_compliance_stats(baseline_chunks)
    baseline_neg_slack_mag = neg_slack_magnitude_stats(baseline_chunks)
    baseline_pending = pending_request_stats(baseline_rows)
    baseline_inter_chunk = inter_chunk_delay_stats(baseline_chunks)

    variant_rows = {
        variant: read_jsonl(output_dir / variant_file(variant, "ttft"))
        for variant in SSLO_VARIANTS
    }
    variant_chunks = {
        variant: read_jsonl(output_dir / variant_file(variant, "chunks"))
        for variant in SSLO_VARIANTS
    }
    variant_stats_rows = {
        variant: read_jsonl(output_dir / variant_file(variant, "stats"))
        for variant in SSLO_VARIANTS
    }
    variant_present = {
        variant: any(
            (output_dir / variant_file(variant, suffix)).exists()
            for suffix in ("ttft", "chunks", "stats")
        )
        for variant in SSLO_VARIANTS
    }
    variant_all_ttft = {
        variant: latency_stats(rows, "ttft")
        for variant, rows in variant_rows.items()
    }
    variant_tpot = {
        variant: tpot_stats(rows)
        for variant, rows in variant_rows.items()
    }
    variant_queue_stall = {
        variant: queue_stall_stats(rows)
        for variant, rows in variant_rows.items()
    }
    variant_slack = {
        variant: slack_stats(rows)
        for variant, rows in variant_chunks.items()
    }
    variant_compliance = {
        variant: request_compliance_stats(rows)
        for variant, rows in variant_chunks.items()
    }
    variant_neg_slack_mag = {
        variant: neg_slack_magnitude_stats(rows)
        for variant, rows in variant_chunks.items()
    }
    variant_running = {
        variant: scheduler_queue_stats(rows, "running")
        for variant, rows in variant_stats_rows.items()
    }
    variant_combined = {
        variant: scheduler_queue_stats(rows, "combined")
        for variant, rows in variant_stats_rows.items()
    }
    variant_pending = {
        variant: pending_request_stats(rows)
        for variant, rows in variant_rows.items()
    }
    variant_inter_chunk = {
        variant: inter_chunk_delay_stats(rows)
        for variant, rows in variant_chunks.items()
    }
    variant_h1 = {
        variant: h1_stats(rows, max_num_seqs)
        for variant, rows in variant_stats_rows.items()
    }
    queue_stall_available = baseline_queue_stall["count"] > 0 or any(
        stats["count"] > 0 for stats in variant_queue_stall.values()
    )
    if (baseline_rows or any(variant_rows.values())) and not queue_stall_available:
        print(
            "Warning: RequestOutput.metrics queue stall timestamps were not "
            "available; queue_stall summary columns will be n/a."
        )

    h2_by_variant = {}
    for variant, rows in variant_rows.items():
        h2_cohort, baseline_h2_rows, variant_h2_rows = h2_rows(
            baseline_rows, rows, max_num_seqs
        )
        baseline_h2_ttft = latency_stats(baseline_h2_rows, "ttft")
        variant_h2_ttft = latency_stats(variant_h2_rows, "ttft")
        h2_by_variant[variant] = {
            "cohort": h2_cohort,
            "baseline": baseline_h2_ttft,
            "variant": variant_h2_ttft,
            "p50_change": pct_change(variant_h2_ttft["p50"], baseline_h2_ttft["p50"]),
            "pass": (
                leq_or_equal(variant_h2_ttft["p50"], baseline_h2_ttft["p50"])
                if variant_present[variant]
                else None
            ),
        }
    baseline_h2_ttft = h2_by_variant["sslo"]["baseline"]
    h2_cohort = h2_by_variant["sslo"]["cohort"]

    is_control = control_case(max_num_seqs, baseline_rows)
    run_complete = (
        not missing_files
        and bool(baseline_rows)
        and bool(variant_rows["sslo"])
        and (
            "sslo_adaptive" not in run_status
            or bool(variant_rows["sslo_adaptive"])
        )
        and all(value == 0 for value in run_status.values())
    )

    summary = {
        "max_num_seqs": max_num_seqs,
        "control_case": is_control,
        "h2_ttft_p50_baseline": baseline_h2_ttft["p50"],
        "h2_ttft_p90_baseline": baseline_h2_ttft["p90"],
        "h2_ttft_p99_baseline": baseline_h2_ttft["p99"],
        "h2_ttft_max_baseline": baseline_h2_ttft["max"],
        "h2_ttft_cohort": h2_cohort,
        "h2_ttft_count_baseline": baseline_h2_ttft["count"],
        "h3_slack_neg_ratio_baseline": baseline_slack["neg_slack_ratio"],
        "h4_tpot_p50_baseline": baseline_tpot["p50"],
        "queue_stall_p50_baseline": baseline_queue_stall["p50"],
        "queue_stall_p90_baseline": baseline_queue_stall["p90"],
        "queue_stall_p99_baseline": baseline_queue_stall["p99"],
        "queue_stall_max_baseline": baseline_queue_stall["max"],
        "slo_compliance_rate_baseline": baseline_compliance["rate"],
        "slo_compliance_count_baseline": baseline_compliance["count"],
        "slo_total_requests_baseline": baseline_compliance["total_requests"],
        "neg_slack_mag_p50_baseline": baseline_neg_slack_mag["p50"],
        "neg_slack_mag_p90_baseline": baseline_neg_slack_mag["p90"],
        "neg_slack_mag_p99_baseline": baseline_neg_slack_mag["p99"],
        "neg_slack_mag_max_baseline": baseline_neg_slack_mag["max"],
        "running_mean_baseline": None,
        "running_p50_baseline": None,
        "running_p99_baseline": None,
        "combined_mean_baseline": None,
        "combined_p50_baseline": None,
        "combined_p99_baseline": None,
        "pending_time_p50_baseline": baseline_pending["time"]["p50"],
        "pending_time_p90_baseline": baseline_pending["time"]["p90"],
        "pending_time_p99_baseline": baseline_pending["time"]["p99"],
        "pending_intervals_p50_baseline": baseline_pending["intervals"]["p50"],
        "pending_intervals_p90_baseline": baseline_pending["intervals"]["p90"],
        "inter_chunk_delay_p50_baseline": baseline_inter_chunk["p50"],
        "inter_chunk_delay_p90_baseline": baseline_inter_chunk["p90"],
        "inter_chunk_delay_p99_baseline": baseline_inter_chunk["p99"],
        "inter_chunk_delay_max_baseline": baseline_inter_chunk["max"],
        "ttft": {
            "baseline_all": baseline_all_ttft,
            "baseline_h2": baseline_h2_ttft,
        },
        "tpot": {
            "baseline": baseline_tpot,
        },
        "queue_stall": {
            "baseline": baseline_queue_stall,
            "available": queue_stall_available,
        },
        "slack": {
            "baseline": baseline_slack,
        },
        "slo_compliance": {
            "baseline": baseline_compliance,
        },
        "neg_slack_magnitude": {
            "baseline": baseline_neg_slack_mag,
        },
        "running": {
            "baseline": {"mean": None, "p50": None, "p99": None},
        },
        "combined": {
            "baseline": {"mean": None, "p50": None, "p99": None},
        },
        "pending": {
            "baseline": baseline_pending,
        },
        "inter_chunk_delay": {
            "baseline": baseline_inter_chunk,
        },
        "run_complete": run_complete,
        "missing_files": missing_files,
        "run_status": run_status,
    }
    for variant in SSLO_VARIANTS:
        h1 = variant_h1[variant]
        h2 = h2_by_variant[variant]
        tpot = variant_tpot[variant]
        queue_stall = variant_queue_stall[variant]
        slack = variant_slack[variant]
        compliance = variant_compliance[variant]
        neg_slack_mag = variant_neg_slack_mag[variant]
        running = variant_running[variant]
        combined = variant_combined[variant]
        pending = variant_pending[variant]
        inter_chunk = variant_inter_chunk[variant]
        h1_pass = (
            h1["pending_plus_running_exceeds_cap"] if variant_present[variant] else None
        )
        h3_pass = (
            leq_or_equal(slack["neg_slack_ratio"], baseline_slack["neg_slack_ratio"])
            if variant_present[variant]
            else None
        )

        summary[f"h1_{variant}_pending_plus_running_exceeds_cap"] = h1_pass
        summary[f"h1_{variant}_note"] = (
            "control_case_no_waiting_queue_pressure" if is_control else None
        )
        summary[f"h1_{variant}_max_combined"] = (
            h1["max_combined"] if variant_present[variant] else None
        )
        summary[f"h1_{variant}_iterations_above_cap"] = (
            h1["iterations_above_cap"] if variant_present[variant] else None
        )
        summary[f"h1_{variant}_max_pending"] = (
            h1["max_pending"] if variant_present[variant] else None
        )
        summary[f"h2_ttft_p50_{variant}"] = h2["variant"]["p50"]
        summary[f"h2_ttft_p50_pct_change_{variant}"] = h2["p50_change"]
        summary[f"h2_ttft_p90_{variant}"] = h2["variant"]["p90"]
        summary[f"h2_ttft_p99_{variant}"] = h2["variant"]["p99"]
        summary[f"h2_ttft_max_{variant}"] = h2["variant"]["max"]
        summary[f"h2_ttft_cohort_{variant}"] = h2["cohort"]
        summary[f"h2_ttft_count_{variant}"] = h2["variant"]["count"]
        summary[f"h2_ttft_p50_not_worse_{variant}"] = h2["pass"]
        summary[f"h3_slack_neg_ratio_{variant}"] = slack["neg_slack_ratio"]
        summary[f"h3_slack_neg_ratio_not_worse_{variant}"] = h3_pass
        summary[f"h4_tpot_p50_{variant}"] = tpot["p50"]
        summary[f"h4_tpot_p50_pct_change_{variant}"] = pct_change(
            tpot["p50"], baseline_tpot["p50"]
        )
        summary[f"queue_stall_p50_{variant}"] = queue_stall["p50"]
        summary[f"queue_stall_p90_{variant}"] = queue_stall["p90"]
        summary[f"queue_stall_p99_{variant}"] = queue_stall["p99"]
        summary[f"queue_stall_max_{variant}"] = queue_stall["max"]
        summary[f"queue_stall_p50_pct_change_{variant}"] = pct_change(
            queue_stall["p50"], baseline_queue_stall["p50"]
        )
        summary[f"slo_compliance_rate_{variant}"] = compliance["rate"]
        summary[f"slo_compliance_count_{variant}"] = compliance["count"]
        summary[f"slo_total_requests_{variant}"] = compliance["total_requests"]
        summary[f"neg_slack_mag_p50_{variant}"] = neg_slack_mag["p50"]
        summary[f"neg_slack_mag_p90_{variant}"] = neg_slack_mag["p90"]
        summary[f"neg_slack_mag_p99_{variant}"] = neg_slack_mag["p99"]
        summary[f"neg_slack_mag_max_{variant}"] = neg_slack_mag["max"]
        summary[f"running_mean_{variant}"] = running["mean"]
        summary[f"running_p50_{variant}"] = running["p50"]
        summary[f"running_p99_{variant}"] = running["p99"]
        summary[f"combined_mean_{variant}"] = combined["mean"]
        summary[f"combined_p50_{variant}"] = combined["p50"]
        summary[f"combined_p99_{variant}"] = combined["p99"]
        summary[f"pending_time_p50_{variant}"] = pending["time"]["p50"]
        summary[f"pending_time_p90_{variant}"] = pending["time"]["p90"]
        summary[f"pending_time_p99_{variant}"] = pending["time"]["p99"]
        summary[f"pending_intervals_p50_{variant}"] = pending["intervals"]["p50"]
        summary[f"pending_intervals_p90_{variant}"] = pending["intervals"]["p90"]
        summary[f"inter_chunk_delay_p50_{variant}"] = inter_chunk["p50"]
        summary[f"inter_chunk_delay_p90_{variant}"] = inter_chunk["p90"]
        summary[f"inter_chunk_delay_p99_{variant}"] = inter_chunk["p99"]
        summary[f"inter_chunk_delay_max_{variant}"] = inter_chunk["max"]
        summary["ttft"][f"{variant}_all"] = variant_all_ttft[variant]
        summary["ttft"][f"{variant}_h2"] = h2["variant"]
        summary["tpot"][variant] = tpot
        summary["queue_stall"][variant] = queue_stall
        summary["slack"][variant] = slack
        summary["slo_compliance"][variant] = compliance
        summary["neg_slack_magnitude"][variant] = neg_slack_mag
        summary["running"][variant] = running
        summary["combined"][variant] = combined
        summary["pending"][variant] = pending
        summary["inter_chunk_delay"][variant] = inter_chunk
        if not variant_present[variant]:
            clear_mode_fields(summary, variant)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print_stats("Baseline", "TTFT", baseline_all_ttft)
    for variant in SSLO_VARIANTS:
        print_stats(variant.upper(), "TTFT", variant_all_ttft[variant])
    print(f"H2 TTFT cohort: {h2_cohort}")
    print_stats("Baseline H2 cohort", "TTFT", baseline_h2_ttft)
    for variant in SSLO_VARIANTS:
        print_stats(
            f"{variant.upper()} H2 cohort",
            "TTFT",
            h2_by_variant[variant]["variant"],
        )
    print_stats("Baseline", "TPOT", baseline_tpot)
    for variant in SSLO_VARIANTS:
        print_stats(variant.upper(), "TPOT", variant_tpot[variant])
    print_stats("Baseline", "Queue stall", baseline_queue_stall)
    for variant in SSLO_VARIANTS:
        print_stats(variant.upper(), "Queue stall", variant_queue_stall[variant])
    print_slack("Baseline", baseline_slack)
    for variant in SSLO_VARIANTS:
        print_slack(variant.upper(), variant_slack[variant])
    if is_control:
        print("Control case: num_prompts <= max_num_seqs, so H1 is expected to fail.")
    if not summary["run_complete"]:
        print(
            "Run completeness: INCOMPLETE "
            f"missing_files={missing_files} run_status={run_status}"
        )
    for variant in SSLO_VARIANTS:
        print(
            f"{variant.upper()} p50 TTFT percent change: "
            f"{format_value(h2_by_variant[variant]['p50_change'])}%"
        )
        print(
            f"{variant.upper()} scheduler stats: "
            "max_running_plus_pending="
            f"{format_value(summary[f'h1_{variant}_max_combined'])} "
            "iterations_above_cap="
            f"{format_value(summary[f'h1_{variant}_iterations_above_cap'])} "
            f"max_pending={format_value(summary[f'h1_{variant}_max_pending'])}"
        )
        print(
            f"{variant.upper()} H1 verdict: "
            f"{format_bool(summary[f'h1_{variant}_pending_plus_running_exceeds_cap'])}"
        )
        print(
            f"{variant.upper()} H2-TTFT verdict: "
            f"{format_bool(h2_by_variant[variant]['pass'])}"
        )
        print(
            f"{variant.upper()} H3-Slack verdict: "
            f"{format_bool(summary[f'h3_slack_neg_ratio_not_worse_{variant}'])}"
        )
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
        "H1_sslo",
        "H1_adapt",
        "H2_sslo",
        "H2_adapt",
        "H3_sslo",
        "H3_adapt",
        "base_ttft_p50",
        "sslo_ttft_p50",
        "adapt_ttft_p50",
        "base_neg_slack",
        "sslo_neg_slack",
        "adapt_neg_slack",
        "base_tpot_p50",
        "sslo_tpot_p50",
        "adapt_tpot_p50",
        "base_qstall_p50",
        "sslo_qstall_p50",
        "adapt_qstall_p50",
        "base_qstall_p90",
        "sslo_qstall_p90",
        "adapt_qstall_p90",
        "sslo_qstall_p50_pct",
        "adapt_qstall_p50_pct",
    ]
    data = []
    for row in rows:
        data.append([
            str(row.get("max_num_seqs", "n/a")),
            "yes" if row.get("control_case") else "no",
            format_bool(row.get("h1_sslo_pending_plus_running_exceeds_cap")),
            format_bool(row.get("h1_sslo_adaptive_pending_plus_running_exceeds_cap")),
            format_bool(row.get("h2_ttft_p50_not_worse_sslo")),
            format_bool(row.get("h2_ttft_p50_not_worse_sslo_adaptive")),
            format_bool(row.get("h3_slack_neg_ratio_not_worse_sslo")),
            format_bool(row.get("h3_slack_neg_ratio_not_worse_sslo_adaptive")),
            format_value(row.get("h2_ttft_p50_baseline")),
            format_value(row.get("h2_ttft_p50_sslo")),
            format_value(row.get("h2_ttft_p50_sslo_adaptive")),
            format_value(row.get("h3_slack_neg_ratio_baseline")),
            format_value(row.get("h3_slack_neg_ratio_sslo")),
            format_value(row.get("h3_slack_neg_ratio_sslo_adaptive")),
            format_value(row.get("h4_tpot_p50_baseline")),
            format_value(row.get("h4_tpot_p50_sslo")),
            format_value(row.get("h4_tpot_p50_sslo_adaptive")),
            format_value(row.get("queue_stall_p50_baseline")),
            format_value(row.get("queue_stall_p50_sslo")),
            format_value(row.get("queue_stall_p50_sslo_adaptive")),
            format_value(row.get("queue_stall_p90_baseline")),
            format_value(row.get("queue_stall_p90_sslo")),
            format_value(row.get("queue_stall_p90_sslo_adaptive")),
            format_value(row.get("queue_stall_p50_pct_change_sslo")),
            format_value(row.get("queue_stall_p50_pct_change_sslo_adaptive")),
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
