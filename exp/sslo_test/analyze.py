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
SSLO_MODES = ("sslo", "sslo_offload", "sslo_adaptive", "sslo_adaptive_offload")
ALL_MODES = ("baseline",) + SSLO_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--chunk-unit", default="sentence")
    parser.add_argument("--request-rate", type=float, default=0.0)
    return parser.parse_args()


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
    return [float(row[key]) for row in rows if row.get(key) is not None]


def distribution_stats(
    values: list[float],
    percentiles: tuple[int, ...] = (50, 90, 99),
    *,
    include_mean: bool = True,
    include_max: bool = True,
) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {"count": len(values)}
    if include_mean:
        stats["mean"] = statistics.fmean(values) if values else None
    for pct in percentiles:
        stats[f"p{pct}"] = percentile(values, pct)
    if include_max:
        stats["max"] = max(values, default=None)
    return stats


def latency_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    return distribution_stats(numeric_values(rows, key), (50, 90, 99))


def tpot_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return distribution_stats(numeric_values(rows, "tpot"), (50, 90, 99))


def queue_stall_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return distribution_stats(numeric_values(rows, "queue_stall"), (50, 90, 99))


def slack_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    # Exclude chunk_idx == 0: cumulative_slack is fixed at 0.0 for the first
    # chunk by definition (deadline starts there), so including it dilutes
    # both the violation ratio and the distribution stats.
    rows = [r for r in rows if r.get("chunk_idx") not in (None, 0)]
    values = numeric_values(rows, "cumulative_slack")
    neg_count = sum(1 for v in values if v < 0)
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "neg_ratio": (neg_count / len(values)) if values else None,
        "p5": percentile(values, 5),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values, default=None),
    }


def neg_slack_magnitude_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = [
        -float(row["cumulative_slack"])
        for row in rows
        if row.get("cumulative_slack") is not None and float(row["cumulative_slack"]) < 0
    ]
    return distribution_stats(values, (50, 90, 99))


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
        1 for rid in request_has_chunk if not request_has_neg_slack.get(rid, False)
    )
    return {
        "rate": (compliant / total) if total else None,
        "count": compliant,
        "total_requests": total,
    }


def scheduler_queue_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    return distribution_stats(numeric_values(rows, key), (50, 90, 99))


def pending_request_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "time": distribution_stats(numeric_values(rows, "total_pending_time_s"), (50, 90, 99)),
        "intervals": distribution_stats(numeric_values(rows, "num_pending_intervals"), (50, 90)),
    }


def inter_chunk_delay_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    by_request: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = row.get("request_id")
        if request_id is None or row.get("chunk_idx") is None or row.get("end_time_ts") is None:
            continue
        by_request.setdefault(str(request_id), []).append(row)
    delays: list[float] = []
    for request_rows in by_request.values():
        ordered = sorted(request_rows, key=lambda r: int(r["chunk_idx"]))
        for prev, cur in zip(ordered, ordered[1:]):
            cur_end = float(cur["end_time_ts"])
            prev_end = float(prev["end_time_ts"])
            if cur_end >= prev_end:
                delays.append(cur_end - prev_end)
    return distribution_stats(delays, (50, 90, 99))


def h2_rows(
    baseline_rows: list[dict[str, Any]],
    mode_rows: list[dict[str, Any]],
    max_num_seqs: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_new = [r for r in baseline_rows if int(r.get("request_idx", -1)) >= max_num_seqs]
    mode_new = [r for r in mode_rows if int(r.get("request_idx", -1)) >= max_num_seqs]
    if baseline_new and mode_new:
        return "post_cap_arrivals", baseline_new, mode_new
    if baseline_new and not mode_new:
        return "post_cap_arrivals_missing_sslo", baseline_new, mode_new
    return "all_requests_control_fallback", baseline_rows, mode_rows


def scheduler_saturation_stats(rows: list[dict[str, Any]], max_num_seqs: int) -> dict[str, int]:
    def int_field(row: dict[str, Any], key: str, default: int = 0) -> int:
        v = row.get(key)
        return int(v) if v is not None else default

    def combined_field(row: dict[str, Any]) -> int:
        v = row.get("combined")
        if v is not None:
            return int(v)
        return int_field(row, "running") + int_field(row, "pending")

    combined_values = [combined_field(r) for r in rows]
    pending_values = [int_field(r, "pending") for r in rows]
    iterations_above_cap = sum(
        1 for r in rows
        if combined_field(r) > max_num_seqs
        and int_field(r, "pending") > 0
        and int_field(r, "running") <= max_num_seqs
    )
    return {
        "max_combined": max(combined_values, default=0),
        "iterations_above_cap": iterations_above_cap,
        "max_pending": max(pending_values, default=0),
    }


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
    keys = [k for k in ("count", "mean", "p5", "p50", "p90", "p95", "p99", "max") if k in stats]
    rendered = " ".join(f"{k}={format_value(stats[k])}" for k in keys)
    print(f"{label} {metric}: {rendered}")


def analyze(
    output_dir: Path,
    max_num_seqs: int,
    chunk_unit: str = "sentence",
    request_rate: float = 0.0,
) -> dict[str, Any]:
    request_rows_all = read_jsonl(output_dir / "requests.jsonl")
    chunk_rows_all = read_jsonl(output_dir / "chunks.jsonl")
    sched_rows_all = read_jsonl(output_dir / "scheduler_stats.jsonl")
    run_status: dict[str, Any] = {}
    if (output_dir / "run_status.json").exists():
        run_status = json.loads((output_dir / "run_status.json").read_text())

    modes_run = sorted({row["mode"] for row in request_rows_all if "mode" in row})

    req_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in ALL_MODES}
    for row in request_rows_all:
        m = row.get("mode")
        if m in req_by_mode:
            req_by_mode[m].append(row)

    chunk_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in ALL_MODES}
    for row in chunk_rows_all:
        m = row.get("mode")
        if m in chunk_by_mode:
            chunk_by_mode[m].append(row)

    sched_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in SSLO_MODES}
    for row in sched_rows_all:
        m = row.get("mode")
        if m in sched_by_mode:
            sched_by_mode[m].append(row)

    baseline_req = req_by_mode["baseline"]
    is_control = bool(baseline_req) and len(baseline_req) <= max_num_seqs
    queue_stall_available = any(r.get("queue_stall") is not None for r in request_rows_all)

    metrics: dict[str, Any] = {
        "ttft": {}, "tpot": {}, "queue_stall": {}, "slack": {},
        "slo_compliance": {}, "scheduler": {}, "pending": {}, "inter_chunk_delay": {},
    }

    for mode in ALL_MODES:
        req_rows = req_by_mode[mode]
        ch_rows = chunk_by_mode[mode]

        if mode == "baseline":
            cohort, h2_bl, _ = h2_rows(baseline_req, req_rows, max_num_seqs)
            h2_req = h2_bl
        else:
            cohort, _, h2_mode = h2_rows(baseline_req, req_rows, max_num_seqs)
            h2_req = h2_mode

        ttft_pc = latency_stats(h2_req, "ttft")
        ttft_pc["cohort"] = cohort
        metrics["ttft"][mode] = {"all": latency_stats(req_rows, "ttft"), "post_cap": ttft_pc}
        metrics["tpot"][mode] = tpot_stats(req_rows)
        metrics["queue_stall"][mode] = queue_stall_stats(req_rows)
        s = slack_stats(ch_rows)
        mag = neg_slack_magnitude_stats(ch_rows)
        metrics["slack"][mode] = {**s, "magnitude": mag}
        metrics["slo_compliance"][mode] = request_compliance_stats(ch_rows)
        metrics["inter_chunk_delay"][mode] = inter_chunk_delay_stats(ch_rows)
        metrics["pending"][mode] = pending_request_stats(req_rows)

    for mode in SSLO_MODES:
        sched_rows = sched_by_mode[mode]
        metrics["scheduler"][mode] = {
            "running": scheduler_queue_stats(sched_rows, "running"),
            "combined": scheduler_queue_stats(sched_rows, "combined"),
        }

    scheduler_saturation: dict[str, Any] = {}
    for mode in SSLO_MODES:
        if sched_by_mode[mode]:
            scheduler_saturation[mode] = scheduler_saturation_stats(sched_by_mode[mode], max_num_seqs)

    baseline_slack = metrics["slack"].get("baseline", {})
    baseline_ttft_pc = (metrics["ttft"].get("baseline") or {}).get("post_cap", {})

    passes: dict[str, Any] = {"pending_used": {}, "ttft_not_worse": {}, "neg_slack_not_worse": {}}
    for mode in SSLO_MODES:
        pend = (metrics["pending"].get(mode) or {}).get("time", {})
        passes["pending_used"][mode] = (pend.get("count") or 0) > 0

        mode_p90 = (metrics["ttft"].get(mode) or {}).get("post_cap", {}).get("p90")
        bl_p90 = baseline_ttft_pc.get("p90")
        passes["ttft_not_worse"][mode] = (
            mode_p90 is not None and bl_p90 is not None and mode_p90 <= bl_p90 * 1.1
        )

        mode_neg = (metrics["slack"].get(mode) or {}).get("neg_ratio")
        bl_neg = baseline_slack.get("neg_ratio")
        passes["neg_slack_not_worse"][mode] = (
            mode_neg is not None and bl_neg is not None and mode_neg <= bl_neg * 1.1
        )

    summary: dict[str, Any] = {
        "config": {
            "max_num_seqs": max_num_seqs,
            "chunk_unit": chunk_unit,
            "request_rate": request_rate,
            "is_control": is_control,
            "modes_run": modes_run,
        },
        "metrics": metrics,
        "queue_stall_available": queue_stall_available,
        "scheduler_saturation": scheduler_saturation,
        "passes": passes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for mode in ALL_MODES:
        print_stats(mode, "TTFT", metrics["ttft"].get(mode, {}).get("all", {}))
    for mode in ALL_MODES:
        print_stats(mode, "TTFT post_cap", metrics["ttft"].get(mode, {}).get("post_cap", {}))
    for mode in SSLO_MODES:
        print(
            f"{mode} passes: pending_used={passes['pending_used'].get(mode)} "
            f"ttft_not_worse={passes['ttft_not_worse'].get(mode)} "
            f"neg_slack_not_worse={passes['neg_slack_not_worse'].get(mode)}"
        )
    return summary


def main() -> None:
    args = parse_args()
    analyze(
        Path(args.output_dir),
        args.max_num_seqs,
        chunk_unit=args.chunk_unit,
        request_rate=args.request_rate,
    )


if __name__ == "__main__":
    main()
