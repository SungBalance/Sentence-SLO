#!/usr/bin/env python3
"""Summarize SSLO scheduler end-to-end validation runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from jsonl_utils import read_jsonl
from metrics_utils import distribution_stats, numeric_values, percentile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import progress_metrics as pm
from validity import validate_run

MAX_NUM_SEQS = 64
DEFAULT_OUTPUT_DIR = "exp/run_sslo/output"
SSLO_MODES = ("sslo", "sslo_offload", "sslo_adaptive", "sslo_adaptive_offload")
ALL_MODES = ("baseline",) + SSLO_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--chunk-unit", default="sentence")
    parser.add_argument("--request-rate", type=float, default=0.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--generation-max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def dist_for_key(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    """Standard mean/p50/p90/p95/p99/max distribution over rows[*][key]."""
    return distribution_stats(numeric_values(rows, key))


def slack_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    # Exclude chunk_idx == 0: chunk_slack is fixed at 0.0 for the first
    # chunk by definition (deadline starts there), so including it dilutes
    # both the violation ratio and the distribution stats.
    rows = [r for r in rows if r.get("chunk_idx") not in (None, 0)]
    values = numeric_values(rows, "chunk_slack")
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


def stall_time_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Per-chunk stall = max(0, -slack) statistics over chunks (idx >= 1).

    Returns:
      - mean        : average stall over chunks where stall > 0 (violated).
      - mean_with_0 : average stall over ALL chunks (idx >= 1), counting
                      on-time chunks as 0. Reflects overall slack burn.
      - p50, p90, p99 : percentiles of stall across ALL chunks (idx >= 1),
                        with on-time chunks contributing 0.
    """
    rows = [r for r in rows if r.get("chunk_idx") not in (None, 0)
            and r.get("chunk_slack") is not None]
    stalls_all = [max(0.0, -float(row["chunk_slack"])) for row in rows]
    stalls_violated = [s for s in stalls_all if s > 0]
    mean_violated = (
        statistics.fmean(stalls_violated) if stalls_violated else None)
    mean_with_0 = (
        statistics.fmean(stalls_all) if stalls_all else None)
    return {
        "count": len(stalls_all),
        "violated_count": len(stalls_violated),
        "mean": mean_violated,
        "mean_with_0": mean_with_0,
        "p50": percentile(stalls_all, 50),
        "p90": percentile(stalls_all, 90),
        "p99": percentile(stalls_all, 99),
    }


def prediction_ratio_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Per-chunk ratio of actual / predicted token count.

    Skips chunk_idx == 0 (predictor has no history yet) and rows where
    expected_len is None or non-positive. ratio == 1 means perfect
    prediction; > 1 means actual exceeded prediction (under-estimate).
    """
    values: list[float] = []
    for row in rows:
        if row.get("chunk_idx") in (None, 0):
            continue
        actual = row.get("num_token")
        expected = row.get("expected_len")
        if actual is None or expected is None:
            continue
        try:
            actual_f = float(actual)
            expected_f = float(expected)
        except (TypeError, ValueError):
            continue
        if expected_f <= 0:
            continue
        values.append(actual_f / expected_f)
    return distribution_stats(values)


def request_compliance_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    seen: set[str] = set()
    violated: set[str] = set()
    for row in rows:
        rid = row.get("request_id")
        if rid is None:
            continue
        rid = str(rid)
        seen.add(rid)
        slack = row.get("chunk_slack")
        if slack is not None and float(slack) < 0:
            violated.add(rid)
    total = len(seen)
    compliant = total - len(violated)
    return {
        "rate": (compliant / total) if total else None,
        "count": compliant,
        "total_requests": total,
    }


def pending_request_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "time": dist_for_key(rows, "total_pending_time_s"),
        "intervals": distribution_stats(
            numeric_values(rows, "num_pending_iters_per_request")),
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
    return distribution_stats(delays)


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
        v = row.get("num_handling_users")
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
    model: str | None = None,
    generation_max_tokens: int | None = None,
    max_model_len: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    # New layout: output_dir is a single (mode, run) directory:
    #   <label>/<chunk>/seqs_<n>/rate_<r>/<mode>/run_<i>/
    # Each holds requests.jsonl, chunks.jsonl, scheduler_stats.jsonl,
    # sslo_config.json — all rows already tagged with `mode` by run_test.py.
    # Mode is detected from the parent directory name.
    mode = output_dir.parent.name
    if mode not in ALL_MODES:
        raise ValueError(
            f"output_dir parent must be a mode in {ALL_MODES}; "
            f"got parent={mode!r} for {output_dir}")

    request_rows_mode = read_jsonl(output_dir / "requests.jsonl")
    chunk_rows_mode = read_jsonl(output_dir / "chunks.jsonl")
    sched_rows_mode = read_jsonl(output_dir / "scheduler_stats.jsonl")
    # scheduler_stats.jsonl is written by the scheduler with no mode awareness;
    # tag rows here for the per-mode split downstream.
    for row in sched_rows_mode:
        row.setdefault("mode", mode)

    sslo_config_by_mode: dict[str, dict[str, Any]] = {}
    cfg_path = output_dir / "sslo_config.json"
    if cfg_path.exists():
        try:
            sslo_config_by_mode[mode] = json.loads(cfg_path.read_text()) or {}
        except json.JSONDecodeError:
            sslo_config_by_mode[mode] = {}

    run_status: dict[str, Any] = {mode: 0}

    request_rows_all = list(request_rows_mode)
    chunk_rows_all = list(chunk_rows_mode)
    sched_rows_all = list(sched_rows_mode)

    modes_run = [mode]

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

    # SSLO diagnostics (`scheduler` key) only fill for SSLO_MODES, but
    # handling_users / measurement_window are needed for baseline too
    # (capacity comparison), so bucket scheduler rows for ALL_MODES.
    sched_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in ALL_MODES}
    for row in sched_rows_all:
        m = row.get("mode")
        if m in sched_by_mode:
            sched_by_mode[m].append(row)

    baseline_req = req_by_mode["baseline"]
    is_control = bool(baseline_req) and len(baseline_req) <= max_num_seqs
    queue_stall_available = any(r.get("queue_stall") is not None for r in request_rows_all)

    metrics: dict[str, Any] = {
        "ttft": {}, "ttfc": {}, "tpot": {}, "queue_stall": {}, "slack": {},
        "slo_compliance": {}, "scheduler": {}, "pending": {}, "inter_chunk_delay": {},
        "prediction_ratio": {}, "stall_time": {},
        "progress_request": {}, "workload": {}, "throughput": {},
        "cp_slo_violation": {}, "measurement_window": {}, "handling_users": {},
        # Phase 5: new top-level section for merged-interval CP-SLO metrics.
        "cpslo": {},
    }

    # SSLO Phase 6: load run_meta.json early so throughput_stats can use
    # the canonical full-run wall-time (measurement_start_ts / _end_ts).
    run_meta: dict[str, Any] = {}
    meta_path = output_dir / "run_meta.json"
    if meta_path.exists():
        try:
            run_meta = json.loads(meta_path.read_text()) or {}
        except json.JSONDecodeError:
            run_meta = {}

    for mode in ALL_MODES:
        req_rows = req_by_mode[mode]
        ch_rows = chunk_by_mode[mode]

        if mode == "baseline":
            cohort, h2_bl, _ = h2_rows(baseline_req, req_rows, max_num_seqs)
            h2_req = h2_bl
        else:
            cohort, _, h2_mode = h2_rows(baseline_req, req_rows, max_num_seqs)
            h2_req = h2_mode

        ttft_pc = dist_for_key(h2_req, "ttft")
        ttft_pc["cohort"] = cohort
        metrics["ttft"][mode] = {"all": dist_for_key(req_rows, "ttft"), "post_cap": ttft_pc}
        metrics["ttfc"][mode] = dist_for_key(req_rows, "ttfc")
        metrics["tpot"][mode] = dist_for_key(req_rows, "tpot")
        metrics["queue_stall"][mode] = dist_for_key(req_rows, "queue_stall")
        metrics["slack"][mode] = slack_stats(ch_rows)
        metrics["stall_time"][mode] = stall_time_stats(ch_rows)
        metrics["slo_compliance"][mode] = request_compliance_stats(ch_rows)
        metrics["inter_chunk_delay"][mode] = inter_chunk_delay_stats(ch_rows)
        metrics["pending"][mode] = pending_request_stats(req_rows)
        metrics["prediction_ratio"][mode] = prediction_ratio_stats(ch_rows)

    # Scheduler stats (running / num_handling_users) emit for ALL modes,
    # including baseline (which now also routes through schedule_sslo via
    # method="baseline" so scheduler_stats.jsonl is populated).
    for mode in ALL_MODES:
        sched_rows = sched_by_mode[mode]
        # Include min in `num_handling_users` so summary captures the
        # running+pending floor alongside mean/max.
        nhu_dist = distribution_stats(
            numeric_values(sched_rows, "num_handling_users"),
            include_min=True)
        metrics["scheduler"][mode] = {
            "running": dist_for_key(sched_rows, "running"),
            "num_handling_users": nhu_dist,
        }

    windows: dict[str, tuple] = {}
    for mode in ALL_MODES:
        req_rows = req_by_mode[mode]
        ch_rows = chunk_by_mode[mode]
        per_req = pm.per_request_progress(req_rows, ch_rows)
        window = pm.measurement_window(req_rows, ch_rows, max_num_seqs)

        def vals(key):
            return [p[key] for p in per_req if p.get(key) is not None]

        metrics["progress_request"][mode] = {
            "total_stall_time":    distribution_stats(vals("total_stall_time")),
            "max_stall_time":      distribution_stats(vals("max_stall_time")),
            "num_stall_intervals": distribution_stats(vals("num_stall_intervals")),
            "stall_fraction":      distribution_stats(vals("stall_fraction")),
            "completion_latency":  distribution_stats(vals("completion_latency")),
            "demand_duration":     distribution_stats(vals("demand_duration")),
        }
        # Phase 5: new merged-interval CP-SLO aggregates per mode.
        metrics["cpslo"][mode] = {
            "max_stall_interval_distribution": distribution_stats(
                vals("max_stall_interval_s")),
            "num_stall_intervals_merged_distribution": distribution_stats(
                vals("num_stall_intervals_merged")),
        }
        metrics["workload"][mode] = {
            "num_prompt_tokens": dist_for_key(req_rows, "num_prompt_tokens"),
            "num_output_tokens": dist_for_key(req_rows, "num_output_tokens"),
            "num_chunks":        dist_for_key(req_rows, "num_chunks"),
        }
        metrics["throughput"][mode] = pm.throughput_stats(
            req_rows, per_req, window, run_meta=run_meta)
        # SSLO Phase 6: scalar run-level workload counts consumed by
        # validate_run (drop/timeout/throughput predicates).
        num_total = len(req_rows)
        num_completed = sum(
            1 for r in req_rows if r.get("terminal_outcome") == "completed")
        num_timeout = sum(
            1 for r in req_rows if r.get("terminal_outcome") == "timeout")
        num_chunks_total = sum(
            int(r.get("num_chunks") or 0) for r in req_rows)
        metrics["workload"][mode].update({
            "num_requests_total": num_total,
            "num_requests_completed": num_completed,
            "num_requests_timeout": num_timeout,
            "num_chunks_total": num_chunks_total,
            "tokens_per_second": (
                metrics["throughput"][mode].get("tokens_per_second")),
        })
        metrics["cp_slo_violation"][mode] = pm.cp_slo_violation_rates(per_req, list(pm.DEFAULT_TAUS))
        # Phase 5: also surface the same violation table under the new
        # cpslo namespace so downstream readers don't have to know the
        # legacy key.
        metrics["cpslo"][mode]["cp_slo_violation_rates_by_tau"] = (
            metrics["cp_slo_violation"][mode])
        duration = (window[1] - window[0]) if (window[0] is not None and window[1] is not None) else None
        metrics["measurement_window"][mode] = {
            "start_ts":   window[0],
            "end_ts":     window[1],
            "duration_s": duration,
        }
        windows[mode] = window

    for mode in ALL_MODES:
        sched_rows = sched_by_mode[mode]
        window = windows[mode]
        hu_stats = pm.handling_users_stats(sched_rows, window)
        metrics["handling_users"][mode] = hu_stats
        # Phase 5: dual-write time-weighted mean onto scheduler.num_handling_users
        # so downstream code can read the canonical mean from the same key
        # without losing the existing simple-sample mean.
        if metrics["scheduler"].get(mode) is not None:
            nhu = metrics["scheduler"][mode].get("num_handling_users")
            if isinstance(nhu, dict):
                nhu["mean_time_weighted"] = hu_stats.get("time_avg")
        # Phase 5: promote time-weighted mean into cpslo headline.
        metrics["cpslo"].setdefault(mode, {})
        metrics["cpslo"][mode]["mean_handling_users_time_weighted"] = (
            hu_stats.get("time_avg"))

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

    # SSLO Phase 6: feed validate_run with per-mode queueing distributions
    # so the no-harm gate can read p99 queue_stall / pending_time.
    for mode in ALL_MODES:
        req_rows = req_by_mode[mode]
        queue_stalls = [r["queue_stall_s"] for r in req_rows
                        if r.get("queue_stall_s") is not None]
        pending_times = [r["total_pending_time_s"] for r in req_rows
                         if r.get("total_pending_time_s") is not None]
        metrics["cpslo"].setdefault(mode, {})
        metrics["cpslo"][mode]["queue_stall_distribution"] = distribution_stats(
            queue_stalls)
        metrics["cpslo"][mode]["pending_time_distribution"] = distribution_stats(
            pending_times)

    summary: dict[str, Any] = {
        "config": {
            "model": model,
            "label": label,
            "max_num_seqs": max_num_seqs,
            "chunk_unit": chunk_unit,
            "request_rate": request_rate,
            "generation_max_tokens": generation_max_tokens,
            "max_model_len": max_model_len,
            "is_control": is_control,
            "modes_run": modes_run,
            "sslo_config": sslo_config_by_mode,
            "run_kind": modes_run[0] if modes_run else None,
        },
        "metrics": metrics,
        "queue_stall_available": queue_stall_available,
        "scheduler_saturation": scheduler_saturation,
        "passes": passes,
        "run_meta": run_meta,
    }

    # SSLO Phase 6: validity + no-harm gate. Single-mode runs have no paired
    # baseline tps available here, so the throughput-regression branch is
    # skipped; _consolidate_mode_outputs.py / sweep-level analysis recompute
    # against the paired baseline when both modes are present.
    baseline_tps = (
        metrics.get("workload", {}).get("baseline", {}).get("tokens_per_second"))
    report = validate_run(summary, baseline_tokens_per_second=baseline_tps)
    summary["validity"] = {
        "validity_pass": report.validity_pass,
        "no_harm_pass": report.no_harm_pass,
        "invalid_reason": report.invalid_reason,
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
        model=args.model,
        generation_max_tokens=args.generation_max_tokens,
        max_model_len=args.max_model_len,
        label=args.label,
    )


if __name__ == "__main__":
    main()
