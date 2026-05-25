#!/usr/bin/env python3
"""Summarize SSLO scheduler end-to-end validation runs."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
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
SSLO_MODES = ("sslo", "sslo_offload", "sslo_adaptive", "sslo_adaptive_offload", "sslo_mlp")
ALL_MODES = ("baseline",) + SSLO_MODES
SIX_STAT_KEYS = ("mean", "p50", "p90", "p95", "p99", "max")


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


def flatten_distribution(
    out: dict[str, Any],
    prefix: str,
    stats: dict[str, float | int | None],
    *,
    suffix: str = "_s",
) -> None:
    for key in SIX_STAT_KEYS:
        out[f"{prefix}_{key}{suffix}"] = stats.get(key)


def finalize_round2_metrics(
    metrics: dict[str, Any],
    modes: tuple[str, ...],
    request_progress_distributions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the Round 2 summary.json metrics shape."""
    final: dict[str, Any] = {}

    for key in (
        "slack",
        "slo_compliance",
        "inter_chunk_delay",
        "prediction_ratio",
        "stall_time",
        "measurement_window",
    ):
        if key in metrics:
            final[key] = metrics[key]

    if "workload" in metrics:
        final["workload"] = {}
        for mode, workload in metrics["workload"].items():
            final["workload"][mode] = {
                k: v for k, v in workload.items()
                if k != "num_chunks_total"
            }

    if "request_cu_slo_violation" in metrics:
        final["request_cu_slo_violation"] = metrics["request_cu_slo_violation"]

    if "scheduler" in metrics:
        final["scheduler"] = {}
        flattened_scheduler_keys = {
            "mean_running_time_weighted",
            "mean_pending_time_weighted",
            "mean_waiting_time_weighted",
            "mean_handling_users_time_weighted",
        }
        for mode, sched in metrics["scheduler"].items():
            final["scheduler"][mode] = {
                k: v for k, v in sched.items()
                if k not in flattened_scheduler_keys
            }

    if "handling_users" in metrics:
        final["handling_users"] = {}
        for mode, handling in metrics["handling_users"].items():
            final["handling_users"][mode] = {
                k: v for k, v in handling.items()
                if k != "time_avg"
            }

    if "throughput" in metrics:
        final["throughput"] = {}
        for mode, throughput in metrics["throughput"].items():
            final["throughput"][mode] = {
                k: v for k, v in throughput.items()
                if k != "tokens_per_second"
            }

    for mode in modes:
        mode_metrics: dict[str, Any] = {}
        progress = request_progress_distributions.get(mode) or {}
        flatten_distribution(
            mode_metrics, "request_max_stall_s",
            progress.get("request_max_stall_s") or {},
            suffix="")
        flatten_distribution(
            mode_metrics, "request_total_stall_s",
            progress.get("request_total_stall_s") or {},
            suffix="")
        flatten_distribution(
            mode_metrics, "request_stall_fraction",
            progress.get("request_stall_fraction") or {},
            suffix="")
        flatten_distribution(
            mode_metrics, "completion_latency",
            progress.get("completion_latency") or {})

        ttft_all = ((metrics.get("ttft") or {}).get(mode) or {}).get("all") or {}
        flatten_distribution(mode_metrics, "ttft", ttft_all)
        flatten_distribution(
            mode_metrics, "ttfc",
            (metrics.get("ttfc") or {}).get(mode) or {})
        flatten_distribution(
            mode_metrics, "tpot",
            (metrics.get("tpot") or {}).get(mode) or {})
        flatten_distribution(
            mode_metrics, "queue_stall",
            (metrics.get("queue_stall") or {}).get(mode) or {})
        pending_time = (
            ((metrics.get("pending") or {}).get(mode) or {}).get("time") or {}
        )
        flatten_distribution(mode_metrics, "pending_time", pending_time)

        scheduler = (metrics.get("scheduler") or {}).get(mode) or {}
        mode_metrics["mean_running"] = scheduler.get("mean_running_time_weighted")
        mode_metrics["mean_pending"] = scheduler.get("mean_pending_time_weighted")
        mode_metrics["mean_waiting"] = scheduler.get("mean_waiting_time_weighted")
        mode_metrics["mean_handling_users"] = scheduler.get(
            "mean_handling_users_time_weighted")

        throughput = (metrics.get("throughput") or {}).get(mode) or {}
        mode_metrics["corrected_processed_tokens_per_s"] = throughput.get(
            "tokens_per_second")

        workload = (metrics.get("workload") or {}).get(mode) or {}
        total = workload.get("num_requests_total") or 0
        completed = workload.get("num_requests_completed") or 0
        mode_metrics["drop_timeout_rate"] = (
            ((total - completed) / total) if total else None
        )
        mode_metrics["starvation_count"] = workload.get("starvation_count")
        mode_metrics["unit_deadline_miss_rate"] = workload.get(
            "unit_deadline_miss_rate")
        final[mode] = mode_metrics

    return final


def slack_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if any("unit_deadline_miss_s" in r for r in rows):
        values = [
            -float(r.get("unit_deadline_miss_s") or 0.0)
            for r in rows
            if r.get("unit_index") is not None
        ]
    else:
        # Exclude chunk_idx == 0: chunk_slack is fixed at 0.0 for the first
        # chunk by definition (deadline starts there), so including it dilutes
        # both the violation ratio and the distribution stats.
        filtered = [r for r in rows if r.get("chunk_idx") not in (None, 0)]
        values = numeric_values(filtered, "chunk_slack")
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
    """Per-unit stall statistics.

    Returns:
      - mean        : average stall over units where stall > 0 (violated).
      - mean_with_0 : average stall over all eligible units, counting on-time
                      units as 0. Reflects overall slack burn.
      - p50, p90, p99 : percentiles of stall across all eligible units.
    """
    if any("unit_deadline_miss_s" in r for r in rows):
        stalls_all = [
            max(0.0, float(row.get("unit_deadline_miss_s") or 0.0))
            for row in rows
            if row.get("unit_index") is not None
        ]
    else:
        filtered = [r for r in rows if r.get("chunk_idx") not in (None, 0)
                    and r.get("chunk_slack") is not None]
        stalls_all = [max(0.0, -float(row["chunk_slack"])) for row in filtered]
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
    """Per-unit ratio of actual / predicted token count.

    Skips unit/chunk 0 (predictor has no history yet) and rows where
    expected_len is None or non-positive. ratio == 1 means perfect
    prediction; > 1 means actual exceeded prediction (under-estimate).
    """
    values: list[float] = []
    for row in rows:
        unit_index = row.get("unit_index")
        if unit_index is None:
            unit_index = row.get("chunk_idx")
        if unit_index in (None, 0):
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
        miss = row.get("unit_deadline_miss_s")
        slack = row.get("chunk_slack")
        if miss is not None and float(miss) > 0:
            violated.add(rid)
        elif slack is not None and float(slack) < 0:
            violated.add(rid)
    total = len(seen)
    compliant = total - len(violated)
    return {
        "rate": (compliant / total) if total else None,
        "count": compliant,
        "total_requests": total,
    }


def pending_request_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    intervals = []
    for row in rows:
        value = row.get("num_pending_intervals")
        if value is None:
            value = row.get("num_pending_iters_per_request")
        if value is not None:
            intervals.append(float(value))
    return {
        "time": dist_for_key(rows, "total_pending_time_s"),
        "intervals": distribution_stats(intervals),
    }


def inter_chunk_delay_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    by_request: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        request_id = row.get("request_id")
        unit_index = row.get("unit_index")
        end_ts = row.get("text_generation_end_time")
        if unit_index is None:
            unit_index = row.get("chunk_idx")
        if end_ts is None:
            end_ts = row.get("end_time_ts")
        if request_id is None or unit_index is None or end_ts is None:
            continue
        normalized = {**row, "_unit_index": unit_index, "_end_ts": end_ts}
        by_request.setdefault(str(request_id), []).append(normalized)
    delays: list[float] = []
    for request_rows in by_request.values():
        ordered = sorted(request_rows, key=lambda r: int(r["_unit_index"]))
        for prev, cur in zip(ordered, ordered[1:]):
            cur_end = float(cur["_end_ts"])
            prev_end = float(prev["_end_ts"])
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


def _filter_by_window(
    req_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    mw0: float,
    mw1: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only requests in the measurement window and their chunks.

    Prefers the per-row `in_window` flag (set by run_test.py from the
    completion timestamp under the new completion-gated workflow). Falls back
    to completion_wall_ts when compact request rows omit the flag, then to
    injection_ts in [mw0, mw1) for older rows."""
    has_flag = any("in_window" in r for r in req_rows)
    if has_flag:
        kept_req = [r for r in req_rows if r.get("in_window")]
    elif any("completion_wall_ts" in r for r in req_rows):
        kept_req = [r for r in req_rows
                    if r.get("completion_wall_ts") is not None
                    and mw0 <= float(r["completion_wall_ts"]) <= mw1]
    else:
        kept_req = [r for r in req_rows
                    if r.get("injection_ts") is not None
                    and mw0 <= float(r["injection_ts"]) < mw1]
    kept_ids = {str(r["request_id"]) for r in kept_req if r.get("request_id") is not None}
    kept_chunks = [c for c in chunk_rows
                   if str(c.get("request_id", "")) in kept_ids]
    return kept_req, kept_chunks


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
    # Mode is detected by walking up the path looking for a directory
    # whose name is in ALL_MODES. Supports any layout depth:
    #   legacy: <label>/.../rate_<r>/<mode>/run_<i>/
    #   v15:    v15_grid/<model>/cap<C>/<mode>/run_<i>/rate_<r>/
    mode = None
    for p in output_dir.parents:
        if p.name in ALL_MODES:
            mode = p.name
            break
    if mode is None:
        raise ValueError(
            f"could not infer mode from path; expected one of {ALL_MODES} "
            f"to appear in an ancestor of {output_dir}")

    # Load run_meta.json early — it is the authoritative source for the
    # measurement window.
    run_meta: dict[str, Any] = {}
    meta_path = output_dir / "run_meta.json"
    if meta_path.exists():
        try:
            run_meta = json.loads(meta_path.read_text()) or {}
        except json.JSONDecodeError:
            run_meta = {}

    mw0 = run_meta.get("measurement_window_start_ts")
    mw1 = run_meta.get("measurement_window_end_ts")
    if mw0 is None or mw1 is None:
        raise RuntimeError(
            "missing measurement_window_*: re-run with new flow"
        )
    mw0 = float(mw0)
    mw1 = float(mw1)
    # scheduler_stats.jsonl uses time.monotonic(); request fields use
    # time.time(). Read mono-clock window bounds if the run wrote them
    # (new flow), else fall back to wall-clock bounds (legacy runs —
    # filter then drops everything, matching prior behavior).
    mw0_mono = run_meta.get("measurement_window_start_mono_ts")
    mw1_mono = run_meta.get("measurement_window_end_mono_ts")
    if mw0_mono is not None and mw1_mono is not None:
        sched_mw0, sched_mw1 = float(mw0_mono), float(mw1_mono)
    else:
        sched_mw0, sched_mw1 = mw0, mw1

    request_rows_mode = read_jsonl(output_dir / "requests.jsonl")
    chunk_rows_mode = read_jsonl(output_dir / "chunks.jsonl")
    sched_rows_mode = read_jsonl(output_dir / "scheduler_stats.jsonl")
    # Per-rate outputs are already single-mode; tag rows here because the
    # compact R1 writers no longer carry a mode field per row.
    for row in request_rows_mode:
        row.setdefault("mode", mode)
    for row in chunk_rows_mode:
        row.setdefault("mode", mode)
    # scheduler_stats.jsonl is written by the scheduler with no mode awareness;
    # tag rows here for the per-mode split downstream.
    for row in sched_rows_mode:
        row.setdefault("mode", mode)

    # Filter scheduler_stats to measurement window (monotonic clock).
    sched_rows_mode = [
        r for r in sched_rows_mode
        if r.get("ts") is not None
        and sched_mw0 <= float(r["ts"]) < sched_mw1
    ]

    sslo_config_by_mode: dict[str, dict[str, Any]] = {}
    cfg_path = output_dir / "sslo_config.json"
    if cfg_path.exists():
        try:
            sslo_config_by_mode[mode] = json.loads(cfg_path.read_text()) or {}
        except json.JSONDecodeError:
            sslo_config_by_mode[mode] = {}

    run_status: dict[str, Any] = {mode: 0}

    # Apply window filter to requests and chunks.
    request_rows_mode_windowed, chunk_rows_mode_windowed = _filter_by_window(
        request_rows_mode, chunk_rows_mode, mw0, mw1)

    request_rows_all = list(request_rows_mode_windowed)
    chunk_rows_all = list(chunk_rows_mode_windowed)
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
    queue_stall_available = any(
        r.get("queue_stall_s") is not None for r in request_rows_all)

    metrics: dict[str, Any] = {
        "ttft": {}, "ttfc": {}, "tpot": {}, "queue_stall": {}, "slack": {},
        "slo_compliance": {}, "scheduler": {}, "pending": {}, "inter_chunk_delay": {},
        "prediction_ratio": {}, "stall_time": {},
        "workload": {}, "throughput": {},
        "request_cu_slo_violation": {}, "measurement_window": {}, "handling_users": {},
    }
    request_progress_distributions: dict[str, dict[str, Any]] = {}

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
        metrics["queue_stall"][mode] = dist_for_key(req_rows, "queue_stall_s")
        metrics["slack"][mode] = slack_stats(ch_rows)
        metrics["stall_time"][mode] = stall_time_stats(ch_rows)
        metrics["slo_compliance"][mode] = request_compliance_stats(ch_rows)
        metrics["inter_chunk_delay"][mode] = inter_chunk_delay_stats(ch_rows)
        metrics["pending"][mode] = pending_request_stats(req_rows)
        metrics["prediction_ratio"][mode] = prediction_ratio_stats(ch_rows)

    # Scheduler stats (running / num_handling_users) emit for ALL modes,
    # including baseline (which now also routes through schedule_sslo via
    # method="baseline" so scheduler_stats.jsonl is populated).
    def _time_weighted_mean(rows, field, w0, w1):
        """Time-weighted mean of `field` over [w0, w1]. Each sample's
        weight = duration until the next sample (or window end)."""
        if w0 is None or w1 is None:
            return None
        rows_w = sorted(
            (r for r in rows
             if r.get("ts") is not None and w0 <= float(r["ts"]) <= w1),
            key=lambda r: float(r["ts"]))
        if not rows_w:
            return None
        wsum = 0.0
        weight_total = 0.0
        for i, r in enumerate(rows_w):
            v = r.get(field)
            if v is None: continue
            ts = float(r["ts"])
            ts_next = float(rows_w[i+1]["ts"]) if i+1 < len(rows_w) else w1
            w = ts_next - ts
            if w <= 0: continue
            wsum += float(v) * w
            weight_total += w
        return wsum / weight_total if weight_total > 0 else None

    for mode in ALL_MODES:
        sched_rows = sched_by_mode[mode]
        # Include min in `num_handling_users` so summary captures the
        # running+pending floor alongside mean/max.
        nhu_dist = distribution_stats(
            numeric_values(sched_rows, "num_handling_users"),
            include_min=True)
        # SSLO: time-weighted queue means + urgent-mode (critical) fraction.
        # IMPORTANT: scheduler_stats.jsonl ts uses time.monotonic(); use
        # the monotonic-clock window bounds, NOT wall-clock mw0/mw1.
        running_tw = _time_weighted_mean(sched_rows, "running", sched_mw0, sched_mw1)
        pending_tw = _time_weighted_mean(sched_rows, "pending", sched_mw0, sched_mw1)
        waiting_tw = _time_weighted_mean(sched_rows, "waiting", sched_mw0, sched_mw1)
        hu_tw = _time_weighted_mean(sched_rows, "num_handling_users", sched_mw0, sched_mw1)
        # has_critical is bool; coerce to 0/1 then time-weight.
        crit_rows = [{**r, "_crit": (1.0 if r.get("has_critical") else 0.0)}
                     for r in sched_rows]
        crit_frac = _time_weighted_mean(crit_rows, "_crit", sched_mw0, sched_mw1)
        # SSLO: unbiased token throughput from scheduler step stats.
        # Sum num_decode_tokens / num_prefill_tokens within the monotonic
        # window, divide by duration. Independent of which reqs finished
        # in the cap*4 measurement cohort (so this avoids the selection
        # bias the request-level tokens_per_second has).
        in_win_sched = [
            r for r in sched_rows
            if r.get("ts") is not None
            and sched_mw0 is not None and sched_mw1 is not None
            and sched_mw0 <= float(r["ts"]) <= sched_mw1
        ]
        win_dur = (sched_mw1 - sched_mw0) if (
            sched_mw0 is not None and sched_mw1 is not None) else None
        def _sum(key):
            return sum(int(r.get(key) or 0) for r in in_win_sched)
        sched_decode_tokens = _sum("num_decode_tokens")
        sched_prefill_tokens = _sum("num_prefill_tokens")
        sched_total_tokens = _sum("num_scheduled_tokens_total")
        metrics["scheduler"][mode] = {
            "running": dist_for_key(sched_rows, "running"),
            "num_handling_users": nhu_dist,
            "mean_running_time_weighted": running_tw,
            "mean_pending_time_weighted": pending_tw,
            "mean_waiting_time_weighted": waiting_tw,
            "mean_handling_users_time_weighted": hu_tw,
            "urgent_mode_fraction": crit_frac,
            # Unbiased token throughputs from scheduler-side counters.
            "decode_tokens_per_second": (
                sched_decode_tokens / win_dur
                if win_dur and win_dur > 0 else None),
            "prefill_tokens_per_second": (
                sched_prefill_tokens / win_dur
                if win_dur and win_dur > 0 else None),
            "scheduled_tokens_per_second": (
                sched_total_tokens / win_dur
                if win_dur and win_dur > 0 else None),
        }

    # Use run_meta window as the single authoritative window for all modes.
    window = (mw0, mw1)

    for mode in ALL_MODES:
        req_rows = req_by_mode[mode]
        ch_rows = chunk_by_mode[mode]
        per_req = pm.per_request_progress(req_rows, ch_rows)

        def vals(key):
            return [p[key] for p in per_req if p.get(key) is not None]

        request_progress_distributions[mode] = {
            "request_total_stall_s": distribution_stats(
                vals("request_total_stall_s")),
            "request_max_stall_s": distribution_stats(
                vals("request_max_stall_s")),
            "num_stall_intervals": distribution_stats(
                vals("num_stall_intervals")),
            "request_stall_fraction": distribution_stats(
                vals("request_stall_fraction")),
            "completion_latency": distribution_stats(vals("completion_latency")),
            "request_demand_duration_s": distribution_stats(
                vals("request_demand_duration_s")),
        }
        metrics["workload"][mode] = {
            "num_prompt_tokens": dist_for_key(req_rows, "num_prompt_tokens"),
            "num_output_tokens": dist_for_key(req_rows, "num_output_tokens"),
            "num_consumable_units": dist_for_key(
                req_rows, "num_consumable_units"),
            # SSLO: pending-time + consume-time distributions for summary CSV.
            "total_pending_time_s": dist_for_key(req_rows, "total_pending_time_s"),
            "num_pending_intervals": dist_for_key(req_rows, "num_pending_intervals"),
            "consume_duration": dist_for_key(ch_rows, "consume_duration"),
        }
        metrics["throughput"][mode] = pm.throughput_stats(
            req_rows, per_req, run_meta=run_meta)
        # SSLO Phase 6: scalar run-level workload counts consumed by
        # validate_run (drop/timeout/throughput predicates).
        num_total = len(req_rows)
        has_terminal_outcome = any("terminal_outcome" in r for r in req_rows)
        if has_terminal_outcome:
            num_completed = sum(
                1 for r in req_rows
                if r.get("terminal_outcome") == "completed")
            num_timeout = sum(
                1 for r in req_rows if r.get("terminal_outcome") == "timeout")
        elif mode == modes_run[0]:
            num_total = int(run_meta.get("num_requests_total", num_total) or 0)
            num_completed = int(
                run_meta.get("num_requests_completed", len(req_rows)) or 0)
            num_timeout = int(run_meta.get("num_requests_timeout", 0) or 0)
        else:
            num_completed = sum(
                1 for r in req_rows if r.get("completion_wall_ts") is not None)
            num_timeout = 0
        num_consumable_units_total = sum(
            int(r.get("num_consumable_units") or 0) for r in req_rows)
        has_pending_intervals = any(
            r.get("num_pending_intervals") is not None
            or r.get("num_pending_iters_per_request") is not None
            for r in req_rows)
        starvation_count = (
            sum(
                int(r.get("num_pending_intervals")
                    if r.get("num_pending_intervals") is not None
                    else (r.get("num_pending_iters_per_request") or 0))
                for r in req_rows)
            if has_pending_intervals
            else None
        )
        # Spec section 5: unit_deadline_miss_rate = sum(unit_deadline_missed)
        # / count(consumable units) over this mode's in-window chunks
        # (chunk_filter already applied upstream). chunk_by_mode is the
        # post-filter, in-window chunk set for this mode.
        mode_chunks = chunk_by_mode.get(mode, [])
        unit_deadline_miss_rate = None
        if mode_chunks:
            missed = sum(int(c.get("unit_deadline_missed") or 0)
                         for c in mode_chunks)
            unit_deadline_miss_rate = missed / len(mode_chunks)
        metrics["workload"][mode].update({
            "num_requests_total": num_total,
            "num_requests_completed": num_completed,
            "num_requests_timeout": num_timeout,
            "num_consumable_units_total": num_consumable_units_total,
            # Internal compatibility for validate_run(); omitted from the
            # final Round 2 metrics shape.
            "num_chunks_total": num_consumable_units_total,
            "starvation_count": starvation_count,
            "unit_deadline_miss_rate": unit_deadline_miss_rate,
            "tokens_per_second": (
                metrics["throughput"][mode].get("tokens_per_second")),
        })
        metrics["request_cu_slo_violation"][mode] = (
            pm.request_cu_slo_violation_rates(per_req, list(pm.DEFAULT_TAUS)))
        duration = mw1 - mw0
        metrics["measurement_window"][mode] = {
            "start_ts":   mw0,
            "end_ts":     mw1,
            "duration_s": duration,
        }

    for mode in ALL_MODES:
        sched_rows = sched_by_mode[mode]
        hu_stats = pm.handling_users_stats(sched_rows, window)
        metrics["handling_users"][mode] = hu_stats
        # Phase 5: dual-write time-weighted mean onto scheduler.num_handling_users
        # so downstream code can read the canonical mean from the same key
        # without losing the existing simple-sample mean.
        if metrics["scheduler"].get(mode) is not None:
            nhu = metrics["scheduler"][mode].get("num_handling_users")
            if isinstance(nhu, dict):
                nhu["mean_time_weighted"] = hu_stats.get("time_avg")

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

    # Surface window-level counts from run_meta in summary.
    injected_count = run_meta.get("injected_count", 0)
    in_window_count = run_meta.get("in_window_count", 0)

    final_metrics = finalize_round2_metrics(
        metrics, ALL_MODES, request_progress_distributions)
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
        "metrics": final_metrics,
        "queue_stall_available": queue_stall_available,
        "scheduler_saturation": scheduler_saturation,
        "passes": passes,
        "run_meta": run_meta,
        # Window-level counts surfaced at top level.
        "pool_size": run_meta.get("pool_size"),
        "injected_count": injected_count,
        "in_window_count": in_window_count,
        "out_of_window_count": injected_count - in_window_count,
        "measurement_window_seconds": run_meta.get("measurement_window_seconds"),
    }

    # SSLO Phase 6: validity + no-harm gate. Single-mode runs have no paired
    # baseline tps available here, so the throughput-regression branch is
    # skipped; _consolidate_mode_outputs.py / sweep-level analysis recompute
    # against the paired baseline when both modes are present.
    baseline_tps = (
        final_metrics.get("workload", {}).get("baseline", {}).get(
            "tokens_per_second")
        if len(modes_run) > 1
        else None
    )
    report = validate_run(
        summary,
        baseline_tokens_per_second=baseline_tps,
        request_rows=request_rows_mode,
    )
    summary["validity"] = asdict(report)

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
