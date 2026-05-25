"""Per-request ProgressServe metric derivation helpers."""
from __future__ import annotations

import statistics
from typing import Any


DEFAULT_TAUS = (0.5, 1.0, 2.0, 5.0)


def compute_stall_intervals(
    units: list[dict[str, Any]],
) -> list[tuple[float, float, float]]:
    """Return exact stall intervals under linear-between-deadlines consume.

    With the linear-between-deadlines consume model, the stall region reduces
    exactly to per-unit intervals [deadline[i], consumer_ready_time[i]) for
    units where unit_deadline_miss_s > 0; no interval merging is needed.
    """
    return [
        (u["deadline"], u["consumer_ready_time"], u["unit_deadline_miss_s"])
        for u in units
        if u.get("unit_deadline_miss_s", 0) > 0
    ]


def compute_token_trace(
    units: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    """Build diagnostic available/consumed token traces."""
    ordered = sorted(
        units,
        key=lambda u: (
            u.get("unit_index") is None,
            u.get("unit_index", 0),
            u.get("deadline", 0),
        ),
    )
    available: list[tuple[float, float]] = []
    consumed: list[tuple[float, float]] = []
    for unit in ordered:
        ready = unit.get("consumer_ready_time")
        boundary = unit.get("token_boundary")
        if ready is not None and boundary is not None:
            available.append((float(ready), float(boundary)))

        start = unit.get("deadline")
        if start is None or boundary is None:
            continue
        consumed.append((float(start), float(boundary)))

    return {
        "available": sorted(available, key=lambda p: p[0]),
        "consumed": sorted(consumed, key=lambda p: p[0]),
    }


def available_tokens_curve(
    chunks: list[dict[str, Any]],
) -> list[tuple[float, int]]:
    """Step function of cumulative available tokens over time.

    Derived from `chunk_generation_end_ts` + `cumulative_tokens_at_end`
    per chunk. Used by M1 reconstruction in downstream plot scripts.
    Not dumped to summary.json (size).
    """
    points: list[tuple[float, int]] = []
    for c in chunks:
        ts = c.get("chunk_generation_end_ts")
        cum = c.get("cumulative_tokens_at_end")
        if ts is None or cum is None:
            continue
        try:
            points.append((float(ts), int(cum)))
        except (TypeError, ValueError):
            continue
    points.sort(key=lambda p: p[0])
    return points


def _percentile(values: list[float], pct: float) -> float | None:
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


def per_request_progress(
    req_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Index consumable units by request_id.
    units_by_req: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        rid = row.get("request_id")
        if rid is not None:
            units_by_req.setdefault(str(rid), []).append(row)

    results = []
    for req in req_rows:
        rid = str(req.get("request_id", ""))
        units = sorted(
            units_by_req.get(rid, []),
            key=lambda u: (
                u.get("unit_index") is None,
                u.get("unit_index", 0),
                u.get("deadline", 0),
            ),
        )

        # arrival_ts
        arrival_ts = req.get("arrival_ts")
        if arrival_ts is not None:
            arrival_ts = float(arrival_ts)
        else:
            dts = req.get("decoding_start_ts")
            ttft = req.get("ttft")
            arrival_ts = (
                (float(dts) - float(ttft))
                if (dts is not None and ttft is not None)
                else None
            )

        # completion_ts
        completion_ts = req.get("completion_wall_ts")
        completion_ts = float(completion_ts) if completion_ts is not None else None

        # completion_latency
        completion_latency = (
            (completion_ts - arrival_ts)
            if (completion_ts is not None and arrival_ts is not None)
            else None
        )

        deadlines = [
            float(u["deadline"]) for u in units
            if u.get("deadline") is not None
        ]
        consume_origin_ts = min(deadlines) if deadlines else None
        final_deadline_ts = max(deadlines) if deadlines else None

        # request_demand_duration_s
        request_demand_duration_s = None
        if consume_origin_ts is not None and final_deadline_ts is not None:
            d = final_deadline_ts - consume_origin_ts
            request_demand_duration_s = d if d > 0 else None

        intervals = compute_stall_intervals(units)
        unit_misses = [
            float(u.get("unit_deadline_miss_s") or 0.0)
            for u in units
        ]
        request_total_stall_s = sum(m for m in unit_misses if m > 0)
        request_max_stall_s = max(
            (float(d) for _, _, d in intervals),
            default=0.0,
        )
        num_stall_intervals = len(intervals)
        request_stall_fraction = (
            (request_total_stall_s / request_demand_duration_s)
            if (
                request_demand_duration_s is not None
                and request_demand_duration_s > 0
            )
            else None
        )

        results.append({
            "request_id": rid,
            "arrival_ts": arrival_ts,
            "completion_ts": completion_ts,
            "completion_latency": completion_latency,
            "consume_origin_ts": consume_origin_ts,
            "final_deadline_ts": final_deadline_ts,
            "request_demand_duration_s": request_demand_duration_s,
            "request_total_stall_s": request_total_stall_s,
            "request_max_stall_s": request_max_stall_s,
            "num_stall_intervals": num_stall_intervals,
            "request_stall_fraction": request_stall_fraction,
            "stall_intervals": intervals,
            "num_output_tokens": req.get("num_output_tokens"),
            "num_prompt_tokens": req.get("num_prompt_tokens"),
        })

    return results


def handling_users_stats(
    sched_rows: list[dict[str, Any]],
    window: tuple[float | None, float | None],
) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "count": 0, "time_avg": None, "p50": None, "p95": None, "p99": None, "max": None,
    }
    w0, w1 = window
    if w0 is None or w1 is None:
        return empty

    in_window = [
        r for r in sched_rows
        if r.get("ts") is not None and w0 <= float(r["ts"]) <= w1
    ]
    if not in_window:
        return empty

    in_window.sort(key=lambda r: float(r["ts"]))
    nhu_values = [
        float(r["num_handling_users"]) for r in in_window
        if r.get("num_handling_users") is not None
    ]

    # Time-weighted mean
    time_avg = None
    if len(in_window) >= 1:
        total_weight = 0.0
        weighted_sum = 0.0
        for i in range(len(in_window) - 1):
            ts_cur = float(in_window[i]["ts"])
            ts_next = float(in_window[i + 1]["ts"])
            weight = ts_next - ts_cur
            nhu = in_window[i].get("num_handling_users")
            if nhu is not None:
                weighted_sum += float(nhu) * weight
                total_weight += weight
        # last sample
        last_ts = float(in_window[-1]["ts"])
        last_weight = (w1 - last_ts) if w1 is not None and w1 >= last_ts else 0.0
        last_nhu = in_window[-1].get("num_handling_users")
        if last_nhu is not None:
            weighted_sum += float(last_nhu) * last_weight
            total_weight += last_weight
        if total_weight > 0:
            time_avg = weighted_sum / total_weight

    return {
        "count": len(in_window),
        "time_avg": time_avg,
        "p50": _percentile(nhu_values, 50),
        "p95": _percentile(nhu_values, 95),
        "p99": _percentile(nhu_values, 99),
        "max": max(nhu_values, default=None),
    }


def throughput_stats(
    req_rows: list[dict[str, Any]],
    per_req_progress: list[dict[str, Any]],
    window: tuple[float | None, float | None] = (None, None),
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokens generated per second over the measurement window.

    Uses measurement_window_start_ts / measurement_window_end_ts from
    run_meta (the first-completion window set by run_test.py). The `window`
    arg is kept for signature compatibility but is no longer used.
    """
    w0: float | None = None
    w1: float | None = None
    if run_meta:
        w0 = run_meta.get("measurement_window_start_ts")
        w1 = run_meta.get("measurement_window_end_ts")
    if w0 is None or w1 is None:
        return {"count": 0, "duration_s": None, "tokens_per_second": None,
                "completed_req_per_s": None, "basis": "first_completion_window",
                "error": "missing measurement_window_*"}

    duration = w1 - w0
    if duration <= 0:
        return {"count": 0, "duration_s": duration, "tokens_per_second": None,
                "completed_req_per_s": None, "basis": "first_completion_window"}

    tokens_out = sum(
        int(p["num_output_tokens"])
        for p in per_req_progress
        if p.get("num_output_tokens") is not None
    )
    tokens_in = sum(
        int(p["num_prompt_tokens"])
        for p in per_req_progress
        if p.get("num_prompt_tokens") is not None
    )
    count = sum(
        1 for p in per_req_progress
        if p.get("completion_ts") is not None
    )

    return {
        "count": count,
        "duration_s": duration,
        # `tokens_per_second` (output-only) kept for backward compat.
        "tokens_per_second": tokens_out / duration,
        "output_tokens_per_second": tokens_out / duration,
        "input_tokens_per_second": tokens_in / duration,
        "total_tokens_per_second": (tokens_in + tokens_out) / duration,
        "completed_req_per_s": count / duration,
        "basis": "first_completion_window",
    }


def request_cu_slo_violation_rates(
    per_req_progress: list[dict[str, Any]],
    taus: list[float],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    valid = [
        p for p in per_req_progress
        if p.get("request_max_stall_s") is not None
    ]
    total = len(valid)
    for tau in taus:
        key = f"tau_{tau:g}"
        violated = sum(1 for p in valid if p["request_max_stall_s"] > tau)
        result[key] = {
            "rate": (violated / total) if total > 0 else None,
            "violated": violated,
            "total": total,
        }
    return result


def chunk_slo_violation_rates(
    chunk_rows: list[dict[str, Any]],
    taus: list[float],
) -> dict[str, dict[str, Any]]:
    """Chunk-level SLO violation: stall_duration_s > τ across chunks
    that have a deadline (chunk_idx >= 1; chunk 0 has no preceding chunk
    to compute a deadline from).
    """
    eligible = [
        r for r in chunk_rows
        if r.get("chunk_idx") not in (None, 0)
        and r.get("stall_duration_s") is not None
    ]
    total = len(eligible)
    result: dict[str, dict[str, Any]] = {}
    for tau in taus:
        key = f"tau_{tau:g}"
        violated = sum(
            1 for r in eligible if float(r["stall_duration_s"]) > tau)
        result[key] = {
            "rate": (violated / total) if total > 0 else None,
            "violated": violated,
            "total": total,
        }
    return result


if __name__ == "__main__":
    synthetic_units = [
        {
            "unit_index": 0,
            "deadline": 10,
            "consumer_ready_time": 9.5,
            "consume_duration": 2,
            "token_boundary": 10,
            "unit_deadline_miss_s": 0,
        },
        {
            "unit_index": 1,
            "deadline": 12,
            "consumer_ready_time": 13.0,
            "consume_duration": 2,
            "token_boundary": 20,
            "unit_deadline_miss_s": 1.0,
        },
        {
            "unit_index": 2,
            "deadline": 15,
            "consumer_ready_time": 14.5,
            "consume_duration": 2,
            "token_boundary": 30,
            "unit_deadline_miss_s": 0,
        },
    ]
    intervals = compute_stall_intervals(synthetic_units)
    max_miss = max((d for _, _, d in intervals), default=0.0)
    assert intervals == [(12, 13.0, 1.0)], f"unexpected intervals: {intervals}"
    assert max_miss == 1.0, f"unexpected max_miss: {max_miss}"
    print("OK")
