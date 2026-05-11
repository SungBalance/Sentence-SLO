"""Shared metrics constants and helpers for SSLO sweep aggregators."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


MODES_DEFAULT = (
    "baseline",
    "sslo",
    "sslo_offload",
    "sslo_adaptive",
    "sslo_adaptive_offload",
)


@dataclass(frozen=True)
class MetricSpec:
    path: tuple[str, ...]
    field: str
    scale: float
    fmt: str
    label: str


DISPLAY_GROUPS: tuple[tuple[str, tuple[MetricSpec, ...]], ...] = (
    ("Latency", (
        MetricSpec(("ttft", "post_cap"), "mean", 1.0,  "{:.3f}", "TTFT post_cap mean (s)"),
        MetricSpec(("ttft", "post_cap"), "p50",  1.0,  "{:.3f}", "TTFT post_cap p50 (s)"),
        MetricSpec(("ttft", "post_cap"), "p90",  1.0,  "{:.3f}", "TTFT post_cap p90 (s)"),
        MetricSpec(("ttft", "post_cap"), "p99",  1.0,  "{:.3f}", "TTFT post_cap p99 (s)"),
        MetricSpec(("ttft", "post_cap"), "max",  1.0,  "{:.3f}", "TTFT post_cap max (s)"),
        MetricSpec(("ttfc",),            "mean", 1.0,  "{:.3f}", "TTFC mean (s)"),
        MetricSpec(("ttfc",),            "p50",  1.0,  "{:.3f}", "TTFC p50 (s)"),
        MetricSpec(("ttfc",),            "p99",  1.0,  "{:.3f}", "TTFC p99 (s)"),
        MetricSpec(("tpot",),            "mean", 1000, "{:.2f}", "TPOT mean (ms)"),
        MetricSpec(("tpot",),            "p50",  1000, "{:.2f}", "TPOT p50 (ms)"),
        MetricSpec(("tpot",),            "p90",  1000, "{:.2f}", "TPOT p90 (ms)"),
        MetricSpec(("tpot",),            "p99",  1000, "{:.2f}", "TPOT p99 (ms)"),
        MetricSpec(("tpot",),            "max",  1000, "{:.2f}", "TPOT max (ms)"),
    )),
    ("Queue stall", (
        MetricSpec(("queue_stall",), "mean", 1000, "{:.2f}", "queue stall mean (ms)"),
        MetricSpec(("queue_stall",), "p50",  1000, "{:.2f}", "queue stall p50 (ms)"),
        MetricSpec(("queue_stall",), "p90",  1000, "{:.2f}", "queue stall p90 (ms)"),
        MetricSpec(("queue_stall",), "p99",  1000, "{:.2f}", "queue stall p99 (ms)"),
        MetricSpec(("queue_stall",), "max",  1000, "{:.2f}", "queue stall max (ms)"),
    )),
    ("Slack", (
        MetricSpec(("slack",),                      "neg_ratio", 100, "{:.4f}", "neg slack ratio (%)"),
        MetricSpec(("slack",),                      "mean",      1.0, "{:.3f}", "slack mean (s)"),
        MetricSpec(("slack",),                      "p50",       1.0, "{:.3f}", "slack p50 (s)"),
        MetricSpec(("slack",),                      "p90",       1.0, "{:.3f}", "slack p90 (s)"),
        MetricSpec(("slack",),                      "p99",       1.0, "{:.3f}", "slack p99 (s)"),
        MetricSpec(("slack",),                      "max",       1.0, "{:.3f}", "slack max (s)"),
        MetricSpec(("slack", "violated-magnitude"), "mean",      1.0, "{:.3f}", "violated magnitude mean (s)"),
        MetricSpec(("slack", "violated-magnitude"), "p50",       1.0, "{:.3f}", "violated magnitude p50 (s)"),
        MetricSpec(("slack", "violated-magnitude"), "p90",       1.0, "{:.3f}", "violated magnitude p90 (s)"),
        MetricSpec(("slack", "violated-magnitude"), "p99",       1.0, "{:.3f}", "violated magnitude p99 (s)"),
        MetricSpec(("slack", "violated-magnitude"), "max",       1.0, "{:.3f}", "violated magnitude max (s)"),
    )),
    ("SLO compliance", (
        MetricSpec(("slo_compliance",), "rate",           100, "{:.2f}", "SLO compliance (%)"),
        MetricSpec(("slo_compliance",), "count",          1.0, "{:.1f}", "SLO compliant reqs"),
        MetricSpec(("slo_compliance",), "total_requests", 1.0, "{:.1f}", "SLO total reqs"),
    )),
    ("Scheduler occupancy", (
        MetricSpec(("scheduler", "running"),  "mean", 1.0, "{:.2f}", "running mean"),
        MetricSpec(("scheduler", "running"),  "p50",  1.0, "{:.2f}", "running p50"),
        MetricSpec(("scheduler", "running"),  "p90",  1.0, "{:.2f}", "running p90"),
        MetricSpec(("scheduler", "running"),  "p99",  1.0, "{:.2f}", "running p99"),
        MetricSpec(("scheduler", "num_handling_users"), "min",  1.0, "{:.2f}", "handling-users min"),
        MetricSpec(("scheduler", "num_handling_users"), "mean", 1.0, "{:.2f}", "handling-users mean"),
        MetricSpec(("scheduler", "num_handling_users"), "max",  1.0, "{:.2f}", "handling-users max"),
        MetricSpec(("scheduler", "num_handling_users"), "p50",  1.0, "{:.2f}", "handling-users p50"),
        MetricSpec(("scheduler", "num_handling_users"), "p90",  1.0, "{:.2f}", "handling-users p90"),
        MetricSpec(("scheduler", "num_handling_users"), "p99",  1.0, "{:.2f}", "handling-users p99"),
    )),
    ("Pending dynamics", (
        MetricSpec(("pending", "time"),      "mean", 1.0, "{:.3f}", "pending time mean (s)"),
        MetricSpec(("pending", "time"),      "p50",  1.0, "{:.3f}", "pending time p50 (s)"),
        MetricSpec(("pending", "time"),      "p90",  1.0, "{:.3f}", "pending time p90 (s)"),
        MetricSpec(("pending", "time"),      "p99",  1.0, "{:.3f}", "pending time p99 (s)"),
        MetricSpec(("pending", "intervals"), "mean", 1.0, "{:.2f}", "pending intervals mean"),
        MetricSpec(("pending", "intervals"), "p50",  1.0, "{:.2f}", "pending intervals p50"),
        MetricSpec(("pending", "intervals"), "p90",  1.0, "{:.2f}", "pending intervals p90"),
    )),
    ("Streaming smoothness", (
        MetricSpec(("inter_chunk_delay",), "mean", 1000, "{:.2f}", "inter-chunk delay mean (ms)"),
        MetricSpec(("inter_chunk_delay",), "p50",  1000, "{:.2f}", "inter-chunk delay p50 (ms)"),
        MetricSpec(("inter_chunk_delay",), "p90",  1000, "{:.2f}", "inter-chunk delay p90 (ms)"),
        MetricSpec(("inter_chunk_delay",), "p99",  1000, "{:.2f}", "inter-chunk delay p99 (ms)"),
        MetricSpec(("inter_chunk_delay",), "max",  1000, "{:.2f}", "inter-chunk delay max (ms)"),
    )),
    ("Stall (request)", (
        MetricSpec(("progress_request", "total_stall_time"), "mean", 1.0, "{:.3f}", "total stall mean (s)"),
        MetricSpec(("progress_request", "total_stall_time"), "p50",  1.0, "{:.3f}", "total stall p50 (s)"),
        MetricSpec(("progress_request", "total_stall_time"), "p95",  1.0, "{:.3f}", "total stall p95 (s)"),
        MetricSpec(("progress_request", "total_stall_time"), "p99",  1.0, "{:.3f}", "total stall p99 (s)"),
        MetricSpec(("progress_request", "max_stall_time"),   "p95",  1.0, "{:.3f}", "max stall p95 (s)"),
        MetricSpec(("progress_request", "max_stall_time"),   "p99",  1.0, "{:.3f}", "max stall p99 (s)"),
        MetricSpec(("progress_request", "stall_fraction"),   "mean", 100, "{:.2f}", "stall fraction mean (%)"),
        MetricSpec(("progress_request", "stall_fraction"),   "p95",  100, "{:.2f}", "stall fraction p95 (%)"),
        MetricSpec(("progress_request", "num_stall_intervals"), "mean", 1.0, "{:.2f}", "stall intervals mean"),
    )),
    ("Latency (request)", (
        MetricSpec(("progress_request", "completion_latency"), "mean", 1.0, "{:.3f}", "completion latency mean (s)"),
        MetricSpec(("progress_request", "completion_latency"), "p50",  1.0, "{:.3f}", "completion latency p50 (s)"),
        MetricSpec(("progress_request", "completion_latency"), "p95",  1.0, "{:.3f}", "completion latency p95 (s)"),
        MetricSpec(("progress_request", "completion_latency"), "p99",  1.0, "{:.3f}", "completion latency p99 (s)"),
        MetricSpec(("progress_request", "demand_duration"),    "mean", 1.0, "{:.3f}", "demand duration mean (s)"),
    )),
    ("Workload", (
        MetricSpec(("workload", "num_prompt_tokens"), "mean", 1.0, "{:.1f}", "prompt tokens mean"),
        MetricSpec(("workload", "num_prompt_tokens"), "p50",  1.0, "{:.1f}", "prompt tokens p50"),
        MetricSpec(("workload", "num_prompt_tokens"), "p99",  1.0, "{:.1f}", "prompt tokens p99"),
        MetricSpec(("workload", "num_output_tokens"), "mean", 1.0, "{:.1f}", "output tokens mean"),
        MetricSpec(("workload", "num_output_tokens"), "p50",  1.0, "{:.1f}", "output tokens p50"),
        MetricSpec(("workload", "num_output_tokens"), "p99",  1.0, "{:.1f}", "output tokens p99"),
        MetricSpec(("workload", "num_chunks"),        "mean", 1.0, "{:.2f}", "chunks/req mean"),
        MetricSpec(("workload", "num_chunks"),        "p99",  1.0, "{:.2f}", "chunks/req p99"),
    )),
    ("Throughput", (
        MetricSpec(("throughput",), "tokens_per_second",   1.0, "{:.2f}", "tokens/s"),
        MetricSpec(("throughput",), "completed_req_per_s", 1.0, "{:.2f}", "completed req/s"),
        MetricSpec(("throughput",), "duration_s",          1.0, "{:.2f}", "measurement window (s)"),
    )),
    ("Handling users", (
        MetricSpec(("handling_users",), "time_avg", 1.0, "{:.2f}", "handling users time-avg"),
        MetricSpec(("handling_users",), "p50",      1.0, "{:.2f}", "handling users p50"),
        MetricSpec(("handling_users",), "p95",      1.0, "{:.2f}", "handling users p95"),
        MetricSpec(("handling_users",), "p99",      1.0, "{:.2f}", "handling users p99"),
        MetricSpec(("handling_users",), "max",      1.0, "{:.2f}", "handling users max"),
    )),
    ("CP-SLO violation", (
        MetricSpec(("cp_slo_violation", "tau_0.5"), "rate", 100, "{:.2f}", "CP-SLO viol @ tau=0.5s (%)"),
        MetricSpec(("cp_slo_violation", "tau_1"),   "rate", 100, "{:.2f}", "CP-SLO viol @ tau=1s (%)"),
        MetricSpec(("cp_slo_violation", "tau_2"),   "rate", 100, "{:.2f}", "CP-SLO viol @ tau=2s (%)"),
        MetricSpec(("cp_slo_violation", "tau_5"),   "rate", 100, "{:.2f}", "CP-SLO viol @ tau=5s (%)"),
    )),
)


def percentile(values: list[float], pct: float) -> float | None:
    """Linearly-interpolated percentile over `values`. None if empty."""
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
    """Extract floats from `rows[*][key]`, skipping None entries."""
    return [float(row[key]) for row in rows if row.get(key) is not None]


def distribution_stats(
    values: list[float],
    percentiles: tuple[int, ...] = (50, 90, 99),
    *,
    include_mean: bool = True,
    include_min: bool = False,
    include_max: bool = True,
) -> dict[str, float | int | None]:
    """Build a standard {count, mean, min?, p<X>..., max?} dict from values."""
    stats: dict[str, float | int | None] = {"count": len(values)}
    if include_mean:
        stats["mean"] = statistics.fmean(values) if values else None
    if include_min:
        stats["min"] = min(values, default=None)
    for pct in percentiles:
        stats[f"p{pct}"] = percentile(values, pct)
    if include_max:
        stats["max"] = max(values, default=None)
    return stats


def lookup(summary: dict, path: tuple[str, ...], field: str | None, mode: str):
    """Return summary['metrics'][path[0]][mode][path[1]]...[field], or None.

    If field is None, return the node itself after path traversal.
    """
    node = summary.get("metrics", {}).get(path[0], {}).get(mode)
    if node is None:
        return None
    for segment in path[1:]:
        node = node.get(segment)
        if node is None:
            return None
    if field is None:
        return node
    return node.get(field)


def fmt_pair(values: list[float], scale: float = 1.0, fmt: str = "{:.4f}") -> str:
    import statistics
    if not values:
        return "n/a"
    if len(values) == 1:
        return fmt.format(values[0] * scale) + "  (n=1)"
    mean = statistics.mean(values) * scale
    stdev = statistics.stdev(values) * scale
    return f"{fmt.format(mean)} +/- {fmt.format(stdev)}"


def parse_modes_arg(arg: str) -> tuple[str, ...]:
    requested = [m.strip() for m in arg.split(",") if m.strip()]
    invalid = [m for m in requested if m not in MODES_DEFAULT]
    if invalid:
        raise SystemExit(f"Unknown mode(s) in --modes: {invalid}")
    return tuple(requested)
