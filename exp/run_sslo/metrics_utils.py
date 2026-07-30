"""Shared metrics constants and helpers for SSLO sweep aggregators."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


MODES_DEFAULT = (
    "baseline",
    # SSLO
    "progress_serve",
    # SSLO
    "progress_serve_adaptive",
)


@dataclass(frozen=True)
class MetricSpec:
    path: tuple[str, ...]
    field: str
    scale: float
    fmt: str
    label: str


# Standard quantile cohort for latency-like distributions.
_DIST_FIELDS: tuple[str, ...] = ("mean", "p50", "p90", "p95", "p99", "max")


def _dist_specs(
    path: tuple[str, ...],
    scale: float,
    fmt: str,
    label_prefix: str,
    fields: tuple[str, ...] = _DIST_FIELDS,
) -> tuple[MetricSpec, ...]:
    """Build mean/p50/p90/p95/p99/max specs for a distribution-bearing node."""
    return tuple(
        MetricSpec(path, f, scale, fmt, f"{label_prefix} {f}") for f in fields
    )


DISPLAY_GROUPS: tuple[tuple[str, tuple[MetricSpec, ...]], ...] = (
    ("Latency", (
        *_dist_specs(("ttft", "post_cap"), 1.0,  "{:.3f}", "TTFT post_cap (s)"),
        *_dist_specs(("ttfc",),            1.0,  "{:.3f}", "TTFC (s)"),
        *_dist_specs(("tpot",),            1000, "{:.2f}", "TPOT (ms)"),
    )),
    ("Queue stall", (
        *_dist_specs(("queue_stall",), 1000, "{:.2f}", "queue stall (ms)"),
    )),
    ("Slack", (
        MetricSpec(("slack",), "neg_ratio", 100, "{:.4f}", "neg slack ratio (%)"),
        *_dist_specs(("slack",),                      1.0, "{:.3f}", "slack (s)"),
        *_dist_specs(("slack", "violated-magnitude"), 1.0, "{:.3f}", "violated magnitude (s)"),
    )),
    ("SLO compliance", (
        MetricSpec(("slo_compliance",), "rate",           100, "{:.2f}", "SLO compliance (%)"),
        MetricSpec(("slo_compliance",), "count",          1.0, "{:.1f}", "SLO compliant reqs"),
        MetricSpec(("slo_compliance",), "total_requests", 1.0, "{:.1f}", "SLO total reqs"),
    )),
    ("Scheduler occupancy", (
        *_dist_specs(("scheduler", "running"), 1.0, "{:.2f}", "running"),
        MetricSpec(("scheduler", "num_handling_users"), "min",  1.0, "{:.2f}", "handling-users min"),
        *_dist_specs(("scheduler", "num_handling_users"), 1.0, "{:.2f}", "handling-users"),
    )),
    ("Pending dynamics", (
        *_dist_specs(("pending", "time"),      1.0, "{:.3f}", "pending time (s)"),
        *_dist_specs(("pending", "intervals"), 1.0, "{:.2f}", "pending intervals"),
    )),
    ("Streaming smoothness", (
        *_dist_specs(("inter_chunk_delay",), 1000, "{:.2f}", "inter-chunk delay (ms)"),
    )),
    ("Stall (request)", (
        *_dist_specs(("progress_request", "total_stall_time"),    1.0, "{:.3f}", "total stall (s)"),
        *_dist_specs(("progress_request", "max_stall_time"),      1.0, "{:.3f}", "max stall (s)"),
        *_dist_specs(("progress_request", "stall_fraction"),      100, "{:.2f}", "stall fraction (%)"),
        *_dist_specs(("progress_request", "num_stall_intervals"), 1.0, "{:.2f}", "stall intervals"),
    )),
    ("Latency (request)", (
        *_dist_specs(("progress_request", "completion_latency"), 1.0, "{:.3f}", "completion latency (s)"),
        *_dist_specs(("progress_request", "demand_duration"),    1.0, "{:.3f}", "demand duration (s)"),
    )),
    ("Workload", (
        *_dist_specs(("workload", "num_prompt_tokens"), 1.0, "{:.1f}", "prompt tokens"),
        *_dist_specs(("workload", "num_output_tokens"), 1.0, "{:.1f}", "output tokens"),
        *_dist_specs(("workload", "num_chunks"),        1.0, "{:.2f}", "chunks/req"),
    )),
    ("Throughput", (
        MetricSpec(("throughput",), "tokens_per_second",   1.0, "{:.2f}", "tokens/s"),
        MetricSpec(("throughput",), "completed_req_per_s", 1.0, "{:.2f}", "completed req/s"),
        MetricSpec(("throughput",), "duration_s",          1.0, "{:.2f}", "measurement window (s)"),
    )),
    ("Handling users (time-weighted)", (
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
    percentiles: tuple[int, ...] = (50, 90, 95, 99),
    *,
    include_mean: bool = True,
    include_min: bool = False,
    include_max: bool = True,
) -> dict[str, float | int | None]:
    """Build a standard {count, mean, min?, p<X>..., max?} dict from values.

    Default percentile set is (50, 90, 95, 99) so latency-like metrics
    consistently expose mean/p50/p90/p95/p99/max in summary.csv. Older
    callsites passing custom tuples may still do so for narrower views.
    """
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
