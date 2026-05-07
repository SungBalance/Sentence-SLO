"""Shared metrics constants and helpers for SSLO sweep aggregators."""
from __future__ import annotations

from dataclasses import dataclass


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
        MetricSpec(("slack",),             "neg_ratio", 100, "{:.4f}", "neg slack ratio (%)"),
        MetricSpec(("slack",),             "mean",      1.0, "{:.3f}", "slack mean (s)"),
        MetricSpec(("slack",),             "p50",       1.0, "{:.3f}", "slack p50 (s)"),
        MetricSpec(("slack",),             "p90",       1.0, "{:.3f}", "slack p90 (s)"),
        MetricSpec(("slack",),             "p99",       1.0, "{:.3f}", "slack p99 (s)"),
        MetricSpec(("slack",),             "max",       1.0, "{:.3f}", "slack max (s)"),
        MetricSpec(("slack", "magnitude"), "mean",      1.0, "{:.3f}", "neg slack magnitude mean (s)"),
        MetricSpec(("slack", "magnitude"), "p50",       1.0, "{:.3f}", "neg slack magnitude p50 (s)"),
        MetricSpec(("slack", "magnitude"), "p90",       1.0, "{:.3f}", "neg slack magnitude p90 (s)"),
        MetricSpec(("slack", "magnitude"), "p99",       1.0, "{:.3f}", "neg slack magnitude p99 (s)"),
        MetricSpec(("slack", "magnitude"), "max",       1.0, "{:.3f}", "neg slack magnitude max (s)"),
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
        MetricSpec(("scheduler", "combined"), "mean", 1.0, "{:.2f}", "combined mean"),
        MetricSpec(("scheduler", "combined"), "p50",  1.0, "{:.2f}", "combined p50"),
        MetricSpec(("scheduler", "combined"), "p90",  1.0, "{:.2f}", "combined p90"),
        MetricSpec(("scheduler", "combined"), "p99",  1.0, "{:.2f}", "combined p99"),
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
)


def lookup(summary: dict, path: tuple[str, ...], field: str, mode: str):
    """Return summary['metrics'][path[0]][mode][path[1]]...[field], or None."""
    node = summary.get("metrics", {}).get(path[0], {}).get(mode)
    if node is None:
        return None
    for segment in path[1:]:
        node = node.get(segment)
        if node is None:
            return None
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
