"""CP-SLO validity + no-harm gating.

A run is excluded from the main frontier if ANY of the threshold
predicates below trip. Excluded runs still appear in
validity_checks.csv with a populated invalid_reason list.

Thresholds declared up-front per the TODO contract:
"All guardrail thresholds must be fixed before selecting frontier
points."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Pre-declared no-harm thresholds (escalation #3 defaults).
DROP_RATE_MAX = 0.01            # 1% of requests may drop
TIMEOUT_RATE_MAX = 0.01         # 1% of requests may timeout
THROUGHPUT_REGRESSION_MAX = 0.10  # tokens/s vs baseline must be >= 0.90
QUEUE_STALL_P99_MAX_S = 5.0     # p99 queue_stall_s upper bound
PENDING_TIME_P99_MAX_S = 5.0    # p99 pending_time_s upper bound


@dataclass
class ValidityReport:
    validity_pass: bool
    no_harm_pass: bool
    invalid_reason: list[str] = field(default_factory=list)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def validate_run(
    summary: dict[str, Any],
    *,
    baseline_tokens_per_second: float | None = None,
) -> ValidityReport:
    """Apply validity + no-harm predicates to a per-run summary.

    Args:
      summary: the dict written to summary.json by analyze.py.
      baseline_tokens_per_second: tokens/s from a paired baseline run;
        if None the throughput-regression check is skipped (e.g.
        single-mode dev runs).
    """
    reasons: list[str] = []
    metrics = summary.get("metrics", {}) or {}

    # Required CP-SLO fields: chunk records must exist for at least one mode.
    workload = metrics.get("workload", {}) or {}
    chunks_present = any(
        (w or {}).get("num_chunks_total", 0) > 0
        for w in workload.values()
    )
    if not chunks_present:
        reasons.append("missing_chunk_records")

    # Per-mode completion / drop / timeout / queueing checks.
    for mode, mode_workload in workload.items():
        if not mode_workload:
            continue
        total = mode_workload.get("num_requests_total", 0) or 0
        completed = mode_workload.get("num_requests_completed", 0) or 0
        drops = total - completed if total else 0
        drop_rate = (drops / total) if total else 0.0
        if drop_rate > DROP_RATE_MAX:
            reasons.append(f"drop_rate_too_high:{mode}={drop_rate:.3f}")
        timeouts = mode_workload.get("num_requests_timeout", 0) or 0
        timeout_rate = (timeouts / total) if total else 0.0
        if timeout_rate > TIMEOUT_RATE_MAX:
            reasons.append(f"timeout_rate_too_high:{mode}={timeout_rate:.3f}")

    # Per-mode queueing guardrails — read from cpslo / scheduler sections.
    cpslo = metrics.get("cpslo", {}) or {}
    for mode, mode_metrics in cpslo.items():
        if not mode_metrics:
            continue
        queue_dist = mode_metrics.get("queue_stall_distribution", {}) or {}
        p99 = _safe_float(queue_dist.get("p99"))
        if p99 is not None and p99 > QUEUE_STALL_P99_MAX_S:
            reasons.append(f"queue_stall_p99_too_high:{mode}={p99:.2f}s")
        pending_dist = mode_metrics.get("pending_time_distribution", {}) or {}
        p99p = _safe_float(pending_dist.get("p99"))
        if p99p is not None and p99p > PENDING_TIME_P99_MAX_S:
            reasons.append(f"pending_time_p99_too_high:{mode}={p99p:.2f}s")

    # No-harm: throughput regression vs baseline (paired).
    no_harm_pass = True
    if baseline_tokens_per_second is not None and baseline_tokens_per_second > 0:
        for mode, mode_workload in workload.items():
            if mode == "baseline" or not mode_workload:
                continue
            tps = _safe_float(mode_workload.get("tokens_per_second"))
            if tps is None:
                continue
            ratio = tps / baseline_tokens_per_second
            if ratio < (1.0 - THROUGHPUT_REGRESSION_MAX):
                no_harm_pass = False
                reasons.append(
                    f"throughput_regression:{mode}={ratio:.3f}")

    return ValidityReport(
        validity_pass=(len(reasons) == 0),
        no_harm_pass=no_harm_pass,
        invalid_reason=reasons,
    )
