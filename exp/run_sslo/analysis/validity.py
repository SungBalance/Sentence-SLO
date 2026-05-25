"""SSLO validity + no-harm gating.

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
    oom_count: int = 0
    preemption_count: int = 0
    recompute_count: int = 0
    offload_count: int = 0
    swap_count: int = 0
    starvation_count: int = 0
    included_in_main_result: bool = False


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _active_mode(summary: dict[str, Any]) -> str | None:
    cfg = summary.get("config", {}) or {}
    modes = cfg.get("modes_run") or []
    if modes:
        return modes[0]
    return cfg.get("run_kind") or (summary.get("run_meta", {}) or {}).get("policy")


def _flat_mode_metrics(metrics: dict[str, Any]):
    for mode, mode_metrics in metrics.items():
        if not isinstance(mode_metrics, dict):
            continue
        if (
            "queue_stall_p99_s" in mode_metrics
            or "pending_time_p99_s" in mode_metrics
            or "starvation_count" in mode_metrics
        ):
            yield mode, mode_metrics


def _mode_names(
    workload: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    seen: set[str] = set()
    modes: list[str] = []
    for mode in workload:
        seen.add(mode)
        modes.append(mode)
    for mode, _ in _flat_mode_metrics(metrics):
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes


def _sum_preemptions(request_rows: list[dict[str, Any]] | None) -> int:
    if request_rows is None:
        return 0
    return sum(_safe_int(row.get("num_preemptions")) or 0 for row in request_rows)


def validate_run(
    summary: dict[str, Any],
    *,
    baseline_tokens_per_second: float | None = None,
    request_rows: list[dict[str, Any]] | None = None,
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

    # Required per-unit fields: chunk records must exist for at least one mode.
    workload = metrics.get("workload", {}) or {}
    chunks_present = any(
        (
            (w or {}).get("num_chunks_total")
            or (w or {}).get("num_consumable_units_total")
            or 0
        ) > 0
        for w in workload.values()
    )
    if not chunks_present:
        reasons.append("missing_chunk_records")

    # Per-mode completion / drop / timeout checks.
    for mode in _mode_names(workload, metrics):
        mode_workload = workload.get(mode, {}) or {}
        mode_metrics = metrics.get(mode, {}) or {}
        total = mode_workload.get("num_requests_total", 0) or 0
        completed = mode_workload.get("num_requests_completed", 0) or 0
        if total:
            drops = total - completed
            drop_rate = drops / total
        else:
            drop_rate = _safe_float(mode_metrics.get("drop_timeout_rate"))
        if drop_rate is not None and drop_rate > DROP_RATE_MAX:
            reasons.append(f"drop_rate_too_high:{mode}={drop_rate:.3f}")
        timeouts = mode_workload.get("num_requests_timeout", 0) or 0
        timeout_rate = (timeouts / total) if total else None
        if timeout_rate is not None and timeout_rate > TIMEOUT_RATE_MAX:
            reasons.append(f"timeout_rate_too_high:{mode}={timeout_rate:.3f}")

    # Per-mode queueing guardrails — read from R2 flat per-mode metrics.
    for mode, mode_metrics in _flat_mode_metrics(metrics):
        p99 = _safe_float(mode_metrics.get("queue_stall_p99_s"))
        if p99 is not None and p99 > QUEUE_STALL_P99_MAX_S:
            reasons.append(f"queue_stall_p99_too_high:{mode}={p99:.2f}s")
        p99p = _safe_float(mode_metrics.get("pending_time_p99_s"))
        if p99p is not None and p99p > PENDING_TIME_P99_MAX_S:
            reasons.append(f"pending_time_p99_too_high:{mode}={p99p:.2f}s")

    # No-harm: throughput regression vs baseline (paired).
    no_harm_pass = True
    if baseline_tokens_per_second is not None and baseline_tokens_per_second > 0:
        for mode in _mode_names(workload, metrics):
            if mode == "baseline":
                continue
            mode_workload = workload.get(mode, {}) or {}
            tps = _safe_float(
                (metrics.get(mode, {}) or {}).get(
                    "corrected_processed_tokens_per_s"))
            if tps is None:
                tps = _safe_float(mode_workload.get("tokens_per_second"))
            if tps is None:
                continue
            ratio = tps / baseline_tokens_per_second
            if ratio < (1.0 - THROUGHPUT_REGRESSION_MAX):
                no_harm_pass = False
                reasons.append(
                    f"throughput_regression:{mode}={ratio:.3f}")

    active_mode = _active_mode(summary)
    active_metrics = (metrics.get(active_mode, {}) or {}) if active_mode else {}
    active_workload = workload.get(active_mode, {}) if active_mode else {}
    starvation_count = (
        _safe_int(active_metrics.get("starvation_count"))
        if isinstance(active_metrics, dict)
        else None
    )
    if starvation_count is None:
        starvation_count = _safe_int(
            (active_workload or {}).get("starvation_count"))
    preemption_count = _sum_preemptions(request_rows)
    if request_rows is None:
        preemption_count = _safe_int(
            (summary.get("run_meta", {}) or {}).get("num_preemptions_total")) or 0
    validity_pass = len(reasons) == 0
    return ValidityReport(
        validity_pass=validity_pass,
        no_harm_pass=no_harm_pass,
        invalid_reason=reasons,
        preemption_count=preemption_count,
        starvation_count=starvation_count or 0,
        included_in_main_result=validity_pass and no_harm_pass,
    )
