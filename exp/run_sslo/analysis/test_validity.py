"""Tests for the Phase 6 validity / no-harm gate."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from validity import (
    PENDING_TIME_P99_MAX_S,
    QUEUE_STALL_P99_MAX_S,
    THROUGHPUT_REGRESSION_MAX,
    ValidityReport,
    validate_run,
)


def _mode_metrics(**overrides) -> dict:
    metrics = {
        "ttft_p95_s": 0.5,
        "ttfc_p95_s": 1.0,
        "tpot_p95_s": 0.05,
        "queue_stall_p99_s": 2.0,
        "pending_time_p99_s": 1.0,
        "mean_handling_users": 50.0,
        "mean_running": 30.0,
        "mean_pending": 20.0,
        "mean_waiting": 5.0,
        "corrected_processed_tokens_per_s": 1000.0,
        "drop_timeout_rate": 0.0,
        "starvation_count": 0,
        "request_max_stall_s_p99": 0.3,
    }
    metrics.update(overrides)
    return metrics


def _mk_summary(
    mode_metrics: dict[str, dict] | None = None,
    request_cu_slo_violation: dict | None = None,
    **workload_per_mode,
) -> dict:
    mode_metrics = dict(mode_metrics or {})
    for mode in workload_per_mode:
        mode_metrics.setdefault(mode, _mode_metrics())
    if request_cu_slo_violation is None:
        request_cu_slo_violation = {
            mode: {"tau_1": {"rate": 0.01}}
            for mode in mode_metrics
        }
    return {
        "metrics": {
            "workload": workload_per_mode,
            "request_cu_slo_violation": request_cu_slo_violation,
            **mode_metrics,
        },
        "config": {},
    }


def test_validity_pass_clean_run():
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 100,
        "num_consumable_units_total": 500,
    })
    r = validate_run(s)
    assert isinstance(r, ValidityReport)
    assert r.validity_pass is True
    assert r.no_harm_pass is True
    assert r.invalid_reason == []


def test_validity_fail_high_drop_rate():
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 80,
        "num_consumable_units_total": 500,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("drop_rate" in x for x in r.invalid_reason)


def test_validity_fail_high_timeout_rate():
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 100,
        "num_requests_timeout": 5,
        "num_consumable_units_total": 500,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("timeout_rate" in x for x in r.invalid_reason)


def test_no_harm_fail_throughput_regression():
    s = _mk_summary(
        baseline={"num_requests_total": 100, "num_requests_completed": 100,
                  "num_consumable_units_total": 500},
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "num_consumable_units_total": 500},
        mode_metrics={
            "baseline": _mode_metrics(corrected_processed_tokens_per_s=1000.0),
            "sslo": _mode_metrics(corrected_processed_tokens_per_s=800.0),
        },
    )
    r = validate_run(s, baseline_tokens_per_second=1000.0)
    assert r.no_harm_pass is False
    assert any("throughput_regression" in x for x in r.invalid_reason)


def test_no_harm_pass_throughput_within_tolerance():
    # 0.95 ratio is within the 10% allowance.
    s = _mk_summary(
        baseline={"num_requests_total": 100, "num_requests_completed": 100,
                  "num_consumable_units_total": 500},
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "num_consumable_units_total": 500},
        mode_metrics={
            "baseline": _mode_metrics(corrected_processed_tokens_per_s=1000.0),
            "sslo": _mode_metrics(corrected_processed_tokens_per_s=950.0),
        },
    )
    r = validate_run(s, baseline_tokens_per_second=1000.0)
    assert r.no_harm_pass is True


def test_missing_chunk_records():
    s = _mk_summary(baseline={
        "num_requests_total": 10,
        "num_requests_completed": 10,
        "num_consumable_units_total": 0,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert "missing_chunk_records" in r.invalid_reason


def test_validity_fail_queue_stall_p99_too_high():
    s = _mk_summary(
        mode_metrics={
            "sslo": _mode_metrics(
                queue_stall_p99_s=QUEUE_STALL_P99_MAX_S + 1.0),
        },
        sslo={"num_requests_total": 10, "num_requests_completed": 10,
              "num_consumable_units_total": 50},
    )
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("queue_stall_p99_too_high" in x for x in r.invalid_reason)


def test_validity_fail_pending_time_p99_too_high():
    s = _mk_summary(
        mode_metrics={
            "sslo": _mode_metrics(
                pending_time_p99_s=PENDING_TIME_P99_MAX_S + 0.5),
        },
        sslo={"num_requests_total": 10, "num_requests_completed": 10,
              "num_consumable_units_total": 50},
    )
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("pending_time_p99_too_high" in x for x in r.invalid_reason)


def test_threshold_constants_overridable_via_monkeypatch(monkeypatch):
    # Verifies tests can lower threshold to trigger a failure.
    import validity as v
    monkeypatch.setattr(v, "DROP_RATE_MAX", 0.0)
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 99,
        "num_consumable_units_total": 500,
    })
    r = v.validate_run(s)
    assert r.validity_pass is False


def test_no_harm_skipped_without_baseline_tps():
    # Regression check is bypassed when caller has no paired baseline.
    s = _mk_summary(
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "num_consumable_units_total": 500},
        mode_metrics={
            "sslo": _mode_metrics(corrected_processed_tokens_per_s=1.0),
        },
    )
    r = validate_run(s, baseline_tokens_per_second=None)
    assert r.no_harm_pass is True


def test_throughput_regression_constant_is_10pct():
    # Pinned so frontier-selection code can rely on the documented bound.
    assert THROUGHPUT_REGRESSION_MAX == pytest.approx(0.10)
