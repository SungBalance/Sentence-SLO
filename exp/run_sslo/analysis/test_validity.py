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


def _mk_summary(cpslo: dict | None = None, **workload_per_mode) -> dict:
    return {
        "metrics": {
            "workload": workload_per_mode,
            "cpslo": cpslo or {},
        },
        "config": {},
    }


def test_validity_pass_clean_run():
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 100,
        "tokens_per_second": 1000.0,
        "num_chunks_total": 500,
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
        "num_chunks_total": 500,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("drop_rate" in x for x in r.invalid_reason)


def test_validity_fail_high_timeout_rate():
    s = _mk_summary(baseline={
        "num_requests_total": 100,
        "num_requests_completed": 100,
        "num_requests_timeout": 5,
        "num_chunks_total": 500,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("timeout_rate" in x for x in r.invalid_reason)


def test_no_harm_fail_throughput_regression():
    s = _mk_summary(
        baseline={"num_requests_total": 100, "num_requests_completed": 100,
                  "tokens_per_second": 1000.0, "num_chunks_total": 500},
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "tokens_per_second": 800.0, "num_chunks_total": 500},
    )
    r = validate_run(s, baseline_tokens_per_second=1000.0)
    assert r.no_harm_pass is False
    assert any("throughput_regression" in x for x in r.invalid_reason)


def test_no_harm_pass_throughput_within_tolerance():
    # 0.95 ratio is within the 10% allowance.
    s = _mk_summary(
        baseline={"num_requests_total": 100, "num_requests_completed": 100,
                  "tokens_per_second": 1000.0, "num_chunks_total": 500},
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "tokens_per_second": 950.0, "num_chunks_total": 500},
    )
    r = validate_run(s, baseline_tokens_per_second=1000.0)
    assert r.no_harm_pass is True


def test_missing_chunk_records():
    s = _mk_summary(baseline={
        "num_requests_total": 10,
        "num_requests_completed": 10,
        "num_chunks_total": 0,
    })
    r = validate_run(s)
    assert r.validity_pass is False
    assert "missing_chunk_records" in r.invalid_reason


def test_validity_fail_queue_stall_p99_too_high():
    s = _mk_summary(
        cpslo={"sslo": {
            "queue_stall_distribution": {"p99": QUEUE_STALL_P99_MAX_S + 1.0},
        }},
        baseline={"num_requests_total": 10, "num_requests_completed": 10,
                  "num_chunks_total": 50},
    )
    r = validate_run(s)
    assert r.validity_pass is False
    assert any("queue_stall_p99_too_high" in x for x in r.invalid_reason)


def test_validity_fail_pending_time_p99_too_high():
    s = _mk_summary(
        cpslo={"sslo": {
            "pending_time_distribution": {"p99": PENDING_TIME_P99_MAX_S + 0.5},
        }},
        baseline={"num_requests_total": 10, "num_requests_completed": 10,
                  "num_chunks_total": 50},
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
        "num_chunks_total": 500,
    })
    r = v.validate_run(s)
    assert r.validity_pass is False


def test_no_harm_skipped_without_baseline_tps():
    # Regression check is bypassed when caller has no paired baseline.
    s = _mk_summary(
        sslo={"num_requests_total": 100, "num_requests_completed": 100,
              "tokens_per_second": 1.0, "num_chunks_total": 500},
    )
    r = validate_run(s, baseline_tokens_per_second=None)
    assert r.no_harm_pass is True


def test_throughput_regression_constant_is_10pct():
    # Pinned so frontier-selection code can rely on the documented bound.
    assert THROUGHPUT_REGRESSION_MAX == pytest.approx(0.10)
