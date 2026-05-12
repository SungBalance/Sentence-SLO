"""Tests for the Phase 5 stall-interval merge helper."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import progress_metrics as pm


def _chunk(
    idx: int,
    stall_start: float | None = None,
    stall_end: float | None = None,
    stall_duration: float | None = None,
) -> dict:
    return {
        "chunk_idx": idx,
        "stall_start_ts": stall_start,
        "stall_end_ts": stall_end,
        "stall_duration_s": stall_duration,
    }


def test_compute_stall_intervals_empty():
    """All chunks on-time → no intervals."""
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    assert pm.compute_stall_intervals(chunks) == []


def test_compute_stall_intervals_separate_with_consume_gap():
    """Two stalls separated by a non-stalled chunk → two intervals.

    Also covers the common case where stall_end(N) != stall_start(N+1)
    because the consume gap pushes the next deadline forward.
    """
    chunks = [
        _chunk(0),
        _chunk(1, stall_start=10.0, stall_end=10.5, stall_duration=0.5),
        _chunk(2),  # on-time, breaks the run
        _chunk(3, stall_start=12.0, stall_end=12.7, stall_duration=0.7),
    ]
    intervals = pm.compute_stall_intervals(chunks)
    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((10.0, 10.5, 0.5))
    assert intervals[1] == pytest.approx((12.0, 12.7, 0.7))


def test_compute_stall_intervals_merges_zero_gap():
    """stall_end(N) == stall_start(N+1) → merged into one interval."""
    chunks = [
        _chunk(0),
        _chunk(1, stall_start=5.0, stall_end=5.4, stall_duration=0.4),
        _chunk(2, stall_start=5.4, stall_end=6.0, stall_duration=0.6),
        _chunk(3, stall_start=6.0, stall_end=6.3, stall_duration=0.3),
    ]
    intervals = pm.compute_stall_intervals(chunks)
    assert len(intervals) == 1
    start, end, duration = intervals[0]
    assert start == pytest.approx(5.0)
    assert end == pytest.approx(6.3)
    assert duration == pytest.approx(1.3)
