# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase A v2 SSLO config."""

import pytest

from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import RequestSLOState


def test_defaults_match_phase_a_v2_plan():
    cfg = SsloConfig()
    assert cfg.enabled is False
    assert cfg.offloading is False
    assert cfg.adaptive_batching is False
    assert cfg.num_warmup_chunks == 4
    assert cfg.tpot_bucket_size == 8
    assert cfg.tpot_ema_alpha == 0.1
    assert cfg.critical_threshold == 1.0
    assert cfg.pending_in_threshold == 0.3
    assert cfg.pending_out_threshold == 0.7
    assert cfg.offloading_in_threshold == 0.5
    assert cfg.offloading_out_threshold == 0.7
    assert cfg.adaptive_batching_min_throughput_ratio == 0.9
    assert cfg.offload_safety_margin_s == 0.05
    assert cfg.offload_bandwidth_bytes_per_s == 1e10
    assert cfg.seconds_per_word == 0.28
    assert cfg.chunk_unit == "sentence"
    assert cfg.min_chunk_tokens == 16
    assert not hasattr(cfg, "adaptive_batch_size")
    assert not hasattr(cfg, "max_pending_num")
    assert not hasattr(cfg, "iter_time_ema_alpha")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_warmup_chunks", 0),
        ("tpot_bucket_size", 0),
        ("tpot_ema_alpha", 0.0),
        ("tpot_ema_alpha", 1.1),
        ("critical_threshold", -0.1),
        ("pending_in_threshold", -0.1),
        ("offloading_in_threshold", -0.1),
        ("adaptive_batching_min_throughput_ratio", 0.0),
        ("offload_safety_margin_s", -0.1),
        ("offload_bandwidth_bytes_per_s", 0.0),
        ("seconds_per_word", -0.1),
        ("min_chunk_tokens", -1),
    ],
)
def test_validation_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        SsloConfig(**{field: value})


def test_validation_rejects_bad_threshold_ordering():
    with pytest.raises(ValueError, match="pending_in_threshold"):
        SsloConfig(pending_in_threshold=0.8, pending_out_threshold=0.7)
    with pytest.raises(ValueError, match="offloading_in_threshold"):
        SsloConfig(offloading_in_threshold=0.8, offloading_out_threshold=0.7)


def test_invalid_chunk_unit_raises():
    with pytest.raises(ValueError, match="chunk_unit"):
        SsloConfig(chunk_unit="token")


def test_build_slo_state_freezes_config_constants():
    cfg = SsloConfig(
        seconds_per_word=0.5,
        num_warmup_chunks=7,
        chunk_unit="paragraph",
        min_chunk_tokens=24,
    )
    state = build_slo_state(cfg)
    assert isinstance(state, RequestSLOState)
    assert state.seconds_per_word == 0.5
    assert state.num_warmup_chunks == 7
    assert state.chunk_unit == "paragraph"
    assert state.min_chunk_tokens == 24
