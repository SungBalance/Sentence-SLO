# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO config."""

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import RequestSLOState


def test_defaults():
    cfg = SsloConfig()
    assert cfg.method == "baseline"
    assert cfg.policy == "threshold"
    assert cfg.adaptive_batching is False
    assert cfg.num_warmup_chunks == 4
    assert cfg.tpot_ema_alpha == 0.1
    assert cfg.critical_threshold == 1.0
    assert cfg.pending_in_threshold == 0.3
    assert cfg.pending_out_threshold == 0.7
    assert cfg.adaptive_batching_min_throughput_ratio == 0.9
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
        ("tpot_ema_alpha", 0.0),
        ("tpot_ema_alpha", 1.1),
        ("critical_threshold", -0.1),
        ("pending_in_threshold", -0.1),
        ("adaptive_batching_min_throughput_ratio", 0.0),
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


def test_invalid_chunk_unit_raises():
    with pytest.raises(ValueError, match="chunk_unit"):
        SsloConfig(chunk_unit="token")


def test_from_config_freezes_constants():
    cfg = SsloConfig(
        seconds_per_word=0.5,
        num_warmup_chunks=7,
        chunk_unit="paragraph",
        min_chunk_tokens=24,
    )
    state = RequestSLOState.from_config(cfg)
    assert isinstance(state, RequestSLOState)
    # num_warmup_chunks persists on the instance (phase property reads it).
    assert state.num_warmup_chunks == 7
    # The other knobs propagate into the helpers built in __post_init__,
    # not onto the instance.
    assert state.chunk_separator.chunk_unit == "paragraph"
    assert state.chunk_separator.min_chunk_tokens == 24
    assert state.consume_estimator.seconds_per_word == 0.5
