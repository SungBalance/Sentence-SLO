# SPDX-License-Identifier: Apache-2.0
"""Tests for SsloConfig and build_slo_state()."""
import pytest

from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import (
    EmaChunkGenerationEstimator,
    PercentileChunkGenerationEstimator,
    RequestSLOState,
)


class TestSsloConfig:
    def test_defaults(self):
        cfg = SsloConfig()
        assert cfg.enabled is False
        assert cfg.seconds_per_word == 0.28
        assert cfg.chunk_unit == "sentence"
        assert cfg.estimator_type == "word_rate"
        assert cfg.offloading is False
        assert cfg.adaptive_batch_size is False
        assert cfg.max_pending_num == 5
        assert cfg.severe_overdue_margin_s == 0.5
        assert cfg.near_deadline_margin_s == 0.1
        assert cfg.min_num_running_reqs == 1
        assert cfg.cap_growth_safe_steps == 32
        assert cfg.cap_growth_step == 1
        assert cfg.forced_pending_weight == 0.1
        assert cfg.iter_time_ema_alpha == 0.1
        assert not hasattr(cfg, "max_consecutive_pending")
        assert cfg.ema_alpha == 0.2
        assert cfg.chunk_gen_estimator == "ema"
        assert cfg.chunk_gen_p99_window == 100
        assert cfg.pending_pressure_lambda == 0.05
        assert cfg.pending_hysteresis_gap == 0.5
        assert cfg.min_chunk_tokens == 16

    def test_phase_a_fields_validate(self):
        with pytest.raises(ValueError, match="max_pending_num"):
            SsloConfig(enabled=True, max_pending_num=0)
        with pytest.raises(ValueError, match="severe_overdue_margin_s"):
            SsloConfig(enabled=True, severe_overdue_margin_s=-0.1)
        with pytest.raises(ValueError, match="iter_time_ema_alpha"):
            SsloConfig(enabled=True, iter_time_ema_alpha=0.0)
        with pytest.raises(ValueError, match="iter_time_ema_alpha"):
            SsloConfig(enabled=True, iter_time_ema_alpha=1.1)

    def test_min_chunk_tokens_custom(self):
        cfg = SsloConfig(min_chunk_tokens=32)
        assert cfg.min_chunk_tokens == 32

    def test_min_chunk_tokens_zero_allowed(self):
        cfg = SsloConfig(min_chunk_tokens=0)
        assert cfg.min_chunk_tokens == 0

    def test_min_chunk_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="min_chunk_tokens"):
            SsloConfig(min_chunk_tokens=-1)

    def test_invalid_chunk_unit_raises(self):
        with pytest.raises(ValueError, match="chunk_unit"):
            SsloConfig(chunk_unit="invalid")

    def test_invalid_estimator_type_raises(self):
        with pytest.raises(ValueError, match="estimator_type"):
            SsloConfig(estimator_type="tts")

    def test_invalid_chunk_gen_estimator_raises(self):
        with pytest.raises(ValueError, match="chunk_gen_estimator"):
            SsloConfig(chunk_gen_estimator="mean")


class TestBuildSloState:
    def test_default_builds_word_rate_sentence(self):
        state = build_slo_state(SsloConfig())
        assert isinstance(state, RequestSLOState)

    def test_paragraph_unit_uses_paragraph_detector(self):
        import time
        # min_chunk_tokens=0 here so this test only exercises the detector
        # choice, not the min-token guard.
        state = build_slo_state(SsloConfig(chunk_unit="paragraph",
                                           min_chunk_tokens=0))
        now = time.monotonic()
        state.on_text_delta("Hello world. More text.", now)
        assert len(state.chunk_records) == 0  # sentence boundary ignored
        state.on_text_delta("\n\nNext paragraph.", now)
        assert len(state.chunk_records) == 1

    def test_ema_alpha_propagated(self):
        state = build_slo_state(SsloConfig(ema_alpha=0.5))
        assert isinstance(state.chunk_gen_estimator, EmaChunkGenerationEstimator)
        assert state.chunk_gen_estimator._alpha == 0.5

    def test_p99_estimator_propagated(self):
        state = build_slo_state(SsloConfig(chunk_gen_estimator="p99"))
        assert isinstance(
            state.chunk_gen_estimator,
            PercentileChunkGenerationEstimator,
        )

    def test_min_chunk_tokens_propagated(self):
        state = build_slo_state(SsloConfig(min_chunk_tokens=24))
        assert state._min_chunk_tokens == 24
