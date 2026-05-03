# SPDX-License-Identifier: Apache-2.0
"""Tests for SsloConfig and build_slo_state()."""
import pytest

from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import RequestSLOState


class TestSsloConfig:
    def test_defaults(self):
        cfg = SsloConfig()
        assert cfg.enabled is False
        assert cfg.seconds_per_word == 0.28
        assert cfg.chunk_unit == "sentence"
        assert cfg.estimator_type == "word_rate"
        assert cfg.offloading is False
        assert cfg.adaptive_batch_size is False
        assert cfg.max_consecutive_pending == 5
        assert cfg.ema_alpha == 0.2
        assert cfg.pending_slack_eps_num_tokens == 5

    def test_invalid_chunk_unit_raises(self):
        with pytest.raises(ValueError, match="chunk_unit"):
            SsloConfig(chunk_unit="invalid")

    def test_invalid_estimator_type_raises(self):
        with pytest.raises(ValueError, match="estimator_type"):
            SsloConfig(estimator_type="tts")


class TestBuildSloState:
    def test_default_builds_word_rate_sentence(self):
        state = build_slo_state(SsloConfig())
        assert isinstance(state, RequestSLOState)

    def test_paragraph_unit_uses_paragraph_detector(self):
        import time
        state = build_slo_state(SsloConfig(chunk_unit="paragraph"))
        now = time.monotonic()
        state.on_text_delta("Hello world. More text.", now)
        assert len(state.chunk_records) == 0  # sentence boundary ignored
        state.on_text_delta("\n\nNext paragraph.", now)
        assert len(state.chunk_records) == 1

    def test_ema_alpha_propagated(self):
        state = build_slo_state(SsloConfig(ema_alpha=0.5))
        assert state._ema_alpha == 0.5

    def test_eps_num_tokens_propagated(self):
        state = build_slo_state(SsloConfig(pending_slack_eps_num_tokens=10))
        assert state._pending_slack_eps_num_tokens == 10
