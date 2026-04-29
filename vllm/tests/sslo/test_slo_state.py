# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm.sslo.slo_state."""

import pytest

from vllm.sslo.slo_state import (
    ConsumeEstimator,
    RequestSLOState,
    WordRateEstimator,
)


# ---------------------------------------------------------------------------
# WordRateEstimator tests
# ---------------------------------------------------------------------------


class TestWordRateEstimator:
    def test_default_rate(self):
        est = WordRateEstimator()
        # "hello world" -> 2 words * 0.28 s/word = 0.56
        assert est("hello world") == pytest.approx(0.56)

    def test_single_word(self):
        est = WordRateEstimator()
        assert est("hello") == pytest.approx(0.28)

    def test_custom_rate(self):
        est = WordRateEstimator(seconds_per_word=0.5)
        # "one two three" -> 3 words * 0.5 = 1.5
        assert est("one two three") == pytest.approx(1.5)

    def test_empty_string(self):
        est = WordRateEstimator()
        assert est("") == pytest.approx(0.0)

    def test_whitespace_only(self):
        est = WordRateEstimator()
        assert est("   ") == pytest.approx(0.0)

    def test_implements_protocol(self):
        est = WordRateEstimator()
        assert isinstance(est, ConsumeEstimator)


# ---------------------------------------------------------------------------
# RequestSLOState tests
# ---------------------------------------------------------------------------


class TestRequestSLOState:
    def test_initial_slack_is_zero(self):
        state = RequestSLOState(decoding_start=10.0)
        assert state.cumulative_slack == 0.0

    def test_no_flush_without_boundary(self):
        state = RequestSLOState(decoding_start=0.0)
        state.on_text_delta("hello world", 1.0)
        # No sentence boundary -> no flush -> slack stays 0.0
        assert state.cumulative_slack == 0.0

    def test_single_chunk_slack(self):
        """Chunk 0 slack is fixed at 0.0 (not computed)."""
        state = RequestSLOState(decoding_start=0.0)
        state.on_text_delta("Hello world.", 5.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_decoding_start_offset(self):
        """Chunk 0 slack is fixed at 0.0 regardless of decoding_start."""
        state = RequestSLOState(decoding_start=10.0)
        state.on_text_delta("Hello world.", 12.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_second_chunk_deadline_includes_first_consume(self):
        """
        decoding_start = 0.0, seconds_per_word = 1.0 for simplicity.

        Chunk 0: "Hello world." (2 words -> 2.0 s consume), ends at t=5.
          slack_0 = 0.0 (fixed, not computed)

        Chunk 1: "How are you?" (3 words -> 3.0 s consume), ends at t=8.
          deadline_1 = 0 + 2.0 (chunk 0's consume) = 2.0
          slack_1 = 2.0 - 8 = -6
        """
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est, decoding_start=0.0)

        state.on_text_delta("Hello world.", 5.0)
        assert state.cumulative_slack == pytest.approx(0.0)

        state.on_text_delta("How are you?", 8.0)
        assert state.cumulative_slack == pytest.approx(-6.0)

    def test_positive_slack(self):
        """Second chunk arrives before deadline -> positive slack.

        Chunk 0: "Hi." (1 word, 0.28 s consume) ends at t=100.1
          slack_0 = 0.0 (fixed, not computed)
        Chunk 1: "Hey?" (1 word, 0.28 s consume) ends at t=100.2
          deadline_1 = 100.0 + 0.28 = 100.28; slack_1 = 100.28 - 100.2 = +0.08
        """
        state = RequestSLOState(decoding_start=100.0)
        state.on_text_delta("Hi.", 100.1)
        state.on_text_delta("Hey?", 100.2)
        # deadline_1 = 100.0 + 0.28 (1 word * 0.28) = 100.28
        expected_slack = 100.0 + 0.28 - 100.2
        assert state.cumulative_slack == pytest.approx(expected_slack, abs=1e-9)

    def test_on_finish_flushes_remaining_text(self):
        """on_finish should flush non-boundary text as a final chunk."""
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est, decoding_start=0.0)
        # No boundary — text is held as pending
        state.on_text_delta("some partial text", 3.0)
        assert state.cumulative_slack == 0.0
        # Finish flushes it as chunk 0 -> slack fixed at 0.0
        state.on_finish(6.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_on_finish_no_pending_is_noop(self):
        """on_finish with nothing pending should not change slack."""
        state = RequestSLOState(decoding_start=0.0)
        state.on_text_delta("Done.", 2.0)
        slack_before = state.cumulative_slack
        state.on_finish(3.0)
        # Pending was flushed by on_text_delta; on_finish has nothing to flush.
        assert state.cumulative_slack == slack_before

    def test_boundary_characters(self):
        """All sentence boundary chars should trigger a flush (chunk 0 -> 0.0)."""
        for char in ".!?。！？…":
            state = RequestSLOState(decoding_start=0.0)
            state.on_text_delta(f"text{char}", 1.0)
            # Chunk 0 slack is fixed at 0.0 regardless of arrival time.
            assert state.cumulative_slack == pytest.approx(0.0), \
                f"Failed for boundary char {char!r}"

    def test_non_boundary_chars_no_flush(self):
        """Non-boundary text should not flush."""
        state = RequestSLOState(decoding_start=0.0)
        state.on_text_delta("hello", 1.0)
        state.on_text_delta(" world", 2.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_custom_consume_estimator(self):
        """Custom ConsumeEstimator can be substituted."""

        class FixedEstimator:
            def __call__(self, text: str) -> float:
                return 3.0  # always 3 seconds regardless of text

        assert isinstance(FixedEstimator(), ConsumeEstimator)

        state = RequestSLOState(estimator=FixedEstimator(), decoding_start=0.0)
        state.on_text_delta("chunk zero.", 1.0)
        # slack_0 = 0.0 (fixed)
        assert state.cumulative_slack == pytest.approx(0.0)

        state.on_text_delta("chunk one.", 2.0)
        # deadline_1 = 0.0 + 3.0 (fixed estimator) = 3.0; slack_1 = 3.0 - 2.0 = 1.0
        assert state.cumulative_slack == pytest.approx(1.0)

    def test_multi_delta_accumulates_before_boundary(self):
        """Multiple on_text_delta calls should accumulate until boundary."""
        state = RequestSLOState(decoding_start=0.0)
        state.on_text_delta("Hello", 1.0)
        state.on_text_delta(" world", 2.0)
        assert state.cumulative_slack == pytest.approx(0.0)
        # Now add the boundary — flushes as chunk 0 -> slack fixed at 0.0
        state.on_text_delta(".", 3.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_is_sentence_boundary(self):
        assert RequestSLOState._is_sentence_boundary("Hello.") is True
        assert RequestSLOState._is_sentence_boundary("Hello!") is True
        assert RequestSLOState._is_sentence_boundary("Hello?") is True
        assert RequestSLOState._is_sentence_boundary("Hello。") is True
        assert RequestSLOState._is_sentence_boundary("Hello！") is True
        assert RequestSLOState._is_sentence_boundary("Hello？") is True
        assert RequestSLOState._is_sentence_boundary("Hello…") is True
        assert RequestSLOState._is_sentence_boundary("Hello. ") is True  # trailing space
        assert RequestSLOState._is_sentence_boundary("Hello") is False
        assert RequestSLOState._is_sentence_boundary("Hello,") is False
        assert RequestSLOState._is_sentence_boundary("") is False
        assert RequestSLOState._is_sentence_boundary("   ") is False


# ---------------------------------------------------------------------------
# Request.slo_state property tests
# ---------------------------------------------------------------------------


import weakref


class _FakeRequestState:
    def __init__(self):
        self.slo_state = None


class TestRequestSLOStateProperty:
    def _make_request(self):
        """Create a minimal Request-like object with the _rs field and slo_state property."""
        # We can test the property logic directly without instantiating the full Request
        # class (which needs a tokenizer, sampling_params, etc.).
        # Instead, verify the property logic via a small inline class that mirrors
        # exactly what Request does.
        from vllm.sslo.slo_state import RequestSLOState, WordRateEstimator
        import weakref

        class _MinimalRequest:
            _rs = None

            @property
            def slo_state(self):
                if self._rs is not None:
                    rs = self._rs()
                    if rs is not None:
                        return rs.slo_state
                return None

            @slo_state.setter
            def slo_state(self, value):
                if self._rs is not None:
                    rs = self._rs()
                    if rs is not None:
                        rs.slo_state = value

        return _MinimalRequest()

    def test_unbound_getter_returns_none(self):
        req = self._make_request()
        assert req.slo_state is None

    def test_unbound_setter_is_noop(self):
        from vllm.sslo.slo_state import RequestSLOState
        req = self._make_request()
        req.slo_state = RequestSLOState(decoding_start=0.0)  # should not raise
        assert req.slo_state is None

    def test_bound_getter_delegates(self):
        from vllm.sslo.slo_state import RequestSLOState
        req = self._make_request()
        rs = _FakeRequestState()
        req._rs = weakref.ref(rs)
        state = RequestSLOState(decoding_start=0.0)
        rs.slo_state = state
        assert req.slo_state is state

    def test_bound_setter_delegates(self):
        from vllm.sslo.slo_state import RequestSLOState
        req = self._make_request()
        rs = _FakeRequestState()
        req._rs = weakref.ref(rs)
        state = RequestSLOState(decoding_start=0.0)
        req.slo_state = state
        assert rs.slo_state is state

    def test_stale_weakref_returns_none(self):
        req = self._make_request()
        rs = _FakeRequestState()
        req._rs = weakref.ref(rs)
        del rs  # referent GC'd
        assert req.slo_state is None


# ---------------------------------------------------------------------------
# LLMEngine._bind_slo_state tests
# ---------------------------------------------------------------------------


class TestBindSLOState:
    def _make_engine_with_inproc(self, req_id, request, req_state):
        """Simulate the parts of LLMEngine._bind_slo_state that matter."""
        import weakref
        import types

        class _FakeReqStates:
            def get(self, key):
                return req_state if key == req_id else None

        class _FakeScheduler:
            requests = {req_id: request}

        class _FakeInprocCore:
            scheduler = _FakeScheduler()

        class _FakeEngineCore:
            engine_core = _FakeInprocCore()

        class _FakeLLM:
            engine_core = _FakeEngineCore()
            output_processor = types.SimpleNamespace(request_states=_FakeReqStates())

            def _bind_slo_state(self, req_id):
                inproc_core = getattr(self.engine_core, "engine_core", None)
                if inproc_core is None:
                    return
                sched_req = inproc_core.scheduler.requests.get(req_id)
                req_state = self.output_processor.request_states.get(req_id)
                if sched_req is not None and req_state is not None:
                    sched_req._rs = weakref.ref(req_state)

        return _FakeLLM()

    def test_binds_weakref_for_inproc(self):
        req_id = "req-1"
        request = _FakeRequestState()  # reuse as stand-in for Request (just needs _rs attr)
        request._rs = None
        rs = _FakeRequestState()

        engine = self._make_engine_with_inproc(req_id, request, rs)
        engine._bind_slo_state(req_id)

        assert request._rs is not None
        assert request._rs() is rs

    def test_noop_when_no_inproc(self):
        import weakref, types

        class _FakeMPEngine:
            engine_core = types.SimpleNamespace()  # no .engine_core attribute
            output_processor = types.SimpleNamespace(request_states={})

            def _bind_slo_state(self, req_id):
                inproc_core = getattr(self.engine_core, "engine_core", None)
                if inproc_core is None:
                    return
                # Would fail if reached

        engine = _FakeMPEngine()
        engine._bind_slo_state("req-1")  # should not raise
