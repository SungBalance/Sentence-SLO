# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm.sslo.slo_state."""

import pytest

from vllm.sslo.slo_state import (
    ConsumeEstimator,
    ParagraphChunkDetector,
    RequestSLOState,
    SentenceChunkDetector,
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
        state = RequestSLOState()
        assert state.cumulative_slack == 0.0

    def test_no_flush_without_boundary(self):
        state = RequestSLOState()
        state.on_text_delta("hello world", 1.0)
        # No sentence boundary -> no flush -> slack stays 0.0
        assert state.cumulative_slack == 0.0

    def test_single_chunk_slack(self):
        """Chunk 0 slack is fixed at 0.0 (not computed)."""
        state = RequestSLOState()
        state.on_text_delta("Hello world.", 5.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_decoding_start_set_from_first_delta(self):
        """decoding_start is set from the now of the first on_text_delta call."""
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est)
        state.on_text_delta("Hello world.", 10.0)  # chunk 0, decoding_start = 10.0
        state.on_text_delta("How are you?", 13.0)  # chunk 1: deadline = 10 + 2 = 12, slack = -1
        assert state.cumulative_slack == pytest.approx(-1.0)

    def test_second_chunk_deadline_includes_first_consume(self):
        """
        decoding_start = 5.0 (first on_text_delta call), seconds_per_word = 1.0.

        Chunk 0: "Hello world." (2 words -> 2.0 s consume), ends at t=5.
          slack_0 = 0.0 (fixed, not computed)

        Chunk 1: "How are you?" (3 words -> 3.0 s consume), ends at t=8.
          deadline_1 = 5.0 + 2.0 (chunk 0's consume) = 7.0
          slack_1 = 7.0 - 8 = -1.0
        """
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est)

        state.on_text_delta("Hello world.", 5.0)
        assert state.cumulative_slack == pytest.approx(0.0)

        state.on_text_delta("How are you?", 8.0)
        assert state.cumulative_slack == pytest.approx(-1.0)

    def test_positive_slack(self):
        """Second chunk arrives before deadline -> positive slack.

        Chunk 0: "Hi." (1 word, 0.28 s consume) ends at t=100.1, decoding_start = 100.1
          slack_0 = 0.0 (fixed, not computed)
        Chunk 1: "Hey?" (1 word, 0.28 s consume) ends at t=100.2
          deadline_1 = 100.1 + 0.28 = 100.38; slack_1 = 100.38 - 100.2 = +0.18
        """
        state = RequestSLOState()
        state.on_text_delta("Hi.", 100.1)
        state.on_text_delta("Hey?", 100.2)
        # deadline_1 = 100.1 + 0.28 (1 word * 0.28) = 100.38
        expected_slack = 100.1 + 0.28 - 100.2
        assert state.cumulative_slack == pytest.approx(expected_slack, abs=1e-9)

    def test_on_finish_flushes_remaining_text(self):
        """on_finish should flush non-boundary text as a final chunk."""
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est)
        # No boundary — text is held as pending
        state.on_text_delta("some partial text", 3.0)
        assert state.cumulative_slack == 0.0
        # Finish flushes it as chunk 0 -> slack fixed at 0.0
        state.on_finish(6.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_on_finish_no_pending_is_noop(self):
        """on_finish with nothing pending should not change slack."""
        state = RequestSLOState()
        state.on_text_delta("Done.", 2.0)
        slack_before = state.cumulative_slack
        state.on_finish(3.0)
        # Pending was flushed by on_text_delta; on_finish has nothing to flush.
        assert state.cumulative_slack == slack_before

    def test_boundary_characters(self):
        """All sentence boundary chars should trigger a flush (chunk 0 -> 0.0)."""
        for char in ".!?。！？…":
            state = RequestSLOState()
            state.on_text_delta(f"text{char}", 1.0)
            # Chunk 0 slack is fixed at 0.0 regardless of arrival time.
            assert state.cumulative_slack == pytest.approx(0.0), \
                f"Failed for boundary char {char!r}"

    def test_non_boundary_chars_no_flush(self):
        """Non-boundary text should not flush."""
        state = RequestSLOState()
        state.on_text_delta("hello", 1.0)
        state.on_text_delta(" world", 2.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_custom_consume_estimator(self):
        """Custom ConsumeEstimator can be substituted."""

        class FixedEstimator:
            def __call__(self, text: str) -> float:
                return 3.0  # always 3 seconds regardless of text

        assert isinstance(FixedEstimator(), ConsumeEstimator)

        state = RequestSLOState(estimator=FixedEstimator())
        state.on_text_delta("chunk zero.", 1.0)
        # slack_0 = 0.0 (fixed)
        assert state.cumulative_slack == pytest.approx(0.0)

        state.on_text_delta("chunk one.", 2.0)
        # deadline_1 = 1.0 (decoding_start) + 3.0 (fixed estimator) = 4.0; slack_1 = 4.0 - 2.0 = 2.0
        assert state.cumulative_slack == pytest.approx(2.0)

    def test_multi_delta_accumulates_before_boundary(self):
        """Multiple on_text_delta calls should accumulate until boundary."""
        state = RequestSLOState()
        state.on_text_delta("Hello", 1.0)
        state.on_text_delta(" world", 2.0)
        assert state.cumulative_slack == pytest.approx(0.0)
        # Now add the boundary — flushes as chunk 0 -> slack fixed at 0.0
        state.on_text_delta(".", 3.0)
        assert state.cumulative_slack == pytest.approx(0.0)

    def test_sentence_chunk_detector(self):
        det = SentenceChunkDetector()
        assert det.find_boundary("Hello.") == 6
        assert det.find_boundary("Hello!") == 6
        assert det.find_boundary("Hello?") == 6
        assert det.find_boundary("Hello。") is not None
        assert det.find_boundary("Hello. ") == 6  # boundary before trailing space
        assert det.find_boundary("Hello") is None
        assert det.find_boundary("Hello,") is None
        assert det.find_boundary("") is None
        assert det.find_boundary("   ") is None

    def test_paragraph_chunk_detector(self):
        det = ParagraphChunkDetector()
        assert det.find_boundary("Para one.\n\nPara two") == 11
        assert det.find_boundary("No paragraph") is None
        assert det.find_boundary("\n\n") == 2
        assert det.find_boundary("text\n\n") == 6
        assert det.find_boundary("\n") is None

    def test_multi_boundary_in_single_delta(self):
        """Two paragraph boundaries in one on_text_delta call produce two flushes."""
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est, detector=ParagraphChunkDetector())
        # "Para.\n\nSecond.\n\nThird" → two boundaries, two flushes at t=1.0
        # chunk 0: "Para.\n\n" (1 word = 1.0s), decoding_start=1.0
        # chunk 1: "Second.\n\n" (1 word), deadline=1.0+1.0=2.0, slack=2.0-1.0=+1.0
        # pending after: "Third"
        state.on_text_delta("Para.\n\nSecond.\n\nThird", 1.0)
        assert state.cumulative_slack == pytest.approx(1.0)
        assert state._pending_text == "Third"

    def test_paragraph_detection(self):
        """ParagraphChunkDetector flushes at \\n\\n boundaries."""
        est = WordRateEstimator(seconds_per_word=1.0)
        state = RequestSLOState(estimator=est, detector=ParagraphChunkDetector())
        state.on_text_delta("Hello world.\n\n", 5.0)  # chunk 0, 2 words
        assert state.cumulative_slack == pytest.approx(0.0)
        assert state._pending_text == ""
        # chunk 1: paragraph ends at t=8.0; deadline=5.0+2.0=7.0, slack=-1.0
        state.on_text_delta("Next para.", 7.0)  # accumulating, not flushed yet
        state.on_finish(8.0)                    # flush at finish
        assert state.cumulative_slack == pytest.approx(-1.0)

    def test_paragraph_remainder_stays_pending(self):
        """Text after \\n\\n stays in pending for the next chunk."""
        state = RequestSLOState(detector=ParagraphChunkDetector())
        state.on_text_delta("Para one.\n\nStart of two", 1.0)
        assert state._pending_text == "Start of two"


class TestEmaTracking:
    def test_ema_initially_none(self):
        state = RequestSLOState()
        assert state._ema_pure_gen_time is None
        assert state._ema_per_token_time is None

    def test_ema_updates_after_chunk(self):
        import time
        state = RequestSLOState(ema_alpha=0.5)
        t0 = time.monotonic()
        state.on_text_delta("hello world", t0)
        state.on_text_delta(". ", t0 + 1.0)
        # First chunk: pure gen = 1.0, words = 2, per_token ~ 0.5
        assert state._ema_pure_gen_time == pytest.approx(1.0)
        assert state._ema_per_token_time == pytest.approx(0.5)


class TestPendingCallbacks:
    def test_pending_subtracted_from_gen_time(self):
        import time
        state = RequestSLOState(ema_alpha=1.0)  # alpha=1 -> EMA == latest sample
        t0 = time.monotonic()
        # text starts at t0
        state.on_text_delta("hello world", t0 + 0.0)  # no boundary yet
        # pending [t0+0.5, t0+0.8] (0.3s)
        state.on_pending_enter(t0 + 0.5)
        state.on_pending_exit(t0 + 0.8)
        # finish at t0+1.0 (boundary at end of text)
        state.on_text_delta(".", t0 + 1.0)
        # pure gen_time = 1.0 - 0.3 = 0.7
        rec = state.chunk_records[0]
        assert rec["gen_time"] == pytest.approx(0.7)
        assert rec["pending_time"] == pytest.approx(0.3)

    def test_pending_resets_after_flush(self):
        import time
        state = RequestSLOState()
        t0 = time.monotonic()
        state.on_pending_enter(t0)
        state.on_pending_exit(t0 + 0.5)
        state.on_text_delta("a. ", t0 + 1.0)  # chunk flushes
        assert state._chunk_pending_time == 0.0


class TestIsPendingEligible:
    def test_false_when_no_ema_yet(self):
        state = RequestSLOState()
        state.cumulative_slack = 100.0
        assert state.is_pending_eligible is False

    def test_true_when_cumulative_slack_exceeds_threshold(self):
        import time
        state = RequestSLOState(ema_alpha=1.0, pending_slack_eps_num_tokens=3)
        t0 = time.monotonic()
        state.on_text_delta("hello world", t0)
        state.on_text_delta(". ", t0 + 1.0)
        # ema_pure_gen_time = 1.0, ema_per_token_time = 0.5
        # threshold = 1.0 + 3 * 0.5 = 2.5
        state.cumulative_slack = 3.0
        assert state.is_pending_eligible is True

    def test_false_when_cumulative_slack_below_threshold(self):
        import time
        state = RequestSLOState(ema_alpha=1.0, pending_slack_eps_num_tokens=3)
        t0 = time.monotonic()
        state.on_text_delta("hello world", t0)
        state.on_text_delta(". ", t0 + 1.0)
        state.cumulative_slack = 2.0
        assert state.is_pending_eligible is False


# ---------------------------------------------------------------------------
# LLMEngine._bind_slo_state tests
# ---------------------------------------------------------------------------


class TestBindSLOState:
    """Tests for the shared-reference binding pattern."""

    def test_shared_reference_after_bind(self):
        """Both sides hold the same RequestSLOState instance after binding."""
        from vllm.sslo.slo_state import RequestSLOState

        class _FakeReq:
            slo_state = None

        class _FakeReqState:
            slo_state = None

        sched_req = _FakeReq()
        req_state = _FakeReqState()

        # Simulate what _bind_slo_state does
        slo = RequestSLOState()
        sched_req.slo_state = slo
        req_state.slo_state = slo

        assert sched_req.slo_state is req_state.slo_state

    def test_output_processor_update_visible_to_scheduler(self):
        """Updates via req_state.slo_state are visible via sched_req.slo_state."""
        from vllm.sslo.slo_state import RequestSLOState, WordRateEstimator

        class _FakeReq:
            slo_state = None

        class _FakeReqState:
            slo_state = None

        sched_req = _FakeReq()
        req_state = _FakeReqState()
        est = WordRateEstimator(seconds_per_word=1.0)
        slo = RequestSLOState(estimator=est)
        sched_req.slo_state = slo
        req_state.slo_state = slo

        # Output processor updates via req_state side
        req_state.slo_state.on_text_delta("Hello world.", 5.0)  # chunk 0
        req_state.slo_state.on_text_delta("How are you?", 8.0)  # chunk 1

        # Scheduler reads via sched_req side
        assert sched_req.slo_state.cumulative_slack == pytest.approx(-1.0)


class TestSsloScore:
    def test_initial_zero(self):
        state = RequestSLOState()
        assert state.sslo_score == 0.0

    def test_equals_cumulative_slack(self):
        import time
        state = RequestSLOState()
        now = time.monotonic()
        state.on_text_delta("Hello world. ", now)
        state.on_text_delta("Next sentence. ", now + 0.1)
        assert state.sslo_score == state.cumulative_slack
