# SPDX-License-Identifier: Apache-2.0
"""SLO state tracking for streaming requests.

Tracks cumulative SLO slack per request based on sentence-chunk boundaries.
Slack for chunk i is defined as:
    cumulative_deadline[i] = decoding_start + sum(consume_time[0..i-1])
    cumulative_slack[i]    = cumulative_deadline[i] - chunk[i].end_time
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

# Sentence-ending punctuation for chunk boundary detection.
_SENTENCE_END_CHARS = frozenset(".!?。！？…")


@runtime_checkable
class ConsumeEstimator(Protocol):
    """Protocol for estimating consumption time of a text chunk."""

    def __call__(self, text: str) -> float:
        """Return estimated seconds to consume *text*."""
        ...


class WordRateEstimator:
    """Estimates consumption time from word count at a fixed word rate."""

    def __init__(self, seconds_per_word: float = 0.28) -> None:
        self.seconds_per_word = seconds_per_word

    def __call__(self, text: str) -> float:
        word_count = len(text.split())
        return word_count * self.seconds_per_word


class RequestSLOState:
    """Tracks cumulative SLO slack for a single streaming request.

    Usage::

        state = RequestSLOState()
        # for each decoded text delta:
        state.on_text_delta(delta_text, time.monotonic())
        # at request finish:
        state.on_finish(time.monotonic())
        # read the latest slack (0.0 until chunk 1 completes):
        slack = state.cumulative_slack

    decoding_start is set automatically from the timestamp of the first
    on_text_delta / on_finish call.
    """

    def __init__(self, estimator: ConsumeEstimator | None = None) -> None:
        self._estimator: ConsumeEstimator = (
            estimator if estimator is not None else WordRateEstimator()
        )
        self._decoding_start: float | None = None
        self._cumulative_consume: float = 0.0
        self._pending_text: str = ""
        self._chunk_count: int = 0
        self.cumulative_slack: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_text_delta(self, text: str, now: float) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        self._pending_text += text
        if self._is_sentence_boundary(self._pending_text):
            self._flush_chunk(now)

    def on_finish(self, now: float) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        if self._pending_text:
            self._flush_chunk(now)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_chunk(self, now: float) -> None:
        """Compute slack for the completed chunk and update state."""
        # Chunk 0 slack is fixed at 0.0; only update from chunk 1 onward.
        if self._chunk_count > 0:
            assert self._decoding_start is not None
            deadline = self._decoding_start + self._cumulative_consume
            self.cumulative_slack = deadline - now
        # Always accumulate consume time (chunk 0's time feeds chunk 1's deadline).
        self._cumulative_consume += self._estimator(self._pending_text)
        self._chunk_count += 1
        self._pending_text = ""

    @staticmethod
    def _is_sentence_boundary(text: str) -> bool:
        """Return True if *text* ends with a sentence-ending character."""
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS
