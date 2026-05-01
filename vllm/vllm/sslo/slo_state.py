# SPDX-License-Identifier: Apache-2.0
"""SLO state tracking for streaming requests.

Tracks cumulative SLO slack per request based on chunk boundaries.
Slack for chunk i is defined as:
    cumulative_deadline[i] = decoding_start + sum(consume_time[0..i-1])
    cumulative_slack[i]    = cumulative_deadline[i] - chunk[i].end_time

Chunk boundaries are determined by a ChunkBoundaryDetector. Built-in
detectors: SentenceChunkDetector (default), ParagraphChunkDetector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

# Sentence-ending punctuation for sentence boundary detection.
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


@runtime_checkable
class ChunkBoundaryDetector(Protocol):
    """Protocol for detecting chunk boundaries in accumulated text."""

    def find_boundary(self, text: str) -> int | None:
        """Return the index just past the boundary in *text*, or None.

        The text up to (not including) this index is the current chunk.
        Any text at or after this index is carried forward as pending text
        for the next chunk.
        """
        ...


class SentenceChunkDetector:
    """Flushes at the first sentence boundary in the accumulated text.

    A sentence boundary is a sentence-ending character (.!?。！？…) followed
    by whitespace or end of text.  Consecutive sentence-ending chars are
    consumed together (e.g. "..." or "?!") so the boundary position sits just
    past all of them.  Mid-token sequences like "3.14" or "Dr.Smith" (no
    following whitespace) are not treated as boundaries.
    """

    def find_boundary(self, text: str) -> int | None:
        i = 0
        n = len(text)
        while i < n:
            if text[i] in _SENTENCE_END_CHARS:
                j = i + 1
                # Consume consecutive sentence-ending chars ("..." / "?!")
                while j < n and text[j] in _SENTENCE_END_CHARS:
                    j += 1
                # Valid boundary: end of text OR followed by whitespace
                if j >= n or text[j].isspace():
                    return j
                # Not a boundary (e.g. "3.14", "Dr.Smith") — keep scanning
                i = j
            else:
                i += 1
        return None


class ParagraphChunkDetector:
    """Flushes when the accumulated text contains a paragraph break (\\n\\n).

    The boundary is placed just after the \\n\\n so the double-newline is
    included in the current chunk and any following text starts a new chunk.
    This correctly handles the case where a single streaming delta contains
    both the end of one paragraph and the beginning of the next.
    """

    def find_boundary(self, text: str) -> int | None:
        idx = text.find("\n\n")
        if idx != -1:
            return idx + 2
        return None


@dataclass
class SLOChunkRecord:
    chunk_idx: int
    text: str
    word_count: int
    decoding_start_ts: float
    end_time_ts: float
    cumulative_consume: float   # sum of consume times for chunks 0..idx-1 (deadline offset)
    cumulative_slack: float     # 0.0 for chunk 0


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

    Pass a ParagraphChunkDetector (or any ChunkBoundaryDetector) to change
    the granularity at which chunks are flushed.
    """

    def __init__(
        self,
        estimator: ConsumeEstimator | None = None,
        detector: ChunkBoundaryDetector | None = None,
        ema_alpha: float = 0.2,
        pending_slack_eps_num_tokens: int = 3,
    ) -> None:
        self._estimator: ConsumeEstimator = (
            estimator if estimator is not None else WordRateEstimator()
        )
        self._detector: ChunkBoundaryDetector = (
            detector if detector is not None else SentenceChunkDetector()
        )
        self._decoding_start: float | None = None
        self._cumulative_consume: float = 0.0
        self._pending_text: str = ""
        self._chunk_count: int = 0
        self.cumulative_slack: float = 0.0
        self._slack_dirty: bool = False
        self._chunk_records: list[SLOChunkRecord] = []
        self._ema_alpha: float = ema_alpha
        self._pending_slack_eps_num_tokens: int = pending_slack_eps_num_tokens
        # EMA / pending fields are populated in Task 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sslo_score(self) -> float:
        """Scheduling urgency score. Lower = more urgent.

        Currently equals cumulative_slack. Abstracted so future scoring
        formulas don't require caller changes.
        """
        return self.cumulative_slack

    def on_text_delta(self, text: str, now: float) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        self._pending_text += text
        while (pos := self._detector.find_boundary(self._pending_text)) is not None:
            self._flush_chunk(now, pos)

    def on_finish(self, now: float) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        if self._pending_text:
            self._flush_chunk(now, len(self._pending_text))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def take_slack_update(self) -> float | None:
        """Return cumulative_slack if updated since last call, else None."""
        if self._slack_dirty:
            self._slack_dirty = False
            return self.cumulative_slack
        return None

    def _flush_chunk(self, now: float, boundary_pos: int) -> None:
        """Compute slack for the completed chunk and update state."""
        chunk_text = self._pending_text[:boundary_pos]
        # Chunk 0 slack is fixed at 0.0; only update from chunk 1 onward.
        if self._chunk_count > 0:
            assert self._decoding_start is not None
            deadline = self._decoding_start + self._cumulative_consume
            self.cumulative_slack = deadline - now
            self._slack_dirty = True
        self._chunk_records.append(SLOChunkRecord(
            chunk_idx=self._chunk_count,
            text=chunk_text,
            word_count=len(chunk_text.split()),
            decoding_start_ts=self._decoding_start,  # type: ignore[arg-type]
            end_time_ts=now,
            cumulative_consume=self._cumulative_consume,
            cumulative_slack=self.cumulative_slack,
        ))
        # Always accumulate consume time (chunk 0's time feeds chunk 1's deadline).
        self._cumulative_consume += self._estimator(chunk_text)
        self._chunk_count += 1
        self._pending_text = self._pending_text[boundary_pos:]

    @property
    def chunk_records(self) -> list[dict]:
        return [asdict(r) for r in self._chunk_records]
