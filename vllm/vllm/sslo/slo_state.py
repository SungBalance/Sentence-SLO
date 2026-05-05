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

import collections
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
class ChunkGenerationEstimator(Protocol):
    def update(self, pure_gen_time: float, word_count: int) -> None:
        ...

    @property
    def gen_time(self) -> float | None:
        ...

    @property
    def per_token_time(self) -> float | None:
        ...

    @property
    def chunk_word_count(self) -> float | None:
        ...

    @property
    def n_samples(self) -> int:
        """Number of valid samples observed so far."""
        ...


class EmaChunkGenerationEstimator:
    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = alpha
        self._gen_time: float | None = None
        self._per_token_time: float | None = None
        self._chunk_word_count: float | None = None
        self._n_samples: int = 0

    def update(self, pure_gen_time: float, word_count: int) -> None:
        if word_count <= 0 or pure_gen_time <= 0:
            return
        per_token = pure_gen_time / word_count
        a = self._alpha
        if self._gen_time is None:
            self._gen_time = pure_gen_time
            self._per_token_time = per_token
            self._chunk_word_count = float(word_count)
        else:
            self._gen_time = a * pure_gen_time + (1 - a) * self._gen_time
            self._per_token_time = a * per_token + (1 - a) * self._per_token_time
            self._chunk_word_count = a * word_count + (1 - a) * self._chunk_word_count
        self._n_samples += 1

    @property
    def gen_time(self) -> float | None:
        return self._gen_time

    @property
    def per_token_time(self) -> float | None:
        return self._per_token_time

    @property
    def chunk_word_count(self) -> float | None:
        return self._chunk_word_count

    @property
    def n_samples(self) -> int:
        return self._n_samples


class PercentileChunkGenerationEstimator:
    def __init__(self, percentile: float = 99.0, window_size: int = 100) -> None:
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self._percentile = percentile
        self._gen_times: collections.deque[float] = collections.deque(
            maxlen=window_size)
        self._per_token_times: collections.deque[float] = collections.deque(
            maxlen=window_size)
        self._word_counts: collections.deque[float] = collections.deque(
            maxlen=window_size)
        self._n_samples: int = 0

    def update(self, pure_gen_time: float, word_count: int) -> None:
        if word_count <= 0 or pure_gen_time <= 0:
            return
        self._gen_times.append(pure_gen_time)
        self._per_token_times.append(pure_gen_time / word_count)
        self._word_counts.append(float(word_count))
        self._n_samples += 1

    def _pct(self, samples: "collections.deque[float]") -> float | None:
        if not samples:
            return None
        ordered = sorted(samples)
        rank = self._percentile / 100.0 * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        frac = rank - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * frac

    @property
    def gen_time(self) -> float | None:
        return self._pct(self._gen_times)

    @property
    def per_token_time(self) -> float | None:
        return self._pct(self._per_token_times)

    @property
    def chunk_word_count(self) -> float | None:
        return self._pct(self._word_counts)

    @property
    def n_samples(self) -> int:
        return self._n_samples


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
class SsloRequestStats:
    final_cumulative_slack: float
    min_cumulative_slack: float
    neg_slack_chunk_count: int
    total_pending_time_s: float
    num_pending_intervals: int
    max_consecutive_pending: int
    final_ema_gen_time_s: float | None
    final_ema_per_word_time_s: float | None
    total_offload_time_s: float
    num_offload_intervals: int


@dataclass
class SLOChunkRecord:
    chunk_idx: int
    text: str
    word_count: int
    decoding_start_ts: float
    end_time_ts: float
    cumulative_consume: float   # sum of consume times for chunks 0..idx-1 (deadline offset)
    cumulative_slack: float     # 0.0 for chunk 0
    gen_time: float             # end_time - chunk_start - pending_time
    pending_time: float         # total pending time during this chunk


class RequestSLOState:
    """Tracks cumulative SLO slack for a single streaming request.

    Slack sign convention: slack > 0 is healthy; slack < 0 is
    overdue. Both realtime_slack and cumulative_slack follow this sign.

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
        chunk_gen_estimator: ChunkGenerationEstimator | None = None,
        pending_warmup_chunks: int = 5,
        pending_enter_factor: float = 2.5,
        pending_pressure_lambda: float = 0.0,
        pending_hysteresis_gap: float = 0.5,
        min_chunk_tokens: int = 0,
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
        self._pending_chunk_tokens: int = 0
        self._search_offset: int = 0
        self._min_chunk_tokens: int = min_chunk_tokens
        self._chunk_count: int = 0
        self.cumulative_slack: float = 0.0
        self._slack_dirty: bool = False
        self._chunk_records: list[SLOChunkRecord] = []
        self._pending_warmup_chunks: int = pending_warmup_chunks
        self._pending_enter_factor: float = pending_enter_factor
        self._pending_pressure_lambda: float = pending_pressure_lambda
        self._pending_hysteresis_gap: float = pending_hysteresis_gap
        self._chunk_gen_estimator = (
            chunk_gen_estimator
            if chunk_gen_estimator is not None
            else EmaChunkGenerationEstimator(alpha=ema_alpha)
        )
        self._last_chunk_end_ts: float | None = None
        self._pending_enter_ts: float | None = None
        self._chunk_pending_time: float = 0.0
        self._num_pending_intervals: int = 0
        self._cur_consecutive_pending: int = 0
        self._max_consecutive_pending: int = 0
        # Offload timing (mirrors pending mechanism)
        self._offload_enter_ts: float | None = None
        self._total_offload_time: float = 0.0
        self._num_offload_intervals: int = 0

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

    @property
    def chunk_gen_estimator(self) -> ChunkGenerationEstimator:
        return self._chunk_gen_estimator

    def on_text_delta(self, text: str, now: float, num_tokens: int = 0) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        self._pending_text += text
        self._pending_chunk_tokens += num_tokens
        while True:
            sub = self._pending_text[self._search_offset:]
            rel_pos = self._detector.find_boundary(sub)
            if rel_pos is None:
                break
            pos = self._search_offset + rel_pos
            if (
                self._min_chunk_tokens > 0
                and self._pending_chunk_tokens <= self._min_chunk_tokens
            ):
                # Defer: the chunk up to this boundary is too short. Skip
                # past it so the next find_boundary searches further along
                # the same accumulated text; keep _pending_chunk_tokens so
                # the eventual flush counts the full merged span.
                self._search_offset = pos
                continue
            self._flush_chunk(now, pos)
            # _flush_chunk slices _pending_text in place, so reset the
            # offset and the per-chunk token counter.
            self._search_offset = 0
            self._pending_chunk_tokens = 0

    def on_finish(self, now: float) -> None:
        if self._decoding_start is None:
            self._decoding_start = now
        if self._pending_text:
            # Last chunk is exempt from the min_chunk_tokens guard — flush
            # remaining text regardless of token count.
            self._flush_chunk(now, len(self._pending_text))
        self._search_offset = 0
        self._pending_chunk_tokens = 0

    def on_pending_enter(self, now: float) -> None:
        """Mark the start of a pending interval."""
        if self._pending_enter_ts is None:
            self._pending_enter_ts = now
            self._num_pending_intervals += 1
            self._cur_consecutive_pending += 1
            if self._cur_consecutive_pending > self._max_consecutive_pending:
                self._max_consecutive_pending = self._cur_consecutive_pending

    def on_pending_exit(self, now: float) -> None:
        """Accumulate pending duration into the current chunk."""
        if self._pending_enter_ts is not None:
            self._chunk_pending_time += now - self._pending_enter_ts
            self._pending_enter_ts = None
            self._cur_consecutive_pending = 0

    def on_offload_enter(self, now: float) -> None:
        """Mark the start of an offload interval (KV moved to CPU)."""
        if self._offload_enter_ts is None:
            self._offload_enter_ts = now
            self._num_offload_intervals += 1

    def on_offload_exit(self, now: float) -> None:
        """Accumulate offload duration. No-op if not currently offloaded."""
        if self._offload_enter_ts is not None:
            self._total_offload_time += now - self._offload_enter_ts
            self._offload_enter_ts = None

    def is_overdue_post_warmup(self) -> bool:
        """True if cumulative_slack < 0 and past the warmup period.

        Warmup uses the same threshold as the pending mechanism
        (pending_warmup_chunks). Acts as a per-request "contributing to
        overdue signal" check used by the scheduler-side adaptive batch
        size policy. Symmetric in spirit to should_enter_pending(now).
        """
        estimator = self._chunk_gen_estimator
        if estimator.n_samples < self._pending_warmup_chunks:
            return False
        return self.cumulative_slack < 0

    def _realtime_slack(self, now: float) -> float | None:
        """Slack to next chunk's deadline using current wall clock.

        deadline_next = decoding_start + cumulative_consume (sum of consume
        times of all flushed chunks). Returns None if decoding has not begun.
        """
        if self._decoding_start is None:
            return None
        return self._decoding_start + self._cumulative_consume - now

    def _predicted_finish_time(self) -> float | None:
        """Estimated time to finish the in-progress chunk, in seconds."""
        per_token = self._chunk_gen_estimator.per_token_time
        word_count_est = self._chunk_gen_estimator.chunk_word_count
        if per_token is None or word_count_est is None:
            return None
        remaining = max(0.0, word_count_est - self.current_chunk_word_count)
        return remaining * per_token

    def should_enter_pending(self, now: float, pending_count: int = 0) -> bool:
        """True if a currently-running request should move to pending.

        Conditions (all required):
        - estimator has at least pending_warmup_chunks samples
        - realtime_slack > effective enter factor × estimator.gen_time
        """
        estimator = self._chunk_gen_estimator
        if estimator.n_samples < self._pending_warmup_chunks:
            return False
        gen_time = estimator.gen_time
        if gen_time is None:
            return False
        slack = self._realtime_slack(now)
        if slack is None:
            return False
        effective_factor = (
            self._pending_enter_factor
            + self._pending_pressure_lambda * pending_count
        )
        return slack > effective_factor * gen_time

    def should_exit_pending(self, now: float, pending_count: int = 0) -> bool:
        """True if a currently-pending request should move back to running.

        Conditions (any one):
        - estimator no longer has data (shouldn't normally happen)
        - realtime_slack ≤ effective exit factor × estimator.gen_time
        - realtime_slack + 1 token margin ≤ predicted time to finish current chunk
        """
        estimator = self._chunk_gen_estimator
        gen_time = estimator.gen_time
        per_token = estimator.per_token_time
        if gen_time is None or per_token is None:
            return True
        slack = self._realtime_slack(now)
        if slack is None:
            return True
        effective_enter = (
            self._pending_enter_factor
            + self._pending_pressure_lambda * pending_count
        )
        effective_exit = effective_enter - self._pending_hysteresis_gap
        # Hysteresis: low-bar exit
        if slack <= effective_exit * gen_time:
            return True
        # Predicted-finish guard: current chunk's remaining time exceeds slack
        finish = self._predicted_finish_time()
        if finish is not None and slack + per_token <= finish:
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def take_slack_update(self) -> float | None:
        """Return cumulative_slack if updated since last call, else None."""
        if self._slack_dirty:
            self._slack_dirty = False
            return self.cumulative_slack
        return None

    def compute_stats(self) -> SsloRequestStats:
        slacks = [r.cumulative_slack for r in self._chunk_records]
        return SsloRequestStats(
            final_cumulative_slack=slacks[-1] if slacks else 0.0,
            min_cumulative_slack=min(slacks) if slacks else 0.0,
            neg_slack_chunk_count=sum(1 for s in slacks if s < 0),
            total_pending_time_s=sum(r.pending_time for r in self._chunk_records),
            num_pending_intervals=self._num_pending_intervals,
            max_consecutive_pending=self._max_consecutive_pending,
            final_ema_gen_time_s=self._chunk_gen_estimator.gen_time,
            final_ema_per_word_time_s=self._chunk_gen_estimator.per_token_time,
            total_offload_time_s=self._total_offload_time,
            num_offload_intervals=self._num_offload_intervals,
        )

    def _flush_chunk(self, now: float, boundary_pos: int) -> None:
        """Compute slack for the completed chunk and update state."""
        chunk_text = self._pending_text[:boundary_pos]
        word_count = len(chunk_text.split())
        # Chunk 0 slack is fixed at 0.0; only update from chunk 1 onward.
        if self._chunk_count > 0:
            assert self._decoding_start is not None
            deadline = self._decoding_start + self._cumulative_consume
            self.cumulative_slack = deadline - now
            self._slack_dirty = True
        chunk_start = (
            self._last_chunk_end_ts
            if self._last_chunk_end_ts is not None
            else self._decoding_start
        )
        assert chunk_start is not None
        pure_gen_time = max(0.0, (now - chunk_start) - self._chunk_pending_time)
        self._chunk_gen_estimator.update(pure_gen_time, word_count)
        self._chunk_records.append(SLOChunkRecord(
            chunk_idx=self._chunk_count,
            text=chunk_text,
            word_count=word_count,
            decoding_start_ts=self._decoding_start,  # type: ignore[arg-type]
            end_time_ts=now,
            cumulative_consume=self._cumulative_consume,
            cumulative_slack=self.cumulative_slack,
            gen_time=pure_gen_time,
            pending_time=self._chunk_pending_time,
        ))
        # Always accumulate consume time (chunk 0's time feeds chunk 1's deadline).
        self._cumulative_consume += self._estimator(chunk_text)
        self._chunk_count += 1
        self._pending_text = self._pending_text[boundary_pos:]
        self._last_chunk_end_ts = now
        self._chunk_pending_time = 0.0

    @property
    def current_chunk_word_count(self) -> int:
        """Words generated for the in-progress (un-flushed) chunk."""
        return len(self._pending_text.split())

    @property
    def chunk_records(self) -> list[dict]:
        return [asdict(r) for r in self._chunk_records]
