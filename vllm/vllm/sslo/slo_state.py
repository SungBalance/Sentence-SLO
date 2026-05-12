# SPDX-License-Identifier: Apache-2.0
"""SSLO request lifecycle state for score-based scheduling."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import InitVar, asdict, dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.sslo.config import SsloConfig

_SENTENCE_END_CHARS = frozenset(".!?。！？…")
_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})

_VALID_CHUNK_LEN_STRATEGIES = frozenset({"ema", "p90", "p99", "past-future"})
_DEFAULT_CHUNK_LEN_STRATEGY = "p90"
_CHUNK_LEN_HISTORY_MAX = 64
_CHUNK_LEN_EMA_ALPHA = 0.2


class Phase(IntEnum):
    PREFILL = 0
    WARMUP = 1
    MEASURED = 2


class ChunkLengthPredictor:
    """Predict the next chunk's generated_len from completed chunks.

    Strategies select different points on the conservatism axis:
      - "ema": EMA (alpha) — smooth, follows the central tendency.
      - "p90"/"p99": percentile over a sliding window — conservative
        upper bound that biases score() toward overestimating remaining
        work (and therefore admission/preemption pressure).
      - "past-future": placeholder for a future strategy that combines
        past-window and future-projection signals. Currently a no-op
        stub; selecting it leaves `value` at None, so pressure() will
        also be None (i.e. that request opts out of scoring).
    """

    def __init__(
        self,
        strategy: str = _DEFAULT_CHUNK_LEN_STRATEGY,
        history_max: int = _CHUNK_LEN_HISTORY_MAX,
        alpha: float = _CHUNK_LEN_EMA_ALPHA,
    ) -> None:
        if strategy not in _VALID_CHUNK_LEN_STRATEGIES:
            raise ValueError(
                f"chunk_len_strategy must be one of "
                f"{sorted(_VALID_CHUNK_LEN_STRATEGIES)}, got {strategy!r}")
        self.strategy = strategy
        self.history_max = history_max
        self.alpha = alpha
        self._history: deque[int] = deque(maxlen=history_max)
        self._ema: float | None = None
        self.value: float | None = None
        # p99 companion value, populated only when strategy == "p90".
        # expected_remaining_len() falls back to it when generated tokens
        # exceed the p90 prediction so pressure() doesn't collapse to 1.
        self.value_high: float | None = None
        if strategy == "ema":
            self.update = self.update_ema
        elif strategy in ("p90", "p99"):
            self.update = self.update_percentile
        elif strategy == "past-future":
            self.update = self.update_past_future

    def update_ema(self, generated_len: int) -> None:
        n = int(generated_len)
        self._ema = float(n) if self._ema is None else (
            self.alpha * n + (1.0 - self.alpha) * self._ema)
        self.value = self._ema

    def update_percentile(self, generated_len: int) -> None:
        self._history.append(int(generated_len))
        arr = np.fromiter(self._history, dtype=float)
        percentile = 90.0 if self.strategy == "p90" else 99.0
        self.value = float(np.percentile(arr, percentile, method="nearest"))
        if self.strategy == "p90":
            self.value_high = float(
                np.percentile(arr, 99.0, method="nearest"))

    def update_past_future(self, generated_len: int) -> None:
        pass


class ChunkConsumeEstimator:
    """Estimate the consumption (audio playback) duration of a chunk.

    The default uses a fixed seconds-per-word rate, matching the legacy
    behaviour. Subclass and override `estimate()` to plug in a TTS-derived
    duration or any other model — the chunk text is passed in for that
    purpose.
    """

    def __init__(self, seconds_per_word: float = 0.28) -> None:
        if seconds_per_word < 0:
            raise ValueError("seconds_per_word must be >= 0")
        self.seconds_per_word = seconds_per_word

    def estimate(self, chunk_text: str, word_count: int) -> float:
        del chunk_text  # unused in the default rate-based estimator
        return word_count * self.seconds_per_word


class ChunkSeparator:
    """Stream-aware chunk boundary detector.

    Accumulates streamed text deltas and yields complete chunks at
    sentence or paragraph boundaries. Boundaries with fewer than
    `min_chunk_tokens` tokens accumulated since the last flush are
    deferred — the boundary advances past them so a short fragment
    (e.g. "Yes.") merges into the next chunk.

    Sentence vs paragraph boundary asymmetry: in sentence mode the
    yielded chunk ends right after the sentence-end punctuation, and
    trailing whitespace stays in the buffer as the next chunk's prefix.
    In paragraph mode the chunk INCLUDES the trailing ``\\n\\n``
    separator so the buffer can advance past it without re-matching the
    boundary on the next feed. Word counts (`split()`) and the default
    consume estimator ignore whitespace, so downstream math is
    unaffected; callers that display chunks verbatim should strip.
    """

    def __init__(
        self,
        chunk_unit: str = "sentence",
        min_chunk_tokens: int = 16,
    ) -> None:
        if chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {chunk_unit!r}")
        if min_chunk_tokens < 0:
            raise ValueError("min_chunk_tokens must be >= 0")
        self.chunk_unit = chunk_unit
        self.min_chunk_tokens = min_chunk_tokens
        self._pending_text = ""
        self._unflushed_token_count = 0
        self._search_offset = 0

    def feed(self, text: str, num_tokens: int) -> Iterator[str]:
        """Feed a streamed text delta; yield chunk strings as boundaries form."""
        if not text and num_tokens <= 0:
            return
        self._unflushed_token_count += num_tokens if num_tokens > 0 else 1
        self._pending_text += text
        while True:
            sub = self._pending_text[self._search_offset:]
            rel = self._find_boundary(sub)
            if rel is None:
                return
            boundary = self._search_offset + rel
            if self._unflushed_token_count < self.min_chunk_tokens:
                # Held back: skip past this boundary so subsequent scans
                # look at the NEXT one. The text remains in _pending_text
                # so it gets merged into the eventual flush.
                self._search_offset = boundary
                continue
            chunk = self._pending_text[:boundary]
            self._pending_text = self._pending_text[boundary:]
            self._unflushed_token_count = 0
            self._search_offset = 0
            yield chunk

    def flush(self) -> str | None:
        """Force-flush any held-back text (call on stream completion)."""
        if not self._pending_text:
            return None
        chunk = self._pending_text
        self._pending_text = ""
        self._unflushed_token_count = 0
        self._search_offset = 0
        return chunk

    def _find_boundary(self, text: str) -> int | None:
        if self.chunk_unit == "paragraph":
            idx = text.find("\n\n")
            return None if idx == -1 else idx + 2

        i = 0
        n = len(text)
        while i < n:
            if text[i] in _SENTENCE_END_CHARS:
                j = i + 1
                while j < n and text[j] in _SENTENCE_END_CHARS:
                    j += 1
                if j >= n or text[j].isspace():
                    return j
                i = j
            else:
                i += 1
        return None


@dataclass
class ChunkRecord:
    chunk_idx: int
    deadline_ts: float
    gen_finish_ts: float
    slack_s: float
    pending_time_s: float
    word_count: int
    # Chunk start wall-clock — prev chunk's gen_finish_ts, or
    # decoding_start_ts for chunk 0.
    start_time_ts: float
    # Tokens generated within this chunk window.
    num_token: int
    # Scheduler-step accounting per chunk window.
    # num_iters = num_running_iters + num_pending_iters.
    num_iters: int
    num_running_iters: int
    num_pending_iters: int
    # Predictor's estimate (in tokens) at chunk start — i.e. value used by
    # pressure() during this chunk's generation. None for chunk 0 (no
    # history yet) and any time before the predictor has a value. Compare
    # to num_token to measure prediction accuracy.
    expected_len: float | None
    # Raw stall interval for this chunk; non-None only when late.
    # Kept as raw timestamps so post-hoc analysis can merge overlapping
    # stall windows across requests.
    stall_start_ts: float | None
    stall_end_ts: float | None
    stall_duration_s: float


class ChunkStatCollector:
    """Per-request chunk-level diagnostic accumulator.

    Owns the per-chunk records list, the running stall_time_total
    aggregate, and the rolling pending-time bucket that gets stamped
    onto each ChunkRecord at chunk completion (then resets).
    """

    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.stall_time_total: float = 0.0
        self._current_pending_s: float = 0.0
        # Per-chunk scheduler-step tallies, reset at each record().
        self._current_running_iters: int = 0
        self._current_pending_iters: int = 0

    def record(
        self,
        *,
        chunk_idx: int,
        deadline_ts: float,
        gen_finish_ts: float,
        slack_s: float,
        word_count: int,
        start_time_ts: float,
        num_token: int,
        expected_len: float | None,
        stall_start_ts: float | None,
        stall_end_ts: float | None,
        stall_duration_s: float,
    ) -> None:
        running_iters = self._current_running_iters
        pending_iters = self._current_pending_iters
        self.records.append(
            ChunkRecord(
                chunk_idx=chunk_idx,
                deadline_ts=deadline_ts,
                gen_finish_ts=gen_finish_ts,
                slack_s=slack_s,
                pending_time_s=self._current_pending_s,
                word_count=word_count,
                start_time_ts=start_time_ts,
                num_token=num_token,
                num_iters=running_iters + pending_iters,
                num_running_iters=running_iters,
                num_pending_iters=pending_iters,
                expected_len=expected_len,
                stall_start_ts=stall_start_ts,
                stall_end_ts=stall_end_ts,
                stall_duration_s=stall_duration_s,
            ))
        # Aggregate stall = max(0, -slack); positive only when late.
        self.stall_time_total += max(0.0, -slack_s)
        self._current_pending_s = 0.0
        self._current_running_iters = 0
        self._current_pending_iters = 0

    def accumulate_pending(self, interval_s: float) -> None:
        self._current_pending_s += interval_s

    def accumulate_running_step(self) -> None:
        self._current_running_iters += 1

    def accumulate_pending_step(self) -> None:
        self._current_pending_iters += 1

    def asdict(self) -> list[dict]:
        return [asdict(record) for record in self.records]


@dataclass
class SsloRequestStats:
    chunk_stall_time_total: float
    total_pending_time_s: float
    num_pending_intervals: int
    chunks_completed: int
    final_chunk_expected_len: float | None
    # Scheduler-side step accounting. total_step_count = scheduler steps in
    # which this request received any tokens. prefill_step_count = subset
    # that mixed prefill (NOT decoding-only). Ratio reveals how often this
    # request shared a batch with prefill — a proxy for batch-composition
    # pressure that slows decode progress.
    total_step_count: int = 0
    prefill_step_count: int = 0
    # Wall-clock when the consumer can start reading: set at chunk 0 finish,
    # equals d(0) under the new contract.
    consume_start_ts: float | None = None


@dataclass
class RequestSLOState:
    # Lifecycle.
    decoding_start_ts: float | None = None
    # Absolute wall-clock deadline for the chunk currently being generated.
    # Updated at each chunk boundary by the stall-aware recurrence
    #   deadline(t) = max(deadline(t-1), finish(t-1)) + consume(t-1)
    # so the next chunk must arrive before this timestamp to be on-time.
    next_deadline_ts: float | None = None
    chunks_completed: int = 0
    current_chunk_generated_len: int = 0

    # Wall-clock when consumption can start; set once at chunk 0 finish.
    # d(0) = chunk_0_gen_finish_ts, not decoding_start_ts, because the
    # consumer cannot start until the first chunk is actually available.
    consume_start_ts: float | None = None

    # Diagnostic output.
    chunk_stats: ChunkStatCollector = field(default_factory=ChunkStatCollector)
    total_pending_time_s: float = 0.0
    num_pending_intervals: int = 0
    pending_enter_ts: float | None = None
    # Scheduler step accounting (incremented by Scheduler each step).
    total_step_count: int = 0
    prefill_step_count: int = 0

    # num_warmup_chunks lives on the instance because `phase` reads it
    # every call. The other config knobs are consumed only at construction
    # to build chunk_separator / consume_estimator / predictor, so they're
    # InitVars and don't persist as instance attributes.
    num_warmup_chunks: int = 4
    seconds_per_word: InitVar[float] = 0.28
    chunk_unit: InitVar[str] = "sentence"
    chunk_len_strategy: InitVar[str] = _DEFAULT_CHUNK_LEN_STRATEGY
    min_chunk_tokens: InitVar[int] = 16
    # Override hooks — resolved in __post_init__ into non-Optional
    # instance attributes so callers don't have to None-check on use.
    chunk_separator: InitVar[ChunkSeparator | None] = None
    consume_estimator: InitVar[ChunkConsumeEstimator | None] = None

    def __post_init__(
        self,
        seconds_per_word: float,
        chunk_unit: str,
        chunk_len_strategy: str,
        min_chunk_tokens: int,
        chunk_separator: ChunkSeparator | None,
        consume_estimator: ChunkConsumeEstimator | None,
    ) -> None:
        if self.num_warmup_chunks < 1:
            raise ValueError("num_warmup_chunks must be >= 1")
        self.chunk_separator: ChunkSeparator = (
            chunk_separator if chunk_separator is not None
            else ChunkSeparator(
                chunk_unit=chunk_unit,
                min_chunk_tokens=min_chunk_tokens))
        self.consume_estimator: ChunkConsumeEstimator = (
            consume_estimator if consume_estimator is not None
            else ChunkConsumeEstimator(seconds_per_word=seconds_per_word))
        self._chunk_len_predictor = ChunkLengthPredictor(
            strategy=chunk_len_strategy)

    @classmethod
    def from_config(cls, config: "SsloConfig") -> "RequestSLOState":
        return cls(
            seconds_per_word=config.seconds_per_word,
            num_warmup_chunks=config.num_warmup_chunks,
            chunk_unit=config.chunk_unit,
            chunk_len_strategy=config.chunk_len_strategy,
            min_chunk_tokens=config.min_chunk_tokens,
        )

    @property
    def phase(self) -> Phase:
        if self.decoding_start_ts is None:
            return Phase.PREFILL
        if self.chunks_completed >= self.num_warmup_chunks:
            return Phase.MEASURED
        return Phase.WARMUP

    @property
    def chunk_records(self) -> list[ChunkRecord]:
        return self.chunk_stats.records

    @property
    def chunk_stall_time_total(self) -> float:
        return self.chunk_stats.stall_time_total

    @property
    def chunk_expected_len(self) -> float | None:
        return self._chunk_len_predictor.value

    def chunk_deadline(self) -> float | None:
        # Bootstrapped to decoding_start_ts at first token / chunk event,
        # overwritten to gen_finish_ts(0) when chunk 0 completes (so
        # d(0) = consume_start_ts), then advanced by the stall-aware
        # recurrence at each subsequent chunk boundary.
        return self.next_deadline_ts

    def time_to_deadline(self, now: float) -> float | None:
        deadline = self.chunk_deadline()
        return None if deadline is None else deadline - now

    def expected_remaining_len(self) -> float | None:
        predicted = self._chunk_len_predictor.value
        if predicted is None:
            return None
        remaining = predicted - self.current_chunk_generated_len
        if remaining <= 0:
            # Over-shoot: this chunk is already longer than the p90
            # prediction. Fall back to the p99 companion so pressure()
            # keeps reflecting real remaining work instead of collapsing
            # to the 1.0 floor below. value_high is only populated under
            # strategy="p90"; for other strategies the floor is used.
            high = self._chunk_len_predictor.value_high
            if high is not None:
                return max(1.0, high - self.current_chunk_generated_len)
        return max(1.0, remaining)

    def pressure(self, now: float, tpot_s: float | None) -> float | None:
        """Forward-looking urgency (= remaining_work / time_to_deadline,
        normalized so 1.0 means "exactly at deadline given current TPOT").

        > 1.0 projects a deadline miss; 0 means lots of slack. Renamed
        from `score` because the value is dimensioned like load pressure
        (utilization-of-budget) and the semantics carry through the
        scheduler.

        Return-value contract:
          - ``None`` — not measurable yet (PREFILL/WARMUP, missing tpot,
            or predictor has no value).
          - ``float('inf')`` — deadline already passed (time_to_deadline
            <= 0). Callers MUST handle inf: it propagates through max()
            and >= comparisons, but breaks averaging (drop inf before
            computing aggregate pressures).
          - finite float — normal urgency.
        """
        if self.phase != Phase.MEASURED:
            return None
        if tpot_s is None:
            return None
        remaining = self.expected_remaining_len()
        if remaining is None:
            return None
        time_to_deadline = self.time_to_deadline(now)
        if time_to_deadline is None:
            return None
        if time_to_deadline <= 0:
            return float("inf")
        return (remaining * tpot_s) / time_to_deadline

    def _ensure_decoding_started(self, now: float) -> None:
        # Lazy bootstrap of decoding_start_ts and next_deadline_ts on the
        # first token / chunk event.
        if self.decoding_start_ts is None:
            self.decoding_start_ts = now
            self.next_deadline_ts = now

    def on_token(self, now: float) -> None:
        self._ensure_decoding_started(now)
        self.current_chunk_generated_len += 1

    def on_chunk_boundary(
        self,
        now: float,
        word_count: int,
        chunk_consume_time_s: float,
    ) -> None:
        self._ensure_decoding_started(now)

        deadline = self.chunk_deadline()
        assert deadline is not None

        if self.chunks_completed == 0:
            # d(0) = chunk 0's own finish time: the consumer can't start
            # until the first chunk is available, so its deadline IS now.
            self.consume_start_ts = now
            self.next_deadline_ts = now
            deadline = now
            start_time_ts = self.decoding_start_ts
        else:
            start_time_ts = self.chunk_stats.records[-1].gen_finish_ts

        slack = deadline - now
        stall_duration_s = max(0.0, -slack)
        if stall_duration_s > 0:
            stall_start_ts: float | None = deadline
            stall_end_ts: float | None = now
        else:
            stall_start_ts = None
            stall_end_ts = None

        # Invariant: tokens are always accumulated (via on_token /
        # on_text_delta) before a chunk boundary is observed. on_finish's
        # tail flush only fires when prior on_text_delta calls added the
        # tokens that produced the held-back text.
        num_token = self.current_chunk_generated_len
        assert num_token > 0, (
            "on_chunk_boundary requires at least one accumulated token")
        # Predictor's value BEFORE this chunk's update — i.e. the
        # estimate that pressure() used during this chunk's generation.
        expected_len = self._chunk_len_predictor.value

        self.chunk_stats.record(
            chunk_idx=self.chunks_completed,
            deadline_ts=deadline,
            gen_finish_ts=now,
            slack_s=slack,
            word_count=word_count,
            start_time_ts=start_time_ts,
            num_token=num_token,
            expected_len=expected_len,
            stall_start_ts=stall_start_ts,
            stall_end_ts=stall_end_ts,
            stall_duration_s=stall_duration_s,
        )
        self._chunk_len_predictor.update(num_token)

        # Stall-aware deadline recurrence (t = chunk index just completed):
        #   deadline(t+1) = max(deadline(t), finish(t)) + consume(t)
        # An early arrival (finish < deadline) does not buy buffer time —
        # consumption starts at deadline. A late arrival (finish > deadline)
        # pushes the next deadline back by exactly the lateness so a single
        # stall doesn't propagate as cumulative violation.
        self.next_deadline_ts = (
            max(deadline, now) + chunk_consume_time_s)
        self.chunks_completed += 1
        self.current_chunk_generated_len = 0

    def on_pending_enter(self, now: float) -> None:
        if self.pending_enter_ts is None:
            self.pending_enter_ts = now
            self.num_pending_intervals += 1

    def on_pending_exit(self, now: float) -> None:
        if self.pending_enter_ts is None:
            return
        interval = now - self.pending_enter_ts
        self.total_pending_time_s += interval
        self.chunk_stats.accumulate_pending(interval)
        self.pending_enter_ts = None

    def on_step(self, decoding_only: bool) -> None:
        """Increment per-request scheduler step counters.

        Called by the scheduler once per scheduling step in which this
        request was scheduled. Diagnostic only — not used by scoring.
        """
        self.total_step_count += 1
        if not decoding_only:
            self.prefill_step_count += 1

    def on_text_delta(
        self,
        text: str,
        now: float,
        num_tokens: int = 0,
    ) -> None:
        if not text and num_tokens <= 0:
            return
        delta_tokens = num_tokens if num_tokens > 0 else 1
        # Hot path: bulk-add tokens instead of calling on_token N times.
        # Each call would re-check decoding_start_ts and burn one
        # attribute write per token; with chunked prefill / speculative
        # decoding N can be 8+ per delta.
        self._ensure_decoding_started(now)
        self.current_chunk_generated_len += delta_tokens
        for chunk_text in self.chunk_separator.feed(text, delta_tokens):
            word_count = len(chunk_text.split())
            self.on_chunk_boundary(
                now=now,
                word_count=word_count,
                chunk_consume_time_s=self.consume_estimator.estimate(
                    chunk_text, word_count),
            )

    def on_finish(self, now: float) -> None:
        # Force-flush any held-back text on request completion so the last
        # chunk's diagnostics aren't lost even if it's shorter than
        # min_chunk_tokens.
        remaining = self.chunk_separator.flush()
        if remaining:
            word_count = len(remaining.split())
            self.on_chunk_boundary(
                now=now,
                word_count=word_count,
                chunk_consume_time_s=self.consume_estimator.estimate(
                    remaining, word_count),
            )
        elif self.decoding_start_ts is None:
            self.decoding_start_ts = now

    def compute_stats(self) -> SsloRequestStats:
        return SsloRequestStats(
            chunk_stall_time_total=self.chunk_stall_time_total,
            total_pending_time_s=self.total_pending_time_s,
            num_pending_intervals=self.num_pending_intervals,
            chunks_completed=self.chunks_completed,
            final_chunk_expected_len=self._chunk_len_predictor.value,
            total_step_count=self.total_step_count,
            prefill_step_count=self.prefill_step_count,
            consume_start_ts=self.consume_start_ts,
        )

    def chunk_records_asdict(self) -> list[dict]:
        return self.chunk_stats.asdict()
