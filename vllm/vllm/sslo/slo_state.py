# SPDX-License-Identifier: Apache-2.0
"""SSLO request lifecycle state for score-based scheduling."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.sslo.config import SsloConfig

_SENTENCE_END_CHARS = frozenset(".!?。！？…")
_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})

_VALID_CHUNK_LEN_STRATEGIES = frozenset({"ema", "p90", "p99"})
_DEFAULT_CHUNK_LEN_STRATEGY = "p90"
_CHUNK_LEN_HISTORY_MAX = 64
_CHUNK_LEN_EMA_ALPHA = 0.2


class Phase(IntEnum):
    PREFILL = 0
    WARMUP = 1
    MEASURED = 2


class ChunkLengthPredictor:
    """Predict the next chunk's generated_len from completed chunks.

    Three strategies select different points on the conservatism axis:
      - "ema": EMA (alpha) — smooth, follows the central tendency.
      - "p90"/"p99": percentile over a sliding window — conservative
        upper bound that biases score() toward overestimating remaining
        work (and therefore admission/preemption pressure).
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
        self._history: list[int] = []
        self._ema: float | None = None
        self.value: float | None = None

    def update(self, generated_len: int) -> None:
        n = int(generated_len)
        if self.strategy == "ema":
            self._ema = float(n) if self._ema is None else (
                self.alpha * n + (1.0 - self.alpha) * self._ema)
            self.value = self._ema
            return
        self._history.append(n)
        if len(self._history) > self.history_max:
            self._history.pop(0)
        sorted_h = sorted(self._history)
        percentile = 90.0 if self.strategy == "p90" else 99.0
        idx = min(
            len(sorted_h) - 1,
            int(len(sorted_h) * percentile / 100.0),
        )
        self.value = float(sorted_h[idx])


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
    stall_s: float
    pending_time_s: float
    word_count: int


@dataclass
class SsloRequestStats:
    chunk_stall_time_total: float
    total_pending_time_s: float
    num_pending_intervals: int
    total_offload_time_s: float
    num_offload_intervals: int
    chunks_completed: int
    final_chunk_expected_len: float | None
    # Scheduler-side step accounting. total_step_count = scheduler steps in
    # which this request received any tokens. prefill_step_count = subset
    # that mixed prefill (NOT decoding-only). Ratio reveals how often this
    # request shared a batch with prefill — a proxy for batch-composition
    # pressure that slows decode progress.
    total_step_count: int = 0
    prefill_step_count: int = 0


@dataclass
class RequestSLOState:
    # Lifecycle.
    decoding_start_ts: float | None = None
    cumulative_consume_time: float = 0.0
    chunks_completed: int = 0
    current_chunk_generated_len: int = 0

    # Offload.
    offload_enter_ts: float | None = None

    # Diagnostic output.
    chunk_records: list[ChunkRecord] = field(default_factory=list)
    chunk_stall_time_total: float = 0.0
    total_pending_time_s: float = 0.0
    num_pending_intervals: int = 0
    pending_enter_ts: float | None = None
    total_offload_time_s: float = 0.0
    num_offload_intervals: int = 0
    # Scheduler step accounting (incremented by Scheduler each step).
    total_step_count: int = 0
    prefill_step_count: int = 0

    # Config snapshots — used to construct the default helpers below.
    # Provide chunk_separator / consume_estimator directly to override.
    seconds_per_word: float = 0.28
    num_warmup_chunks: int = 4
    chunk_unit: str = "sentence"
    chunk_len_strategy: str = _DEFAULT_CHUNK_LEN_STRATEGY
    min_chunk_tokens: int = 16
    chunk_separator: ChunkSeparator | None = None
    consume_estimator: ChunkConsumeEstimator | None = None

    def __post_init__(self) -> None:
        if self.num_warmup_chunks < 0:
            raise ValueError("num_warmup_chunks must be >= 0")
        if self.chunk_separator is None:
            self.chunk_separator = ChunkSeparator(
                chunk_unit=self.chunk_unit,
                min_chunk_tokens=self.min_chunk_tokens)
        if self.consume_estimator is None:
            self.consume_estimator = ChunkConsumeEstimator(
                seconds_per_word=self.seconds_per_word)
        self._chunk_len_predictor = ChunkLengthPredictor(
            strategy=self.chunk_len_strategy)
        self._current_chunk_pending_time_s = 0.0

    @classmethod
    def from_config(cls, config: "SsloConfig") -> "RequestSLOState":
        return cls(
            seconds_per_word=config.seconds_per_word,
            num_warmup_chunks=config.num_warmup_chunks,
            chunk_unit=config.chunk_unit,
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
    def is_offloaded(self) -> bool:
        return self.offload_enter_ts is not None

    @property
    def chunk_expected_len(self) -> float | None:
        return self._chunk_len_predictor.value

    @chunk_expected_len.setter
    def chunk_expected_len(self, value: float | None) -> None:
        # Direct assignment used by tests to inject a value bypassing the
        # predictor; keeps the predictor's `value` consistent.
        self._chunk_len_predictor.value = value

    def chunk_deadline(self) -> float | None:
        if self.decoding_start_ts is None:
            return None
        return self.decoding_start_ts + self.cumulative_consume_time

    def time_to_deadline(self, now: float) -> float | None:
        deadline = self.chunk_deadline()
        return None if deadline is None else deadline - now

    def expected_remaining_len(self) -> float | None:
        predicted = self._chunk_len_predictor.value
        if predicted is None:
            return None
        return max(1.0, predicted - self.current_chunk_generated_len)

    def score(self, now: float, tpot_s: float | None) -> float | None:
        """Forward-looking urgency. Values >= 1.0 project a deadline miss."""
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

    def on_token(self, now: float) -> None:
        if self.decoding_start_ts is None:
            self.decoding_start_ts = now
        self.current_chunk_generated_len += 1

    def on_chunk_boundary(
        self,
        now: float,
        word_count: int,
        chunk_consume_time_s: float,
    ) -> None:
        if self.decoding_start_ts is None:
            self.decoding_start_ts = now

        deadline = self.chunk_deadline()
        assert deadline is not None
        # Chunk 0 has no preceding consumption budget — its deadline is the
        # decoding start itself, so slack/stall are not meaningful. Treat
        # chunk 0 as always on-time (slack=0) so request-level compliance
        # measures only chunks 1+ where the consumer has a real budget.
        if self.chunks_completed == 0:
            slack = 0.0
            stall = 0.0
        else:
            slack = deadline - now
            stall = max(0.0, now - deadline)
        generated_len = self.current_chunk_generated_len or max(1, word_count)

        self.chunk_records.append(
            ChunkRecord(
                chunk_idx=self.chunks_completed,
                deadline_ts=deadline,
                gen_finish_ts=now,
                slack_s=slack,
                stall_s=stall,
                pending_time_s=self._current_chunk_pending_time_s,
                word_count=word_count,
            ))
        self.chunk_stall_time_total += stall
        self._chunk_len_predictor.update(int(generated_len))

        # Stall-aware deadline propagation. The next chunk's deadline is
        # max(this chunk's gen_finish_ts, the previous deadline) plus this
        # chunk's audio (consume) time — i.e., chunk N+1 must be ready by
        # the time the user finishes consuming chunk N, where consumption
        # starts no earlier than chunk N's gen_finish_ts. A late chunk
        # therefore extends the next deadline by exactly the lateness, so
        # carryover overage doesn't compound across chunks.
        end_offset = max(0.0, now - self.decoding_start_ts)
        self.cumulative_consume_time = (max(end_offset,
                                            self.cumulative_consume_time)
                                        + chunk_consume_time_s)
        self.chunks_completed += 1
        self.current_chunk_generated_len = 0
        self._current_chunk_pending_time_s = 0.0

    def on_pending_enter(self, now: float) -> None:
        if self.pending_enter_ts is None:
            self.pending_enter_ts = now
            self.num_pending_intervals += 1

    def on_pending_exit(self, now: float) -> None:
        if self.pending_enter_ts is None:
            return
        interval = now - self.pending_enter_ts
        self.total_pending_time_s += interval
        self._current_chunk_pending_time_s += interval
        self.pending_enter_ts = None

    def on_step(self, decoding_only: bool) -> None:
        """Increment per-request scheduler step counters.

        Called by the scheduler once per scheduling step in which this
        request was scheduled. Diagnostic only — not used by scoring.
        """
        self.total_step_count += 1
        if not decoding_only:
            self.prefill_step_count += 1

    def on_offload_enter(self, now: float) -> None:
        if self.offload_enter_ts is None:
            self.offload_enter_ts = now
            self.num_offload_intervals += 1

    def on_offload_exit(self, now: float) -> None:
        if self.offload_enter_ts is not None:
            self.total_offload_time_s += now - self.offload_enter_ts
        self.offload_enter_ts = None

    def on_text_delta(
        self,
        text: str,
        now: float,
        num_tokens: int = 0,
    ) -> None:
        if not text and num_tokens <= 0:
            return
        delta_tokens = num_tokens if num_tokens > 0 else 1
        for _ in range(delta_tokens):
            self.on_token(now)
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
            total_offload_time_s=self.total_offload_time_s,
            num_offload_intervals=self.num_offload_intervals,
            chunks_completed=self.chunks_completed,
            final_chunk_expected_len=self._chunk_len_predictor.value,
            total_step_count=self.total_step_count,
            prefill_step_count=self.prefill_step_count,
        )

    def chunk_records_asdict(self) -> list[dict]:
        return [asdict(record) for record in self.chunk_records]
