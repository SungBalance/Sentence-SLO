# SPDX-License-Identifier: Apache-2.0
"""SSLO request lifecycle state for score-based scheduling."""
from __future__ import annotations

import csv
import math
import os
from bisect import bisect_left
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
# SSLO: lifted from 64 → 4096. ShareGPT-style reqs only yield ~10 chunks,
# but the global predictor (shared across reqs) accumulates many more —
# keep enough history to stabilise percentile estimates over long runs.
_CHUNK_LEN_HISTORY_MAX = 4096
_CHUNK_LEN_EMA_ALPHA = 0.2
# SSLO: number of per-req chunk samples below which expected_remaining_len
# uses the global predictor as a warm-up substitute. At sample_count >=
# this threshold the per-req predictor takes over (hard switch).
# SSLO: hybrid-predictor warmup threshold now lives in SsloConfig as
# `num_warmup_chunks` (read via RequestSLOState.num_warmup_chunks).


class Phase(IntEnum):
    PREFILL = 0
    WARMUP = 1
    MEASURED = 2



_VALID_PRESSURE_MISSING_REASONS = frozenset(
    {"prefill", "warmup", "no_tpot", "no_predictor"})

# Shared epsilon for multi-level pressure denominators so ttd→0 (or
# defer_buffer→0) does not blow up to NaN; saturated cases still surface
# as +inf via the explicit ≤ 0 branches in
# multi_level_pressure_components.
_PRESSURE_DENOM_EPSILON = 1e-9


@dataclass
class PressureComponents:
    """Decomposed pressure signal for logging and analysis.

    All time values are in seconds. Fields are None when the corresponding
    input was unavailable — callers must not substitute 1.0 here; that
    substitution belongs in the policy path only.
    """
    remaining_tokens: float | None
    estimated_step_time_s: float | None
    buffer_slack_s: float | None
    estimated_refill_time_s: float | None
    refill_slack_s: float | None
    depletion_pressure: float | None
    # True when depletion_pressure was computed from real signals.
    # False means the policy path will substitute 1.0.
    pressure_available: bool
    # When pressure_available is False, one of:
    #   "prefill"      — phase == PREFILL
    #   "warmup"       — phase == WARMUP
    #   "no_tpot"      — tpot input was None
    #   "no_predictor" — predictor.value is None
    # None when pressure_available is True.
    pressure_missing_reason: str | None
    # Multi-level pressure fields (policy="multi_level_pressure"). All
    # optional / None for legacy policies. When the request is measurable:
    #   epoch_time_s        — expected wall-clock for one scheduler step at
    #                          the candidate batch size (= tpot_ema[n]).
    #   defer_buffer_slack_s — ttd - epoch_time_s; remaining slack if this
    #                          step is deferred. None when epoch_s is None
    #                          (callers asking for serve-only).
    #   serve_pressure      — refill_time / max(ttd, ε). 1.0 ⇒ at-deadline.
    #   defer_pressure      — refill_time / max(defer_buffer, ε). > serve
    #                          whenever epoch_s > 0; +inf if defer_buffer ≤ 0.
    epoch_time_s: float | None = None
    defer_buffer_slack_s: float | None = None
    serve_pressure: float | None = None
    defer_pressure: float | None = None


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
        escalate_threshold: float = 0.9,
        overshoot_safety_factor: float = 2.5,
    ) -> None:
        if strategy not in _VALID_CHUNK_LEN_STRATEGIES:
            raise ValueError(
                f"chunk_len_strategy must be one of "
                f"{sorted(_VALID_CHUNK_LEN_STRATEGIES)}, got {strategy!r}")
        if not (0 < escalate_threshold <= 1.0):
            raise ValueError(
                "escalate_threshold must be in (0, 1], "
                f"got {escalate_threshold}")
        if overshoot_safety_factor < 0:
            raise ValueError(
                "overshoot_safety_factor must be >= 0, "
                f"got {overshoot_safety_factor}")
        self.strategy = strategy
        self.history_max = history_max
        self.alpha = alpha
        # SSLO: tier escalation + post-topmost overshoot knobs.
        self.escalate_threshold = escalate_threshold
        self.overshoot_safety_factor = overshoot_safety_factor
        self._history: deque[int] = deque(maxlen=history_max)
        self._ema: float | None = None
        self.value: float | None = None
        # Escalation ladder for strategy == "p90": when the running
        # chunk's generated_len overshoots the p90 prediction we step up
        # to value_mid (p95); past that we step up to value_high (p99).
        # Other strategies leave both companions at None so the floor
        # logic in expected_remaining_len() falls through.
        self.value_mid: float | None = None
        self.value_high: float | None = None
        # SSLO: full percentile ladder for strategy=="p90":
        # [p50, p60, p70, p80, p90, p95, p99]. expected_remaining_len
        # escalates through these on overshoot. None for other strategies.
        self.tier_values: list[float] | None = None
        # SSLO: strategy-agnostic sample counter (deque length is wrong
        # for ema strategy where _history isn't used).
        self._sample_count: int = 0
        if strategy == "ema":
            self.update = self.update_ema
        elif strategy in ("p90", "p99"):
            self.update = self.update_percentile
        elif strategy == "past-future":
            self.update = self.update_past_future

    @property
    def sample_count(self) -> int:
        # SSLO: number of observed chunks. Used by RequestSLOState to
        # decide whether to fall back to the global predictor.
        return self._sample_count

    def update_ema(self, generated_len: int) -> None:
        n = int(generated_len)
        self._ema = float(n) if self._ema is None else (
            self.alpha * n + (1.0 - self.alpha) * self._ema)
        self.value = self._ema
        self._sample_count += 1

    def update_percentile(self, generated_len: int) -> None:
        self._history.append(int(generated_len))
        self._sample_count += 1
        arr = np.fromiter(self._history, dtype=float)
        percentile = 90.0 if self.strategy == "p90" else 99.0
        self.value = float(np.percentile(arr, percentile, method="nearest"))
        if self.strategy == "p90":
            # Ladder: p90 → p95 → p99 → overshoot(2.5×).
            self.value_mid = float(
                np.percentile(arr, 95.0, method="nearest"))
            self.value_high = float(
                np.percentile(arr, 99.0, method="nearest"))
            self.tier_values = [self.value, self.value_mid, self.value_high]

    def update_past_future(self, generated_len: int) -> None:
        pass

    def reset(self) -> None:
        """Clear all accumulated state. Used between request-rate sweeps
        within a single engine lifetime so each rate run starts from a
        cold predictor (matches a fresh-engine baseline)."""
        self._history.clear()
        self._ema = None
        self.value = None
        self.value_mid = None
        self.value_high = None
        self._sample_count = 0


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

    def estimate(
        self,
        chunk_text: str,
        word_count: int,
    ) -> tuple[float, float | None]:
        del chunk_text  # unused in the default rate-based estimator
        return word_count * self.seconds_per_word, None

    def predict_conversion(self, word_count: int) -> float:
        """Predict conversion_time for a chunk of `word_count` words.
        Read mode has no conversion → 0.0. TTS subclass overrides."""
        del word_count
        return 0.0


class TtsProfileConsumeEstimator(ChunkConsumeEstimator):
    """Profile-driven TTS audio duration and conversion-time estimates."""

    def __init__(self, profile_path: str, model: str) -> None:
        if not isinstance(profile_path, str) or not profile_path:
            raise ValueError("profile_path must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(profile_path)
        self.profile_path = profile_path
        self.model = model
        self._consume_by_wc: dict[int, float] = {}
        self._conversion_by_wc: dict[int, float] = {}
        models_present: set[str] = set()

        required_columns = {
            "model",
            "word_count_low",
            "conversion_time_s_mean",
            "audio_duration_s_mean",
        }
        with open(profile_path, newline="") as f:
            reader = csv.DictReader(f)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"TTS profile missing required columns: "
                    f"{sorted(missing)}")
            for row in reader:
                row_model = row["model"]
                models_present.add(row_model)
                if row_model != model:
                    continue
                wc = int(row["word_count_low"])
                self._consume_by_wc[wc] = float(
                    row["audio_duration_s_mean"])
                self._conversion_by_wc[wc] = float(
                    row["conversion_time_s_mean"])

        if not self._consume_by_wc:
            raise ValueError(
                f"TTS profile has no rows for model {model!r}; "
                f"models present: {sorted(models_present)}")
        self._wc_sorted = sorted(self._consume_by_wc.keys())

    def _nearest_word_count(self, word_count: int) -> int:
        idx = bisect_left(self._wc_sorted, word_count)
        if idx <= 0:
            return self._wc_sorted[0]
        if idx >= len(self._wc_sorted):
            return self._wc_sorted[-1]
        lower = self._wc_sorted[idx - 1]
        upper = self._wc_sorted[idx]
        if word_count - lower <= upper - word_count:
            return lower
        return upper

    def estimate(
        self,
        chunk_text: str,
        word_count: int,
    ) -> tuple[float, float]:
        del chunk_text
        wc = self._nearest_word_count(word_count)
        return self._consume_by_wc[wc], self._conversion_by_wc[wc]

    def predict_conversion(self, word_count: int) -> float:
        wc = self._nearest_word_count(max(1, int(word_count)))
        return self._conversion_by_wc[wc]


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

    @property
    def pending_word_count(self) -> int:
        """Words accumulated in the current (not-yet-flushed) chunk."""
        return len(self._pending_text.split())

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

        # Sentence mode boundaries (whichever appears first):
        #   (a) one or more sentence-end chars followed by whitespace/EOS
        #   (b) any newline `\n` — handles scripts/structures without ASCII
        #       sentence-end punctuation (Thai/Korean lists, markdown table
        #       rows, `=keyword=` lists, degenerate `\boxed{X}\n` repetitions).
        # `min_chunk_tokens` already merges short newline-separated
        # fragments (table cells, code lines) into the next chunk, so the
        # newline rule does not over-fragment normal prose.
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "\n":
                return i + 1
            if ch in _SENTENCE_END_CHARS:
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
    unit_index: int
    deadline: float
    text_generation_end_time: float
    consumer_ready_time: float
    unit_deadline_miss_s: float
    unit_deadline_missed: int
    pending_time_s: float
    word_count: int
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
    # Cumulative token indices: half-open [token_start, token_end).
    token_start: int = 0
    token_end: int = 0
    token_boundary: int = 0
    # Consume time persisted here for CP-SLO export; same value used to
    # advance next_deadline_ts in on_chunk_boundary.
    consume_duration: float = 0.0
    conversion_time: float = 0.0
    # p99 companion from the predictor — only populated under p90 strategy.
    expected_chunk_len_high: float | None = None
    # Which predictor strategy produced expected_len.
    predictor_source: str = ""
    # Raw chunk text — used for diagnostics (e.g. inspect what kind of
    # text drives outlier-length chunks). Off the critical path; kept
    # because chunks.jsonl is already large enough that one more field
    # doesn't materially affect dump size.
    text: str = ""


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
        unit_index: int,
        deadline: float,
        text_generation_end_time: float,
        consumer_ready_time: float,
        unit_deadline_miss_s: float,
        word_count: int,
        num_token: int,
        expected_len: float | None,
        token_start: int,
        token_end: int,
        token_boundary: int,
        consume_duration: float,
        conversion_time: float,
        expected_chunk_len_high: float | None,
        predictor_source: str,
        text: str = "",
    ) -> None:
        running_iters = self._current_running_iters
        pending_iters = self._current_pending_iters
        self.records.append(
            ChunkRecord(
                unit_index=unit_index,
                deadline=deadline,
                text_generation_end_time=text_generation_end_time,
                consumer_ready_time=consumer_ready_time,
                unit_deadline_miss_s=unit_deadline_miss_s,
                unit_deadline_missed=int(unit_deadline_miss_s > 0),
                pending_time_s=self._current_pending_s,
                word_count=word_count,
                num_token=num_token,
                num_iters=running_iters + pending_iters,
                num_running_iters=running_iters,
                num_pending_iters=pending_iters,
                expected_len=expected_len,
                token_start=token_start,
                token_end=token_end,
                token_boundary=token_boundary,
                consume_duration=consume_duration,
                conversion_time=conversion_time,
                expected_chunk_len_high=expected_chunk_len_high,
                predictor_source=predictor_source,
                text=text,
            ))
        self.stall_time_total += unit_deadline_miss_s
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
    consume_start_time: float | None = None
    # Wall-clock of first waiting→running transition; None if never admitted.
    admitted_ts: float | None = None
    # Final disposition of the request.
    terminal_outcome: str = "in_progress"


def _quantized_pressure(remaining: float, time_budget: float,
                        tpot_s: float) -> float:
    """remaining tokens / floor(time_budget / tpot_s).
    Returns +inf when fewer than one iteration fits.

    Adds a tiny epsilon (1e-9) before floor to suppress floating-point
    pathology — e.g., ``1.0 - 0.3 == 0.6999999999999999`` would otherwise
    floor 0.6999.../0.1 to 6 instead of 7."""
    iters = math.floor(time_budget / tpot_s + 1e-9)
    if iters <= 0:
        return float("inf")
    return remaining / iters


@dataclass
class RequestSLOState:
    # Lifecycle.
    decoding_start_ts: float | None = None
    # Consumer-side wall-clock deadline for the chunk currently being
    # generated. Updated at each chunk boundary by the stall-aware recurrence
    #   deadline_consumer(t) = max(deadline_consumer(t-1),
    #                              text_end(t-1) + conv(t-1)) + consume(t-1)
    # so the next chunk's audio can start before this timestamp.
    next_deadline_ts: float | None = None
    chunks_completed: int = 0
    current_chunk_generated_len: int = 0
    last_num_token: int | None = None
    last_word_count: int | None = None

    # Wall-clock when consumption can start; set once at chunk 0 audio-ready
    # time, because the consumer cannot start until the first chunk is
    # converted.
    consume_start_time: float | None = None

    # Wall-clock of first waiting→running transition. Idempotent — only
    # set once; re-entries (running→pending→running) do not update it.
    admitted_ts: float | None = None

    # Final request disposition, updated at cleanup time.
    terminal_outcome: str = "in_progress"

    # Diagnostic output.
    chunk_stats: ChunkStatCollector = field(default_factory=ChunkStatCollector)
    total_pending_time_s: float = 0.0
    num_pending_intervals: int = 0
    pending_enter_ts: float | None = None
    # Scheduler step accounting (incremented by Scheduler each step).
    total_step_count: int = 0
    prefill_step_count: int = 0

    # num_warmup_chunks: threshold (in chunk samples) at which the per-req
    # chunk-length predictor is considered stable. Below this count
    # expected_remaining_len falls back to the shared global predictor.
    # No longer gates phase transitions — phase=MEASURED triggers as soon
    # as chunks_completed >= 1.
    num_warmup_chunks: int = 16
    # SSLO: cold-start (global predictor < threshold) fallback knobs.
    global_warmup_predictor_samples: int = 128
    cold_start_max_remaining_tokens: int = 2048
    seconds_per_word: InitVar[float] = 0.28
    chunk_unit: InitVar[str] = "sentence"
    chunk_len_strategy: InitVar[str] = _DEFAULT_CHUNK_LEN_STRATEGY
    min_chunk_tokens: InitVar[int] = 16
    # SSLO: predictor escalation knobs (default = legacy 0.9/2.5).
    predictor_escalate_threshold: InitVar[float] = 0.9
    predictor_overshoot_safety_factor: InitVar[float] = 2.5
    # SSLO: optional global predictor reference shared across reqs. When
    # the per-req predictor has fewer than _GLOBAL_PREDICTOR_WARMUP_SAMPLES
    # samples, expected_remaining_len falls back to this. Updated on every
    # chunk completion across all requests (system-wide distribution).
    global_chunk_len_predictor: InitVar["ChunkLengthPredictor | None"] = None
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
        predictor_escalate_threshold: float,
        predictor_overshoot_safety_factor: float,
        global_chunk_len_predictor: "ChunkLengthPredictor | None",
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
            strategy=chunk_len_strategy,
            escalate_threshold=predictor_escalate_threshold,
            overshoot_safety_factor=predictor_overshoot_safety_factor)
        # SSLO: shared global predictor (set by Scheduler via from_config).
        self._global_chunk_len_predictor = global_chunk_len_predictor
        # Running cumulative token count; reset at each chunk boundary.
        self._cumulative_tokens: int = 0

    @classmethod
    def from_config(
        cls,
        config: "SsloConfig",
        global_chunk_len_predictor: "ChunkLengthPredictor | None" = None,
    ) -> "RequestSLOState":
        consume_estimator: ChunkConsumeEstimator | None = None
        if config.consume_mode == "tts":
            assert config.tts_profile_path is not None
            assert config.tts_model is not None
            consume_estimator = TtsProfileConsumeEstimator(
                config.tts_profile_path, config.tts_model)
        return cls(
            seconds_per_word=config.seconds_per_word,
            num_warmup_chunks=config.num_warmup_chunks,
            global_warmup_predictor_samples=(
                config.global_warmup_predictor_samples),
            cold_start_max_remaining_tokens=(
                config.cold_start_max_remaining_tokens),
            chunk_unit=config.chunk_unit,
            chunk_len_strategy=config.chunk_len_strategy,
            min_chunk_tokens=config.min_chunk_tokens,
            # SSLO
            predictor_escalate_threshold=(
                config.mlp_predictor_escalate_threshold),
            # SSLO
            predictor_overshoot_safety_factor=(
                config.mlp_predictor_overshoot_safety_factor),
            # SSLO: shared system-wide predictor for warm-up substitute.
            global_chunk_len_predictor=global_chunk_len_predictor,
            consume_estimator=consume_estimator,
        )

    @property
    def phase(self) -> Phase:
        if self.decoding_start_ts is None:
            return Phase.PREFILL
        # SSLO: WARMUP only while the first chunk hasn't completed.
        # As soon as chunks_completed >= 1 the per-req predictor has a
        # sample and the hybrid predictor can fall through to global,
        # so the policy treats the request as MEASURED.
        if self.chunks_completed == 0:
            return Phase.WARMUP
        return Phase.MEASURED

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
        # overwritten to chunk 0 consumer-ready time when chunk 0 completes,
        # then advanced by the consumer-side stall-aware recurrence at each
        # subsequent chunk boundary.
        return self.next_deadline_ts

    def time_to_deadline(self, now: float) -> float | None:
        deadline = self.chunk_deadline()
        if deadline is None:
            return None
        conv_est = self._predict_current_conversion()
        return deadline - 1.2 * conv_est - now

    def _predict_current_conversion(self) -> float:
        """Estimate next-chunk conv based on the CURRENT chunk's in-progress
        text. Scale current word count by (expected_tokens / generated_tokens
        so far) to project the total wc, then look up profile."""
        consume_estimator = getattr(self, "consume_estimator", None)
        if consume_estimator is None:
            return 0.0
        predicted_tokens = self._chunk_len_predictor.value
        if predicted_tokens is None or predicted_tokens <= 0:
            return 0.0
        cur_tokens = self.current_chunk_generated_len
        if cur_tokens <= 0:
            return 0.0
        cur_words = self.chunk_separator.pending_word_count
        if cur_words <= 0:
            return 0.0
        # Scale current-chunk word count by expected vs generated token ratio.
        ratio = predicted_tokens / cur_tokens
        estimated_total_wc = max(1, int(round(cur_words * ratio)))
        return consume_estimator.predict_conversion(estimated_total_wc)

    def expected_remaining_len(self) -> float | None:
        # SSLO: predictor ladder (strategy=="p90") with early escalation
        # and post-topmost overshoot. cur가 현재 tier value의
        # `escalate_threshold` 비율(default 0.9)에 도달하면 다음 tier
        # (p90 → p95 → p99)로 진입한다. cur > topmost tier value 이후엔
        # `(cur - anchor) × overshoot_safety_factor` 로 remaining 산출 —
        # 1.0 saturate를 막아 long-tail chunks에서도 MLP가 pressure를
        # 반영. Other strategies (ema, p99) leave value_mid / value_high
        # at None so the topmost-tier branch fires directly.
        #
        # SSLO: hybrid predictor — use global predictor as warm-up
        # substitute when per-req has fewer than num_warmup_chunks
        # samples. Hard switch (not blended) per user spec.
        per = self._chunk_len_predictor
        glob = self._global_chunk_len_predictor
        # SSLO: while the shared global predictor itself is cold
        # (sample_count < global_warmup_predictor_samples), assume the
        # worst case for remaining-chunk length (max generation length
        # minus what's already produced in this chunk). Saturates
        # pressure high so SSLO admission throttles aggressively on the
        # first chunks where the global predictor has no signal yet.
        cur = self.current_chunk_generated_len
        if (glob is not None and glob.sample_count
                < self.global_warmup_predictor_samples):
            return max(
                1.0, self.cold_start_max_remaining_tokens - cur)
        if (glob is not None and glob.value is not None
                and per.sample_count < self.num_warmup_chunks):
            pred = glob
        else:
            pred = per
        if pred.value is None:
            return None
        thr = pred.escalate_threshold
        tiers = pred.tier_values
        if tiers is not None:
            # Full ladder: p50→p60→p70→p80→p90→p95→p99→overshoot
            # Find first tier where cur is still below thr×tier.
            for v in tiers:
                if cur < v * thr:
                    return max(1.0, v - cur)
            # Past topmost tier: overshoot mode anchored at p99.
            anchor = tiers[-1]
            return max(1.0, (cur - anchor) * pred.overshoot_safety_factor)
        # Single-tier strategies (p99, ema): legacy single-tier check then
        # overshoot at pred.value.
        if cur < pred.value * thr:
            return max(1.0, pred.value - cur)
        return max(1.0, (cur - pred.value) * pred.overshoot_safety_factor)

    def pressure_components(
        self, now: float, tpot_s: float | None
    ) -> PressureComponents:
        """Compute the raw pressure components for logging.

        Unlike pressure(), this never returns a scalar fallback — instead
        it surfaces which input was missing so the analysis layer can
        distinguish "not measurable" from "exactly at deadline".
        """
        if self.phase == Phase.PREFILL:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="prefill")
        if self.phase == Phase.WARMUP:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="warmup")
        if tpot_s is None:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=None,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="no_tpot")
        remaining = self.expected_remaining_len()
        if remaining is None:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False,
                pressure_missing_reason="no_predictor")

        ttd = self.time_to_deadline(now)
        refill = remaining * tpot_s
        refill_slack = (None if ttd is None else ttd - refill)
        if ttd is None:
            depletion = None
        elif ttd <= 0:
            depletion = float("inf")
        else:
            depletion = _quantized_pressure(remaining, ttd, tpot_s)
        return PressureComponents(
            remaining_tokens=remaining,
            estimated_step_time_s=tpot_s,
            buffer_slack_s=ttd,
            estimated_refill_time_s=refill,
            refill_slack_s=refill_slack,
            depletion_pressure=depletion,
            pressure_available=True,
            pressure_missing_reason=None)

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
        return self.pressure_components(now, tpot_s).depletion_pressure

    def multi_level_pressure_components(
        self,
        now: float,
        tpot_s: float | None,
        epoch_s: float | None,
    ) -> PressureComponents:
        """Two-level pressure signal used by policy="multi_level_pressure".

        Mirrors pressure_components phase gating (PREFILL/WARMUP/no_tpot/
        no_predictor → pressure_available=False with reason) so the
        decision log can distinguish unmeasurable requests from
        at-deadline ones. When measurable, populates:
          - serve_pressure = refill / max(ttd, ε)
          - defer_pressure = refill / max(ttd - epoch_s, ε)
        epoch_s=None means "treat the next step's wall-clock as 0", so
        defer collapses onto serve — used when the caller is only
        interested in the serve-side urgency at a given batch.
        """
        if self.phase == Phase.PREFILL:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="prefill",
                epoch_time_s=epoch_s,
                defer_buffer_slack_s=None,
                serve_pressure=None, defer_pressure=None)
        if self.phase == Phase.WARMUP:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="warmup",
                epoch_time_s=epoch_s,
                defer_buffer_slack_s=None,
                serve_pressure=None, defer_pressure=None)
        if tpot_s is None:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=None,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False, pressure_missing_reason="no_tpot",
                epoch_time_s=epoch_s,
                defer_buffer_slack_s=None,
                serve_pressure=None, defer_pressure=None)
        remaining = self.expected_remaining_len()
        if remaining is None:
            return PressureComponents(
                remaining_tokens=None, estimated_step_time_s=tpot_s,
                buffer_slack_s=None, estimated_refill_time_s=None,
                refill_slack_s=None, depletion_pressure=None,
                pressure_available=False,
                pressure_missing_reason="no_predictor",
                epoch_time_s=epoch_s,
                defer_buffer_slack_s=None,
                serve_pressure=None, defer_pressure=None)

        ttd = self.time_to_deadline(now)
        refill = remaining * tpot_s
        refill_slack = (None if ttd is None else ttd - refill)
        if ttd is None:
            depletion = None
            serve = None
            defer = None
            defer_buffer = None
        elif ttd <= 0:
            depletion = float("inf")
            serve = float("inf")
            defer = float("inf")
            # Defer buffer is still defined (epoch shifts the already-past
            # deadline further into the red); report it for logging.
            defer_buffer = (None if epoch_s is None else ttd - epoch_s)
        else:
            depletion = _quantized_pressure(remaining, ttd, tpot_s)
            serve = _quantized_pressure(remaining, ttd, tpot_s)
            if epoch_s is None:
                defer_buffer = None
                defer = serve
            else:
                defer_buffer = ttd - epoch_s
                if defer_buffer <= 0:
                    defer = float("inf")
                else:
                    defer = _quantized_pressure(
                        remaining, defer_buffer, tpot_s)
        return PressureComponents(
            remaining_tokens=remaining,
            estimated_step_time_s=tpot_s,
            buffer_slack_s=ttd,
            estimated_refill_time_s=refill,
            refill_slack_s=refill_slack,
            depletion_pressure=depletion,
            pressure_available=True,
            pressure_missing_reason=None,
            epoch_time_s=epoch_s,
            defer_buffer_slack_s=defer_buffer,
            serve_pressure=serve,
            defer_pressure=defer)

    def serve_pressure(
        self, now: float, tpot_s: float | None
    ) -> float | None:
        """Scalar wrapper around multi_level_pressure_components.

        epoch_s is fixed to None — caller only wants the serve side.
        Returns None for unmeasurable requests (PREFILL/WARMUP/no_tpot/
        no_predictor); finite or +inf otherwise.
        """
        return self.multi_level_pressure_components(
            now, tpot_s, None).serve_pressure

    def defer_pressure(
        self,
        now: float,
        tpot_s: float | None,
        epoch_s: float | None,
    ) -> float | None:
        """Scalar wrapper around multi_level_pressure_components.

        Same None contract as serve_pressure(). When epoch_s is None the
        defer side collapses to the serve side.
        """
        return self.multi_level_pressure_components(
            now, tpot_s, epoch_s).defer_pressure

    def mark_admitted(self, now: float) -> None:
        if self.admitted_ts is None:
            self.admitted_ts = now

    def mark_terminal(self, outcome: str) -> None:
        self.terminal_outcome = outcome

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
        consume_duration: float,
        conversion_time: float | None = None,
        text: str = "",
    ) -> None:
        self._ensure_decoding_started(now)

        deadline = self.chunk_deadline()
        assert deadline is not None

        if conversion_time is None:
            record_conversion_time = 0.0
            consumer_ready_time = now
        else:
            record_conversion_time = conversion_time
            consumer_ready_time = now + record_conversion_time

        if self.chunks_completed == 0:
            # next_deadline_ts is consumer-side (when playback can start).
            # Chunk 0's text-side deadline is consumer deadline - conversion,
            # which equals now, so it is on-time by construction.
            self.consume_start_time = consumer_ready_time
            self.next_deadline_ts = consumer_ready_time
            deadline = consumer_ready_time

        # Miss is computed in TEXT-side time: was the text late vs the time
        # it had to finish so conversion could meet the consumer deadline?
        deadline_text = deadline - record_conversion_time
        unit_deadline_miss_s = max(0.0, now - deadline_text)

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

        token_start = self._cumulative_tokens
        token_end = self._cumulative_tokens + num_token
        self._cumulative_tokens = token_end

        self.chunk_stats.record(
            unit_index=self.chunks_completed,
            deadline=deadline,
            text_generation_end_time=now,
            consumer_ready_time=consumer_ready_time,
            unit_deadline_miss_s=unit_deadline_miss_s,
            word_count=word_count,
            num_token=num_token,
            expected_len=expected_len,
            token_start=token_start,
            token_end=token_end,
            token_boundary=token_end,
            consume_duration=consume_duration,
            conversion_time=record_conversion_time,
            expected_chunk_len_high=self._chunk_len_predictor.value_high,
            predictor_source=self._chunk_len_predictor.strategy,
            text=text,
        )
        self._chunk_len_predictor.update(num_token)
        # SSLO: feed the global predictor too so other reqs benefit from
        # this sample during their warm-up window.
        if self._global_chunk_len_predictor is not None:
            self._global_chunk_len_predictor.update(num_token)
        self.last_num_token = num_token
        self.last_word_count = word_count

        # Sequential-TTS recurrence (user spec):
        #   deadline(t+1) = max(deadline(t), text_end(t)) + conv(t) + consume(t)
        # Conv added OUTSIDE max so each chunk contributes conv + consume,
        # treating conversion as serial after previous chunk's playback
        # (no pipelining overlap). Read mode (conv=0) unaffected.
        self.next_deadline_ts = (
            max(deadline, now) + record_conversion_time + consume_duration)
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
            consume_s, conversion_s = self.consume_estimator.estimate(
                chunk_text, word_count)
            self.on_chunk_boundary(
                now=now,
                word_count=word_count,
                consume_duration=consume_s,
                conversion_time=conversion_s,
                text=chunk_text,
            )

    def on_finish(self, now: float) -> None:
        # Force-flush any held-back text on request completion so the last
        # chunk's diagnostics aren't lost even if it's shorter than
        # min_chunk_tokens. The chunk_separator can carry trailing
        # whitespace after a yield (e.g. " " left after "Hello."), in
        # which case flush() returns text but no tokens were accumulated
        # since the previous boundary — skip the on_chunk_boundary call
        # so the assertion (num_token > 0) doesn't trip.
        remaining = self.chunk_separator.flush()
        if remaining and self.current_chunk_generated_len > 0:
            word_count = len(remaining.split())
            consume_s, conversion_s = self.consume_estimator.estimate(
                remaining, word_count)
            self.on_chunk_boundary(
                now=now,
                word_count=word_count,
                consume_duration=consume_s,
                conversion_time=conversion_s,
                text=remaining,
            )
        elif self.decoding_start_ts is None:
            self.decoding_start_ts = now
        # output_processor calls on_finish on natural completion (the
        # only place this method fires from production). Abort/timeout
        # paths skip the output processor and stay at "in_progress".
        if self.terminal_outcome == "in_progress":
            self.terminal_outcome = "completed"

    def compute_stats(self) -> SsloRequestStats:
        return SsloRequestStats(
            chunk_stall_time_total=self.chunk_stall_time_total,
            total_pending_time_s=self.total_pending_time_s,
            num_pending_intervals=self.num_pending_intervals,
            chunks_completed=self.chunks_completed,
            final_chunk_expected_len=self._chunk_len_predictor.value,
            total_step_count=self.total_step_count,
            prefill_step_count=self.prefill_step_count,
            consume_start_time=self.consume_start_time,
            admitted_ts=self.admitted_ts,
            terminal_outcome=self.terminal_outcome,
        )

    def chunk_records_asdict(self) -> list[dict]:
        return self.chunk_stats.asdict()
