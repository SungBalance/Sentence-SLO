# SPDX-License-Identifier: Apache-2.0
"""SSLO request lifecycle state for score-based scheduling."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum

_SENTENCE_END_CHARS = frozenset(".!?。！？…")
_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})
_CHUNK_LEN_EMA_ALPHA = 0.2


class Phase(IntEnum):
    PREFILL = 0
    WARMUP = 1
    MEASURED = 2


@dataclass
class ChunkRecord:
    chunk_idx: int
    deadline_ts: float
    gen_finish_ts: float
    consume_finish_ts: float
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
    final_chunk_expected_len_ema: float | None


@dataclass
class RequestSLOState:
    # Lifecycle.
    phase: Phase = Phase.PREFILL
    decoding_start_ts: float | None = None
    cumulative_consume_time: float = 0.0
    chunks_completed: int = 0
    current_chunk_generated_len: int = 0
    chunk_expected_len_ema: float | None = None

    # Offload.
    is_offloaded: bool = False
    offload_enter_ts: float | None = None

    # Diagnostic output.
    chunk_records: list[ChunkRecord] = field(default_factory=list)
    chunk_stall_time_total: float = 0.0
    total_pending_time_s: float = 0.0
    num_pending_intervals: int = 0
    pending_enter_ts: float | None = None
    total_offload_time_s: float = 0.0
    num_offload_intervals: int = 0

    # Config constants.
    seconds_per_word: float = 0.28
    num_warmup_chunks: int = 4
    chunk_unit: str = "sentence"

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}")
        if self.seconds_per_word < 0:
            raise ValueError("seconds_per_word must be >= 0")
        if self.num_warmup_chunks < 0:
            raise ValueError("num_warmup_chunks must be >= 0")
        self._pending_text = ""
        self._current_chunk_pending_time_s = 0.0

    @property
    def cumulative_slack(self) -> float:
        if not self.chunk_records:
            return 0.0
        return self.chunk_records[-1].slack_s

    def chunk_deadline(self) -> float | None:
        if self.decoding_start_ts is None:
            return None
        return self.decoding_start_ts + self.cumulative_consume_time

    def time_to_deadline(self, now: float) -> float | None:
        deadline = self.chunk_deadline()
        return None if deadline is None else deadline - now

    def expected_remaining_len(self) -> float | None:
        if self.chunk_expected_len_ema is None:
            return None
        return max(1.0, self.chunk_expected_len_ema -
                   self.current_chunk_generated_len)

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

    def on_token(self, now: float, token_id: int | None = None) -> None:
        del token_id
        if self.phase == Phase.PREFILL:
            self.phase = Phase.WARMUP
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
        if self.phase == Phase.PREFILL:
            self.phase = Phase.WARMUP

        deadline = self.chunk_deadline()
        assert deadline is not None
        slack = deadline - now
        stall = max(0.0, now - deadline)
        consume_finish = deadline + chunk_consume_time_s
        generated_len = self.current_chunk_generated_len or max(1, word_count)

        self.chunk_records.append(
            ChunkRecord(
                chunk_idx=self.chunks_completed,
                deadline_ts=deadline,
                gen_finish_ts=now,
                consume_finish_ts=consume_finish,
                slack_s=slack,
                stall_s=stall,
                pending_time_s=self._current_chunk_pending_time_s,
                word_count=word_count,
            ))
        self.chunk_stall_time_total += stall
        if self.chunk_expected_len_ema is None:
            self.chunk_expected_len_ema = float(generated_len)
        else:
            alpha = _CHUNK_LEN_EMA_ALPHA
            self.chunk_expected_len_ema = (
                alpha * generated_len +
                (1 - alpha) * self.chunk_expected_len_ema)

        self.cumulative_consume_time += chunk_consume_time_s
        self.chunks_completed += 1
        if self.chunks_completed >= self.num_warmup_chunks:
            self.phase = Phase.MEASURED
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

    def on_offload_enter(self, now: float) -> None:
        if self.offload_enter_ts is None:
            self.offload_enter_ts = now
            self.is_offloaded = True
            self.num_offload_intervals += 1

    def on_offload_exit(self, now: float) -> None:
        if self.offload_enter_ts is not None:
            self.total_offload_time_s += now - self.offload_enter_ts
        self.offload_enter_ts = None
        self.is_offloaded = False

    def on_text_delta(self, text: str, now: float, num_tokens: int = 0) -> None:
        if not text and num_tokens <= 0:
            return
        if num_tokens > 0:
            for _ in range(num_tokens):
                self.on_token(now)
        else:
            self.on_token(now)
        self._pending_text += text

        while True:
            boundary = self._find_boundary(self._pending_text)
            if boundary is None:
                break
            self._flush_text_chunk(now, boundary)

    def on_finish(self, now: float) -> None:
        if self._pending_text:
            self._flush_text_chunk(now, len(self._pending_text))
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
            final_chunk_expected_len_ema=self.chunk_expected_len_ema,
        )

    def chunk_records_asdict(self) -> list[dict]:
        return [asdict(record) for record in self.chunk_records]

    def _flush_text_chunk(self, now: float, boundary_pos: int) -> None:
        chunk_text = self._pending_text[:boundary_pos]
        word_count = len(chunk_text.split())
        self.on_chunk_boundary(
            now=now,
            word_count=word_count,
            chunk_consume_time_s=word_count * self.seconds_per_word,
        )
        self._pending_text = self._pending_text[boundary_pos:]

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
