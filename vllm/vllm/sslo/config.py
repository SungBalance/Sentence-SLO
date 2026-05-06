# SPDX-License-Identifier: Apache-2.0
"""Configuration for SSLO score-based scheduling."""
from __future__ import annotations

from dataclasses import dataclass

_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})


@dataclass
class SsloConfig:
    enabled: bool = False
    offloading: bool = False
    adaptive_batching: bool = False
    num_warmup_chunks: int = 4
    tpot_bucket_size: int = 8
    tpot_ema_alpha: float = 0.1
    critical_threshold: float = 1.0
    pending_in_threshold: float = 0.3
    pending_out_threshold: float = 0.7
    offloading_in_threshold: float = 0.5
    offloading_out_threshold: float = 0.7
    adaptive_batching_min_throughput_ratio: float = 0.9
    offload_safety_margin_s: float = 0.05
    offload_bandwidth_bytes_per_s: float = 1e10
    seconds_per_word: float = 0.28
    chunk_unit: str = "sentence"
    # Defer flushing a found chunk boundary until the accumulated token count
    # since the last flush reaches this threshold. Short chunks (e.g. "Yes.")
    # are merged into the next chunk so the consume_time and chunk EMA are
    # not skewed by sub-token sentences.
    min_chunk_tokens: int = 16

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}")
        for name in ("num_warmup_chunks", "tpot_bucket_size"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        for name in ("tpot_ema_alpha",
                     "adaptive_batching_min_throughput_ratio"):
            value = getattr(self, name)
            if not (0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        for name in (
                "critical_threshold",
                "pending_in_threshold",
                "pending_out_threshold",
                "offloading_in_threshold",
                "offloading_out_threshold",
                "offload_safety_margin_s",
                "seconds_per_word",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.min_chunk_tokens < 0:
            raise ValueError(
                f"min_chunk_tokens must be >= 0, got {self.min_chunk_tokens}")
        if self.offload_bandwidth_bytes_per_s <= 0:
            raise ValueError(
                f"offload_bandwidth_bytes_per_s must be > 0, "
                f"got {self.offload_bandwidth_bytes_per_s}")
        if self.pending_in_threshold > self.pending_out_threshold:
            raise ValueError(
                "pending_in_threshold must be <= pending_out_threshold")
        if self.offloading_in_threshold > self.offloading_out_threshold:
            raise ValueError(
                "offloading_in_threshold must be <= offloading_out_threshold")
