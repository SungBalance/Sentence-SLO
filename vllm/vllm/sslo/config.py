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

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}")
        for name in (
            "num_warmup_chunks",
            "tpot_bucket_size",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        for name in (
            "seconds_per_word",
            "critical_threshold",
            "pending_in_threshold",
            "pending_out_threshold",
            "offloading_in_threshold",
            "offloading_out_threshold",
            "adaptive_batching_min_throughput_ratio",
            "offload_safety_margin_s",
            "offload_bandwidth_bytes_per_s",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        for name in (
            "tpot_ema_alpha",
            "adaptive_batching_min_throughput_ratio",
        ):
            value = getattr(self, name)
            if not (0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
            if self.tpot_bucket_size <= 0:
                raise ValueError("tpot_bucket_size must be > 0")
        if self.pending_in_threshold > self.pending_out_threshold:
            raise ValueError(
                "pending_in_threshold must be <= pending_out_threshold")
        if self.offloading_in_threshold > self.offloading_out_threshold:
            raise ValueError(
                "offloading_in_threshold must be <= offloading_out_threshold")
        if self.offload_bandwidth_bytes_per_s <= 0:
            raise ValueError("offload_bandwidth_bytes_per_s must be > 0")


def build_slo_state(config: SsloConfig) -> "RequestSLOState":
    """Create a RequestSLOState from SsloConfig."""
    from vllm.sslo.slo_state import RequestSLOState

    return RequestSLOState(
        seconds_per_word=config.seconds_per_word,
        num_warmup_chunks=config.num_warmup_chunks,
        chunk_unit=config.chunk_unit,
    )
