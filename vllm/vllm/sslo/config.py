# SPDX-License-Identifier: Apache-2.0
"""SsloConfig: configuration dataclass for SSLO (Sentence-level SLO)."""
from __future__ import annotations

from dataclasses import dataclass

_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})
_VALID_ESTIMATOR_TYPES = frozenset({"word_rate"})
_VALID_CHUNK_GEN_ESTIMATORS = frozenset({"ema", "p99"})


@dataclass
class SsloConfig:
    enabled: bool = False
    seconds_per_word: float = 0.28
    chunk_unit: str = "sentence"
    estimator_type: str = "word_rate"
    offloading: bool = False
    adaptive_batch_size: bool = False
    max_consecutive_pending: int = 5
    ema_alpha: float = 0.2
    chunk_gen_estimator: str = "ema"
    chunk_gen_p99_window: int = 100
    pending_warmup_chunks: int = 5
    pending_pressure_lambda: float = 0.05
    pending_hysteresis_gap: float = 0.5

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}"
            )
        if self.estimator_type not in _VALID_ESTIMATOR_TYPES:
            raise ValueError(
                f"estimator_type must be one of {sorted(_VALID_ESTIMATOR_TYPES)}, "
                f"got {self.estimator_type!r}"
            )
        if self.chunk_gen_estimator not in _VALID_CHUNK_GEN_ESTIMATORS:
            raise ValueError(
                "chunk_gen_estimator must be one of "
                f"{_VALID_CHUNK_GEN_ESTIMATORS}, "
                f"got {self.chunk_gen_estimator!r}"
            )


def build_slo_state(config: SsloConfig) -> "RequestSLOState":
    """Create a RequestSLOState from SsloConfig (single factory for both sides)."""
    from vllm.sslo.slo_state import (
        EmaChunkGenerationEstimator,
        ParagraphChunkDetector,
        PercentileChunkGenerationEstimator,
        RequestSLOState,
        WordRateEstimator,
    )
    estimator = WordRateEstimator(seconds_per_word=config.seconds_per_word)
    detector = ParagraphChunkDetector() if config.chunk_unit == "paragraph" else None
    if config.chunk_gen_estimator == "p99":
        cge = PercentileChunkGenerationEstimator(
            percentile=99.0,
            window_size=config.chunk_gen_p99_window,
        )
    else:
        cge = EmaChunkGenerationEstimator(alpha=config.ema_alpha)
    return RequestSLOState(
        estimator=estimator,
        detector=detector,
        ema_alpha=config.ema_alpha,
        chunk_gen_estimator=cge,
        pending_warmup_chunks=config.pending_warmup_chunks,
        pending_pressure_lambda=config.pending_pressure_lambda,
        pending_hysteresis_gap=config.pending_hysteresis_gap,
    )
