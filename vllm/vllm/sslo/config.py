# SPDX-License-Identifier: Apache-2.0
"""SsloConfig: configuration dataclass for SSLO (Sentence-level SLO)."""
from __future__ import annotations

from dataclasses import dataclass

_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})
_VALID_ESTIMATOR_TYPES = frozenset({"word_rate"})


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
    pending_slack_eps_num_tokens: int = 3

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


def build_slo_state(config: SsloConfig) -> "RequestSLOState":
    """Create a RequestSLOState from SsloConfig (single factory for both sides)."""
    from vllm.sslo.slo_state import (
        ParagraphChunkDetector,
        RequestSLOState,
        WordRateEstimator,
    )
    estimator = WordRateEstimator(seconds_per_word=config.seconds_per_word)
    detector = ParagraphChunkDetector() if config.chunk_unit == "paragraph" else None
    return RequestSLOState(
        estimator=estimator,
        detector=detector,
        ema_alpha=config.ema_alpha,
        pending_slack_eps_num_tokens=config.pending_slack_eps_num_tokens,
    )
