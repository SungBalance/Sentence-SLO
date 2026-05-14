# SPDX-License-Identifier: Apache-2.0
"""Configuration for SSLO score-based scheduling."""
from __future__ import annotations

from dataclasses import dataclass

from vllm.sslo.slo_state import (
    _DEFAULT_CHUNK_LEN_STRATEGY,
    _VALID_CHUNK_LEN_STRATEGIES,
    _VALID_CHUNK_UNITS,
)


@dataclass
class SsloConfig:
    # Top-level scheduling mode:
    #   "baseline" — schedule_sslo() runs (so chunk_records, tpot EMA, and
    #                scheduler_stats are emitted for fair comparison) but
    #                the SSLO placement decisions (critical/non-critical,
    #                pending pool, waiting throttle) are SKIPPED. Equivalent
    #                to vanilla vLLM admission, with metrics.
    #   "sslo"     — full SSLO scheduling, algorithm chosen by `policy`.
    method: str = "baseline"
    # Placement algorithm (only meaningful when method=="sslo"):
    #   None        — placeholder; raises if method=="sslo".
    #   "threshold" — hysteresis (in/out thresholds).
    #   "pressure"  — pressure-budget admission.
    policy: str | None = "threshold"
    adaptive_batching: bool = False
    num_warmup_chunks: int = 4
    tpot_ema_alpha: float = 0.1
    critical_threshold: float = 1.0
    # Hysteresis thresholds for non-critical placement under
    # policy="threshold":
    #   pressure ≤ in  → park to pending
    #   pressure ≥ out → resume to running
    #   in (0.3, 0.7) → keep current state (hysteresis band)
    pending_in_threshold: float = 0.3
    pending_out_threshold: float = 0.7
    adaptive_batching_min_throughput_ratio: float = 0.9
    # Hard floor for adaptive batching: the smallest profiled bucket whose
    # throughput is still ≥ this fraction of `max_num_seqs / base_tpot`.
    # Below it the cap shouldn't shrink — losing more than (1 - ratio) of
    # the base throughput isn't worth the latency relief.
    adaptive_batching_low_cap_throughput_ratio: float = 0.25
    seconds_per_word: float = 0.28
    chunk_unit: str = "sentence"
    # Chunk-length predictor strategy: "ema" / "p90" / "p99" /
    # "past-future" (placeholder). Picks the conservatism point for
    # the predicted-remaining-tokens used by pressure().
    chunk_len_strategy: str = _DEFAULT_CHUNK_LEN_STRATEGY
    # Defer flushing a found chunk boundary until the accumulated token count
    # since the last flush reaches this threshold. Short chunks (e.g. "Yes.")
    # are merged into the next chunk so the consume_time and chunk EMA are
    # not skewed by sub-token sentences.
    min_chunk_tokens: int = 16
    # Decision log control. Emits per-(step, admitted_request) rows to
    # decisions.jsonl for M5 (refill-slack diagnostic) and M6 (cumulative
    # ablation) analysis.
    #   "off"           — never emit (default for baseline runs).
    #   "step"          — every admitted request, every scheduler step.
    #                     ~1M rows/5min run; use for M5 deep dives.
    #   "tier_changes"  — emit on tier transition or admit/preempt event,
    #                     plus a full snapshot every `decision_heartbeat_steps`.
    #   "admit_only"    — emit only on the step where the request was
    #                     admitted from waiting or preempted from running.
    decision_log_mode: str = "tier_changes"
    # How often (in scheduler steps) the tier_changes mode emits a
    # full-admitted heartbeat to keep non-transitioning requests sampled.
    decision_heartbeat_steps: int = 200
    # multi_level_pressure policy knobs. mlp_pressure_epsilon guards the
    # denominator in serve / defer pressure when ttd or defer_buffer
    # collapse toward zero; mlp_critical_serve_threshold triggers the
    # critical branch when any MEASURED request's serve_pressure crosses
    # it at base_n; mlp_defer_constraint forces requests with
    # defer_pressure ≥ constraint into running so deferring one step
    # would miss the deadline.
    mlp_pressure_epsilon: float = 1e-9
    mlp_critical_serve_threshold: float = 1.0
    mlp_defer_constraint: float = 1.0

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}")
        if self.chunk_len_strategy not in _VALID_CHUNK_LEN_STRATEGIES:
            raise ValueError(
                f"chunk_len_strategy must be one of "
                f"{sorted(_VALID_CHUNK_LEN_STRATEGIES)}, "
                f"got {self.chunk_len_strategy!r}")
        if self.num_warmup_chunks < 1:
            raise ValueError(
                f"num_warmup_chunks must be >= 1, "
                f"got {self.num_warmup_chunks}")
        for name in ("tpot_ema_alpha",
                     "adaptive_batching_min_throughput_ratio",
                     "adaptive_batching_low_cap_throughput_ratio"):
            value = getattr(self, name)
            if not (0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        for name in (
                "critical_threshold",
                "pending_in_threshold",
                "pending_out_threshold",
                "seconds_per_word",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.min_chunk_tokens < 0:
            raise ValueError(
                f"min_chunk_tokens must be >= 0, got {self.min_chunk_tokens}")
        if self.pending_in_threshold > self.pending_out_threshold:
            raise ValueError(
                "pending_in_threshold must be <= pending_out_threshold")
        if self.method not in ("baseline", "sslo"):
            raise ValueError(
                f"method must be 'baseline' or 'sslo', "
                f"got {self.method!r}")
        valid_policies = (None, "threshold", "pressure", "multi_level_pressure")
        if self.policy not in valid_policies:
            raise ValueError(
                f"policy must be one of {valid_policies}, "
                f"got {self.policy!r}")
        if self.method == "sslo" and self.policy is None:
            raise ValueError(
                "policy must be set (not None) when method='sslo'")
        valid_modes = ("off", "step", "tier_changes", "admit_only")
        if self.decision_log_mode not in valid_modes:
            raise ValueError(
                f"decision_log_mode must be one of {valid_modes}, "
                f"got {self.decision_log_mode!r}")
        if self.decision_heartbeat_steps < 1:
            raise ValueError(
                f"decision_heartbeat_steps must be >= 1, "
                f"got {self.decision_heartbeat_steps}")
        for name in (
                "mlp_pressure_epsilon",
                "mlp_critical_serve_threshold",
                "mlp_defer_constraint",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if (
            self.method == "sslo"
            and self.policy == "multi_level_pressure"
            and not self.adaptive_batching
        ):
            raise ValueError(
                "policy='multi_level_pressure' requires "
                "adaptive_batching=True (the policy's critical branch "
                "shrinks the cap to clear deadline misses)")
