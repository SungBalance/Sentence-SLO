# SPDX-License-Identifier: Apache-2.0
"""Configuration for SSLO ProgressServe scheduling."""
from __future__ import annotations

from dataclasses import dataclass

from vllm.sslo.slo_state import (
    _DEFAULT_CHUNK_LEN_STRATEGY,
    _VALID_CHUNK_LEN_STRATEGIES,
    _VALID_CHUNK_UNITS,
)


@dataclass
class SsloConfig:
    # Scheduling mode:
    #   "baseline"        — schedule_sslo() runs (chunk_records, wall-step EMA,
    #                       and scheduler_stats are emitted for fair
    #                       comparison) but SSLO placement is SKIPPED.
    #                       Equivalent to vanilla vLLM admission, with metrics.
    #   "progress_serve"  — posterior tail-risk admission + running selection.
    method: str = "baseline"
    # Threshold (in chunk samples) below which the per-request chunk-length
    # predictor is treated as warming up. ProgressServe's posterior uses the
    # shared global distribution, but this still gates the per-req diagnostic
    # predictor. No longer gates phase transitions.
    num_warmup_chunks: int = 16
    # EMA smoothing for the per-batch wall-step time (Δ source).
    tpot_ema_alpha: float = 0.1
    seconds_per_word: float = 0.28
    consume_mode: str = "read"
    tts_profile_path: str | None = None
    tts_model: str | None = None
    chunk_unit: str = "sentence"
    # Chunk-length predictor strategy: "ema" / "p90" / "p99" /
    # "past-future" (placeholder). ProgressServe needs the empirical history,
    # so a history-keeping strategy ("p90"/"p99") is required.
    chunk_len_strategy: str = _DEFAULT_CHUNK_LEN_STRATEGY
    # Defer flushing a found chunk boundary until the accumulated token count
    # since the last flush reaches this threshold. Short chunks (e.g. "Yes.")
    # are merged into the next chunk so consume_time / chunk history are not
    # skewed by sub-token sentences.
    min_chunk_tokens: int = 16
    # Decision log control. Emits per-(step, admitted_request) rows to
    # decisions.jsonl for diagnostics.
    #   "off" / "step" / "tier_changes" / "admit_only".
    decision_log_mode: str = "tier_changes"
    # How often (in scheduler steps) the tier_changes mode emits a
    # full-admitted heartbeat to keep non-transitioning requests sampled.
    decision_heartbeat_steps: int = 200
    # Diagnostic chunk-length predictor escalation knobs (feed the per-req
    # point-estimate `expected_remaining_len`, used for ChunkRecord logging).
    mlp_predictor_escalate_threshold: float = 0.9
    mlp_predictor_overshoot_safety_factor: float = 3.5
    # KV-aware admission cap. The FCFS waiting prefix admitted per step is
    # capped at `free_kv_blocks // kv_blocks_per_new_admit` so we don't admit
    # more reqs than the KV pool can absorb. Each admit needs prompt blocks
    # (~7-10 for ShareGPT) + 1 first-decode block. Set 0 to disable (no cap).
    kv_blocks_per_new_admit: int = 8
    # Max samples retained by ChunkLengthPredictor's history. The shared
    # global predictor accumulates many samples — keep enough to stabilise
    # the empirical tail.
    chunk_len_predictor_history_max: int = 4096
    # Until the shared global chunk-length predictor has accumulated
    # `global_warmup_predictor_samples` samples, ProgressServe's posterior
    # falls back to an analytic cold-start tail bounded by
    # `cold_start_max_remaining_tokens`.
    global_warmup_predictor_samples: int = 128
    cold_start_max_remaining_tokens: int = 2048
    # ProgressServe: min global-history samples above c_q required to trust
    # the empirical tail posterior (else the analytic cold-start fallback).
    progress_serve_min_denom: int = 4
    # When True, ProgressServe may shrink the decode batch (to a smaller
    # CUDA-graph-captured size) when E_viol(B) >= 1, trading throughput for a
    # lower per-iteration latency so the few urgent requests meet their
    # deadlines. See vllm.sslo.progress_serve.pick_adaptive_batch.
    adaptive_batching: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("baseline", "progress_serve"):
            raise ValueError(
                f"method must be 'baseline' or 'progress_serve', "
                f"got {self.method!r}")
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}")
        if self.chunk_len_strategy not in _VALID_CHUNK_LEN_STRATEGIES:
            raise ValueError(
                f"chunk_len_strategy must be one of "
                f"{sorted(_VALID_CHUNK_LEN_STRATEGIES)}, "
                f"got {self.chunk_len_strategy!r}")
        if self.consume_mode not in {"read", "tts"}:
            raise ValueError(
                "consume_mode must be one of ['read', 'tts'], "
                f"got {self.consume_mode!r}")
        if self.consume_mode == "tts" and self.tts_profile_path is None:
            raise ValueError(
                "tts_profile_path must be set when consume_mode='tts'")
        if self.consume_mode == "tts" and self.tts_model is None:
            raise ValueError(
                "tts_model must be set when consume_mode='tts'")
        if (self.consume_mode == "read"
                and (self.tts_profile_path is not None
                     or self.tts_model is not None)):
            raise ValueError(
                "tts_profile_path and tts_model must be None when "
                "consume_mode='read'")
        if self.num_warmup_chunks < 1:
            raise ValueError(
                f"num_warmup_chunks must be >= 1, "
                f"got {self.num_warmup_chunks}")
        if not (0 < self.tpot_ema_alpha <= 1):
            raise ValueError(
                f"tpot_ema_alpha must be in (0, 1], got {self.tpot_ema_alpha}")
        if self.seconds_per_word < 0:
            raise ValueError(
                f"seconds_per_word must be >= 0, got {self.seconds_per_word}")
        if self.min_chunk_tokens < 0:
            raise ValueError(
                f"min_chunk_tokens must be >= 0, got {self.min_chunk_tokens}")
        valid_modes = ("off", "step", "tier_changes", "admit_only")
        if self.decision_log_mode not in valid_modes:
            raise ValueError(
                f"decision_log_mode must be one of {valid_modes}, "
                f"got {self.decision_log_mode!r}")
        if self.decision_heartbeat_steps < 1:
            raise ValueError(
                f"decision_heartbeat_steps must be >= 1, "
                f"got {self.decision_heartbeat_steps}")
        if not (0 < self.mlp_predictor_escalate_threshold <= 1.0):
            raise ValueError(
                "mlp_predictor_escalate_threshold must be in (0, 1], "
                f"got {self.mlp_predictor_escalate_threshold}")
        if self.mlp_predictor_overshoot_safety_factor < 0:
            raise ValueError(
                "mlp_predictor_overshoot_safety_factor must be >= 0, "
                f"got {self.mlp_predictor_overshoot_safety_factor}")
        if self.kv_blocks_per_new_admit < 0:
            raise ValueError(
                "kv_blocks_per_new_admit must be >= 0 (0 disables), "
                f"got {self.kv_blocks_per_new_admit}")
        if self.chunk_len_predictor_history_max < 1:
            raise ValueError(
                "chunk_len_predictor_history_max must be >= 1, "
                f"got {self.chunk_len_predictor_history_max}")
        if self.progress_serve_min_denom < 1:
            raise ValueError(
                "progress_serve_min_denom must be >= 1, "
                f"got {self.progress_serve_min_denom}")
