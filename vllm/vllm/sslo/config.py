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
    # Admission criterion. False (default): admit while the ABSOLUTE
    # E_viol(k) < 1. True: admit while the MARGINAL E_viol(k) - E_viol(0) < 1,
    # i.e. budget only the extra expected violations this step's admissions
    # cause, treating the risk already carried by in-flight requests as sunk.
    # The absolute rule self-locks — once a few high-risk in-flight requests
    # sum past 1, k* pins to 0 even with idle KV / compute. Under low load
    # (E_viol(0) ~ 0) the two rules coincide.
    # REJECTED experiment option (ablation A3, WORKLOG 2026-08-07): kept at
    # False. Dropping the absolute test removes the only brake on the system's
    # total risk — per-step e_viol accumulated to 16-41 (vs ~1) and violation
    # rates worsened monotonically in request rate, degrading even plain
    # kv_offload (e.g. cap128 offload 1.3-2.0% → 3.7-7.0%).
    admission_delta_criterion: bool = False
    # When True, ProgressServe may shrink the decode batch (to a smaller
    # CUDA-graph-captured size) when E_viol(B) >= 1, trading throughput for a
    # lower per-iteration latency so the few urgent requests meet their
    # deadlines. See vllm.sslo.progress_serve.pick_adaptive_batch.
    adaptive_batching: bool = False
    # KV offload tier. When True, ProgressServe may offload the KV blocks of
    # slack-deep deferred requests to CPU (freeing GPU headroom) and prefetch
    # them back before their deadline approaches. Only valid under
    # method == "progress_serve".
    kv_offload: bool = False
    # Onload lead: restore a CPU-resident request this many decode iterations
    # before its deadline horizon so the CPU→GPU transfer is hidden. Enters
    # the risk math as N_defer_cpu = floor(max(H_q - 1 - lead, 0) * s).
    kv_onload_lead_iters: int = 2
    # Risk-penalty epsilon (in tail-posterior probability units) separating
    # offload-eligible from onload-eligible requests. A CPU stay is cheap
    # (M_cpu <= eps) → offload candidate; once the lead penalty surfaces in the
    # risk (M_cpu > eps) → onload candidate.
    kv_offload_risk_eps: float = 1e-3
    # Adopted default behaviour of the KV offload tier: count CPU-parked
    # (offloaded) requests in the service-share denominator |A+|. Offloading
    # frees memory, not compute: a parked request returns and reclaims its
    # share, so excluding it inflates s, which underestimates M_cpu (→
    # over-offload) and distorts the adaptive batch choice. Measured (ablation
    # A1, WORKLOG 2026-08-07): fixes kv_offload's low-cap over-offload — cap32
    # offloads 8→1 with +9-26% throughput and lower violation rate; higher caps
    # unchanged.
    # Affects the s term only — parked reqs still hold no decode slot and still
    # enter E_viol as R_defer_cpu. Set False only to reproduce the pre-A1
    # semantics.
    kv_offload_share_includes_parked: bool = True
    # Anti-thrash guard: once onload completes, a request may not be
    # re-offloaded within this many scheduler steps. 0 disables. The step-count
    # check itself lives in the scheduler (stage 2); this is the knob only.
    kv_offload_min_residency_steps: int = 0
    # Deadline-aware Token Budget TB*. Both axes buy step time with the same
    # currency the admission rule spends — the expected violation count E_viol
    # of the in-flight set, evaluated at the step time the choice implies:
    #   [D axis] defer slack-deep defer-safe requests while each defer strictly
    #            lowers E_viol at the resulting Δ_dec(D')
    #            (progress_serve.select_decode_defer).
    #   [P axis] TB*_pre caps the TOTAL prefill tokens of a step —
    #            chunked-prefill carry-over in the running loop plus new admits
    #            in the waiting loop, sharing one budget — at the largest P
    #            whose E_viol(Δ_dec(D') + κ_p·P) stays within
    #            token_budget_risk_eps of E_viol at P=0
    #            (progress_serve.token_budget_prefill_risk). Measured
    #            (cap128/rate2 baseline): prefill-carrying steps are only 6% of
    #            steps but 18.4% of the wall clock (Δ p90 463 ms vs 78 ms
    #            decode-only) — deadline misses come from the prefill spike,
    #            not decode concurrency.
    # Decode tokens are never charged to TB*_pre (the D axis works in whole
    # requests, never in tokens). Requires method="progress_serve"; a single
    # flag gates both axes because they share one ledger.
    token_budget_control: bool = False
    # Lower clamp on TB*_pre (tokens). Must be > 0 so prefill always makes
    # progress under chunked prefill (no prefill starvation). Keep it well
    # below the engine's max_num_batched_tokens: token_budget_prefill_risk()
    # clamps the floor to that base budget, so a floor at or above it turns the
    # control into a silent no-op. SsloConfig cannot cross-validate this — it
    # does not see SchedulerConfig.
    token_budget_prefill_floor: int = 512
    # P-axis dosing budget ε_p, in expected-violation units: TB*_pre is the
    # largest prefill allowance whose extra expected violations
    # E_viol(Δ(P)) - E_viol(Δ(0)) stay within ε_p. Must be > 0 — at 0 any
    # in-flight request whose risk moves at all would pin TB*_pre to the floor.
    token_budget_risk_eps: float = 0.01
    # DEPRECATED (2026-08-11): worst-case safety factor on t_min, the input of
    # the rejected progress_serve.token_budget_prefill(). Kept (with that
    # function) only to replay the old rule against the risk dosing above; no
    # live scheduler path reads it. Rejected because one near-deadline survivor
    # pinned TB*_pre to the floor on 81% of recovery steps (WORKLOG
    # 2026-08-11).
    token_budget_gamma: float = 0.5
    # D-axis defer-safety epsilon (in tail-posterior probability units, same
    # grammar as kv_offload_risk_eps): only a MEASURED request whose deferred
    # risk R_defer <= eps may be dropped from the decode set to shrink Δ_dec.
    # Forced / onloading / at-risk requests are never deferred — they are the
    # natural floor D_floor.
    token_budget_decode_risk_eps: float = 1e-3

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
        if self.kv_offload and self.method != "progress_serve":
            raise ValueError(
                "kv_offload requires method='progress_serve', "
                f"got method={self.method!r}")
        if self.kv_onload_lead_iters < 0:
            raise ValueError(
                "kv_onload_lead_iters must be >= 0, "
                f"got {self.kv_onload_lead_iters}")
        if self.kv_offload_risk_eps < 0:
            raise ValueError(
                "kv_offload_risk_eps must be >= 0, "
                f"got {self.kv_offload_risk_eps}")
        if self.kv_offload_min_residency_steps < 0:
            raise ValueError(
                "kv_offload_min_residency_steps must be >= 0 (0 disables), "
                f"got {self.kv_offload_min_residency_steps}")
        if self.token_budget_control and self.method != "progress_serve":
            raise ValueError(
                "token_budget_control requires method='progress_serve', "
                f"got method={self.method!r}")
        if self.token_budget_prefill_floor < 1:
            raise ValueError(
                "token_budget_prefill_floor must be >= 1 (prefill "
                f"starvation), got {self.token_budget_prefill_floor}")
        if self.token_budget_risk_eps <= 0:
            raise ValueError(
                "token_budget_risk_eps must be > 0, "
                f"got {self.token_budget_risk_eps}")
        if not (0 < self.token_budget_gamma <= 1):
            raise ValueError(
                "token_budget_gamma must be in (0, 1], "
                f"got {self.token_budget_gamma}")
        if self.token_budget_decode_risk_eps < 0:
            raise ValueError(
                "token_budget_decode_risk_eps must be >= 0, "
                f"got {self.token_budget_decode_risk_eps}")
