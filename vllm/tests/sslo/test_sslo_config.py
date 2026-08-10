# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO config (ProgressServe)."""

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import ChunkLengthPredictor, RequestSLOState


def test_defaults():
    cfg = SsloConfig()
    assert cfg.method == "baseline"
    assert cfg.num_warmup_chunks == 16
    assert cfg.tpot_ema_alpha == 0.1
    assert cfg.seconds_per_word == 0.28
    assert cfg.chunk_unit == "sentence"
    assert cfg.min_chunk_tokens == 16
    assert cfg.kv_blocks_per_new_admit == 8
    assert cfg.progress_serve_min_denom == 4
    # Removed policy knobs must be gone.
    assert not hasattr(cfg, "policy")
    assert not hasattr(cfg, "critical_threshold")
    assert not hasattr(cfg, "pending_in_threshold")
    assert not hasattr(cfg, "mlp_defer_constraint")
    assert not hasattr(cfg, "allow_admit_critical")


def test_method_progress_serve_valid():
    cfg = SsloConfig(method="progress_serve")
    assert cfg.method == "progress_serve"


def test_method_validation_rejects_old_modes():
    for bad in ("sslo", "sslo_mlp", "threshold"):
        with pytest.raises(ValueError, match="method"):
            SsloConfig(method=bad)


def test_removed_knobs_rejected_as_kwargs():
    for bad in ("policy", "critical_threshold",
                "pending_in_threshold", "mlp_defer_constraint",
                "mlp_kv_blocks_per_new_admit"):
        with pytest.raises(TypeError):
            SsloConfig(**{bad: 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_warmup_chunks", 0),
        ("tpot_ema_alpha", 0.0),
        ("tpot_ema_alpha", 1.1),
        ("seconds_per_word", -0.1),
        ("min_chunk_tokens", -1),
        ("kv_blocks_per_new_admit", -1),
        ("progress_serve_min_denom", 0),
    ],
)
def test_validation_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        SsloConfig(**{field: value})


def test_invalid_chunk_unit_raises():
    with pytest.raises(ValueError, match="chunk_unit"):
        SsloConfig(chunk_unit="token")


def test_decision_log_mode_default_and_validation():
    cfg = SsloConfig()
    assert cfg.decision_log_mode == "tier_changes"
    assert cfg.decision_heartbeat_steps == 200
    with pytest.raises(ValueError, match="decision_log_mode"):
        SsloConfig(decision_log_mode="invalid")
    with pytest.raises(ValueError, match="decision_heartbeat_steps"):
        SsloConfig(decision_heartbeat_steps=0)


def test_tts_mode_requires_profile_and_model():
    with pytest.raises(ValueError, match="tts_profile_path"):
        SsloConfig(consume_mode="tts")


def test_from_config_freezes_constants():
    cfg = SsloConfig(
        method="progress_serve",
        seconds_per_word=0.5,
        num_warmup_chunks=7,
        chunk_unit="paragraph",
        min_chunk_tokens=24,
        progress_serve_min_denom=6,
    )
    state = RequestSLOState.from_config(cfg)
    assert isinstance(state, RequestSLOState)
    assert state.num_warmup_chunks == 7
    assert state.progress_serve_min_denom == 6
    assert state.chunk_separator.chunk_unit == "paragraph"
    assert state.chunk_separator.min_chunk_tokens == 24
    assert state.consume_estimator.seconds_per_word == 0.5


def test_from_config_wires_global_predictor_for_tail():
    cfg = SsloConfig(method="progress_serve",
                     global_warmup_predictor_samples=3,
                     progress_serve_min_denom=2)
    glob = ChunkLengthPredictor(strategy="p90")
    state = RequestSLOState.from_config(cfg, global_chunk_len_predictor=glob)
    for v in (10, 20, 30, 40, 50):
        glob.update(v)
    # c_q=5: P(L>25 | L>5) = |{>25}|/|{>5}| = 3/5
    state.current_chunk_generated_len = 5
    assert abs(state.length_tail_prob(25) - 0.6) < 1e-9
    # cold-start fallback when c_q exceeds all samples (denom 0).
    state.current_chunk_generated_len = 100
    assert state.length_tail_prob(120) == 1.0  # < c_q + cold_start_max


def test_adaptive_batching_default_and_settable():
    assert SsloConfig().adaptive_batching is False
    assert SsloConfig(method="progress_serve",
                      adaptive_batching=True).adaptive_batching is True


def test_kv_offload_defaults():
    cfg = SsloConfig()
    assert cfg.kv_offload is False
    assert cfg.kv_onload_lead_iters == 2
    assert cfg.kv_offload_risk_eps == 1e-3
    assert cfg.kv_offload_min_residency_steps == 0


def test_kv_offload_requires_progress_serve():
    with pytest.raises(ValueError, match="kv_offload"):
        SsloConfig(kv_offload=True)  # method defaults to baseline
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    assert cfg.kv_offload is True


def test_prefill_budget_defaults():
    cfg = SsloConfig()
    assert cfg.prefill_budget_control is False
    assert cfg.prefill_budget_floor == 512
    assert cfg.prefill_budget_gamma == 0.5


def test_prefill_budget_control_requires_progress_serve():
    with pytest.raises(ValueError, match="prefill_budget_control"):
        SsloConfig(prefill_budget_control=True)  # method defaults to baseline
    cfg = SsloConfig(method="progress_serve", prefill_budget_control=True)
    assert cfg.prefill_budget_control is True


@pytest.mark.parametrize("value", [0, -1])
def test_prefill_budget_floor_must_be_positive(value):
    with pytest.raises(ValueError, match="prefill_budget_floor"):
        SsloConfig(prefill_budget_floor=value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_prefill_budget_gamma_range(value):
    with pytest.raises(ValueError, match="prefill_budget_gamma"):
        SsloConfig(prefill_budget_gamma=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kv_onload_lead_iters", -1),
        ("kv_offload_risk_eps", -0.1),
        ("kv_offload_min_residency_steps", -1),
    ],
)
def test_kv_offload_validation_rejects_negative(field, value):
    with pytest.raises(ValueError, match=field):
        SsloConfig(**{field: value})
