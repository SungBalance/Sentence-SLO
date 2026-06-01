# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure ProgressServe scheduling primitives."""
from __future__ import annotations

from vllm.sslo.progress_serve import (
    E_VIOL_FEASIBLE,
    ProgressView,
    build_plan,
    horizon,
    n_run_defer,
    pick_adaptive_batch,
    request_risk,
    schedule_step,
    service_share,
)

PHASE_PREFILL = 0
PHASE_MEASURED = 2


def _measurable(rid: str, c_q: int, t_q: float, tail) -> ProgressView:
    return ProgressView(rid, c_q, t_q, PHASE_MEASURED, False, tail)


def _forced(rid: str) -> ProgressView:
    return ProgressView(rid, 0, None, PHASE_PREFILL, False, lambda x: 1.0)


def _new_admit(rid: str) -> ProgressView:
    return ProgressView(rid, 0, None, PHASE_PREFILL, True, lambda x: 1.0)


# ---- horizon ----------------------------------------------------------

def test_horizon_floor():
    assert horizon(10.0, 0.1) == 100
    assert horizon(0.95, 0.1) == 9


def test_horizon_clamps_negative_and_subiteration():
    assert horizon(-5.0, 0.1) == 0  # past deadline
    assert horizon(0.05, 0.1) == 0  # less than one iteration fits


def test_horizon_zero_delta_guard():
    assert horizon(10.0, 0.0) == 0


# ---- service_share ----------------------------------------------------

def test_service_share():
    assert service_share(4, 8) == 0.5
    assert service_share(4, 2) == 1.0  # capped at 1
    assert service_share(4, 0) == 1.0  # empty A+ guard


# ---- n_run_defer ------------------------------------------------------

def test_n_run_defer_horizon_zero():
    assert n_run_defer(0, 1.0) == (0, 0)


def test_n_run_defer_horizon_one():
    # I_q=1, N_future=floor(0*s)=0
    assert n_run_defer(1, 1.0) == (1, 0)


def test_n_run_defer_general():
    # H=5, s=0.5 → N_future=floor(4*0.5)=2 → (3, 2)
    assert n_run_defer(5, 0.5) == (3, 2)


# ---- request_risk -----------------------------------------------------

def test_request_risk_marginal_benefit_nonnegative():
    # Decreasing tail → R_run <= R_defer → M >= 0.
    tail = lambda x: max(0.0, 1.0 - x / 1000.0)
    v = _measurable("q", c_q=10, t_q=10.0, tail=tail)
    r_run, r_defer, m = request_risk(v, delta=0.1, s=1.0)
    assert r_run <= r_defer
    assert m >= 0.0


def test_request_risk_certain_defer_miss_forces_max_benefit():
    # R_defer == 1 (e.g. tail still saturated at the deferred horizon, or
    # deadline imminent) → M is forced to 1.0 even when R_run == 1, so the
    # request is prioritized instead of abandoned with M = 0.
    v = _measurable("q", c_q=0, t_q=100.0, tail=lambda x: 1.0)
    r_run, r_defer, m = request_risk(v, delta=1.0, s=1.0)
    assert r_run == 1.0 and r_defer == 1.0
    assert m == 1.0

    # When R_defer < 1 the normal marginal-difference path applies.
    tail = lambda x: max(0.0, 1.0 - x / 50.0)
    v2 = _measurable("q2", c_q=0, t_q=11.0, tail=tail)
    r_run2, r_defer2, m2 = request_risk(v2, delta=1.0, s=1.0)
    assert r_defer2 < 1.0
    assert abs(m2 - (r_defer2 - r_run2)) < 1e-9


# ---- build_plan -------------------------------------------------------

# Continuous decreasing tails; t2 is steeper ⇒ larger marginal benefit M.
_T1 = lambda x: max(0.0, 1.0 - x / 100.0)
_T2 = lambda x: max(0.0, 1.0 - x / 30.0)


def _expected_eviol_one_deferred(t_q, delta, n_aplus, b):
    """Oracle for a 2-measurable case where the steeper-tail req (v2) is
    scheduled and the gentler (v1) deferred: R_run(v2) + R_defer(v1)."""
    s = service_share(b, n_aplus)
    nr, nd = n_run_defer(horizon(t_q, delta), s)
    return _T2(nr) + _T1(nd)


def test_build_plan_partitions_by_marginal_benefit():
    v1 = _measurable("v1", 0, 11.0, _T1)
    v2 = _measurable("v2", 0, 11.0, _T2)
    plan = build_plan([v1, v2], [], b=1, delta=1.0)
    # v2 (steeper tail) has higher M → scheduled; v1 deferred.
    assert plan.scheduled_A == ["v2"]
    assert plan.deferred_A == ["v1"]
    assert abs(plan.e_viol
               - _expected_eviol_one_deferred(11.0, 1.0, 2, 1)) < 1e-9


def test_build_plan_forced_always_scheduled_and_excluded_from_eviol():
    v1 = _measurable("v1", 0, 11.0, _T1)
    v2 = _measurable("v2", 0, 11.0, _T2)
    f = _forced("f")
    # B=2: decode_cap = 2 - 1(forced) - 0 = 1 → schedule f + v2, defer v1.
    plan = build_plan([f, v1, v2], [], b=2, delta=1.0)
    assert "f" in plan.scheduled_A
    assert "v2" in plan.scheduled_A
    assert plan.deferred_A == ["v1"]
    # forced contributes nothing to E_viol; |A+| = 3.
    assert abs(plan.e_viol
               - _expected_eviol_one_deferred(11.0, 1.0, 3, 2)) < 1e-9
    assert "f" not in plan.risks


def test_build_plan_new_admits_consume_slots_but_excluded_from_eviol():
    v1 = _measurable("v1", 0, 11.0, _T1)
    v2 = _measurable("v2", 0, 11.0, _T2)
    # B=2 with one new admit → decode_cap = 2 - 0 - 1 = 1; |A+| = 3.
    plan = build_plan([v1, v2], [_new_admit("n1")], b=2, delta=1.0)
    assert plan.scheduled_A == ["v2"]
    assert plan.deferred_A == ["v1"]
    # new admit never appears in scheduled_A / deferred_A / risks.
    assert "n1" not in plan.scheduled_A
    assert "n1" not in plan.deferred_A
    assert "n1" not in plan.risks
    assert abs(plan.e_viol
               - _expected_eviol_one_deferred(11.0, 1.0, 3, 2)) < 1e-9


# ---- schedule_step ----------------------------------------------------

def test_schedule_step_empty_A_bounded_by_kv():
    # No in-flight; E_viol(k)=0<1 for all k, so k* bounded by kv_feasible.
    waiting = [_new_admit(f"w{i}") for i in range(10)]
    res = schedule_step([], waiting, b=8, delta=1.0,
                        kv_feasible=lambda k: k <= 3)
    assert res.k_star == 3
    assert res.e_viol == 0.0


def test_schedule_step_kv_caps_below_risk():
    v = _measurable("v", 0, 100.0, lambda x: 0.0)  # zero risk
    waiting = [_new_admit(f"w{i}") for i in range(10)]
    res = schedule_step([v], waiting, b=8, delta=1.0,
                        kv_feasible=lambda k: k <= 2)
    assert res.k_star == 2


def test_schedule_step_matches_build_plan_boundary():
    # Decreasing tail so deferring a measurable raises E_viol. As k grows,
    # decode_cap shrinks → more deferral → E_viol increases (monotone).
    tail = lambda x: max(0.0, 1.0 - x / 6.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    res = schedule_step(a, waiting, b=3, delta=1.0, kv_feasible=lambda k: True)
    # Oracle: largest k with build_plan(...,waiting[:k]).e_viol < 1.
    expected_k = 0
    for k in range(len(waiting) + 1):
        if build_plan(a, waiting[:k], 3, 1.0).e_viol < 1.0:
            expected_k = k
        else:
            break
    assert res.k_star == expected_k


def test_schedule_step_defer_only_when_infeasible_at_zero():
    # Three measurable, B=1, high deferral risk → E_viol(k=0) already >= 1.
    tail = lambda x: 1.0  # everything still going → max risk
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    res = schedule_step(a, [_new_admit("w0")], b=1, delta=1.0,
                        kv_feasible=lambda k: True)
    assert res.k_star == 0
    # one scheduled (R_run=1), two deferred (R_defer=1) → e_viol=3
    assert res.e_viol >= 1.0
    assert len(res.scheduled_A) == 1
    assert len(res.deferred_A) == 2


# ---- pick_adaptive_batch -----------------------------------------------

def _expected_pick(views, base_b, base_delta, candidates, delta_of):
    """Independent re-derivation of the search for oracle comparison."""
    base_e = build_plan(views, [], base_b, base_delta).e_viol
    if base_e < E_VIOL_FEASIBLE:
        return base_b
    best_b, prev = base_b, base_e
    for b in candidates:
        if b >= base_b:
            continue
        d = delta_of(b)
        if d is None or d <= 0:
            continue
        e = build_plan(views, [], b, d).e_viol
        if e > prev:
            break
        best_b, prev = b, e
    return best_b


def test_pick_adaptive_batch_no_trigger_keeps_base():
    # Low risk → E_viol(base) < 1 → no search, keep base regardless of cands.
    views = [_measurable(f"v{i}", 0, 100.0, lambda x: 0.0) for i in range(3)]
    b, plan = pick_adaptive_batch(
        views, 128, 0.1, [96, 64, 32], delta_of=lambda b: 0.05)
    assert b == 128
    assert plan.e_viol < E_VIOL_FEASIBLE


def test_pick_adaptive_batch_stops_at_first_increase():
    # Slow base (large Δ) → H small → high risk → trigger. Shrinking lowers Δ
    # (more iters before deadline) until b=16's Δ bump raises E_viol again.
    tail = lambda x: max(0.0, 1.0 - x / 40.0)
    views = [_measurable(f"v{i}", 0, 4.0, tail) for i in range(3)]
    deltas = {128: 2.0, 96: 1.5, 64: 1.0, 32: 0.5, 16: 3.0}
    cands = [96, 64, 32, 16]
    delta_of = lambda b: deltas.get(b)
    exp = _expected_pick(views, 128, deltas[128], cands, delta_of)
    got, _ = pick_adaptive_batch(views, 128, deltas[128], cands, delta_of)
    assert got == exp
    assert got == 32  # hand-checked: 16's Δ bump stops the descent


def test_pick_adaptive_batch_skips_missing_ema():
    tail = lambda x: max(0.0, 1.0 - x / 40.0)
    views = [_measurable(f"v{i}", 0, 4.0, tail) for i in range(3)]
    deltas = {128: 2.0, 96: 1.5, 64: None, 32: 0.5, 16: 3.0}
    cands = [96, 64, 32, 16]
    delta_of = lambda b: deltas.get(b)
    got, _ = pick_adaptive_batch(views, 128, deltas[128], cands, delta_of)
    # 64 (no EMA) is skipped; descent still reaches 32 and stops at 16.
    assert got == 32
