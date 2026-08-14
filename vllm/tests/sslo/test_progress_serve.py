# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure ProgressServe scheduling primitives."""
from __future__ import annotations

from vllm.sslo.progress_serve import (
    E_VIOL_FEASIBLE,
    LOC_CPU,
    LOC_ONLOADING,
    ProgressView,
    build_plan,
    horizon,
    n_defer_cpu,
    n_run_defer,
    pick_adaptive_batch,
    request_risk,
    request_risk_cpu,
    schedule_step,
    select_onload,
    select_offload,
    service_share,
    token_budget_prefill,
    token_budget_prefill_risk,
)

PHASE_PREFILL = 0
PHASE_MEASURED = 2


def _measurable(rid: str, c_q: int, t_q: float, tail) -> ProgressView:
    return ProgressView(rid, c_q, t_q, PHASE_MEASURED, False, tail)


def _forced(rid: str) -> ProgressView:
    return ProgressView(rid, 0, None, PHASE_PREFILL, False, lambda x: 1.0)


def _new_admit(rid: str) -> ProgressView:
    return ProgressView(rid, 0, None, PHASE_PREFILL, True, lambda x: 1.0)


def _cpu(rid: str, c_q: int, t_q: float, tail) -> ProgressView:
    return ProgressView(rid, c_q, t_q, PHASE_MEASURED, False, tail, LOC_CPU)


# schedule_step's KV accounting is a cumulative sum of per-candidate block
# counts. `_one_block` gives every candidate the same unit demand, so
# free_kv_blocks=n reads directly as "at most n admits".
_one_block = lambda rid: 1
_KV_UNLIMITED = dict(free_kv_blocks=10**9, blocks_of=_one_block)


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
    # No in-flight; E_viol(k)=0<1 for all k, so k* is bounded by the KV blocks.
    waiting = [_new_admit(f"w{i}") for i in range(10)]
    res = schedule_step([], waiting, b=8, delta=1.0,
                        free_kv_blocks=3, blocks_of=_one_block)
    assert res.k_star == 3
    assert res.e_viol == 0.0


def test_schedule_step_kv_caps_below_risk():
    v = _measurable("v", 0, 100.0, lambda x: 0.0)  # zero risk
    waiting = [_new_admit(f"w{i}") for i in range(10)]
    res = schedule_step([v], waiting, b=8, delta=1.0,
                        free_kv_blocks=2, blocks_of=_one_block)
    assert res.k_star == 2


def test_schedule_step_matches_build_plan_boundary():
    # Decreasing tail so deferring a measurable raises E_viol. As k grows,
    # decode_cap shrinks → more deferral → E_viol increases (monotone).
    tail = lambda x: max(0.0, 1.0 - x / 6.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    res = schedule_step(a, waiting, b=3, delta=1.0, **_KV_UNLIMITED)
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
                        **_KV_UNLIMITED)
    assert res.k_star == 0
    # one scheduled (R_run=1), two deferred (R_defer=1) → e_viol=3
    assert res.e_viol >= 1.0
    assert len(res.scheduled_A) == 1
    assert len(res.deferred_A) == 2


def test_schedule_step_delta_criterion_unlocks_sunk_risk():
    # Same self-locking input as above: E_viol(0) = 3 >= 1 from in-flight risk
    # alone. The absolute rule pins k*=0; the marginal rule sees a flat tail
    # (admits add no extra expected violations) and admits the whole prefix.
    tail = lambda x: 1.0
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(2)]
    absolute = schedule_step(a, waiting, b=1, delta=1.0, **_KV_UNLIMITED)
    delta_res = schedule_step(a, waiting, b=1, delta=1.0, **_KV_UNLIMITED,
                              admission_delta_criterion=True)
    assert absolute.k_star == 0
    assert delta_res.k_star == 2
    # e_viol stays the ABSOLUTE value at k*, not the marginal one.
    assert delta_res.e_viol >= E_VIOL_FEASIBLE


def test_schedule_step_delta_criterion_matches_absolute_at_low_risk():
    # E_viol(0) == 0 → sunk cost is nothing → both criteria are identical.
    tail = lambda x: max(0.0, 1.0 - x / 2.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    assert build_plan(a, [], 3, 1.0).e_viol == 0.0
    absolute = schedule_step(a, waiting, b=3, delta=1.0, **_KV_UNLIMITED)
    delta_res = schedule_step(a, waiting, b=3, delta=1.0, **_KV_UNLIMITED,
                              admission_delta_criterion=True)
    assert 0 < absolute.k_star < len(waiting)  # the risk budget did bind
    assert delta_res.k_star == absolute.k_star
    assert delta_res.e_viol == absolute.e_viol


def test_schedule_step_delta_criterion_still_bounded_by_marginal_budget():
    # Two doomed reqs (tail == 1) make E_viol(0) = 2 >= 1 → absolute locks at
    # k=0. Three more reqs whose deferral risk grows with k supply the
    # marginal cost, so the delta scan admits some and then stops.
    tail = lambda x: max(0.0, 1.0 - x / 2.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    a += [_measurable(f"d{i}", 0, 3.0, lambda x: 1.0) for i in range(2)]
    waiting = [_new_admit(f"w{i}") for i in range(4)]
    absolute = schedule_step(a, waiting, b=5, delta=1.0, **_KV_UNLIMITED)
    delta_res = schedule_step(a, waiting, b=5, delta=1.0, **_KV_UNLIMITED,
                              admission_delta_criterion=True)
    assert absolute.k_star == 0
    # Oracle: largest k with E_viol(k) - E_viol(0) < 1.
    base = build_plan(a, [], 5, 1.0).e_viol
    expected_k = 0
    for k in range(1, len(waiting) + 1):
        if build_plan(a, waiting[:k], 5, 1.0).e_viol - base < 1.0:
            expected_k = k
        else:
            break
    assert 0 < expected_k < len(waiting)  # the marginal budget did bind
    assert delta_res.k_star == expected_k


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


# ---- n_defer_cpu / request_risk_cpu -----------------------------------

def test_n_defer_cpu_equals_n_defer_at_zero_lead():
    for h in (0, 1, 5, 10):
        for s in (0.5, 1.0):
            assert n_defer_cpu(h, s, 0) == n_run_defer(h, s)[1]


def test_n_defer_cpu_lead_reduces_and_clamps():
    assert n_defer_cpu(10, 1.0, 2) == 7  # floor(max(10-1-2,0)*1)
    assert n_defer_cpu(2, 1.0, 5) == 0  # max(...,0) clamp


def test_request_risk_cpu_monotone():
    # R_run <= R_defer <= R_defer_cpu (tail non-increasing, fewer usable
    # iters on CPU), M_cpu >= 0.
    tail = lambda x: max(0.0, 1.0 - x / 1000.0)
    v = _measurable("q", c_q=10, t_q=10.0, tail=tail)
    r_run, r_defer, _m = request_risk(v, delta=0.1, s=1.0)
    r_defer2, r_defer_cpu, m_cpu = request_risk_cpu(
        v, delta=0.1, s=1.0, lead=2)
    assert abs(r_defer - r_defer2) < 1e-9
    assert r_run <= r_defer <= r_defer_cpu
    assert abs(m_cpu - (r_defer_cpu - r_defer)) < 1e-9
    assert m_cpu >= 0.0


# ---- build_plan: KV-offload locations ---------------------------------

def test_build_plan_cpu_no_slot_contributes_rdefer_cpu():
    cpu = _cpu("c", 0, 11.0, _T2)
    v1 = _measurable("v1", 0, 11.0, _T1)
    plan = build_plan([v1, cpu], [], b=1, delta=1.0, lead=0)
    # CPU req holds no slot: absent from scheduled/deferred/risks, present
    # in offloaded_A + cpu_risks.
    assert plan.offloaded_A == ["c"]
    assert "c" in plan.cpu_risks
    assert "c" not in plan.scheduled_A
    assert "c" not in plan.deferred_A
    assert "c" not in plan.risks
    # E_viol = R_run(v1) + R_defer_cpu(c).
    r_defer_cpu = plan.cpu_risks["c"][1]
    r_run_v1 = plan.risks["v1"][0]
    assert abs(plan.e_viol - (r_run_v1 + r_defer_cpu)) < 1e-9


def test_build_plan_service_share_includes_cpu_by_default():
    # A lone GPU measurable with one CPU sibling: parked reqs stay in the
    # service race by default, so |A+| == 2 and s = min(1, B/2).
    v1 = _measurable("v1", 0, 11.0, _T1)
    cpu = _cpu("c", 0, 11.0, _T2)
    plan = build_plan([v1, cpu], [], b=1, delta=1.0)
    s = service_share(1, 2)  # n_aplus == 2 (GPU + CPU)
    nr, _nd = n_run_defer(horizon(11.0, 1.0), s)
    assert abs(plan.risks["v1"][0] - _T1(nr)) < 1e-9


def test_build_plan_service_share_excludes_cpu_when_opted_out():
    # Legacy semantics: |A+| == 1 (CPU out) → larger s → smaller R_run.
    v1 = _measurable("v1", 0, 11.0, _T1)
    cpu = _cpu("c", 0, 11.0, _T2)
    plan = build_plan([v1, cpu], [], b=1, delta=1.0,
                      share_includes_parked=False)
    s = service_share(1, 1)  # n_aplus == 1
    nr, _nd = n_run_defer(horizon(11.0, 1.0), s)
    assert abs(plan.risks["v1"][0] - _T1(nr)) < 1e-9
    # The two semantics must actually differ here.
    inc = build_plan([v1, cpu], [], b=1, delta=1.0)
    assert plan.risks["v1"][0] < inc.risks["v1"][0]


def test_build_plan_parked_share_does_not_change_slots():
    # Including parked reqs in |A+| touches s only: decode_cap is unchanged
    # (the CPU req still holds no slot) and it still contributes R_defer_cpu.
    v1 = _measurable("v1", 0, 11.0, _T1)
    v2 = _measurable("v2", 0, 11.0, _T2)
    cpu = _cpu("c", 0, 11.0, _T2)
    inc = build_plan([v1, v2, cpu], [], b=1, delta=1.0)
    exc = build_plan([v1, v2, cpu], [], b=1, delta=1.0,
                     share_includes_parked=False)
    assert inc.scheduled_A == exc.scheduled_A == ["v2"]
    assert inc.deferred_A == exc.deferred_A == ["v1"]
    assert inc.offloaded_A == exc.offloaded_A == ["c"]
    # Smaller s → less future service on CPU → higher R_defer_cpu.
    assert inc.cpu_risks["c"][1] > exc.cpu_risks["c"][1]


def test_build_plan_onloading_takes_slot_excluded_from_eviol():
    # ONLOADING measurable with saturated tail: if mistreated as a GPU
    # measurable it would add 1.0 to E_viol. As ONLOADING it takes a slot
    # and is excluded from the sum (forced-like) but stays in |A+|.
    on = ProgressView(
        "on", 0, 10.0, PHASE_MEASURED, False, lambda x: 1.0, LOC_ONLOADING)
    v1 = _measurable("v1", 0, 11.0, _T1)
    plan = build_plan([on, v1], [], b=2, delta=1.0)
    assert "on" in plan.scheduled_A
    assert "on" not in plan.risks
    # decode_cap = 2 - 0(forced) - 1(onloading) - 0 = 1 → v1 scheduled.
    assert "v1" in plan.scheduled_A
    assert abs(plan.e_viol - plan.risks["v1"][0]) < 1e-9


# ---- schedule_step: kv_capped -----------------------------------------

def test_schedule_step_kv_capped_reports_unconstrained():
    # E_viol always 0 (zero risk), KV caps at 2 but 5 could be admitted.
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    res = schedule_step([], waiting, b=8, delta=1.0,
                        free_kv_blocks=2, blocks_of=_one_block)
    assert res.k_star == 2
    assert res.k_star_unconstrained == 5
    assert res.kv_capped is True


def test_schedule_step_not_capped_when_eviol_bounds():
    # Decreasing tail → E_viol stops the scan before KV ever bites.
    tail = lambda x: max(0.0, 1.0 - x / 6.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    res = schedule_step(a, waiting, b=3, delta=1.0, **_KV_UNLIMITED)
    assert res.kv_capped is False
    assert res.k_star_unconstrained == res.k_star


# ---- schedule_step: measured KV block accounting ----------------------

# Uneven prompts: a flat per-admit constant cannot express this, which is the
# whole point of the 2026-08-13 change. Cumulative demand: 141, 161, 461, 471.
_BLOCKS = {"w0": 141, "w1": 20, "w2": 300, "w3": 10}


def test_schedule_step_stops_at_cumulative_block_sum():
    # Zero-risk in-flight → E_viol never binds, so only KV can stop the scan.
    waiting = [_new_admit(f"w{i}") for i in range(4)]
    # 400 free blocks covers w0+w1 (161) but not w0+w1+w2 (461).
    res = schedule_step([], waiting, b=8, delta=1.0, free_kv_blocks=400,
                        blocks_of=_BLOCKS.__getitem__)
    assert res.k_star == 2
    assert res.kv_capped is True
    assert res.k_star_unconstrained == 4
    # The offload tier must free the DEFICIT, not the incremental demand:
    # the whole prefix wants 471 and 400 are already free.
    assert res.kv_blocks_needed == 71
    # One block short of the first candidate → nobody is admitted.
    tight = schedule_step([], waiting, b=8, delta=1.0, free_kv_blocks=140,
                          blocks_of=_BLOCKS.__getitem__)
    assert tight.k_star == 0
    assert tight.kv_capped is True
    assert tight.kv_blocks_needed == 471 - 140


def test_schedule_step_kv_boundary_is_inclusive():
    # free_kv_blocks lands EXACTLY on a cumulative sum: that candidate still
    # fits (the rule is <=), and the deficit is measured from there.
    waiting = [_new_admit(f"w{i}") for i in range(4)]
    res = schedule_step([], waiting, b=8, delta=1.0, free_kv_blocks=161,
                        blocks_of=_BLOCKS.__getitem__)
    assert res.k_star == 2  # w0 + w1 == 161 fits exactly
    assert res.kv_blocks_needed == 471 - 161


def test_schedule_step_no_blocks_needed_when_eviol_stops_at_the_kv_break():
    # KV stops the scan at k=1, but E_viol would have stopped there too, so
    # k*_unconstrained == k* — nothing to offload FOR, and the tier stays off.
    tail = lambda x: 0.0 if x >= 3 else 0.9
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(2)]
    waiting = [_new_admit(f"w{i}") for i in range(4)]
    assert build_plan(a, [], 2, 1.0).e_viol < 1.0            # k=0 feasible
    assert build_plan(a, waiting[:1], 2, 1.0).e_viol >= 1.0  # k=1 is not
    res = schedule_step(a, waiting, b=2, delta=1.0, free_kv_blocks=0,
                        blocks_of=_BLOCKS.__getitem__)
    assert res.k_star == 0
    assert res.k_star_unconstrained == 0
    assert res.kv_capped is False
    assert res.kv_blocks_needed == 0


def test_schedule_step_kv_not_capped_when_blocks_suffice():
    # Same candidates, enough blocks for all of them → E_viol decides k* and
    # the offload trigger stays quiet.
    tail = lambda x: max(0.0, 1.0 - x / 2.0)
    a = [_measurable(f"a{i}", 0, 3.0, tail) for i in range(3)]
    waiting = [_new_admit(f"w{i}") for i in range(4)]
    res = schedule_step(a, waiting, b=3, delta=1.0, free_kv_blocks=471,
                        blocks_of=_BLOCKS.__getitem__)
    expected_k = 0
    for k in range(len(waiting) + 1):
        if build_plan(a, waiting[:k], 3, 1.0).e_viol < 1.0:
            expected_k = k
        else:
            break
    assert 0 < expected_k < len(waiting)  # the risk budget really did bind
    assert res.k_star == expected_k
    assert res.kv_capped is False
    assert res.kv_blocks_needed == 0


def test_schedule_step_threads_share_includes_parked():
    # Parked reqs in |A+| shrink s → higher risk → a smaller admission k*.
    # Step tail: P(L>x)=0.6 for x<7 (two such reqs ⇒ E_viol >= 1), else 0.
    def tail(x):
        return 0.6 if x < 7 else 0.0

    # Two GPU measurables (H_q=20) plus two risk-free parked reqs that move
    # |A+| only.
    views = [_measurable(f"v{i}", 0, 20.0, tail) for i in range(2)] + [
        _cpu(f"c{i}", 0, 20.0, lambda x: 0.0) for i in range(2)]
    waiting = [_new_admit(f"w{i}") for i in range(5)]
    inc = schedule_step(views, waiting, b=2, delta=1.0, **_KV_UNLIMITED)
    exc = schedule_step(views, waiting, b=2, delta=1.0, **_KV_UNLIMITED,
                        share_includes_parked=False)
    assert inc.k_star < exc.k_star
    # Each result must match build_plan under the same flag (threaded, not
    # silently defaulted).
    assert abs(inc.e_viol - build_plan(
        views, waiting[:inc.k_star], 2, 1.0).e_viol) < 1e-9
    assert abs(exc.e_viol - build_plan(
        views, waiting[:exc.k_star], 2, 1.0,
        share_includes_parked=False).e_viol) < 1e-9


# ---- select_offload ---------------------------------------------------

# Gentle vs steep tails → different M_cpu at (delta=1, s=1, lead=2, h=10):
#   N_defer=9, N_defer_cpu=7.
#   gentle 1-x/100: M_cpu = tail(7)-tail(9) = 0.93-0.91 = 0.02
#   steep  1-x/20 : M_cpu = tail(7)-tail(9) = 0.65-0.55 = 0.10
_GENTLE = lambda x: max(0.0, 1.0 - x / 100.0)
_STEEP = lambda x: max(0.0, 1.0 - x / 20.0)


def test_select_offload_zero_blocks_needed_is_empty():
    a = _measurable("a", 0, 10.0, _GENTLE)
    assert select_offload([a], 1.0, 1.0, 1.0, 2, eps=1.0, blocks_needed=0,
                          blocks_of=lambda r: 5) == []


def test_select_offload_filters_by_eps():
    a = _measurable("a", 0, 10.0, _GENTLE)  # M_cpu ~0.02
    b = _measurable("b", 0, 10.0, _STEEP)  # M_cpu ~0.10
    # eps excludes the steep one; only 'a' is offload-eligible.
    got = select_offload([a, b], 1.0, 1.0, 1.0, 2, eps=0.05, blocks_needed=100,
                        blocks_of=lambda r: 1)
    assert got == ["a"]


def test_select_offload_sorts_by_mcpu_then_meets_blocks():
    a = _measurable("a", 0, 10.0, _GENTLE)  # M_cpu ~0.02 (smaller)
    b = _measurable("b", 0, 10.0, _STEEP)  # M_cpu ~0.10 (larger)
    blocks = {"a": 3, "b": 5}
    # Ascending M_cpu → a first. blocks_needed=2 met by a alone.
    got1 = select_offload([b, a], 1.0, 1.0, 1.0, 2, eps=1.0, blocks_needed=2,
                         blocks_of=lambda r: blocks[r])
    assert got1 == ["a"]
    # blocks_needed=4 → a(3) then b(8) to cross the threshold.
    got2 = select_offload([b, a], 1.0, 1.0, 1.0, 2, eps=1.0, blocks_needed=4,
                         blocks_of=lambda r: blocks[r])
    assert got2 == ["a", "b"]


def test_select_offload_doomed_guard_excludes_certain_miss():
    # Saturated tail → M_cpu = 0 (clears eps) but R_defer_cpu = 1: parking it
    # is pointless because select_onload's safety net recalls it at once. The
    # mirror image of test_select_onload_certain_miss_safety_net.
    miss = _measurable("miss", 0, 10.0, lambda x: 1.0)
    assert select_offload([miss], 1.0, 1.0, 1.0, 2, eps=1.0, blocks_needed=1,
                          blocks_of=lambda r: 1) == []


def test_select_offload_tiebreak_prefers_larger_block_count():
    # Equal M_cpu (same tail) → larger block count first.
    a = _measurable("a", 0, 10.0, _GENTLE)
    b = _measurable("b", 0, 10.0, _GENTLE)
    blocks = {"a": 2, "b": 9}
    got = select_offload([a, b], 1.0, 1.0, 1.0, 2, eps=1.0, blocks_needed=5,
                        blocks_of=lambda r: blocks[r])
    assert got == ["b"]  # b's 9 blocks alone meets the need


# ---- select_onload ----------------------------------------------------

def test_select_onload_boundary_by_eps():
    cheap = _cpu("cheap", 0, 10.0, _GENTLE)  # M_cpu ~0.02 → stay
    costly = _cpu("costly", 0, 10.0, _STEEP)  # M_cpu ~0.10 → onload
    got = select_onload([cheap, costly], 1.0, 1.0, 2, eps=0.05)
    assert got == ["costly"]


def test_select_onload_certain_miss_safety_net():
    # Saturated tail → M_cpu = 0 (<= eps) but R_defer_cpu = 1 forces onload.
    miss = _cpu("miss", 0, 10.0, lambda x: 1.0)
    assert select_onload([miss], 1.0, 1.0, 2, eps=0.05) == ["miss"]


def test_select_onload_skips_non_measurable_cpu():
    prefill = ProgressView(
        "p", 0, None, PHASE_PREFILL, False, lambda x: 1.0, LOC_CPU)
    assert select_onload([prefill], 1.0, 1.0, 2, eps=0.05) == []


def test_pick_adaptive_batch_threads_lead_to_cpu_risk():
    # A CPU-resident (LOC_CPU) request's R_defer_cpu depends on `lead` (it
    # loses `lead` future iterations before its horizon). pick_adaptive_batch
    # must thread lead into build_plan so its E_viol matches
    # schedule_step(lead=...). Step tail: P(L>x)=0.5 for x<7, else 0.
    def tail(x):
        return 0.5 if x < 7 else 0.0

    views = [_cpu("c0", c_q=0, t_q=10.0, tail=tail)]
    # delta=1 → H_q=10, s=1. N_defer_cpu(lead=0)=9 → tail(9)=0 → e_viol=0.
    _b0, plan0 = pick_adaptive_batch(views, 4, 1.0, [], lambda b: None, lead=0)
    # N_defer_cpu(lead=5)=4 → tail(4)=0.5 → e_viol=0.5.
    _b5, plan5 = pick_adaptive_batch(views, 4, 1.0, [], lambda b: None, lead=5)
    assert plan0.e_viol == 0.0
    assert plan5.e_viol == 0.5
    # Default lead=0 keeps back-compat with existing call sites.
    _bd, plan_default = pick_adaptive_batch(views, 4, 1.0, [], lambda b: None)
    assert plan_default.e_viol == plan0.e_viol


def test_pick_adaptive_batch_threads_share_includes_parked():
    # 1 GPU + 3 parked, b=2: |A+| = 4 (parked in) → s = 0.5 vs |A+| = 1
    # (parked out) → s = 1. The GPU req's R_run must follow the flag.
    views = [_measurable("v1", 0, 11.0, _T1)] + [
        _cpu(f"c{i}", 0, 11.0, _T2) for i in range(3)]
    _bi, plan_inc = pick_adaptive_batch(views, 2, 1.0, [], lambda b: None)
    _be, plan_exc = pick_adaptive_batch(views, 2, 1.0, [], lambda b: None,
                                        share_includes_parked=False)
    assert plan_inc.risks["v1"][0] > plan_exc.risks["v1"][0]


# ---------------------------------------------------------------------------
# Token Budget — P axis (TB*_pre, risk dosing)
# ---------------------------------------------------------------------------

_SAFE = lambda x: 0.0  # deep slack → R_defer = 0, and no risk to save either
_ATRISK = lambda x: 1.0  # certain miss → R_defer = 1
# An urgent-but-savable request: its risk actually falls when the step gets
# shorter, which is what makes a smaller prefill dose pay for itself.
_URGENT = lambda x: max(0.0, 1.0 - x / 40.0)

# Δ_dec 80 ms, κ_p 1 ms/prefill-token, base 8192 tokens, floor 512, and a
# 4096-token burst W → the burst's sunk delay κ_p·W is 4.096 s and its window
# wall(P) = 327.68/P + 4.096 s spans 4.14 s (at P = base) to 4.74 s (at floor).
_PBR = dict(b=8, delta_dec=0.08, kappa_p=0.001, base_budget=8192, floor=512,
            eps_p=0.01, prefill_work=4096)


def _tb_e_viol(views, p, **kw):
    """E(P): the ledger re-priced under a burst of W/P steps at Δ_dec + κ_p·P."""
    return build_plan(views, [], kw["b"], kw["delta_dec"],
                      delta_burst=kw["delta_dec"] + kw["kappa_p"] * p,
                      burst_steps=kw["prefill_work"] / p).e_viol


def _tb_oracle(views, **kw):
    """Brute-force max{P in [floor, base] : E(P) - E(floor) <= eps_p}, plus the
    feasibility sequence so the test can assert the monotonicity the bisection
    relies on."""
    ref = _tb_e_viol(views, kw["floor"], **kw)
    feasible = [_tb_e_viol(views, p, **kw) - ref <= kw["eps_p"]
                for p in range(kw["floor"], kw["base_budget"] + 1)]
    best = kw["floor"]
    for i, ok in enumerate(feasible):
        if ok:
            best = kw["floor"] + i
    return best, feasible


def test_token_budget_prefill_risk_doomed_in_flight_does_not_throttle():
    # The recovery case that killed the γ·t_min rule: the survivors hold
    # near-zero time to their deadlines but are unsavable (R = 1 either way),
    # so E_viol does not move with P and the full base budget is released for
    # the refill. The old worst-case rule read the same state as t_min = 0.2 s
    # and pinned the step to the floor.
    views = [_measurable(f"d{i}", 0, 0.2, _ATRISK) for i in range(4)]
    assert token_budget_prefill_risk(views, **_PBR) == 8192
    assert token_budget_prefill(t_min=0.2, delta_decode=0.08, kappa=0.001,
                                base_budget=8192, floor=512, gamma=0.5) == 512


def test_token_budget_prefill_risk_slack_returns_base():
    # Nothing at risk → the dosing never binds.
    views = [_measurable(f"v{i}", 0, 100.0, _SAFE) for i in range(4)]
    assert token_budget_prefill_risk(views, **_PBR) == 8192


def test_token_budget_prefill_risk_stops_at_eps_and_matches_brute_force():
    # One sensitive in-flight request: every extra prefill token stretches the
    # step, shrinking its horizon, so E_viol climbs with P and the dose stops
    # where the climb reaches eps_p. Bisection must land on the same integer a
    # linear scan does (small range so the scan is cheap).
    kw = {**_PBR, "base_budget": 400, "floor": 10, "eps_p": 0.005}
    gentle = lambda x: max(0.0, 1.0 - x / 2000.0)
    views = [_measurable("u", 0, 2.0, gentle)] + [
        _measurable(f"s{i}", 0, 100.0, _SAFE) for i in range(3)]
    got = token_budget_prefill_risk(views, **kw)
    oracle, feasible = _tb_oracle(views, **kw)
    assert got == oracle
    assert kw["floor"] < got < kw["base_budget"]  # the dosing actually binds
    # E_viol is non-decreasing in P, i.e. feasibility is a prefix — the
    # premise the bisection stands on.
    assert feasible == sorted(feasible, reverse=True)


def test_token_budget_prefill_risk_e_viol_is_monotone_in_p():
    # The premise the bisection stands on, over a mixed ledger: one request
    # whose deadline falls inside the burst window (priced), one past it
    # (flat), one doomed (flat at R = 1). H_q is non-increasing in P for each,
    # so the sum only ever climbs.
    views = [_measurable("near", 0, 2.0, lambda x: max(0.0, 1 - x / 2000.0)),
             _measurable("far", 0, 20.0, lambda x: max(0.0, 1 - x / 20000.0)),
             _measurable("doomed", 0, 0.1, _ATRISK)]
    evals = [_tb_e_viol(views, p, **_PBR)
             for p in (512, 700, 1024, 2048, 4096, 6000, 8192)]
    assert evals == sorted(evals)
    assert evals[0] < evals[-1]  # the ledger really does move with P


def test_token_budget_prefill_risk_far_deadlines_are_not_priced():
    # Every deadline sits past wall(floor) = 4.74 s, so the horizon is
    # (T_q - κ_p·W)/Δ_dec — set by the burst's sunk total delay, not by P.
    # E(P) is flat, shrinking the budget buys nothing, and base is released.
    views = [_measurable("f", 0, 20.0, lambda x: max(0.0, 1 - x / 20000.0))]
    evals = [_tb_e_viol(views, p, **_PBR) for p in (512, 1024, 4096, 8192)]
    assert evals == [evals[0]] * len(evals)
    assert 0.0 < evals[0] < 1.0  # non-degenerate: the request does carry risk
    assert token_budget_prefill_risk(views, **_PBR) == 8192


def test_token_budget_prefill_risk_sunk_burst_cost_does_not_pin_to_floor():
    # Regression guard on the reference point: E_ref is E(floor), NOT a
    # burst-free E. Here the burst's unavoidable delay κ_p·W is 200 s, but every
    # deadline sits past wall(floor) = 231 s, so no P is any better than another
    # and the whole base budget must still be released. Priced against a
    # burst-free reference the fixed sunk term alone blows eps_p at EVERY P,
    # which would pin the dose to the floor on every step.
    kw = {**_PBR, "prefill_work": 200_000}
    views = [_measurable("f", 0, 401.0, lambda x: max(0.0, 1 - x / 5000.0))]
    assert token_budget_prefill_risk(views, **kw) == 8192
    burst_free = build_plan(views, [], kw["b"], kw["delta_dec"]).e_viol
    assert _tb_e_viol(views, kw["base_budget"], **kw) - burst_free > kw["eps_p"]


def test_token_budget_prefill_risk_clamps_for_in_burst_deadline():
    # A deadline INSIDE the burst window (4 s vs wall(P) >= 4.14 s): its token
    # rate is 1/Δ_b(P), so P is priced against what the floor already delivers
    # and the dose stops strictly inside [floor, base].
    views = [_measurable("n", 0, 4.0, lambda x: max(0.0, 1 - x / 10.0))]
    got = token_budget_prefill_risk(views, **_PBR)
    assert 512 < got < 8192
    assert got == _tb_oracle(views, **_PBR)[0]


def test_token_budget_prefill_risk_disabled_returns_base():
    views = [_measurable("u", 0, 2.0, _URGENT)]
    # No κ estimate yet / degenerate κ → control off.
    assert token_budget_prefill_risk(views, **{**_PBR, "kappa_p": 0.0}) == 8192
    assert token_budget_prefill_risk(views, **{**_PBR, "kappa_p": -1.0}) == 8192
    # No prefill work → no burst to price → nothing to dose.
    assert token_budget_prefill_risk(
        views, **{**_PBR, "prefill_work": 0}) == 8192
    # Nothing measurable in flight → no ledger to protect → control off.
    assert token_budget_prefill_risk([_forced("f0")], **_PBR) == 8192


def test_token_budget_prefill_risk_stays_within_floor_and_base():
    views = [_measurable("u", 0, 0.5, _URGENT),
             _measurable("d", 0, 0.1, _ATRISK)]
    for floor in (1, 512, 8192, 20000):  # incl. floor above base (no-op)
        p = token_budget_prefill_risk(views, **{**_PBR, "floor": floor})
        assert min(floor, 8192) <= p <= 8192


# ---------------------------------------------------------------------------
# Token Budget — P axis, DEPRECATED worst-case rule (token_budget_prefill)
# Kept for the A/B replay of γ·t_min against the risk dosing above.
# ---------------------------------------------------------------------------

# Δ_decode 80 ms, κ 1 ms/prefill-token, base 8192 tokens, floor 512, γ = 0.5.
_PB = dict(delta_decode=0.08, kappa=0.001, base_budget=8192, floor=512,
           gamma=0.5)


def test_token_budget_prefill_grows_with_slack_and_saturates():
    # t_min = 2 s → (0.5*2 - 0.08)/0.001 = 920 tokens.
    assert token_budget_prefill(t_min=2.0, **_PB) == 920
    # More slack → strictly more prefill allowed.
    assert (token_budget_prefill(t_min=4.0, **_PB)
            > token_budget_prefill(t_min=2.0, **_PB))
    # Enough slack → saturates at the engine's base budget.
    assert token_budget_prefill(t_min=1000.0, **_PB) == _PB["base_budget"]


def test_token_budget_prefill_monotone_non_decreasing_in_t_min():
    prev = 0
    for t_min in (0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        cur = token_budget_prefill(t_min=t_min, **_PB)
        assert cur >= prev
        prev = cur


def test_token_budget_prefill_overdue_clamps_to_floor():
    # Already past the deadline → smallest budget that still makes progress.
    assert token_budget_prefill(t_min=0.0, **_PB) == _PB["floor"]
    assert token_budget_prefill(t_min=-3.0, **_PB) == _PB["floor"]
    # Positive but tighter than the decode cost alone → also floor.
    assert token_budget_prefill(t_min=0.01, **_PB) == _PB["floor"]


def test_token_budget_prefill_disabled_returns_base():
    # No κ estimate yet / degenerate κ → control off.
    assert token_budget_prefill(t_min=1.0, **{**_PB, "kappa": 0.0}) == 8192
    assert token_budget_prefill(t_min=1.0, **{**_PB, "kappa": -1.0}) == 8192
    # Nothing measurable in flight → nothing to protect → control off.
    assert token_budget_prefill(t_min=None, **_PB) == 8192


def test_token_budget_prefill_stays_within_floor_and_base():
    for t_min in (None, -10.0, 0.0, 0.05, 0.3, 1.0, 7.5, 1e6):
        p = token_budget_prefill(t_min=t_min, **_PB)
        assert _PB["floor"] <= p <= _PB["base_budget"]


def test_token_budget_prefill_floor_above_base_never_exceeds_base():
    # Misconfiguration guard: a floor above the engine's base budget must not
    # push the result past base_budget (the control degrades to a no-op).
    over = {**_PB, "floor": 20000}
    for t_min in (None, -1.0, 0.0, 0.5, 3.0, 1e6):
        assert token_budget_prefill(t_min=t_min, **over) == over["base_budget"]


# ---- two-share offload / onload (spec §3④) ----------------------------

# Onload is judged on s_now (|A+| counting k*); offload must clear eps under
# BOTH s_now and s_post (|A+| counting k*_unconstrained — the state once the
# freed blocks are spent on admits) and must not be doomed under either. The
# max() form makes the anti-thrash property structural against select_onload's
# M_cpu trigger: eligibility implies M_cpu(s_now) <= eps (not an onload
# candidate on the same step) AND M_cpu(s_post) <= eps (not one on the next
# step either, once the admits have landed and s_post is the live share). The
# doomed guard covers select_onload's other trigger, the R_defer_cpu >= 1
# safety net, which max() cannot reach: a doomed request has M_cpu ~ 0 and so
# clears eps, but would be called straight back.
_EPS = 1e-3
_LEAD = 2
_DELTA = 0.04


def _geom(q):
    return lambda x: q**x


def _lin(n):
    return lambda x: max(0.0, 1.0 - x / n)


# The grid that disproved the monotonicity the first draft assumed. Shares
# span the range the scheduler can actually produce (|A+| <= ~2B + parked);
# tails mix fast/slow geometric decay with locally flat (linear) ones; T_q
# spans a 120x horizon range. 5 tails x 5 horizons x 3 c_q x 28 ordered
# share pairs = 2100 combinations.
_GRID_TAILS = [_geom(0.999), _geom(0.99), _geom(0.95), _lin(1000), _lin(200)]
_GRID_T_Q = [0.5, 2.0, 5.0, 20.0, 60.0]
_GRID_C_Q = [0, 20, 100]
_GRID_SHARES = [0.35, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


# Doomed extension. The base grid never reaches R_defer_cpu >= 1, so it could
# not exercise the doomed guard: T_q below Delta collapses H_q to 0 (both
# N_defer and N_defer_cpu are 0, so R_defer_cpu = tail(c_q) = 1 at c_q = 0),
# and a saturated tail is doomed at any horizon.
_DOOMED_T_Q = [0.01, 0.03]
_DOOMED_TAILS = [lambda x: 1.0]


def _grid_cases(tails=None, t_qs=None):
    for tail in tails or _GRID_TAILS:
        for t_q in t_qs or _GRID_T_Q:
            for c_q in _GRID_C_Q:
                for i, s_now in enumerate(_GRID_SHARES):
                    for s_post in _GRID_SHARES[:i + 1]:  # s_post <= s_now
                        yield tail, t_q, c_q, s_now, s_post


def _all_grid_cases():
    yield from _grid_cases()
    yield from _grid_cases(t_qs=_DOOMED_T_Q)
    yield from _grid_cases(tails=_DOOMED_TAILS,
                           t_qs=_GRID_T_Q + _DOOMED_T_Q)


def test_offload_never_round_trips_over_the_full_share_grid():
    # Core regression line. Over every combination — no regime restriction,
    # since max() needs no monotonicity — an offload-eligible request is an
    # onload candidate under NEITHER share: not under s_now (same step) and
    # not under s_post (next step, after the admits it freed room for land).
    # Covers both of select_onload's triggers: the M_cpu one (negated by
    # max()) and the R_defer_cpu >= 1 safety net (negated by the doomed
    # guard), the latter only reachable on the doomed extension.
    cases = offloads = doomed_but_cheap = 0
    for tail, t_q, c_q, s_now, s_post in _all_grid_cases():
        cases += 1
        gpu = _measurable("r", c_q, t_q, tail)
        cpu = _cpu("r", c_q, t_q, tail)
        # Cases the doomed guard exists for: cheap enough to clear eps, yet
        # certain to miss on CPU, so onload's safety net would recall them.
        now = request_risk_cpu(gpu, _DELTA, s_now, _LEAD)
        post = request_risk_cpu(gpu, _DELTA, s_post, _LEAD)
        if max(now[2], post[2]) <= _EPS and max(now[1], post[1]) >= 1.0:
            doomed_but_cheap += 1
        picked = select_offload([gpu], _DELTA, s_now, s_post, _LEAD, _EPS,
                                blocks_needed=1, blocks_of=lambda r: 1)
        if not picked:
            continue
        offloads += 1
        assert select_onload([cpu], _DELTA, s_now, _LEAD, _EPS) == []
        assert select_onload([cpu], _DELTA, s_post, _LEAD, _EPS) == []
    assert cases == 2100 + 840 + 588
    assert offloads > 0  # the grid actually exercises the offload path
    assert doomed_but_cheap > 0  # ...and the doomed guard's own cases


def test_doomed_guard_leaves_non_doomed_selection_untouched():
    # Regression fence for the guard: wherever R_defer_cpu < 1 under both
    # shares, the outcome is still exactly the pre-guard rule
    # (max-M_cpu <= eps). The guard only ever removes doomed reqs.
    checked = 0
    for tail, t_q, c_q, s_now, s_post in _all_grid_cases():
        gpu = _measurable("r", c_q, t_q, tail)
        now = request_risk_cpu(gpu, _DELTA, s_now, _LEAD)
        post = request_risk_cpu(gpu, _DELTA, s_post, _LEAD)
        if max(now[1], post[1]) >= 1.0:
            continue  # doomed — this one is meant to change
        checked += 1
        assert select_offload(
            [gpu], _DELTA, s_now, s_post, _LEAD, _EPS, blocks_needed=1,
            blocks_of=lambda r: 1) == (
                ["r"] if max(now[2], post[2]) <= _EPS else [])
    assert checked > 0


def test_single_share_forms_leave_round_trips_on_the_same_grid():
    # Why max() and not either share alone. Judging on s_post only leaves
    # same-step round trips (the measured 229/2100 that killed the first
    # draft); judging on s_now only — the original code — leaves next-step
    # ones, which is the offload→onload median-2-step thrash.
    same_step = next_step = 0
    for tail, t_q, c_q, s_now, s_post in _grid_cases():
        gpu = _measurable("r", c_q, t_q, tail)
        cpu = _cpu("r", c_q, t_q, tail)
        by_post = select_offload([gpu], _DELTA, s_post, s_post, _LEAD, _EPS,
                                 blocks_needed=1, blocks_of=lambda r: 1)
        by_now = select_offload([gpu], _DELTA, s_now, s_now, _LEAD, _EPS,
                                blocks_needed=1, blocks_of=lambda r: 1)
        if by_post and select_onload([cpu], _DELTA, s_now, _LEAD, _EPS):
            same_step += 1
        if by_now and select_onload([cpu], _DELTA, s_post, _LEAD, _EPS):
            next_step += 1
    assert same_step == 229
    assert next_step > 0


def test_max_form_closes_the_regime_counterexamples():
    # The two mechanisms that break "M_cpu(s_post) >= M_cpu(s_now)". Under the
    # s_post-only draft each produced a request that offloads and is onload-
    # eligible in the same breath; under max() neither offloads at all.
    # (a) locally flat tail: M_cpu ~ f * lead * s *grows* with s.
    flat_gpu = _measurable("r", 0, 20.0, _lin(1500))
    flat_cpu = _cpu("r", 0, 20.0, _lin(1500))
    assert select_offload([flat_gpu], _DELTA, 1.0, 0.5, _LEAD, _EPS,
                          blocks_needed=1, blocks_of=lambda r: 1) == []
    assert select_onload([flat_cpu], _DELTA, 1.0, _LEAD, _EPS) == ["r"]
    # (b) sub-token lead: at s_post < 1/lead the floor()s in N_defer /
    #     N_defer_cpu collapse onto the same token, so M_cpu(s_post) is 0 by
    #     discretisation alone. Reachable once parked reqs push |A+| past
    #     lead * B (R1 keeps them in the denominator).
    h_q = horizon(5.0, _DELTA)
    assert n_defer_cpu(h_q, 0.395, _LEAD) == n_run_defer(h_q, 0.395)[1]
    steep_gpu = _measurable("r", 0, 5.0, _geom(0.95))
    steep_cpu = _cpu("r", 0, 5.0, _geom(0.95))
    assert select_offload([steep_gpu], _DELTA, 0.4, 0.395, _LEAD, _EPS,
                          blocks_needed=1, blocks_of=lambda r: 1) == []
    assert select_onload([steep_cpu], _DELTA, 0.4, _LEAD, _EPS) == ["r"]


def test_single_share_thrashes_across_steps_two_shares_do_not():
    # The measured defect, end to end: step t offloads on the pre-admit share,
    # step t+1 runs at the post-admit share and onloads it right back.
    b, n_base, k_star, k_star_unconstrained = 64, 64, 0, 16
    s_now = service_share(b, n_base + k_star)  # 1.0
    s_post = service_share(b, n_base + k_star_unconstrained)  # 0.8
    gpu = _measurable("r", 20, 3.0, _geom(0.95))
    cpu = _cpu("r", 20, 3.0, _geom(0.95))
    picked = lambda sn, sp: select_offload([gpu], _DELTA, sn, sp, _LEAD, _EPS,
                                           blocks_needed=1,
                                           blocks_of=lambda r: 1)

    # Old behaviour: both judgements used the k* share, so offload passes...
    assert picked(s_now, s_now) == ["r"]
    # ...and once the admits land, the live share is s_post, under which the
    # request it just parked is already onload-eligible → round trip.
    assert select_onload([cpu], _DELTA, s_post, _LEAD, _EPS) == ["r"]

    # Fixed: the s_post leg of max() rejects it, so it never leaves the GPU.
    assert picked(s_now, s_post) == []


def test_two_shares_coincide_when_not_kv_capped():
    # k*_unconstrained == k* (E_viol, not KV, stopped the scan) → s_post is
    # s_now, max() degenerates to the single M_cpu the old code used, and
    # every selection is identical to the single-share behaviour.
    for b, n_aplus in [(64, 64), (64, 80), (64, 96), (32, 40), (32, 48)]:
        s = service_share(b, n_aplus)
        for tail in _GRID_TAILS:
            for t_q in _GRID_T_Q:
                gpu = _measurable("r", 0, t_q, tail)
                cpu = _cpu("r", 0, t_q, tail)
                offloaded = select_offload(
                    [gpu], _DELTA, s, s, _LEAD, _EPS, blocks_needed=1,
                    blocks_of=lambda r: 1)
                m_cpu = request_risk_cpu(gpu, _DELTA, s, _LEAD)[2]
                assert offloaded == (["r"] if m_cpu <= _EPS else [])
                assert select_onload([cpu], _DELTA, s, _LEAD, _EPS) == (
                    [] if m_cpu <= _EPS else ["r"])
