"""ProgressServe: posterior tail-risk scheduling primitives.

Pure functions and lightweight view structs implementing the per-epoch
admission / running-selection math from the ProgressServe spec
(§5.3.3–5.4.4). This module deliberately has NO vLLM engine imports so the
BuildPlan / ScheduleStep logic is unit-testable in isolation; the scheduler
adapts engine state into ``ProgressView`` objects and supplies KV / waiting
callbacks.

Symbols map to the spec as:
  c_q       current-unit tokens already generated
  T_q       time remaining to the current unit's consumption deadline
  Δ         per-iteration latency estimate (IterTime)
  H_q       deadline horizon = floor(T_q / Δ)
  s         service-share approximation = min(1, B/|A+|)
  N_run/N_defer  deadline-feasible token counts under run / defer
  R_run/R_defer  unit-deadline violation risk = tail posterior probabilities
  M         marginal service benefit = R_defer - R_run
  E_viol    expected violation count after scheduling (operating rule < 1)

Note: there is no speculative decoding, so one decode iteration produces one
token; N_run / N_defer (iteration counts) are used directly as token counts.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

# Default minimum number of global-history samples that must exceed c_q for
# the empirical tail posterior to be trusted; below this the caller falls
# back to the analytic cold-start tail. Wired from
# SsloConfig.progress_serve_min_denom.
DEFAULT_MIN_DENOM = 4

# Phase.MEASURED value (see vllm.sslo.slo_state.Phase). Kept as a local
# constant so this module stays engine-import-free; a request is "measurable"
# only when MEASURED and holding a finite deadline.
PHASE_MEASURED = 2


@dataclass
class ProgressView:
    """Per-request snapshot the scheduler hands to the algorithm."""

    request_id: str
    c_q: int
    T_q: float | None  # time_to_deadline(now); None ⇒ no deadline yet
    phase: int  # Phase value: 0=PREFILL, 1=WARMUP, 2=MEASURED
    is_new_admit: bool  # True for A_new(k) candidates drawn from W
    # P(L_q > x | L_q > c_q); non-increasing in x. Closure over the request's
    # length_tail_prob (c_q baked in).
    tail_prob: Callable[[int], float]

    def is_measurable(self) -> bool:
        return self.phase == PHASE_MEASURED and self.T_q is not None


@dataclass
class Plan:
    scheduled_A: list[str]  # existing in-flight request_ids given a slot
    deferred_A: list[str]  # existing in-flight request_ids deferred
    e_viol: float
    # request_id -> (R_run, R_defer, M) for measurable existing in-flight,
    # for decision logging.
    risks: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class ScheduleResult:
    scheduled_A: list[str]
    deferred_A: list[str]
    k_star: int
    e_viol: float


def horizon(t_q: float, delta: float) -> int:
    """H_q = max(0, floor(T_q / Δ)). Negative (past-deadline) clamps to 0."""
    if delta <= 0:
        return 0
    return max(0, math.floor(t_q / delta))


def service_share(b: int, n_aplus: int) -> float:
    """s = min(1, B / |A+|). Returns 1.0 when |A+| == 0."""
    if n_aplus <= 0:
        return 1.0
    return min(1.0, b / n_aplus)


def n_run_defer(h_q: int, s: float) -> tuple[int, int]:
    """(N_run, N_defer).

    I_q = 1[H_q >= 1]; N_future = floor(max(H_q-1, 0) * s);
    N_run = I_q + N_future; N_defer = N_future.
    """
    i_q = 1 if h_q >= 1 else 0
    n_future = math.floor(max(h_q - 1, 0) * s)
    return i_q + n_future, n_future


def request_risk(
    view: ProgressView, delta: float, s: float
) -> tuple[float, float, float]:
    """(R_run, R_defer, M) for a measurable request.

    R_run = P(L > c_q + N_run | L > c_q); R_defer likewise with N_defer.
    M = R_defer - R_run >= 0 (since N_run >= N_defer and tail is
    non-increasing).
    """
    assert view.T_q is not None
    h_q = horizon(view.T_q, delta)
    n_run, n_defer = n_run_defer(h_q, s)
    r_run = view.tail_prob(view.c_q + n_run)
    r_defer = view.tail_prob(view.c_q + n_defer)
    # If deferring is a certain miss (R_defer == 1 — deadline imminent /
    # already passed, or future service-share alone can't finish the unit),
    # give the request the maximum marginal benefit (M = 1) instead of
    # R_defer - R_run. Otherwise a request that also has R_run == 1 (can't
    # finish even with this step) would get M = 0 and be abandoned, which
    # compounds its stall. Forcing M = 1 makes it win a decode slot so it
    # keeps making progress.
    if r_defer >= 1.0:
        return r_run, r_defer, 1.0
    return r_run, r_defer, r_defer - r_run


def build_plan(
    views_A: list[ProgressView],
    views_Anew: list[ProgressView],
    b: int,
    delta: float,
) -> Plan:
    """Partition existing in-flight A into scheduled / deferred under a fixed
    batch size ``b`` and candidate in-flight set A+ = A ∪ A_new.

    Forced reqs (PREFILL / WARMUP / no deadline) always take a slot and are
    excluded from the E_viol sum. New admits A_new also occupy slots (prefill)
    but are not in A, so they affect risk only through |A+| and the slot
    budget — never counted in E_viol. Remaining decode slots go to measurable
    existing reqs by largest marginal benefit M; the rest are deferred.

    E_viol = Σ_{scheduled measurable} R_run + Σ_{deferred measurable} R_defer.
    """
    n_aplus = len(views_A) + len(views_Anew)
    s = service_share(b, n_aplus)

    forced = [v for v in views_A if not v.is_measurable()]
    measurable = [v for v in views_A if v.is_measurable()]

    risks: dict[str, tuple[float, float, float]] = {}
    scored = []
    for v in measurable:
        r_run, r_defer, m = request_risk(v, delta, s)
        risks[v.request_id] = (r_run, r_defer, m)
        scored.append((m, r_run, r_defer, v))
    # Largest marginal benefit first; reduces E_viol the most per slot.
    scored.sort(key=lambda t: t[0], reverse=True)

    decode_cap = max(0, b - len(forced) - len(views_Anew))
    scheduled_ids = [v.request_id for v in forced]
    deferred_ids: list[str] = []
    e_viol = 0.0
    for idx, (_m, r_run, r_defer, v) in enumerate(scored):
        if idx < decode_cap:
            scheduled_ids.append(v.request_id)
            e_viol += r_run
        else:
            deferred_ids.append(v.request_id)
            e_viol += r_defer
    return Plan(
        scheduled_A=scheduled_ids,
        deferred_A=deferred_ids,
        e_viol=e_viol,
        risks=risks,
    )


def schedule_step(
    views_A: list[ProgressView],
    waiting_views: list[ProgressView],
    b: int,
    delta: float,
    kv_feasible: Callable[[int], bool],
) -> ScheduleResult:
    """Find k* = max{k : E_viol(A+(k), B) < 1} over the FCFS prefix of W.

    E_viol is monotone non-decreasing in k (more admits shrink the service
    share and the decode budget), so scan k upward and stop at the first
    infeasible k or when KV admission is exhausted. Returns the scheduled /
    deferred partition at k* and k* itself (an upper bound on admits; the
    engine's allocate_slots is the hard KV backstop).
    """
    best = build_plan(views_A, [], b, delta)
    best_k = 0
    # If even admitting nobody is already over budget, defer-only at k=0.
    if best.e_viol < 1.0:
        n = len(waiting_views)
        for k in range(1, n + 1):
            if not kv_feasible(k):
                break
            plan = build_plan(views_A, waiting_views[:k], b, delta)
            if plan.e_viol < 1.0:
                best, best_k = plan, k
            else:
                break
    return ScheduleResult(
        scheduled_A=best.scheduled_A,
        deferred_A=best.deferred_A,
        k_star=best_k,
        e_viol=best.e_viol,
    )
