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

# Operating rule: an expected-violation count below this is "feasible".
# Shared single source for BOTH admission (schedule_step) and the adaptive
# batch trigger (pick_adaptive_batch) so the two stay in lock-step.
E_VIOL_FEASIBLE = 1.0

# Request KV-cache location for the offload tier. Kept as local ints (like
# PHASE_MEASURED) so this module stays engine-import-free:
#   LOC_GPU        in-flight on GPU (the classic RUN/DEFER population).
#   LOC_CPU        KV vacated to CPU — holds no decode slot and does not
#                  compete for this epoch's service share; its deferred-on-CPU
#                  risk R_defer_cpu still counts toward E_viol.
#   LOC_ONLOADING  promoted CPU→GPU, slot reserved for the restore. Occupies a
#                  slot like a new admit and is excluded from E_viol (forced).
LOC_GPU = 0
LOC_CPU = 1
LOC_ONLOADING = 2


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
    location: int = LOC_GPU  # LOC_GPU / LOC_CPU / LOC_ONLOADING

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
    # request_ids resident on CPU (LOC_CPU) — no slot, no service share.
    offloaded_A: list[str] = field(default_factory=list)
    # request_id -> (R_defer, R_defer_cpu, M_cpu) for measurable CPU-resident
    # requests, for offload decision logging.
    cpu_risks: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)


@dataclass
class ScheduleResult:
    scheduled_A: list[str]
    deferred_A: list[str]
    k_star: int
    e_viol: float
    # k* if the KV admission cap were lifted (E_viol constraint only). Equals
    # k_star when the scan stopped on E_viol rather than KV.
    k_star_unconstrained: int = 0
    # True when KV admission — not E_viol — bounded k* (k_star_unconstrained
    # > k_star). The vacate trigger fires only under this condition.
    kv_capped: bool = False
    offloaded_A: list[str] = field(default_factory=list)
    cpu_risks: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)


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


def n_defer_cpu(h_q: int, s: float, lead: int) -> int:
    """N_defer_cpu = floor(max(H_q - 1 - lead, 0) * s).

    Same future-service estimate as N_defer but with ``lead`` fewer usable
    iterations, modelling that a CPU-resident request must be onloaded
    ``lead`` iterations before its horizon. For lead == 0 this equals the
    N_defer of ``n_run_defer``.
    """
    return math.floor(max(h_q - 1 - lead, 0) * s)


def request_risk_cpu(
    view: ProgressView, delta: float, s: float, lead: int
) -> tuple[float, float, float]:
    """(R_defer, R_defer_cpu, M_cpu) for a measurable request.

    R_defer = P(L > c_q + N_defer | L > c_q); R_defer_cpu likewise with
    N_defer_cpu. Since N_defer_cpu <= N_defer and the tail is non-increasing,
    R_defer <= R_defer_cpu, so M_cpu = R_defer_cpu - R_defer >= 0 — the risk
    penalty of the request staying on CPU (losing ``lead`` iterations).
    """
    assert view.T_q is not None
    h_q = horizon(view.T_q, delta)
    _n_run, n_defer = n_run_defer(h_q, s)
    n_dc = n_defer_cpu(h_q, s, lead)
    r_defer = view.tail_prob(view.c_q + n_defer)
    r_defer_cpu = view.tail_prob(view.c_q + n_dc)
    return r_defer, r_defer_cpu, r_defer_cpu - r_defer


def build_plan(
    views_A: list[ProgressView],
    views_Anew: list[ProgressView],
    b: int,
    delta: float,
    lead: int = 0,
) -> Plan:
    """Partition existing in-flight A into scheduled / deferred under a fixed
    batch size ``b`` and candidate in-flight set A+ = A ∪ A_new.

    Forced reqs (PREFILL / WARMUP / no deadline) always take a slot and are
    excluded from the E_viol sum. New admits A_new also occupy slots (prefill)
    but are not in A, so they affect risk only through |A+| and the slot
    budget — never counted in E_viol. Remaining decode slots go to measurable
    existing reqs by largest marginal benefit M; the rest are deferred.

    KV-offload locations:
      - LOC_ONLOADING reqs behave like forced/new admits: they take a slot
        (reserved for the restore) and are excluded from E_viol, but they DO
        count in |A+| (service share).
      - LOC_CPU reqs take no slot and are absent from |A+| (they are not in
        this epoch's service race). Each measurable CPU req adds its
        R_defer_cpu to E_viol and is recorded in ``offloaded_A`` / cpu_risks.

    E_viol = Σ_{scheduled measurable} R_run + Σ_{deferred measurable} R_defer
             + Σ_{CPU measurable} R_defer_cpu.
    """
    gpu_views = [v for v in views_A if v.location == LOC_GPU]
    cpu_views = [v for v in views_A if v.location == LOC_CPU]
    onloading_views = [v for v in views_A if v.location == LOC_ONLOADING]

    # CPU reqs are out of the service race; ONLOADING reqs stay in |A+|.
    n_aplus = len(gpu_views) + len(onloading_views) + len(views_Anew)
    s = service_share(b, n_aplus)

    forced = [v for v in gpu_views if not v.is_measurable()]
    measurable = [v for v in gpu_views if v.is_measurable()]

    risks: dict[str, tuple[float, float, float]] = {}
    scored = []
    for v in measurable:
        r_run, r_defer, m = request_risk(v, delta, s)
        risks[v.request_id] = (r_run, r_defer, m)
        scored.append((m, r_run, r_defer, v))
    # Largest marginal benefit first; reduces E_viol the most per slot.
    scored.sort(key=lambda t: t[0], reverse=True)

    # Forced + ONLOADING each hold a slot and are excluded from E_viol.
    scheduled_ids = [v.request_id for v in forced]
    scheduled_ids += [v.request_id for v in onloading_views]
    decode_cap = max(
        0, b - len(forced) - len(onloading_views) - len(views_Anew))
    deferred_ids: list[str] = []
    e_viol = 0.0
    for idx, (_m, r_run, r_defer, v) in enumerate(scored):
        if idx < decode_cap:
            scheduled_ids.append(v.request_id)
            e_viol += r_run
        else:
            deferred_ids.append(v.request_id)
            e_viol += r_defer

    offloaded_ids: list[str] = []
    cpu_risks: dict[str, tuple[float, float, float]] = {}
    for v in cpu_views:
        offloaded_ids.append(v.request_id)
        if v.is_measurable():
            r_defer, r_defer_cpu, m_cpu = request_risk_cpu(v, delta, s, lead)
            e_viol += r_defer_cpu
            cpu_risks[v.request_id] = (r_defer, r_defer_cpu, m_cpu)
    return Plan(
        scheduled_A=scheduled_ids,
        deferred_A=deferred_ids,
        e_viol=e_viol,
        risks=risks,
        offloaded_A=offloaded_ids,
        cpu_risks=cpu_risks,
    )


def schedule_step(
    views_A: list[ProgressView],
    waiting_views: list[ProgressView],
    b: int,
    delta: float,
    kv_feasible: Callable[[int], bool],
    lead: int = 0,
) -> ScheduleResult:
    """Find k* = max{k : E_viol(A+(k), B) < 1} over the FCFS prefix of W.

    E_viol is monotone non-decreasing in k (more admits shrink the service
    share and the decode budget), so scan k upward and stop at the first
    infeasible k or when KV admission is exhausted. Returns the scheduled /
    deferred partition at k* and k* itself (an upper bound on admits; the
    engine's allocate_slots is the hard KV backstop).

    Also reports ``k_star_unconstrained``: had the KV cap been lifted, how far
    the E_viol scan would have reached. When the scan stopped on the KV cap
    (not on E_viol), it continues under E_viol only to compute this, and
    ``kv_capped`` becomes True — the signal the scheduler uses to trigger
    vacate.
    """
    best = build_plan(views_A, [], b, delta, lead)
    best_k = 0
    kv_break_k: int | None = None
    n = len(waiting_views)
    # If even admitting nobody is already over budget, defer-only at k=0.
    if best.e_viol < E_VIOL_FEASIBLE:
        for k in range(1, n + 1):
            if not kv_feasible(k):
                kv_break_k = k
                break
            plan = build_plan(views_A, waiting_views[:k], b, delta, lead)
            if plan.e_viol < E_VIOL_FEASIBLE:
                best, best_k = plan, k
            else:
                break
    k_star_unconstrained = best_k
    if kv_break_k is not None:
        # KV — not E_viol — stopped the scan; keep going ignoring KV.
        for k in range(kv_break_k, n + 1):
            if build_plan(
                    views_A, waiting_views[:k], b, delta,
                    lead).e_viol < E_VIOL_FEASIBLE:
                k_star_unconstrained = k
            else:
                break
    return ScheduleResult(
        scheduled_A=best.scheduled_A,
        deferred_A=best.deferred_A,
        k_star=best_k,
        e_viol=best.e_viol,
        k_star_unconstrained=k_star_unconstrained,
        kv_capped=k_star_unconstrained > best_k,
        offloaded_A=best.offloaded_A,
        cpu_risks=best.cpu_risks,
    )


def pick_adaptive_batch(
    views_A: list[ProgressView],
    base_b: int,
    base_delta: float,
    candidates: list[int],
    delta_of: Callable[[int], float | None],
    lead: int = 0,
) -> tuple[int, Plan]:
    """Choose the decode batch size that minimizes E_viol over the in-flight
    set, shrinking from ``base_b`` only when it is infeasible.

    Rationale: a smaller batch → lower per-iteration latency Δ → larger
    deadline horizon H_q → lower violation risk for the urgent few (at the
    cost of deferring more of the slack-rich rest). E_viol(b) is NOT monotone
    in b (Δ↓ helps, but decode_cap↓ and service-share↓ hurt), so this is a
    hill-climb: step down through CUDA-graph-captured sizes and stop at the
    first size where E_viol rises, returning the previous (minimizing) size.

    - Trigger: only searches when E_viol(base_b) >= E_VIOL_FEASIBLE (same
      threshold admission uses). Otherwise keeps base_b for throughput.
    - ``candidates``: captured sizes strictly below base_b, DESCENDING.
    - ``delta_of(b)``: Δ estimate for size b (None ⇒ no sample yet, skip).
    The search uses the in-flight set only (no new admits); admission is
    handled afterward by schedule_step at the chosen size. ``lead`` is passed
    through to build_plan so LOC_CPU risk (R_defer_cpu) matches schedule_step.
    """
    best_b = base_b
    best_plan = build_plan(views_A, [], base_b, base_delta, lead)
    if best_plan.e_viol < E_VIOL_FEASIBLE:
        return best_b, best_plan  # not under pressure → keep full batch
    # Forced (PREFILL/WARMUP) requests are always scheduled, so the batch
    # cannot shrink below their count — otherwise build_plan would schedule
    # more requests than the cap and the engine's len(running) <= cap assert
    # would fire.
    n_forced = sum(1 for v in views_A if not v.is_measurable())
    prev_e = best_plan.e_viol
    for b in candidates:
        if b >= base_b or b < n_forced:
            continue
        d = delta_of(b)
        if d is None or d <= 0:
            continue
        plan = build_plan(views_A, [], b, d, lead)
        if plan.e_viol > prev_e:
            break  # shrinking made it worse → stop, keep previous size
        best_b, best_plan, prev_e = b, plan, plan.e_viol
    return best_b, best_plan


def select_vacate(
    deferred_views: list[ProgressView],
    delta: float,
    s: float,
    lead: int,
    eps: float,
    blocks_needed: int,
    blocks_of: Callable[[str], int],
) -> list[str]:
    """Pick deferred GPU reqs whose KV to vacate to CPU to free ``blocks_needed``.

    Only reqs with a cheap CPU stay (M_cpu <= eps) are eligible. They are
    chosen by ascending M_cpu (cheapest first), ties broken by descending
    block count so fewer requests are moved. Selection stops once the
    cumulative freed block count reaches ``blocks_needed``. Returns [] when
    ``blocks_needed`` <= 0. ``deferred_views`` are all LOC_GPU and measurable.
    """
    if blocks_needed <= 0:
        return []
    candidates = []
    for v in deferred_views:
        _r_defer, _r_defer_cpu, m_cpu = request_risk_cpu(v, delta, s, lead)
        if m_cpu <= eps:
            candidates.append((m_cpu, blocks_of(v.request_id), v.request_id))
    # Ascending M_cpu; ties → larger block count first.
    candidates.sort(key=lambda t: (t[0], -t[1]))
    selected: list[str] = []
    acc = 0
    for _m_cpu, nblocks, rid in candidates:
        selected.append(rid)
        acc += nblocks
        if acc >= blocks_needed:
            break
    return selected


def select_promote(
    cpu_views: list[ProgressView],
    delta: float,
    s: float,
    lead: int,
    eps: float,
) -> list[str]:
    """Pick CPU-resident reqs to restore (CPU→GPU) ahead of their deadline.

    A req is promoted once its CPU stay stops being free — the lead penalty
    surfaces in the risk (M_cpu > eps) — which is exactly the deadline-aware
    prefetch point. Safety net: R_defer_cpu >= 1 (a certain miss direction)
    forces promotion regardless. Non-measurable CPU views are skipped.
    """
    promoted: list[str] = []
    for v in cpu_views:
        if v.location != LOC_CPU or not v.is_measurable():
            continue
        _r_defer, r_defer_cpu, m_cpu = request_risk_cpu(v, delta, s, lead)
        if m_cpu > eps or r_defer_cpu >= 1.0:
            promoted.append(v.request_id)
    return promoted
