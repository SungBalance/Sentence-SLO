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
#   LOC_CPU        KV offloaded to CPU — holds no decode slot; its deferred-
#                  on-CPU risk R_defer_cpu still counts toward E_viol. Whether
#                  it counts in |A+| (service share) is the caller's choice,
#                  see build_plan(share_includes_parked=...).
#   LOC_ONLOADING  onloading CPU→GPU, slot reserved for the restore. Occupies
#                  a slot like a new admit and is excluded from E_viol
#                  (forced).
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
    # > k_star). The offload trigger fires only under this condition.
    kv_capped: bool = False
    # Measured KV blocks the offload tier must free for the admits KV blocked:
    # max(0, demand of the k_star_unconstrained prefix - free_kv_blocks).
    # 0 when not kv_capped.
    kv_blocks_needed: int = 0
    offloaded_A: list[str] = field(default_factory=list)
    cpu_risks: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)


def horizon(t_q: float, delta: float) -> int:
    """H_q = max(0, floor(T_q / Δ)). Negative (past-deadline) clamps to 0."""
    if delta <= 0:
        return 0
    return max(0, math.floor(t_q / delta))


def burst_horizon(
    t_q: float, delta_dec: float, delta_burst: float, burst_steps: float
) -> int:
    """H_q when a prefill burst runs first, then the step time returns to Δ_dec.

    The burst is ``burst_steps`` = W/P steps of ``delta_burst`` = Δ_dec + κ_p·P
    each, i.e. it ends at wall(P) = burst_steps·delta_burst; after that every
    step costs Δ_dec again:

        T_q <= wall:  H_q = floor(T_q / delta_burst)
        T_q >  wall:  H_q = floor(burst_steps + (T_q - wall) / delta_dec)

    The two branches agree at T_q == wall, so H_q is continuous in T_q. The
    second branch simplifies to floor((T_q - κ_p·W) / Δ_dec) — independent of
    P: past the burst only the burst's TOTAL added time κ_p·W is felt, and that
    is fixed by the arrival W, not by the budget. Only requests whose deadline
    falls INSIDE the burst window pay for P.
    """
    if delta_burst <= 0 or delta_dec <= 0:
        return 0
    wall = burst_steps * delta_burst
    if t_q <= wall:
        return max(0, math.floor(t_q / delta_burst))
    return max(0, math.floor(burst_steps + (t_q - wall) / delta_dec))


def _deadline_horizon(
    t_q: float, delta: float, delta_burst: float | None, burst_steps: float
) -> int:
    """H_q under the plain (``delta_burst is None``) or burst step-time model."""
    if delta_burst is None:
        return horizon(t_q, delta)
    return burst_horizon(t_q, delta, delta_burst, burst_steps)


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
    view: ProgressView,
    delta: float,
    s: float,
    delta_burst: float | None = None,
    burst_steps: float = 0.0,
) -> tuple[float, float, float]:
    """(R_run, R_defer, M) for a measurable request.

    R_run = P(L > c_q + N_run | L > c_q); R_defer likewise with N_defer.
    M = R_defer - R_run >= 0 (since N_run >= N_defer and tail is
    non-increasing).

    ``delta_burst`` / ``burst_steps`` switch H_q to the burst-horizon model
    (see ``burst_horizon``), in which ``delta`` is the POST-burst Δ_dec.
    ``delta_burst is None`` (the default) keeps the plain H_q = T_q/Δ.
    """
    assert view.T_q is not None
    h_q = _deadline_horizon(view.T_q, delta, delta_burst, burst_steps)
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
    view: ProgressView,
    delta: float,
    s: float,
    lead: int,
    delta_burst: float | None = None,
    burst_steps: float = 0.0,
) -> tuple[float, float, float]:
    """(R_defer, R_defer_cpu, M_cpu) for a measurable request.

    R_defer = P(L > c_q + N_defer | L > c_q); R_defer_cpu likewise with
    N_defer_cpu. Since N_defer_cpu <= N_defer and the tail is non-increasing,
    R_defer <= R_defer_cpu, so M_cpu = R_defer_cpu - R_defer >= 0 — the risk
    penalty of the request staying on CPU (losing ``lead`` iterations).

    ``delta_burst`` / ``burst_steps`` switch H_q to the burst-horizon model,
    exactly as in ``request_risk``.
    """
    assert view.T_q is not None
    h_q = _deadline_horizon(view.T_q, delta, delta_burst, burst_steps)
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
    share_includes_parked: bool = True,
    delta_burst: float | None = None,
    burst_steps: float = 0.0,
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
      - LOC_CPU reqs take no slot (decode_cap is unaffected). Each measurable
        CPU req adds its R_defer_cpu to E_viol and is recorded in
        ``offloaded_A`` / cpu_risks. ``share_includes_parked`` controls only
        whether they count in |A+|: True (default) keeps them in the service
        share because offloading frees memory, not compute — they come back and
        reclaim their share.

    E_viol = Σ_{scheduled measurable} R_run + Σ_{deferred measurable} R_defer
             + Σ_{CPU measurable} R_defer_cpu.

    ``delta_burst`` / ``burst_steps`` are passed straight through to
    ``request_risk`` / ``request_risk_cpu``, so the whole ledger is evaluated
    against the burst-horizon step-time model instead of a constant Δ. They
    change only H_q — the service share, the slot budget and the assignment
    rule are unaffected.
    """
    gpu_views = [v for v in views_A if v.location == LOC_GPU]
    cpu_views = [v for v in views_A if v.location == LOC_CPU]
    onloading_views = [v for v in views_A if v.location == LOC_ONLOADING]

    # ONLOADING reqs always stay in |A+|; CPU reqs only when the caller keeps
    # parked requests in the service race. This affects s alone — parked reqs
    # still hold no decode slot and still enter E_viol as R_defer_cpu.
    n_aplus = len(gpu_views) + len(onloading_views) + len(views_Anew)
    if share_includes_parked:
        n_aplus += len(cpu_views)
    s = service_share(b, n_aplus)

    forced = [v for v in gpu_views if not v.is_measurable()]
    measurable = [v for v in gpu_views if v.is_measurable()]

    risks: dict[str, tuple[float, float, float]] = {}
    scored = []
    for v in measurable:
        r_run, r_defer, m = request_risk(v, delta, s, delta_burst, burst_steps)
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
            r_defer, r_defer_cpu, m_cpu = request_risk_cpu(
                v, delta, s, lead, delta_burst, burst_steps)
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
    free_kv_blocks: int,
    blocks_of: Callable[[str], int],
    lead: int = 0,
    share_includes_parked: bool = True,
    admission_delta_criterion: bool = False,
) -> ScheduleResult:
    """Find k* = max{k : E_viol(A+(k), B) < 1} over the FCFS prefix of W.

    E_viol is monotone non-decreasing in k (more admits shrink the service
    share and the decode budget), so scan k upward and stop at the first
    infeasible k or when KV admission is exhausted. Returns the scheduled /
    deferred partition at k* and k* itself (an upper bound on admits; the
    engine's allocate_slots is the hard KV backstop).

    KV feasibility is judged on the MEASURED block demand of the candidates
    themselves: ``blocks_of(request_id)`` = ceil(remaining prompt tokens /
    block_size) (injected like ``select_offload``'s ``blocks_of`` so this
    module stays engine-import-free), and admitting k is feasible iff
    Σ_{i<k} blocks(q_i) <= ``free_kv_blocks``. A per-admit CONSTANT was used
    until 2026-08-13 and underestimated the real demand ~18x, so the scan
    never stopped on KV and ``kv_capped`` — the offload tier's trigger — was
    permanently False even at 99% KV occupancy (paper/scheduling.md §3③).

    With ``admission_delta_criterion`` the feasibility test becomes the
    MARGINAL one, E_viol(k) - E_viol(0) < 1: the risk the in-flight set
    already carries is sunk cost, so the budget applies only to the extra
    expected violations this step's admissions cause. This also removes the
    k=0 early-exit (E_viol(0) - E_viol(0) = 0 is always feasible), which is
    the point — under the absolute rule a handful of at-risk in-flight
    requests can pin k* to 0 while KV and compute sit idle.
    ``ScheduleResult.e_viol`` stays the ABSOLUTE E_viol at k* either way; the
    delta is used for the admission decision only.

    Also reports ``k_star_unconstrained``: had the KV cap been lifted, how far
    the E_viol scan would have reached. When the scan stopped on the KV cap
    (not on E_viol), it continues under E_viol only to compute this, and
    ``kv_capped`` becomes True — the signal the scheduler uses to trigger
    offload.
    """
    best = build_plan(views_A, [], b, delta, lead, share_includes_parked)
    best_k = 0
    kv_break_k: int | None = None
    n = len(waiting_views)
    # cum_blocks[k-1] = blocks the first k candidates need together.
    cum_blocks: list[int] = []
    acc = 0
    for v in waiting_views:
        acc += blocks_of(v.request_id)
        cum_blocks.append(acc)

    def kv_feasible(k: int) -> bool:
        return cum_blocks[k - 1] <= free_kv_blocks

    baseline_e = best.e_viol if admission_delta_criterion else 0.0
    # If even admitting nobody is already over budget, defer-only at k=0.
    if best.e_viol - baseline_e < E_VIOL_FEASIBLE:
        for k in range(1, n + 1):
            if not kv_feasible(k):
                kv_break_k = k
                break
            plan = build_plan(views_A, waiting_views[:k], b, delta, lead,
                              share_includes_parked)
            if plan.e_viol - baseline_e < E_VIOL_FEASIBLE:
                best, best_k = plan, k
            else:
                break
    k_star_unconstrained = best_k
    if kv_break_k is not None:
        # KV — not E_viol — stopped the scan; keep going ignoring KV.
        for k in range(kv_break_k, n + 1):
            if build_plan(
                    views_A, waiting_views[:k], b, delta, lead,
                    share_includes_parked).e_viol - baseline_e < (
                        E_VIOL_FEASIBLE):
                k_star_unconstrained = k
            else:
                break
    # What the offload tier has to free: the DEFICIT between the demand of the
    # whole k*_unconstrained prefix and the blocks already free — "only as many
    # blocks as the admission needs" (spec §3④). Charging the full incremental
    # demand instead would ignore the free space left over at k* and vacate up
    # to one extra pending request every capped step.
    blocks_needed = 0
    if k_star_unconstrained > best_k:
        blocks_needed = max(
            0, cum_blocks[k_star_unconstrained - 1] - free_kv_blocks)
    return ScheduleResult(
        scheduled_A=best.scheduled_A,
        deferred_A=best.deferred_A,
        k_star=best_k,
        e_viol=best.e_viol,
        k_star_unconstrained=k_star_unconstrained,
        kv_capped=k_star_unconstrained > best_k,
        kv_blocks_needed=blocks_needed,
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
    share_includes_parked: bool = True,
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
    handled afterward by schedule_step at the chosen size. ``lead`` and
    ``share_includes_parked`` are passed through to build_plan so LOC_CPU risk
    (R_defer_cpu) and |A+| match schedule_step.
    """
    best_b = base_b
    best_plan = build_plan(views_A, [], base_b, base_delta, lead,
                           share_includes_parked)
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
        plan = build_plan(views_A, [], b, d, lead, share_includes_parked)
        if plan.e_viol > prev_e:
            break  # shrinking made it worse → stop, keep previous size
        best_b, best_plan, prev_e = b, plan, plan.e_viol
    return best_b, best_plan


def token_budget_prefill_risk(
    views_A: list[ProgressView],
    b: int,
    delta_dec: float,
    kappa_p: float,
    base_budget: int,
    floor: int,
    eps_p: float,
    prefill_work: int,
    lead: int = 0,
    share_includes_parked: bool = True,
) -> int:
    """Per-step prefill token budget TB*_pre (P axis), dosed on expected risk.

    ``prefill_work`` = W, this burst's remaining prefill work in tokens (the
    chunked-prefill carry-over plus the prompts this step may admit). Choosing
    P spreads W over

        n_b(P)   = W / P                    burst steps
        Δ_b(P)   = delta_dec + kappa_p·P    seconds each
        wall(P)  = n_b·Δ_b = (W/P)·Δ_dec + kappa_p·W

    after which the step time returns to ``delta_dec`` (``burst_horizon``):

        E(P)    = build_plan(..., delta_burst=Δ_b(P), burst_steps=n_b(P)).e_viol
        TB*_pre = max{ P ∈ [floor, base_budget] : E(P) - E(floor) <= eps_p }

    i.e. buy prefill tokens until the in-flight set's expected violation count
    has risen by ``eps_p`` over what the SMALLEST allowed budget would already
    cost — a marginal budget in the same currency the admission rule spends
    (E_viol).

    Why the reference is E(floor) and not a burst-free E: the floor is granted
    unconditionally, so it is the only always-reachable operating point, and
    every P pays the same unavoidable share of the burst. Referencing E(floor)
    cancels that share out. A burst-free reference would instead charge the
    sunk kappa_p·W to every candidate, and once that fixed offset alone exceeds
    ``eps_p`` NO P clears the test and the rule pins to the floor forever —
    the same floor-residency failure this axis exists to remove.

    What P actually buys: the burst's TOTAL added time kappa_p·W is sunk (it is
    fixed by the arrival W, not by the budget), so past wall(P) the horizon is
    P-independent. P therefore sets the burst's CONCENTRATION, not its cost —
    only requests whose deadline falls INSIDE the burst window are priced for
    it. Two consequences: a far-deadline request contributes the same E(P) at
    every P, so it costs exactly 0 against the E(floor) reference, and a doomed
    one (R ≈ 1 either way) barely moves E(P); neither holds prefill hostage.
    This replaces the recurring-price rule (2026-08-11), which applied
    Δ(P) = Δ_dec + kappa_p·P to EVERY step until each deadline and so charged a
    one-step cost as a permanent slowdown (~20x overprice; floor residency
    43-83%, -23% throughput — see paper/scheduling.md §6).

    Units: ``delta_dec`` seconds per decode-only step at the CURRENT decode set
    size (the run/defer split's outcome), ``kappa_p`` SECONDS of
    extra step time per prefill token (``_sslo_kappa`` is a ratio of a seconds
    EMA to a token EMA, so Δ_b(P) stays in seconds), ``base_budget`` / ``floor``
    / ``prefill_work`` tokens, ``eps_p`` in expected-violation units.

    E(P) is non-decreasing in P: H_q is floor(T_q/Δ_b) inside the burst window
    (falling in P) and P-independent outside it, and wall(P) shrinks with P so
    requests only ever move from the first branch to the second — where the two
    agree at the crossing. Every H_q is therefore non-increasing in P, hence
    every N_run / N_defer, hence every (non-increasing) tail probability is
    non-decreasing. So the largest feasible P is found by bisection: 2 +
    ceil(log2(base_budget - floor)) build_plan calls (15 over the default
    512..8192 range) — small next to the up-to-k*+1 calls schedule_step already
    spends per step. P = floor scores 0 by construction, so the search always
    has a feasible point and never has to test one.

    ``floor`` is clamped into [1, ``base_budget``] and the result is always in
    [floor, base_budget]: the floor is granted unconditionally so prefill keeps
    making progress even when no P is affordable, and it must stay >= 1 both
    because a zero dose would stall prefill outright and because it is the
    reference point E(floor) — a burst of W/0 steps is undefined. Control is
    disabled — returns
    ``base_budget`` — when ``kappa_p <= 0`` (no usable estimate), when
    ``prefill_work <= 0`` (no burst to price) or when nothing in ``views_A`` is
    measurable (no risk ledger to protect).
    """
    floor = max(1, min(floor, base_budget))
    if kappa_p <= 0:
        return base_budget
    if prefill_work <= 0:
        return base_budget
    if not any(v.is_measurable() for v in views_A):
        return base_budget

    def e_viol(p: int) -> float:
        return build_plan(views_A, [], b, delta_dec, lead,
                          share_includes_parked,
                          delta_burst=delta_dec + kappa_p * p,
                          burst_steps=prefill_work / p).e_viol

    baseline = e_viol(floor)
    if e_viol(base_budget) - baseline <= eps_p:
        return base_budget
    # Invariant: `hi` is known over budget, `lo` is the largest P known within
    # it — starting at the floor, which scores 0 against itself.
    lo, hi = floor, base_budget
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if e_viol(mid) - baseline <= eps_p:
            lo = mid
        else:
            hi = mid
    return lo


def token_budget_prefill(
    t_min: float | None,
    delta_decode: float,
    kappa: float,
    base_budget: int,
    floor: int,
    gamma: float,
) -> int:
    """DEPRECATED (2026-08-11) — the worst-case P axis, kept for A/B replay.

    Rejected: γ·t_min prices the whole step off the single most urgent
    in-flight request, so in a recovery burst one near-deadline survivor (often
    one that cannot be saved at all) pinned TB*_pre to the floor on 81% of
    steps and stretched the refill ~4x (WORKLOG 2026-08-11).
    ``token_budget_prefill_risk`` replaces it — same clamp, but dosed on the
    E_viol ledger, where a doomed request's marginal contribution is ~0. No
    live caller; kept so the two rules stay replayable against each other.

    Deadline-aware per-step prefill token budget TB*_pre (P axis).

    Rationale (measured, cap128 / rate 2 baseline): decode-only steps are
    extremely stable (Δ p50 74 ms / p90 78 ms), while the 6% of steps that
    carry prefill eat 18.4% of the wall clock at Δ p90 = 463 ms. Chunk
    deadlines are broken by the prefill spike, not by decode concurrency, so
    the spike — not the request concurrency — is what to bound.

        TB*_pre = clamp(int((gamma * t_min - delta_decode) / kappa),
                        floor, base_budget)

    i.e. spend at most a ``gamma`` fraction of the most urgent in-flight
    request's remaining time on this step, minus the decode cost that step
    already owes.

    Units: ``t_min`` seconds to the nearest in-flight chunk deadline (None ⇒
    no measurable request in flight), ``delta_decode`` seconds per decode-only
    step at the current decode set size,
    ``kappa`` seconds of extra step time per prefill token,
    ``base_budget`` / ``floor`` tokens, ``gamma`` dimensionless in (0, 1].

    ``floor`` is itself clamped to ``base_budget`` so the result never exceeds
    what the engine would schedule anyway. For the control to have any effect,
    configure ``floor`` well below the engine's ``max_num_batched_tokens``
    (which is what the caller passes as ``base_budget``) — a floor at or above
    it makes every step return ``base_budget``, i.e. a silent no-op.

    Control is disabled — returns ``base_budget`` — when ``kappa <= 0`` or
    ``t_min is None``. An already-overdue ``t_min <= 0`` returns ``floor``.
    """
    floor = min(floor, base_budget)
    if kappa <= 0 or t_min is None:
        return base_budget
    if t_min <= 0:
        return floor
    p = int((gamma * t_min - delta_decode) / kappa)
    return max(floor, min(base_budget, p))


def select_offload(
    deferred_views: list[ProgressView],
    delta: float,
    s: float,
    lead: int,
    eps: float,
    blocks_needed: int,
    blocks_of: Callable[[str], int],
) -> list[str]:
    """Pick deferred GPU reqs whose KV to offload to CPU to free ``blocks_needed``.

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


def select_onload(
    cpu_views: list[ProgressView],
    delta: float,
    s: float,
    lead: int,
    eps: float,
) -> list[str]:
    """Pick CPU-resident reqs to restore (CPU→GPU) ahead of their deadline.

    A req is onloaded once its CPU stay stops being free — the lead penalty
    surfaces in the risk (M_cpu > eps) — which is exactly the deadline-aware
    prefetch point. Safety net: R_defer_cpu >= 1 (a certain miss direction)
    forces onload regardless. Non-measurable CPU views are skipped.
    """
    onloaded: list[str] = []
    for v in cpu_views:
        if v.location != LOC_CPU or not v.is_measurable():
            continue
        _r_defer, r_defer_cpu, m_cpu = request_risk_cpu(v, delta, s, lead)
        if m_cpu > eps or r_defer_cpu >= 1.0:
            onloaded.append(v.request_id)
    return onloaded
