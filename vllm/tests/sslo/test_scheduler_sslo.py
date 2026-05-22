# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO scheduler policy helpers."""

from types import SimpleNamespace

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import Phase, RequestSLOState
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler, SsloStepState
from vllm.v1.core.sched.output import SchedulerOutput


def make_state(
    *,
    phase=Phase.MEASURED,
    deadline=10.0,
    expected_len=10.0,
    generated=0,
) -> RequestSLOState:
    state = RequestSLOState(num_warmup_chunks=1)
    # phase is derived from decoding_start_ts + chunks_completed; set the
    # underlying state to land in the requested phase.
    if phase == Phase.PREFILL:
        state.decoding_start_ts = None
        state.next_deadline_ts = None
    else:
        state.decoding_start_ts = 0.0
        # next_deadline_ts is absolute; with decoding_start_ts=0.0,
        # passing `deadline=10` means deadline_ts = 10.
        state.next_deadline_ts = float(deadline)
        if phase == Phase.MEASURED:
            state.chunks_completed = max(state.num_warmup_chunks, 1)
    state._chunk_len_predictor.value = expected_len
    state.current_chunk_generated_len = generated
    return state


def make_request(request_id: str, state: RequestSLOState):
    return SimpleNamespace(
        request_id=request_id,
        slo_state=state,
        status=None,
        has_encoder_inputs=False,
    )


class FakeKVCacheManager:

    empty_kv_cache_blocks = None
    freed = None

    def __init__(self):
        # SSLO: mlp_kv_blocks_per_new_admit reads
        # block_pool.get_num_free_blocks(). Mock with a generous pool so
        # the KV cap doesn't artificially throttle test admission.
        self.block_pool = SimpleNamespace(
            get_num_free_blocks=lambda: 1_000_000)

    def new_step_starts(self):
        pass

    def take_new_block_ids(self):
        return []

    def get_num_common_prefix_blocks(self, request_id):
        return []

    def get_blocks(self, request_id):
        return SimpleNamespace(
            get_block_ids=lambda allow_none=False: ([1, 2, 3], ))

    def free(self, request):
        self.freed = request.request_id


class FakeEncoderCacheManager:

    def get_freed_mm_hashes(self):
        return []

    def free(self, request):
        pass


def make_scheduler(
    *,
    running=None,
    pending=None,
    max_num_running_reqs=4,
    cfg=None,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running = list(running or [])
    scheduler.sslo_pending = list(pending or [])
    scheduler.requests = {
        req.request_id: req
        for req in scheduler.running + scheduler.sslo_pending
    }
    # Default fixture: SsloConfig with hysteresis placement (in/out
    # thresholds 0.3 / 0.7). Pass `cfg=...` to override.
    scheduler.sslo_config = cfg or SsloConfig(method="sslo")
    scheduler.tpot_ema = {max_num_running_reqs: 1.0}
    # Pre-seed wall-step EMA so _sslo_step_ema_lookup returns 1.0 via the
    # global-average fallback for any batch_size. Tests that want
    # score-suspend (no EMA) override this to {}.
    scheduler._sslo_step_wall_ema = {max_num_running_reqs: {0: 1.0}}
    scheduler._sslo_prev_step_num_prefills = 0
    scheduler._sslo_prev_step_batch = None
    scheduler._sslo_prev_step_decoding_only = False
    scheduler._sslo_prev_step_start_ts = None
    scheduler._sslo_step = SsloStepState(cur_max_num_requests=max_num_running_reqs)
    scheduler._sslo_done_logged = set()
    scheduler._sslo_log_dir_created = False
    # SSLO: decision-log state normally seeded by Scheduler.__init__.
    scheduler._sslo_prev_tier = {}
    scheduler._sslo_prev_selected = {}
    scheduler._sslo_step_idx = 0
    scheduler._sslo_decision_buffer = []
    scheduler._sslo_decision_buffer_max = 256
    scheduler._sslo_decision_log_path = None
    scheduler._sslo_decision_log_dir_created = False
    scheduler.max_num_running_reqs = max_num_running_reqs
    # Tests treat every multiple of 8 ≤ max_num_running_reqs as a captured
    # CUDA graph size. Override per-test where needed.
    capture_sizes = tuple(
        n for n in range(8, max_num_running_reqs + 1, 8))
    scheduler._sslo_cudagraph_sizes = frozenset(capture_sizes)
    scheduler._sslo_capture_sizes_below_base = tuple(
        sorted((n for n in capture_sizes if n < max_num_running_reqs),
               reverse=True))
    scheduler.max_num_scheduled_tokens = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.kv_cache_manager = FakeKVCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.lora_config = None
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.use_v2_model_runner = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.use_pp = False
    scheduler.log_stats = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler._update_after_schedule = lambda output: None
    # SSLO: __init__ normally patches _apply_sslo_policy based on
    # (method, policy); tests bypass __init__ via __new__, so do it here.
    scheduler._apply_sslo_policy = scheduler._resolve_sslo_policy_dispatch()
    return scheduler


def request_ids(reqs):
    return [req.request_id for req in reqs]


def test_tpot_ema_update_only_multiple_bucket_and_decoding_only():
    scheduler = make_scheduler(max_num_running_reqs=8)
    scheduler.sslo_config = SsloConfig(method="sslo", tpot_ema_alpha=0.5)
    scheduler._sslo_prev_step_batch = 7
    scheduler._sslo_prev_step_decoding_only = True
    scheduler._sslo_prev_step_start_ts = 1.0
    scheduler._update_tpot_ema(3.0)
    assert scheduler.tpot_ema == {8: 1.0}

    scheduler._sslo_prev_step_batch = 8
    scheduler._sslo_prev_step_decoding_only = False
    scheduler._update_tpot_ema(3.0)
    assert scheduler.tpot_ema == {8: 1.0}

    scheduler._sslo_prev_step_decoding_only = True
    scheduler._update_tpot_ema(3.0)
    assert scheduler.tpot_ema[8] == pytest.approx(1.5)


def test_record_step_decoding_only_uses_prompt_progress():
    # Pre-step snapshot: req was already past prompt → step is decoding-only.
    req = make_request("r", make_state())
    req.num_computed_tokens = 11
    req.num_prompt_tokens = 10
    scheduler = make_scheduler(running=[req])
    scheduler.requests = {"r": req}
    scheduler._sslo_pre_step_computed = {"r": 10}  # snapshot at step start
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"r": 1}
    output.total_num_scheduled_tokens = 1

    scheduler._record_step_for_next_ema(output, 2.0)

    assert scheduler._sslo_prev_step_batch == 1
    assert scheduler._sslo_prev_step_decoding_only is True
    assert scheduler._sslo_prev_step_start_ts == pytest.approx(2.0)


def test_record_step_chunked_prefill_not_decoding_only():
    # Pre-step: req still in prefill (computed < prompt) → not decoding-only.
    req = make_request("r", make_state())
    req.num_computed_tokens = 8
    req.num_prompt_tokens = 10
    scheduler = make_scheduler(running=[req])
    scheduler.requests = {"r": req}
    scheduler._sslo_pre_step_computed = {"r": 5}  # 5 < 10 → prefill at step start
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"r": 3}
    output.total_num_scheduled_tokens = 3

    scheduler._record_step_for_next_ema(output, 2.0)

    assert scheduler._sslo_prev_step_decoding_only is False


def test_critical_mode_no_waiting_admission():
    critical = make_request("critical", make_state(deadline=1.0, expected_len=10))
    scheduler = make_scheduler(running=[critical], max_num_running_reqs=1)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is True


def test_critical_mode_priority_fill_cap():
    a = make_request("a", make_state(deadline=10.0, expected_len=20))
    b = make_request("b", make_state(deadline=10.0, expected_len=30))
    c = make_request("c", make_state(deadline=10.0, expected_len=5))
    scheduler = make_scheduler(running=[a, b, c], max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert request_ids(scheduler.running) == ["b", "a"]
    assert request_ids(scheduler.sslo_pending) == ["c"]


def test_non_critical_warmup_always_running():
    warm = make_request("warm", make_state(phase=Phase.WARMUP))
    measured = make_request("measured", make_state(deadline=100, expected_len=1))
    scheduler = make_scheduler(pending=[warm, measured], max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert warm in scheduler.running


def test_non_critical_pending_to_running_at_07():
    req = make_request("pending", make_state(deadline=10, expected_len=7))
    scheduler = make_scheduler(pending=[req], max_num_running_reqs=1)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler.running == [req]


def test_non_critical_running_to_pending_at_03():
    # Demoted req stays in pending only when admission claims the backfill
    # slot. Under the new chunk-SLO admission formula
    # (admission_capacity = cap - Σ pressures), a 1-slot system with a
    # single low-pressure req has Σ pressures ≈ 0.2 < cap=1, so the budget
    # is min(waiting, int(1 - 0.2), slack) = min(1, 0, 1) = 0 → no slack
    # consumed by admission → backfill pulls the req back to running. To
    # exercise the "stays-in-pending" branch we widen cap so admission
    # actually fits (cap=2, two waiting reqs, single demoted req).
    reqs = [
        make_request(f"r{i}", make_state(deadline=10, expected_len=2))
        for i in range(2)
    ]
    scheduler = make_scheduler(running=reqs, max_num_running_reqs=2)
    scheduler.waiting = [object(), object()]

    scheduler._apply_sslo_policy(0.0)

    # All reqs at pressure 0.2 (≤ in_thr=0.3): classified to pending.
    # Σ pressures = 0.4, cap=2 → admission_capacity = int(2-0.4) = 1, slack
    # = 2 → budget=min(2, 1, 2)=1. backfill = slack - budget = 1, so one
    # req returns to running, the other stays pending.
    assert len(scheduler.sslo_pending) == 1
    assert len(scheduler.running) == 1


def test_non_critical_hysteresis_keeps_state_in_band():
    # Both reqs have pressure 0.5 (in band [0.3, 0.7]). With running at cap,
    # backfill has no slack, so hysteresis governs placement: each stays
    # where it was. The default config now collapses the band to a single
    # point (0.8/0.8), so this test pins the original thresholds.
    cfg = SsloConfig(
        method="sslo", pending_in_threshold=0.3, pending_out_threshold=0.7)
    pending = make_request("pending", make_state(deadline=10, expected_len=5))
    running = make_request("running", make_state(deadline=10, expected_len=5))
    scheduler = make_scheduler(running=[running], pending=[pending],
                               max_num_running_reqs=1, cfg=cfg)

    scheduler._apply_sslo_policy(0.0)

    assert running in scheduler.running
    assert pending in scheduler.sslo_pending


def test_non_critical_pending_backfills_when_waiting_empty():
    # No waiting → pending request that's normally held in pending gets
    # backfilled into running so this step's batch hits cap.
    pending = make_request("pending", make_state(deadline=10, expected_len=5))
    scheduler = make_scheduler(pending=[pending], max_num_running_reqs=1)
    # Waiting empty (default) → backfill_budget == slack == 1.

    scheduler._apply_sslo_policy(0.0)

    assert pending in scheduler.running
    assert scheduler.sslo_pending == []


def test_cap_overflow_priority_sort():
    # Pin thresholds so this test exercises the overflow-sort branch
    # (cands_running > cap → priority sort) rather than the exact-edge
    # case where one request hits in_threshold and gets demoted before
    # overflow occurs.
    cfg = SsloConfig(
        method="sslo", pending_in_threshold=0.3, pending_out_threshold=0.7)
    low = make_request("low", make_state(deadline=10, expected_len=4))
    high = make_request("high", make_state(deadline=10, expected_len=6))
    warm = make_request("warm", make_state(phase=Phase.WARMUP))
    scheduler = make_scheduler(running=[low, high, warm], max_num_running_reqs=2,
                               cfg=cfg)

    scheduler._apply_sslo_policy(0.0)

    assert request_ids(scheduler.running) == ["warm", "high"]
    assert request_ids(scheduler.sslo_pending) == ["low"]


def test_score_suspend_when_tpot_unobserved():
    # Score-suspend keeps the pressure-based hysteresis off, but backfill still
    # runs to keep batch at cap (the "non-adaptive batch is constant" rule).
    # waiting is non-empty here to block backfill so we can verify the
    # placement-preserving aspect of suspend mode.
    running = make_request("running", make_state(deadline=10, expected_len=2))
    pending = make_request("pending", make_state(deadline=10, expected_len=9))
    scheduler = make_scheduler(running=[running], pending=[pending],
                               max_num_running_reqs=2)
    scheduler.tpot_ema = {}
    scheduler._sslo_step_wall_ema = {}  # no step observed yet → suspend  # no step observed yet
    scheduler.waiting = [object()]  # block backfill

    scheduler._apply_sslo_policy(0.0)

    assert scheduler.running == [running]
    assert scheduler.sslo_pending == [pending]


def test_no_waiting_backfills_pending_by_priority():
    high = make_request("high", make_state(deadline=10, expected_len=6))
    low = make_request("low", make_state(deadline=10, expected_len=4))
    scheduler = make_scheduler(pending=[low, high], max_num_running_reqs=1)

    scheduler.schedule_sslo()

    assert scheduler.running == [high]
    assert scheduler.sslo_pending == [low]


def test_adaptive_n_disabled_uses_base():
    req = make_request("critical", make_state(deadline=1, expected_len=10))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=8)
    scheduler.tpot_ema = {8: 1.0, 16: 0.2}

    scheduler.schedule_sslo()

    assert len(scheduler.running) == 1


def test_adaptive_n_picks_largest_resolving_critical():
    # Pick the LARGEST profiled bucket below base_n that resolves all
    # critical. base_n=32, ema covers {8, 16, 24}. Score at each:
    # remaining*tpot / time_to_deadline = 2*tpot / 1.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=32, cfg=cfg)
    scheduler.tpot_ema = {32: 1.0, 24: 0.2, 16: 0.4, 8: 0.8}
    tiers = {"critical": 0}

    # 24 has tpot=0.2 → pressure = 2*0.2/1 = 0.4 < 1.0 → resolves. Largest
    # profiled bucket < base_n that resolves: 24.
    assert scheduler._pick_adaptive_n([req], tiers, 0.0, 1.0) == 24


def test_adaptive_n_minimizes_worst_score_when_unresolvable():
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=20))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=32, cfg=cfg)
    # No bucket can resolve (every pressure ≥ 1.0). Pick the one with the
    # smallest worst-case pressure (lowest tpot).
    scheduler.tpot_ema = {32: 1.0, 24: 0.6, 16: 0.8}

    assert scheduler._pick_adaptive_n(
        [req], {"critical": 0}, 0.0, 1.0) == 24


def test_adaptive_n_respects_throughput_floor():
    # Profiled bucket 16 has throughput 16/100 = 0.16 vs base 32/1.0 = 32.
    # Ratio 0.005 < 0.9 min_throughput_ratio → bucket 16 excluded from
    # candidates. No other profiled bucket below base → returns None.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=32, cfg=cfg)
    scheduler.tpot_ema = {32: 1.0, 16: 100.0}

    assert scheduler._pick_adaptive_n(
        [req], {"critical": 0}, 0.0, 1.0) is None


def test_adaptive_n_at_least_num_critical():
    # 10 critical requests → n_critical=10 floors the cap. Profiled
    # buckets {8, 16, 24} below base_n=32; 8 < 10 (floor) so excluded.
    # Largest remaining that resolves: 24 has tpot=0.2 → pressure=0.4 < 1.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    reqs = [
        make_request(f"r{i}", make_state(deadline=1, expected_len=2))
        for i in range(10)
    ]
    scheduler = make_scheduler(running=reqs, max_num_running_reqs=32, cfg=cfg)
    scheduler.tpot_ema = {32: 1.0, 8: 0.5, 16: 0.4, 24: 0.2}
    tiers = {req.request_id: 0 for req in reqs}

    assert scheduler._pick_adaptive_n(reqs, tiers, 0.0, 1.0) == 24


def test_adaptive_n_returns_none_when_only_base_profiled():
    # No sub-base profile data → cascading fallback finds nothing →
    # returns None (caller falls through to base_n cap).
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=64, cfg=cfg)
    scheduler.tpot_ema = {64: 1.0}
    tiers = {"critical": 0}

    assert scheduler._pick_adaptive_n([req], tiers, 0.0, 1.0) is None


def test_adaptive_n_cascades_through_unprofiled_buckets():
    # 8-step-down cascade: base_n=64. Only 24 and 48 are profiled below
    # base. 56, 40, 32, 16 unprofiled — must skip past them and pick the
    # largest profiled that resolves: 48.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=64, cfg=cfg)
    scheduler.tpot_ema = {64: 1.0, 48: 0.4, 24: 0.2}
    tiers = {"critical": 0}

    assert scheduler._pick_adaptive_n([req], tiers, 0.0, 1.0) == 48


def test_adaptive_n_low_cap_throughput_floor():
    # Bucket 16 has throughput 16/2.0 = 8 vs base 64/1.0 = 64. Ratio
    # 0.125 < 0.25 low_cap_ratio → low_cap pushes the floor above 16.
    # Bucket 32 has throughput 32/0.6 ≈ 53.3 vs base 64 → ratio 0.83 ≥
    # 0.25, so low_cap = 32. Adaptive must not pick anything below 32.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=64, cfg=cfg)
    scheduler.tpot_ema = {64: 1.0, 32: 0.6, 16: 2.0}
    tiers = {"critical": 0}

    picked = scheduler._pick_adaptive_n([req], tiers, 0.0, 1.0)
    assert picked is None or picked >= 32


def test_cur_max_num_requests_resets_each_step():
    # First step shrinks the cap; the second step starts from a freshly
    # reset SsloStepState (cur_max_num_requests = max_num_running_reqs) before the
    # policy logic runs.
    cfg = SsloConfig(method="sslo", adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=32, cfg=cfg)
    scheduler.tpot_ema = {32: 1.0, 24: 0.2}

    scheduler._apply_sslo_policy(0.0)
    assert scheduler._sslo_step.cur_max_num_requests <= 32

    # Force the cap to a known shrunk value, then re-enter policy. The
    # SsloStepState reset at the top of _apply_sslo_policy must wipe the
    # previous step's adapted cap before re-deciding.
    scheduler._sslo_step.cur_max_num_requests = 8  # stale value from prior step
    scheduler.tpot_ema = {32: 1.0}  # no sub-base data → adaptive returns None
    scheduler._apply_sslo_policy(1.0)
    assert scheduler._sslo_step.cur_max_num_requests == 32


# ---------------------------------------------------------------------------
# v2 scheduling path tests
# ---------------------------------------------------------------------------

def _make_pressure_scheduler(**kwargs):
    """Helper: make_scheduler pre-configured with policy="pressure"."""
    cfg = kwargs.pop("cfg", None) or SsloConfig(
        method="sslo", policy="pressure")
    return make_scheduler(cfg=cfg, **kwargs)


def test_v2_critical_mode_top_cap_to_running():
    # 3 admitted reqs, M=2, one with pressure >= 1 → critical mode.
    # Top-2 by pressure go to running, rest pending, waiting_budget=0.
    hi = make_request("hi", make_state(deadline=1.0, expected_len=10.0))   # pressure >= 1
    mid = make_request("mid", make_state(deadline=10.0, expected_len=20.0))
    lo = make_request("lo", make_state(deadline=10.0, expected_len=5.0))
    scheduler = _make_pressure_scheduler(running=[hi, mid, lo], max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is True
    assert scheduler._sslo_step.waiting_admission_budget == 0
    assert len(scheduler.running) == 2
    # hi must be in running (highest pressure)
    assert any(r.request_id == "hi" for r in scheduler.running)
    # lo is NOT in running (sorted by _admit_priority_key — hi+mid have higher pressure)
    assert len(scheduler.sslo_pending) == 1


def test_v2_non_critical_demand_room_admission():
    # 4 reqs at p=0.3 (D=1.2), M=4, waiting=10.
    # waiting_budget = max(0, int(4 - 1.2)) = 2, cands_running = M - 2 = 2.
    reqs = [
        make_request(f"r{i}", make_state(deadline=10.0, expected_len=3.0))  # p≈0.3
        for i in range(4)
    ]
    scheduler = _make_pressure_scheduler(running=reqs, max_num_running_reqs=4)
    scheduler.waiting = [object()] * 10

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is False
    assert scheduler._sslo_step.waiting_admission_budget == 2
    assert len(scheduler.running) == 2
    assert len(scheduler.sslo_pending) == 2


def test_v2_warmup_contributes_one_to_demand_and_sorts_top():
    # 2 warmup + 2 measured(p≈0.3), M=4, waiting=10.
    # D = 2*1 + 2*0.3 = 2.6, waiting_budget = int(4-2.6) = 1.
    # cands_running = 4-1 = 3 → 2 warmup + 1 measured (top by _admit_priority_key).
    warmups = [
        make_request(f"w{i}", make_state(phase=Phase.WARMUP))
        for i in range(2)
    ]
    measured = [
        make_request(f"m{i}", make_state(deadline=10.0, expected_len=3.0))
        for i in range(2)
    ]
    scheduler = _make_pressure_scheduler(
        running=warmups + measured, max_num_running_reqs=4)
    scheduler.waiting = [object()] * 10

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is False
    assert scheduler._sslo_step.waiting_admission_budget == 1
    assert len(scheduler.running) == 3
    assert len(scheduler.sslo_pending) == 1
    # Both warmup reqs must be in running (highest priority via _admit_priority_key)
    running_ids = {r.request_id for r in scheduler.running}
    assert "w0" in running_ids and "w1" in running_ids


def test_v2_score_suspend_keeps_state_when_no_tpot():
    # When _sslo_step_wall_ema is empty, policy suspends — running/pending unchanged.
    # waiting_budget = M - len(running).
    r1 = make_request("r1", make_state())
    r2 = make_request("r2", make_state())
    p1 = make_request("p1", make_state())
    scheduler = _make_pressure_scheduler(
        running=[r1, r2], pending=[p1], max_num_running_reqs=4)
    scheduler._sslo_step_wall_ema = {}  # no step observed yet → suspend

    scheduler._apply_sslo_policy(0.0)

    assert scheduler.running == [r1, r2]
    assert scheduler.sslo_pending == [p1]
    # M - len(running) = 4 - 2 = 2; waiting=0 so budget = min(0, 2) = 0.
    assert scheduler._sslo_step.waiting_admission_budget == 0


def test_v2_score_suspend_waiting_budget_limited_by_waiting_count():
    # Same as above but with actual waiting requests present.
    r1 = make_request("r1", make_state())
    scheduler = _make_pressure_scheduler(running=[r1], max_num_running_reqs=4)
    scheduler._sslo_step_wall_ema = {}  # no step observed yet → suspend
    scheduler.waiting = [object(), object(), object()]  # 3 waiting

    scheduler._apply_sslo_policy(0.0)

    # M - len(running) = 3 slots; min(3 waiting, 3 slots) = 3
    assert scheduler._sslo_step.waiting_admission_budget == 3




# ---------------------------------------------------------------------------
# Phase 2B tests: admitted_ts and terminal_outcome via scheduler paths
# ---------------------------------------------------------------------------

def test_admitted_ts_set_on_first_admission_via_mark():
    # Verify the state machine directly: mark_admitted fires once,
    # is idempotent, and propagates to compute_stats().
    state = make_state()
    t1 = 1.0
    t2 = 2.0
    state.mark_admitted(t1)
    state.mark_admitted(t2)  # second call must not overwrite
    assert state.admitted_ts == pytest.approx(t1)
    stats = state.compute_stats()
    assert stats.admitted_ts == pytest.approx(t1)


def test_terminal_outcome_completed_on_finish():
    # state.on_finish marks terminal_outcome="completed" so the value
    # rides on RequestOutput.sslo_metrics (output_processor calls
    # on_finish during _new_completion_output, before the client sees
    # the final output). The scheduler-side _free_blocks runs later
    # and would mark too late.
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=10.0)
    assert state.terminal_outcome == "in_progress"
    state.on_finish(now=1.0)
    assert state.terminal_outcome == "completed"
    assert state.compute_stats().terminal_outcome == "completed"


def test_terminal_outcome_default_in_progress_before_free():
    # terminal_outcome must start as "in_progress" and not change until free.
    state = make_state()
    assert state.terminal_outcome == "in_progress"
    stats = state.compute_stats()
    assert stats.terminal_outcome == "in_progress"


# ---------------------------------------------------------------------------
# Phase 3 tests: _policy_score_with_fallback regression
# ---------------------------------------------------------------------------

def test_policy_fallback_returns_one_for_missing():
    """_policy_score_with_fallback must return 1.0 for warmup (None pressure)."""
    warmup_state = make_state(phase=Phase.WARMUP)
    req = make_request("warmup_req", warmup_state)
    scheduler = _make_pressure_scheduler(running=[req], max_num_running_reqs=2)
    # No TPOT available yet
    score = scheduler._policy_score_with_fallback(warmup_state, now=0.5, tpot=None)
    assert score == pytest.approx(1.0)
    # Also confirm that pressure() returns None (the raw signal)
    assert warmup_state.pressure(0.5, None) is None


# ---------------------------------------------------------------------------
# Phase 4 tests: decision log emission
# ---------------------------------------------------------------------------


def test_decision_log_off_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(method="sslo", decision_log_mode="off")
    req = make_request("r", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=2, cfg=cfg)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_decision_buffer == []
    # path resolve is gated by the "off" early-return; the lazy resolver
    # still works in isolation, so check the buffer instead.
    assert not (tmp_path / "decisions.jsonl").exists()


def test_decision_log_step_emits_every_admitted(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(method="sslo", decision_log_mode="step")
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    b = make_request("b", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[a, b], max_num_running_reqs=2, cfg=cfg)

    scheduler._apply_sslo_policy(0.0)

    assert len(scheduler._sslo_decision_buffer) == 2


def test_decision_log_tier_changes_emits_on_transition(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    # Large heartbeat so the second step only fires on tier changes.
    cfg = SsloConfig(
        method="sslo", decision_log_mode="tier_changes",
        decision_heartbeat_steps=10000)
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[a], max_num_running_reqs=2, cfg=cfg)

    scheduler._apply_sslo_policy(0.0)
    # Step 0 is a heartbeat (idx % heartbeat == 0) and all reqs are new.
    assert len(scheduler._sslo_decision_buffer) == 1
    scheduler._sslo_decision_buffer.clear()

    # Step 1: no tier change, no selection change, no heartbeat → silent.
    scheduler._apply_sslo_policy(1.0)
    assert scheduler._sslo_decision_buffer == []


def test_decision_log_admit_only_subset(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(
        method="sslo", decision_log_mode="admit_only",
        decision_heartbeat_steps=10000)
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[a], max_num_running_reqs=2, cfg=cfg)

    # Step 0: first-seen event for "a".
    scheduler._apply_sslo_policy(0.0)
    admit_only_first = len(scheduler._sslo_decision_buffer)
    scheduler._sslo_decision_buffer.clear()
    # Step 1: no event → no emit.
    scheduler._apply_sslo_policy(1.0)
    assert scheduler._sslo_decision_buffer == []

    # Cross-check against tier_changes with same heartbeat: same count
    # on step 0 (both fire for new request), but admit_only never fires
    # the heartbeat path so it stays a strict subset over time.
    cfg2 = SsloConfig(
        method="sslo", decision_log_mode="tier_changes",
        decision_heartbeat_steps=10000)
    b = make_request("b", make_state(deadline=10.0, expected_len=2.0))
    sched2 = make_scheduler(running=[b], max_num_running_reqs=2, cfg=cfg2)
    sched2._apply_sslo_policy(0.0)
    assert len(sched2._sslo_decision_buffer) >= admit_only_first


def test_decision_log_buffer_flushes_on_overflow(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(method="sslo", decision_log_mode="step")
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    b = make_request("b", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[a, b], max_num_running_reqs=2, cfg=cfg)
    scheduler._sslo_decision_buffer_max = 1  # force flush on first row

    scheduler._apply_sslo_policy(0.0)

    decisions_path = tmp_path / "decisions.jsonl"
    assert decisions_path.exists()
    lines = decisions_path.read_text().splitlines()
    assert len(lines) == 2
    assert scheduler._sslo_decision_buffer == []


# ---------------------------------------------------------------------------
# multi_level_pressure policy tests (F5)
# ---------------------------------------------------------------------------

def _make_mlp_scheduler(max_num_running_reqs=32, adaptive_batching=True, **kwargs):
    """Helper: make_scheduler pre-configured for policy="multi_level_pressure".

    Pre-seeds tpot_ema with multiple captured sizes so the cap search has
    real choices; tests that need a specific tpot map override after.
    """
    cfg = kwargs.pop("cfg", None) or SsloConfig(
        method="sslo",
        policy="multi_level_pressure",
        adaptive_batching=adaptive_batching,
    )
    sched = make_scheduler(
        cfg=cfg, max_num_running_reqs=max_num_running_reqs, **kwargs)
    # Default seed: every multiple-of-8 captured size carries a tpot
    # entry. Override per-test as needed.
    sched.tpot_ema = {n: 1.0 for n in sched._sslo_cudagraph_sizes}
    return sched


def test_mlp_critical_only_measured_phase_triggers():
    # WARMUP req would carry serve=None — must not flip the critical
    # gate. The lone MEASURED req has serve=0.2 (well below 1.0).
    warm = make_request("w", make_state(phase=Phase.WARMUP))
    measured = make_request(
        "m", make_state(deadline=10.0, expected_len=2.0))  # serve = 0.2
    sched = _make_mlp_scheduler(
        running=[warm, measured], max_num_running_reqs=4)

    sched._apply_sslo_policy(0.0)

    assert sched._sslo_step.has_critical is False


def test_mlp_critical_fires_on_measured_serve_ge_one():
    # System-scaled serve = raw_serve * (N / base_n). With N=1, base_n=32
    # we need raw ≥ 32 to push scaled ≥ 1.0. expected_len=50, tpot=1.0,
    # ttd=1.0 → raw = 50; scaled = 50/32 ≈ 1.56 → critical.
    measured = make_request(
        "m", make_state(deadline=1.0, expected_len=50))
    sched = _make_mlp_scheduler(
        running=[measured], max_num_running_reqs=32)
    sched.tpot_ema = {32: 1.0, 24: 0.5, 16: 0.5, 8: 0.5}

    sched._apply_sslo_policy(0.0)

    assert sched._sslo_step.has_critical is True
    assert sched._sslo_step.cur_max_num_requests <= 32
    assert sched._sslo_step.waiting_admission_budget == 0


def test_mlp_pick_n_no_throughput_floor():
    # Only the smallest profiled bucket (8) can shrink the cap; its
    # throughput 8/4.0 = 2 vs base 32/1.0 = 32 (ratio 0.0625) would be
    # rejected by the legacy throughput-floor logic. MLP must not apply
    # that floor.
    measured = make_request(
        "m", make_state(deadline=1.0, expected_len=1.5))
    sched = _make_mlp_scheduler(
        running=[measured], max_num_running_reqs=32)
    sched.tpot_ema = {32: 1.0, 8: 4.0}

    picked_n, running, pending, _serve, _defer = sched._mlp_pick_adaptive_n(
        [measured], now=0.0, base_tpot=1.0)
    # 8 is feasible (objective is finite) and the only option below base.
    assert picked_n in (8, 32)
    assert running  # the measured req lands in running


def test_mlp_pick_n_system_scale_blocks_pointless_shrink():
    # Memory-bound simulation: tpot[32] == tpot[8] (small batch saves
    # no step time). With 32 admitted reqs evaluated at cap=8, system
    # scale = 32/8 = 4 → effective pressure 4× larger than at cap=32.
    # MLP objective should naturally pick cap=32, not cap=8.
    reqs = [
        make_request(f"r{i}", make_state(deadline=1.0, expected_len=2.0))
        for i in range(32)
    ]
    sched = _make_mlp_scheduler(
        running=reqs, max_num_running_reqs=32)
    sched.tpot_ema = {32: 1.0, 8: 1.0}  # memory-bound

    picked_n, _running, _pending, _serve, _defer = sched._mlp_pick_adaptive_n(
        reqs, now=0.0, base_tpot=1.0)
    assert picked_n == 32, (
        "system-scale must reject pointless shrink: smaller cap inflates "
        f"per-req scale by admitted/cap; picked={picked_n}")


def test_mlp_pick_n_system_scale_allows_meaningful_shrink():
    # Compute-bound simulation: tpot scales linearly with batch size.
    # tpot[32]=8.0 vs tpot[8]=1.0 → per-req refill at cap=8 is 1/8 of
    # cap=32. After multiplying by system scale (32/8=4 at cap=8 vs 1.0
    # at cap=32), effective per-req pressure at cap=8 is 4/8 = 0.5× of
    # cap=32. Use generous deadline=100 so defer stays < 1 at both caps
    # (feasibility check passes for both); MLP then picks by objective.
    reqs = [
        make_request(f"r{i}", make_state(deadline=100.0, expected_len=2.0))
        for i in range(32)
    ]
    sched = _make_mlp_scheduler(
        running=reqs, max_num_running_reqs=32)
    # Picker now reads wall_ema (any prefill composition); populate the
    # prefill=0 cells so per-batch latencies stay deterministic.
    sched._sslo_step_wall_ema = {32: {0: 8.0}, 8: {0: 1.0}}

    picked_n, _running, _pending, _serve, _defer = sched._mlp_pick_adaptive_n(
        reqs, now=0.0, base_tpot=8.0)
    assert picked_n == 8, (
        "system-scale must still permit shrink when smaller cap genuinely "
        f"saves step time; picked={picked_n}")


def test_mlp_defer_constraint_forces_running():
    # ttd = 0.5, epoch_s (= tpot_ema[base]) = 1.0 → defer_buffer = -0.5
    # → defer = inf ≥ mlp_defer_constraint (1.0). Even with low serve
    # this req must end up in running under _mlp_partition.
    forced = make_request(
        "forced", make_state(deadline=0.5, expected_len=0.1))
    low = make_request("low", make_state(deadline=100.0, expected_len=1.0))
    sched = _make_mlp_scheduler(
        running=[forced, low], max_num_running_reqs=1)
    sched.tpot_ema = {n: 1.0 for n in sched._sslo_cudagraph_sizes}

    sched._apply_sslo_policy(0.0)

    running_ids = {r.request_id for r in sched.running}
    assert "forced" in running_ids


def test_mlp_warmup_force_running_independent_of_pressure():
    # 2 warmup + 2 measured (low serve). cap=2 → warmup wins
    # (forced=empty since no MEASURED defer-violator).
    warmups = [
        make_request(f"w{i}", make_state(phase=Phase.WARMUP))
        for i in range(2)
    ]
    measured = [
        make_request(
            f"m{i}", make_state(deadline=100.0, expected_len=1.0))
        for i in range(2)
    ]
    sched = _make_mlp_scheduler(
        running=warmups + measured, max_num_running_reqs=2)

    sched._apply_sslo_policy(0.0)

    running_ids = {r.request_id for r in sched.running}
    assert "w0" in running_ids and "w1" in running_ids


def test_mlp_srjf_fallback_when_no_feasible_n():
    # All candidates infeasible: every MEASURED req has serve ≥ 1 AND
    # defer = inf across every tpot. _mlp_pick_adaptive_n must still
    # return a valid (n, partition) tuple via the SRJF fallback path.
    reqs = [
        make_request(f"r{i}", make_state(deadline=0.1, expected_len=10.0))
        for i in range(3)
    ]
    sched = _make_mlp_scheduler(running=reqs, max_num_running_reqs=32)
    sched.tpot_ema = {32: 1.0, 24: 1.0, 16: 1.0, 8: 1.0}

    picked_n, running, pending, _serve, _defer = sched._mlp_pick_adaptive_n(
        reqs, now=0.0, base_tpot=1.0)

    assert picked_n in (32, 24, 16, 8)
    assert len(running) + len(pending) == len(reqs)


def test_mlp_non_critical_admission_budget_uses_measured_only():
    # 1 warmup + 3 measured (deadline=10, expected_len=3, tpot=1.0 from
    # helper). N=4 admitted, base_n=32 → scale = 4/32 = 0.125.
    #   Step 1: warmup=[w] → running. measured=[m0,m1,m2] (defer<1) → pending.
    #   serve/defer scaling: raw 0.3 → scaled 0.3 * 0.125 = 0.0375 each.
    #   load = Σ serve_running (0) + Σ defer_pending (3·0.0375 = 0.1125)
    #        + unmeasured_running (warmup, raw 1.0) + unmeasured_pending (0)
    #        = 1.1125
    #   admission_capacity = int(32 - 1.1125) = 30
    #   slack = 32 - 1 = 31 → budget = min(100, 30, 31) = 30
    warm = make_request("w", make_state(phase=Phase.WARMUP))
    measured = [
        make_request(
            f"m{i}", make_state(deadline=10.0, expected_len=3.0))
        for i in range(3)
    ]
    sched = _make_mlp_scheduler(
        running=[warm] + measured, max_num_running_reqs=32)
    sched.waiting = [object()] * 100

    sched._apply_sslo_policy(0.0)

    assert sched._sslo_step.has_critical is False
    assert sched._sslo_step.waiting_admission_budget == 30
    # Backfill moved to schedule() main loop (post-waiting-admit).
    # MLP non-critical now leaves only forced+warmup in running; all
    # eligible measured stay in pending until the main-loop backfill
    # claims them with leftover token_budget.
    assert len(sched.running) == 1
    assert len(sched.sslo_pending) == 3
    # defer_base snapshot stored for downstream backfill ranking.
    assert sched._sslo_step.defer_base is not None
    assert len(sched._sslo_step.defer_base) == 4


def test_mlp_non_critical_no_admit_when_all_warmup_running():
    # cap=2; two warmups force-running, no measured candidate to demote.
    # The voluntary-demotion mechanism only displaces MEASURED reqs;
    # warmups stay put. Therefore no admission slack is created.
    warmups = [
        make_request(f"w{i}", make_state(phase=Phase.WARMUP))
        for i in range(2)
    ]
    sched = _make_mlp_scheduler(running=warmups, max_num_running_reqs=2)
    sched.waiting = [object()] * 10

    sched._apply_sslo_policy(0.0)

    assert sched._sslo_step.waiting_admission_budget == 0
    assert len(sched.sslo_pending) == 0


def test_mlp_dispatch_resolves():
    cfg = SsloConfig(
        method="sslo",
        policy="multi_level_pressure",
        adaptive_batching=True,
    )
    sched = make_scheduler(max_num_running_reqs=8, cfg=cfg)
    assert sched._apply_sslo_policy.__name__ == (
        "_apply_sslo_multi_level_pressure")


def test_mlp_decision_log_emits_new_fields(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(
        method="sslo",
        policy="multi_level_pressure",
        adaptive_batching=True,
        decision_log_mode="step",
    )
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    sched = make_scheduler(running=[a], max_num_running_reqs=8, cfg=cfg)

    sched._apply_sslo_policy(0.0)

    assert len(sched._sslo_decision_buffer) == 1
    import json
    row = json.loads(sched._sslo_decision_buffer[0])
    # New MLP fields present in the schema.
    assert "serve_pressure" in row
    assert "defer_pressure" in row
    assert "epoch_time_s" in row
    assert "defer_buffer_slack_s" in row
    # MEASURED-phase request → serve is populated.
    assert row["phase"] == "MEASURED"
    assert row["serve_pressure"] is not None
    # tier kept for back-compat (MLP writes 0).
    assert row["tier"] == 0
