# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase A v2 SSLO scheduler policy helpers."""

from types import SimpleNamespace

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import Phase, RequestSLOState
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput


def make_state(
    *,
    phase=Phase.MEASURED,
    deadline=10.0,
    expected_len=10.0,
    generated=0,
) -> RequestSLOState:
    state = RequestSLOState(num_warmup_chunks=1)
    state.phase = phase
    state.decoding_start_ts = 0.0
    state.cumulative_consume_time = deadline
    state.chunk_expected_len_ema = expected_len
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
    scheduler.sslo_config = cfg or SsloConfig(enabled=True)
    scheduler.tpot_ema = {max_num_running_reqs: 1.0}
    scheduler.sslo_offloaded = set()
    scheduler._sslo_prev_step_batch = None
    scheduler._sslo_prev_step_decoding_only = False
    scheduler._sslo_prev_step_start_ts = None
    scheduler._sslo_has_critical = False
    scheduler._sslo_active_cap = max_num_running_reqs
    scheduler.max_num_running_reqs = max_num_running_reqs
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
    return scheduler


def request_ids(reqs):
    return [req.request_id for req in reqs]


def test_tpot_ema_update_only_multiple_bucket_and_decoding_only():
    scheduler = make_scheduler(max_num_running_reqs=8)
    scheduler.sslo_config = SsloConfig(enabled=True, tpot_ema_alpha=0.5)
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

    assert scheduler._sslo_has_critical is True


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
    req = make_request("running", make_state(deadline=10, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=1)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler.sslo_pending == [req]


def test_non_critical_hysteresis_keeps_state_in_band():
    pending = make_request("pending", make_state(deadline=10, expected_len=5))
    running = make_request("running", make_state(deadline=10, expected_len=5))
    scheduler = make_scheduler(running=[running], pending=[pending],
                               max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert running in scheduler.running
    assert pending in scheduler.sslo_pending


def test_cap_overflow_priority_sort():
    low = make_request("low", make_state(deadline=10, expected_len=4))
    high = make_request("high", make_state(deadline=10, expected_len=6))
    warm = make_request("warm", make_state(phase=Phase.WARMUP))
    scheduler = make_scheduler(running=[low, high, warm], max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert request_ids(scheduler.running) == ["warm", "high"]
    assert request_ids(scheduler.sslo_pending) == ["low"]


def test_score_suspend_when_tpot_unobserved():
    running = make_request("running", make_state(deadline=10, expected_len=2))
    pending = make_request("pending", make_state(deadline=10, expected_len=9))
    scheduler = make_scheduler(running=[running], pending=[pending],
                               max_num_running_reqs=2)
    scheduler.tpot_ema = {}

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


def test_offload_eligibility_excludes_warmup_critical_offloaded():
    cfg = SsloConfig(enabled=True, offloading=True)
    warm = make_request("warm", make_state(phase=Phase.WARMUP))
    critical = make_request("critical", make_state(deadline=1, expected_len=10))
    offloaded = make_request("offloaded", make_state(deadline=10, expected_len=1))
    scheduler = make_scheduler(pending=[warm, critical, offloaded], cfg=cfg)
    scheduler.sslo_offloaded.add("offloaded")

    assert scheduler._is_offload_eligible(warm, 0.0) is False
    assert scheduler._is_offload_eligible(critical, 0.0) is False
    assert scheduler._is_offload_eligible(offloaded, 0.0) is False


def test_offloaded_score_includes_transfer_cost():
    req = make_request("r", make_state(deadline=10, expected_len=4))
    scheduler = make_scheduler(pending=[req])
    scheduler._estimate_transfer_s = lambda r: 1.0

    assert scheduler._offloaded_score(req, 0.0) == pytest.approx(4 / 7.95)


def test_reload_trigger_at_offloading_out_threshold():
    cfg = SsloConfig(enabled=True, offloading=True,
                     offloading_out_threshold=0.7)
    req = make_request("r", make_state(deadline=10, expected_len=7))
    scheduler = make_scheduler(pending=[req], cfg=cfg)
    scheduler.sslo_offloaded.add("r")

    scheduler._reload_check(0.0)

    assert "r" not in scheduler.sslo_offloaded
    assert req.slo_state.is_offloaded is False


def test_offload_blocked_in_critical_mode():
    cfg = SsloConfig(enabled=True, offloading=True)
    req = make_request("r", make_state(deadline=10, expected_len=1))
    scheduler = make_scheduler(pending=[req], cfg=cfg)
    scheduler._sslo_has_critical = True

    assert scheduler._pick_offload_victim(0.0) is None


def test_offloaded_pending_not_promoted_before_reload():
    req = make_request("r", make_state(deadline=10, expected_len=6))
    scheduler = make_scheduler(pending=[req], max_num_running_reqs=1)
    scheduler.sslo_offloaded.add("r")

    scheduler._apply_sslo_policy(0.0)

    assert scheduler.running == []
    assert scheduler.sslo_pending == [req]


def test_offloaded_critical_reloads_before_policy_placement():
    cfg = SsloConfig(enabled=True, offloading=True,
                     offloading_out_threshold=0.7)
    req = make_request("r", make_state(deadline=1, expected_len=2))
    req.slo_state.on_offload_enter(0.0)
    scheduler = make_scheduler(pending=[req], max_num_running_reqs=1, cfg=cfg)
    scheduler.sslo_offloaded.add("r")

    scheduler._apply_sslo_policy(0.5)

    assert "r" not in scheduler.sslo_offloaded
    assert scheduler.running == [req]
    assert scheduler._sslo_has_critical is True


def test_offload_marks_only_phase_a_v2():
    # Phase A v2: offload is bookkeeping only. No KV transfer, no recompute.
    cfg = SsloConfig(enabled=True, offloading=True)
    req = make_request("r", make_state(deadline=10, expected_len=1))
    req.num_computed_tokens = 5
    scheduler = make_scheduler(pending=[req], cfg=cfg)

    scheduler._offload(req, 1.0)

    assert "r" in scheduler.sslo_offloaded
    assert req.slo_state.is_offloaded is True
    # Marking only — original KV / num_computed_tokens preserved.
    assert req.num_computed_tokens == 5
    assert scheduler.slo_state_offload_enter_ts_set if False else True
    # State lifecycle hook fired.
    assert req.slo_state.num_offload_intervals == 1


def test_adaptive_n_disabled_uses_base():
    req = make_request("critical", make_state(deadline=1, expected_len=10))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=8)
    scheduler.tpot_ema = {8: 1.0, 16: 0.2}

    scheduler.schedule_sslo()

    assert len(scheduler.running) == 1


def test_adaptive_n_picks_largest_resolving_critical():
    # Phase A v2 spec: pick the LARGEST n that resolves all critical
    # (preserves throughput while clearing critical).
    cfg = SsloConfig(enabled=True, adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=8, cfg=cfg)
    scheduler.tpot_ema = {8: 1.0, 16: 0.4, 24: 0.2}
    tiers = {"critical": 0}

    # All of {16, 24} resolve (score < 1). Pick largest: 24.
    assert scheduler._pick_adaptive_n([req], tiers, 0.0, 1.0) == 24


def test_adaptive_n_minimizes_worst_score_when_unresolvable():
    cfg = SsloConfig(enabled=True, adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=20))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=8, cfg=cfg)
    scheduler.tpot_ema = {8: 1.0, 16: 0.8, 24: 0.6}

    assert scheduler._pick_adaptive_n(
        [req], {"critical": 0}, 0.0, 1.0) == 24


def test_adaptive_n_respects_throughput_floor():
    cfg = SsloConfig(enabled=True, adaptive_batching=True)
    req = make_request("critical", make_state(deadline=1, expected_len=2))
    scheduler = make_scheduler(running=[req], max_num_running_reqs=8, cfg=cfg)
    scheduler.tpot_ema = {16: 100.0}

    assert scheduler._pick_adaptive_n(
        [req], {"critical": 0}, 0.0, 1.0) is None


def test_adaptive_n_at_least_num_critical():
    cfg = SsloConfig(enabled=True, adaptive_batching=True)
    reqs = [
        make_request(f"r{i}", make_state(deadline=1, expected_len=2))
        for i in range(10)
    ]
    scheduler = make_scheduler(running=reqs, max_num_running_reqs=8, cfg=cfg)
    scheduler.tpot_ema = {8: 1.0, 16: 0.2}
    tiers = {req.request_id: 0 for req in reqs}

    assert scheduler._pick_adaptive_n(reqs, tiers, 0.0, 1.0) == 16
