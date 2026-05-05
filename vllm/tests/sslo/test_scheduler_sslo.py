# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO scheduler helpers and pending redistribution."""

from types import SimpleNamespace

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import RequestSLOState
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler, _sslo_score_key


def make_mock_request(request_id="req", slo_state=None):
    return SimpleNamespace(request_id=request_id, slo_state=slo_state)


def make_state(
    *,
    cumulative_slack=1.0,
    realtime_slack=100.0,
    predicted_finish=0.1,
    overdue=False,
    should_exit=False,
):
    state = RequestSLOState()
    state.cumulative_slack = cumulative_slack
    state._realtime_slack = lambda now: realtime_slack
    state._predicted_finish_time = lambda: predicted_finish
    state.is_overdue_post_warmup = lambda: overdue
    state.should_exit_pending = lambda now, pending_count=0: should_exit
    return state


class FakeKVCacheManager:

    def new_step_starts(self):
        pass

    def take_new_block_ids(self):
        return []

    def get_num_common_prefix_blocks(self, request_id):
        return []


class FakeEncoderCacheManager:

    def get_freed_mm_hashes(self):
        return []


def make_scheduler(
    running=None,
    sslo_pending=None,
    waiting_len=0,
    max_num_running_reqs=100,
    sslo_config=None,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running = list(running or [])
    scheduler.sslo_pending = list(sslo_pending or [])
    scheduler.sslo_consecutive_pending = {}
    scheduler.sslo_pending_entered_at = {}
    scheduler.sslo_forced_pending_due_to_cap = {}
    scheduler.sslo_config = sslo_config or SsloConfig(enabled=True)
    scheduler.waiting = [object()] * waiting_len
    scheduler.skipped_waiting = []
    scheduler.max_num_scheduled_tokens = 0
    scheduler.max_num_running_reqs = max_num_running_reqs
    scheduler.current_max_num_running_reqs = max_num_running_reqs
    scheduler._sslo_cap_safe_steps = 0
    scheduler._sslo_iter_time_ema_s = 0.0
    scheduler._sslo_last_schedule_ts = None
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.kv_cache_manager = FakeKVCacheManager()
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler.lora_config = None
    scheduler.policy = SchedulingPolicy.FCFS
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


class TestSsloScoreKey:

    def test_none_slo_state_returns_inf(self):
        request = make_mock_request(slo_state=None)

        assert _sslo_score_key(request) == float("inf")

    def test_finite_score_returned(self):
        state = make_state(cumulative_slack=-2.5)
        request = make_mock_request(slo_state=state)

        assert _sslo_score_key(request) == pytest.approx(-2.5)

    def test_sort_order(self):
        urgent = make_mock_request("urgent", make_state(cumulative_slack=-3.0))
        relaxed = make_mock_request("relaxed", make_state(cumulative_slack=4.0))
        no_state = make_mock_request("none", None)

        requests = sorted([no_state, relaxed, urgent], key=_sslo_score_key)

        assert [request.request_id for request in requests] == [
            "urgent",
            "relaxed",
            "none",
        ]


class TestPhaseAHelpers:

    def test_iter_time_ema(self):
        cfg = SsloConfig(enabled=True, iter_time_ema_alpha=0.25,
                         max_pending_num=5)
        scheduler = make_scheduler(sslo_config=cfg)

        scheduler._update_sslo_iter_time_ema(10.0)
        assert scheduler._sslo_iter_time_ema_s == 0.0
        scheduler._update_sslo_iter_time_ema(14.0)
        assert scheduler._sslo_iter_time_ema_s == pytest.approx(4.0)
        scheduler._update_sslo_iter_time_ema(16.0)
        assert scheduler._sslo_iter_time_ema_s == pytest.approx(3.5)
        assert scheduler._sslo_max_pending_duration_s() == pytest.approx(17.5)

    def test_classify_tier_0_post_warmup_overdue(self):
        req = make_mock_request("r", make_state(overdue=True))
        scheduler = make_scheduler(running=[req])

        assert scheduler._classify_must_run_tier(req, now=0.0) == 0

    def test_classify_tier_0_predicted_finish_exceeds_slack(self):
        req = make_mock_request(
            "r", make_state(realtime_slack=0.05, predicted_finish=0.5))
        scheduler = make_scheduler(running=[req])

        assert scheduler._classify_must_run_tier(req, now=0.0) == 0

    def test_classify_tier_1_pending_should_exit(self):
        req = make_mock_request("r", make_state(should_exit=True))
        scheduler = make_scheduler(sslo_pending=[req])

        assert scheduler._classify_must_run_tier(req, now=0.0) == 1

    def test_classify_tier_2_pending_duration_overflow(self):
        req = make_mock_request("r", make_state())
        scheduler = make_scheduler(sslo_pending=[req])
        scheduler.sslo_pending_entered_at[req.request_id] = 1.0
        scheduler._sslo_iter_time_ema_s = 0.01
        scheduler.sslo_config.max_pending_num = 5

        assert scheduler._classify_must_run_tier(req, now=2.0) == 2

    def test_classify_tier_3_normal_running(self):
        req = make_mock_request("r", make_state())
        scheduler = make_scheduler(running=[req])

        assert scheduler._classify_must_run_tier(req, now=0.0) == 3

    def test_urgency_key_sorts_by_tier_effective_slack_and_id(self):
        tier0 = make_mock_request("b", make_state(cumulative_slack=10.0))
        urgent = make_mock_request("c", make_state(cumulative_slack=0.2))
        forced = make_mock_request("a", make_state(cumulative_slack=0.25))
        scheduler = make_scheduler(running=[tier0, urgent, forced])
        scheduler.sslo_forced_pending_due_to_cap[forced.request_id] = 2

        ordered = sorted(
            [(tier0, 0), (urgent, 3), (forced, 3)],
            key=lambda item: scheduler._sslo_urgency_key(
                item[0], now=0.0, tier=item[1]),
        )

        assert [req.request_id for req, _ in ordered] == ["b", "a", "c"]


class TestPendingRedistribution:

    def test_redistribution_must_tier_wins_cap(self):
        must_a = make_mock_request("must-a", make_state(overdue=True))
        must_b = make_mock_request("must-b", make_state(realtime_slack=0.1,
                                                        predicted_finish=0.5))
        normal = make_mock_request("normal", make_state(cumulative_slack=-10.0))
        scheduler = make_scheduler(running=[normal, must_b, must_a],
                                   max_num_running_reqs=2)

        scheduler.schedule_sslo()

        assert [req.request_id for req in scheduler.running] == [
            "must-b", "must-a"]
        assert [req.request_id for req in scheduler.sslo_pending] == ["normal"]

    def test_redistribution_cap_not_exceeded(self):
        reqs = [
            make_mock_request(f"must-{idx}", make_state(overdue=True,
                                                        cumulative_slack=idx))
            for idx in range(6)
        ]
        scheduler = make_scheduler(running=reqs, max_num_running_reqs=4)

        scheduler.schedule_sslo()

        assert len(scheduler.running) == 4
        assert len(scheduler.sslo_pending) == 2
        assert scheduler.sslo_forced_pending_due_to_cap == {
            "must-4": 1,
            "must-5": 1,
        }

    def test_redistribution_tier2_yields_to_tier01(self):
        tier0 = make_mock_request("tier0", make_state(overdue=True))
        tier1 = make_mock_request("tier1", make_state(should_exit=True))
        tier2 = make_mock_request("tier2", make_state(cumulative_slack=-100.0))
        scheduler = make_scheduler(sslo_pending=[tier2, tier1], running=[tier0],
                                   max_num_running_reqs=2)
        scheduler.sslo_pending_entered_at[tier2.request_id] = 0.0
        scheduler._sslo_iter_time_ema_s = 0.01

        scheduler.schedule_sslo()

        assert [req.request_id for req in scheduler.running] == ["tier0", "tier1"]
        assert [req.request_id for req in scheduler.sslo_pending] == ["tier2"]

    def test_redistribution_normal_fills_remaining_cap(self):
        must = make_mock_request("must", make_state(overdue=True))
        normals = [
            make_mock_request(f"normal-{idx}", make_state(cumulative_slack=slack))
            for idx, slack in enumerate([5.0, 1.0, 3.0, 2.0, 4.0])
        ]
        scheduler = make_scheduler(running=[must] + normals,
                                   max_num_running_reqs=4)

        scheduler.schedule_sslo()

        assert [req.request_id for req in scheduler.running] == [
            "must", "normal-1", "normal-3", "normal-2"]
        assert [req.request_id for req in scheduler.sslo_pending] == [
            "normal-4", "normal-0"]

    def test_redistribution_resets_forced_counter_on_running(self):
        req = make_mock_request("must", make_state(overdue=True))
        scheduler = make_scheduler(running=[req], max_num_running_reqs=1)
        scheduler.sslo_forced_pending_due_to_cap[req.request_id] = 3

        scheduler.schedule_sslo()

        assert scheduler.running == [req]
        assert req.request_id not in scheduler.sslo_forced_pending_due_to_cap

    def test_redistribution_no_waiting_demote_allowed(self):
        must = make_mock_request("must", make_state(overdue=True))
        normal = make_mock_request("normal", make_state(cumulative_slack=10.0))
        scheduler = make_scheduler(running=[normal, must], waiting_len=0,
                                   max_num_running_reqs=1)

        scheduler.schedule_sslo()

        assert scheduler.running == [must]
        assert scheduler.sslo_pending == [normal]


class TestAdaptiveBatchSize:

    def test_cap_severe_overdue_halves(self):
        cfg = SsloConfig(enabled=True, adaptive_batch_size=True,
                         severe_overdue_margin_s=0.5)
        req = make_mock_request("r", make_state(realtime_slack=-0.6))
        scheduler = make_scheduler(running=[req], max_num_running_reqs=16,
                                   sslo_config=cfg)
        scheduler.current_max_num_running_reqs = 10

        scheduler._update_dynamic_cap(now=0.0)

        assert scheduler.current_max_num_running_reqs == 5

    def test_cap_pending_severe_overdue_halves(self):
        cfg = SsloConfig(enabled=True, adaptive_batch_size=True,
                         severe_overdue_margin_s=0.5)
        running = make_mock_request("running", make_state(realtime_slack=1.0))
        pending = make_mock_request("pending", make_state(realtime_slack=-0.6))
        scheduler = make_scheduler(running=[running], sslo_pending=[pending],
                                   max_num_running_reqs=16, sslo_config=cfg)
        scheduler.current_max_num_running_reqs = 10

        scheduler._update_dynamic_cap(now=0.0)

        assert scheduler.current_max_num_running_reqs == 5

    def test_cap_mild_overdue_three_quarters(self):
        cfg = SsloConfig(enabled=True, adaptive_batch_size=True,
                         severe_overdue_margin_s=0.5)
        req = make_mock_request("r", make_state(realtime_slack=-0.2))
        scheduler = make_scheduler(running=[req], max_num_running_reqs=16,
                                   sslo_config=cfg)
        scheduler.current_max_num_running_reqs = 10

        scheduler._update_dynamic_cap(now=0.0)

        assert scheduler.current_max_num_running_reqs == 7

    def test_cap_grows_after_safe_steps(self):
        cfg = SsloConfig(enabled=True, adaptive_batch_size=True,
                         cap_growth_safe_steps=2, cap_growth_step=3)
        req = make_mock_request("r", make_state(realtime_slack=1.0))
        scheduler = make_scheduler(running=[req], max_num_running_reqs=16,
                                   sslo_config=cfg)
        scheduler.current_max_num_running_reqs = 10

        scheduler._update_dynamic_cap(now=0.0)
        scheduler._update_dynamic_cap(now=1.0)

        assert scheduler.current_max_num_running_reqs == 13

    def test_cap_resets_safe_counter_on_overdue(self):
        cfg = SsloConfig(enabled=True, adaptive_batch_size=True)
        req = make_mock_request("r", make_state(realtime_slack=-0.1))
        scheduler = make_scheduler(running=[req], max_num_running_reqs=16,
                                   sslo_config=cfg)
        scheduler._sslo_cap_safe_steps = 10

        scheduler._update_dynamic_cap(now=0.0)

        assert scheduler._sslo_cap_safe_steps == 0


class TestPendingCleanup:

    def test_clear_pending_state_closes_interval_and_counters(self):
        state = RequestSLOState()
        state.on_pending_enter(1.0)
        req = make_mock_request("pending", state)
        scheduler = make_scheduler(sslo_pending=[req])
        scheduler.sslo_consecutive_pending[req.request_id] = 2
        scheduler.sslo_pending_entered_at[req.request_id] = 1.0
        scheduler.sslo_forced_pending_due_to_cap[req.request_id] = 3

        scheduler._sslo_clear_pending_state(req, now=2.5)

        assert scheduler.sslo_pending == []
        assert req.request_id not in scheduler.sslo_consecutive_pending
        assert req.request_id not in scheduler.sslo_pending_entered_at
        assert req.request_id not in scheduler.sslo_forced_pending_due_to_cap
        assert state._pending_enter_ts is None
        assert state._chunk_pending_time == pytest.approx(1.5)


class TestWaitingAdmission:

    def test_waiting_admission_reserved_slots(self):
        running = [
            make_mock_request(f"run-{idx}", make_state()) for idx in range(4)
        ]
        pending_t0 = make_mock_request("pending-t0", make_state(overdue=True))
        pending_t1 = make_mock_request("pending-t1", make_state(should_exit=True))
        pending_t2 = make_mock_request("pending-t2", make_state())
        scheduler = make_scheduler(
            running=running,
            sslo_pending=[pending_t0, pending_t1, pending_t2],
            waiting_len=10,
            max_num_running_reqs=8,
        )
        scheduler.sslo_pending_entered_at[pending_t2.request_id] = 0.0
        scheduler._sslo_iter_time_ema_s = 0.01

        assert scheduler._compute_waiting_admission_budget(now=1.0) == 2
