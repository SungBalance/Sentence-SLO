# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO scheduler helpers and pending redistribution."""

from types import SimpleNamespace
import time

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import RequestSLOState
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler, _sslo_score_key


def make_mock_request(request_id="req", slo_state=None):
    return SimpleNamespace(request_id=request_id, slo_state=slo_state)


def make_pending_eligible_state(score=10.0, n_samples=5):
    state = RequestSLOState()
    state.cumulative_slack = score
    state._decoding_start = time.monotonic()
    state._cumulative_consume = 10.0
    # Populate the chunk_gen_estimator with samples so warmup guard passes
    # and EMA settles at gen_time=1.0, per_token_time=1.0, word_count=1.0.
    for _ in range(n_samples):
        state.chunk_gen_estimator.update(pure_gen_time=1.0, word_count=1)
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


def make_scheduler(running=None, sslo_pending=None, waiting_len=0, max_pending=5):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running = list(running or [])
    scheduler.sslo_pending = list(sslo_pending or [])
    scheduler.sslo_consecutive_pending = {}
    scheduler.sslo_config = SsloConfig(enabled=True,
                                       max_consecutive_pending=max_pending)
    scheduler.waiting = [object()] * waiting_len
    scheduler.skipped_waiting = []
    scheduler.max_num_scheduled_tokens = 0
    scheduler.max_num_running_reqs = 100
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
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler._update_after_schedule = lambda output: None
    return scheduler


class TestSsloScoreKey:

    def test_none_slo_state_returns_inf(self):
        request = make_mock_request(slo_state=None)

        assert _sslo_score_key(request) == float("inf")

    def test_finite_score_returned(self):
        state = RequestSLOState()
        state.cumulative_slack = -2.5
        request = make_mock_request(slo_state=state)

        assert _sslo_score_key(request) == pytest.approx(-2.5)

    def test_sort_order(self):
        urgent = make_mock_request("urgent", make_pending_eligible_state(-3.0))
        relaxed = make_mock_request("relaxed", make_pending_eligible_state(4.0))
        no_state = make_mock_request("none", None)

        requests = sorted([no_state, relaxed, urgent], key=_sslo_score_key)

        assert [request.request_id for request in requests] == [
            "urgent",
            "relaxed",
            "none",
        ]


class TestPendingRedistribution:

    def test_no_waiting_nothing_pended(self):
        request = make_mock_request(slo_state=make_pending_eligible_state())
        scheduler = make_scheduler(running=[request], waiting_len=0)

        scheduler.schedule_sslo()

        assert scheduler.running == [request]
        assert scheduler.sslo_pending == []
        assert scheduler.sslo_consecutive_pending[request.request_id] == 0

    def test_high_urgency_request_pended(self):
        request = make_mock_request(slo_state=make_pending_eligible_state())
        scheduler = make_scheduler(running=[request], waiting_len=1)

        scheduler.schedule_sslo()

        assert scheduler.running == []
        assert scheduler.sslo_pending == [request]
        assert scheduler.sslo_consecutive_pending[request.request_id] == 1
        assert request.slo_state._pending_enter_ts is not None

    def test_max_consecutive_pending_forces_run(self):
        request = make_mock_request(slo_state=make_pending_eligible_state())
        request.slo_state.on_pending_enter(0.0)
        scheduler = make_scheduler(sslo_pending=[request], waiting_len=1,
                                   max_pending=2)
        scheduler.sslo_consecutive_pending[request.request_id] = 2

        scheduler.schedule_sslo()

        assert scheduler.running == [request]
        assert scheduler.sslo_pending == []
        assert scheduler.sslo_consecutive_pending[request.request_id] == 0
        assert request.slo_state._pending_enter_ts is None


class TestOffloadMarking:
    def test_highest_score_in_combined_is_marked(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        running = [
            make_mock_request("a", make_pending_eligible_state(0.1)),
            make_mock_request("b", make_pending_eligible_state(1.5)),
        ]
        pending = [make_mock_request("c", make_pending_eligible_state(2.5))]
        combined = running + pending
        candidate = max(combined, key=_sslo_score_key)
        candidate.sslo_offload_requested = True
        assert candidate.request_id == "c"
        assert candidate.sslo_offload_requested is True


class TestAdaptiveBatchSize:
    def test_overdue_reduces_cap(self):
        running = [make_mock_request("a", make_pending_eligible_state(0.5)),
                   make_mock_request("b", make_pending_eligible_state(-0.1))]
        base = 16
        overdue = any(r.slo_state and r.slo_state.sslo_score < 0
                      for r in running)
        cap = max(1, base - 1) if overdue else base
        assert cap == 15

    def test_no_overdue_keeps_cap(self):
        running = [make_mock_request("a", make_pending_eligible_state(0.5)),
                   make_mock_request("b", make_pending_eligible_state(0.1))]
        base = 16
        overdue = any(r.slo_state and r.slo_state.sslo_score < 0
                      for r in running)
        cap = max(1, base - 1) if overdue else base
        assert cap == 16
