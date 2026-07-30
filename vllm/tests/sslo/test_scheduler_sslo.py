# SPDX-License-Identifier: Apache-2.0
"""Tests for surviving SSLO scheduler machinery.

The SSLO scheduler now exposes only two methods — ``baseline`` (metrics
only) and ``progress_serve`` (posterior tail-risk admission). The old
pressure / multi-level-pressure / adaptive-batching policies were deleted;
their tests went with them. What remains here covers the shared per-step
machinery: wall-step EMA recording, admitted_ts / terminal_outcome, the
decision-log emitter, and a couple of light progress_serve placement checks.
The ProgressServe algorithm itself is covered by test_progress_serve.py.
"""

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
        # passing `deadline=10` means deadline = 10.
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
        # ProgressServe's KV-aware admission reads
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
    scheduler.sslo_config = cfg or SsloConfig(method="progress_serve")
    # Pre-seed wall-step EMA so _sslo_step_ema_lookup returns 1.0 (the
    # global-average fallback) for any batch_size. Tests that want the
    # cold-start (base_tpot=None) branch override this to {}.
    scheduler._sslo_step_wall_ema = {max_num_running_reqs: {0: 1.0}}
    scheduler._sslo_prev_step_num_prefills = 0
    scheduler._sslo_prev_step_batch = None
    # SSLO: capture-time decode-latency profile (empty unless a test loads it).
    scheduler._sslo_decode_profile = {}
    scheduler._sslo_decode_profile_keys = []
    scheduler._sslo_prev_step_decoding_only = False
    scheduler._sslo_prev_step_start_ts = None
    scheduler._sslo_step = SsloStepState(
        cur_max_num_requests=max_num_running_reqs)
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
    # SSLO: __init__ normally patches _apply_sslo_policy based on method;
    # tests bypass __init__ via __new__, so do it here.
    scheduler._apply_sslo_policy = scheduler._resolve_sslo_policy_dispatch()
    return scheduler


def request_ids(reqs):
    return [req.request_id for req in reqs]


# ---------------------------------------------------------------------------
# Wall-step EMA recording (surviving machinery)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ProgressServe placement (light coverage; algorithm tested in
# test_progress_serve.py)
# ---------------------------------------------------------------------------

def test_progress_serve_warmup_admits_fcfs_up_to_b():
    # No wall-step EMA yet → base_tpot is None → warmup branch admits FCFS
    # up to B - len(running) from the waiting queue.
    req = make_request("r", make_state())
    scheduler = make_scheduler(running=[req], max_num_running_reqs=4)
    scheduler._sslo_step_wall_ema = {}  # cold start: no Δ signal
    scheduler.waiting = [object()] * 10

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is False
    assert scheduler._sslo_step.cur_max_num_requests == 4
    # B - len(running) = 4 - 1 = 3 slots; min(10 waiting, 3) = 3.
    assert scheduler._sslo_step.waiting_admission_budget == 3
    assert scheduler._sslo_step.k_star == 3


def test_progress_serve_partitions_running_and_pending():
    # With a wall-step EMA present (base_tpot=1.0), the algorithm runs and
    # partitions the in-flight set into running / sslo_pending with the
    # union preserved.
    reqs = [
        make_request(f"r{i}", make_state(deadline=10.0, expected_len=3.0))
        for i in range(3)
    ]
    scheduler = make_scheduler(running=reqs, max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_step.has_critical is False
    placed = request_ids(scheduler.running) + request_ids(scheduler.sslo_pending)
    assert sorted(placed) == ["r0", "r1", "r2"]


# ---------------------------------------------------------------------------
# Hybrid Δ(b) for adaptive batching
# ---------------------------------------------------------------------------

def test_hybrid_delta_rescales_profile_by_live_ema():
    sched = make_scheduler(max_num_running_reqs=128)
    # Profile shape: forward latency grows sublinearly with batch.
    sched.set_cudagraph_decode_profile({64: 40.0, 128: 60.0})
    # Live scheduler-step wall-EMA at the currently-running batch (128).
    sched._sslo_step_wall_ema = {128: {0: 90.0}}
    sched._sslo_prev_step_batch = 128
    # scale = wall_ema(128)/profile(128) = 90/60 = 1.5
    # Δ(64) = profile(64) * 1.5 = 60.0 ; Δ(128) = 90.0 (== live)
    assert sched._sslo_hybrid_delta(64) == pytest.approx(60.0)
    assert sched._sslo_hybrid_delta(128) == pytest.approx(90.0)


def test_hybrid_delta_falls_back_to_wall_ema_without_profile():
    sched = make_scheduler(max_num_running_reqs=128)
    sched._sslo_step_wall_ema = {128: {0: 90.0}}
    sched._sslo_prev_step_batch = 128
    # No profile loaded → exact wall-EMA for known batch, None otherwise.
    assert sched._sslo_hybrid_delta(128) == pytest.approx(90.0)
    assert sched._sslo_hybrid_delta(64) is None


def test_hybrid_delta_nearest_size_and_uncalibrated():
    sched = make_scheduler(max_num_running_reqs=128)
    sched.set_cudagraph_decode_profile({64: 40.0, 128: 60.0})
    # No live EMA anchor → return raw profile shape (uncalibrated).
    sched._sslo_step_wall_ema = {}
    sched._sslo_prev_step_batch = None
    assert sched._sslo_hybrid_delta(64) == pytest.approx(40.0)
    # Non-captured size 96 → nearest captured (64 and 128 equidistant → min=64).
    assert sched._sslo_hybrid_delta(96) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# admitted_ts and terminal_outcome (surviving state machinery)
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
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=10.0)
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
# Decision log emission (surviving machinery; driven via progress_serve)
# ---------------------------------------------------------------------------

def test_decision_log_off_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(method="progress_serve", decision_log_mode="off")
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
    cfg = SsloConfig(method="progress_serve", decision_log_mode="step")
    a = make_request("a", make_state(deadline=10.0, expected_len=2.0))
    b = make_request("b", make_state(deadline=10.0, expected_len=2.0))
    scheduler = make_scheduler(running=[a, b], max_num_running_reqs=2, cfg=cfg)

    scheduler._apply_sslo_policy(0.0)

    assert len(scheduler._sslo_decision_buffer) == 2


def test_decision_log_tier_changes_emits_on_transition(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    # Large heartbeat so the second step only fires on tier/selection changes.
    cfg = SsloConfig(
        method="progress_serve", decision_log_mode="tier_changes",
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
        method="progress_serve", decision_log_mode="admit_only",
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
        method="progress_serve", decision_log_mode="tier_changes",
        decision_heartbeat_steps=10000)
    b = make_request("b", make_state(deadline=10.0, expected_len=2.0))
    sched2 = make_scheduler(running=[b], max_num_running_reqs=2, cfg=cfg2)
    sched2._apply_sslo_policy(0.0)
    assert len(sched2._sslo_decision_buffer) >= admit_only_first


def test_decision_log_buffer_flushes_on_overflow(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SSLO_STATS_LOG_PATH", str(tmp_path / "scheduler_stats.jsonl"))
    cfg = SsloConfig(method="progress_serve", decision_log_mode="step")
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
