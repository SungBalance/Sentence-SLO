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

from vllm.sslo import progress_serve as ps
from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import Phase, RequestSLOState
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler, SsloStepState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus


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

    # For end-to-end schedule_sslo() waiting-loop tests: trivial cache-miss
    # allocation so a WAITING request is always admissible.
    def get_computed_blocks(self, request):
        return self.empty_kv_cache_blocks, 0

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        return SimpleNamespace(
            get_block_ids=lambda allow_none=False: ([1], ))


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
    # SSLO: KV-offload tier state normally seeded by Scheduler.__init__.
    scheduler._sslo_offloaded = {}
    scheduler._sslo_promoting = {}
    scheduler._sslo_last_onload_step = {}
    scheduler._sslo_vacated_req_ids = set()
    scheduler._sslo_offload_conn = None
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


# ---------------------------------------------------------------------------
# KV-offload tier (stage 2 engine integration)
# ---------------------------------------------------------------------------


class FakeWaitingQueue(list):
    """Minimal RequestQueue stand-in supporting front insertion + iteration."""

    def prepend_request(self, request):
        self.insert(0, request)

    def remove_requests(self, requests):
        rm = {r.request_id for r in requests}
        self[:] = [r for r in self if r.request_id not in rm]

    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.pop(0)


class FakeOffloadConnector:
    """SimpleCPUOffloadConnector stand-in with controllable mirror/pin state."""

    def __init__(self, mirrored=None, pin_ok=None):
        self.mirrored = mirrored or {}
        self.pin_ok = pin_ok or {}
        self.pinned = set()
        self.unpinned = []

    def is_fully_mirrored(self, request):
        return self.mirrored.get(request.request_id, False)

    def pin_request_cpu_blocks(self, request):
        ok = self.pin_ok.get(request.request_id, True)
        if ok:
            self.pinned.add(request.request_id)
        return ok

    def unpin_request_cpu_blocks(self, request):
        self.unpinned.append(request.request_id)
        self.pinned.discard(request.request_id)


def make_offload_request(request_id, state, *, status):
    req = make_request(request_id, state)
    req.status = status
    req.num_computed_tokens = 0
    req.num_output_placeholders = 0
    req.num_preemptions = 0
    req.spec_token_ids = []
    req.block_hashes = []
    return req


def _result(*, deferred, k_star, k_star_unconstrained, kv_capped):
    return ps.ScheduleResult(
        scheduled_A=[], deferred_A=list(deferred), k_star=k_star, e_viol=0.0,
        k_star_unconstrained=k_star_unconstrained, kv_capped=kv_capped)


def test_kv_offload_disabled_leaves_offload_state_empty():
    # Default cfg has kv_offload=False: the offload path is fully gated off.
    reqs = [
        make_request(f"r{i}", make_state(deadline=10.0, expected_len=3.0))
        for i in range(3)
    ]
    scheduler = make_scheduler(running=reqs, max_num_running_reqs=2)

    scheduler._apply_sslo_policy(0.0)

    assert scheduler._sslo_offload_conn is None
    assert scheduler._sslo_offloaded == {}
    assert scheduler._sslo_promoting == {}
    assert scheduler._sslo_vacated_req_ids == set()


def test_vacate_only_fully_mirrored_when_kv_capped():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    p0 = make_offload_request(
        "p0", make_state(deadline=1000.0), status=RequestStatus.RUNNING)
    p1 = make_offload_request(
        "p1", make_state(deadline=1000.0), status=RequestStatus.RUNNING)
    p2 = make_offload_request(
        "p2", make_state(deadline=1000.0), status=RequestStatus.RUNNING)
    # Slack-deep tail (M_cpu ~ 0 <= eps) → all vacate-eligible.
    for r in (p0, p1, p2):
        r.slo_state.length_tail_prob = lambda x: 0.0
    scheduler = make_scheduler(
        pending=[p0, p1, p2], max_num_running_reqs=3, cfg=cfg)
    scheduler.waiting = FakeWaitingQueue()  # _preempt_request enqueues here
    scheduler.log_stats = False
    # p1 is fully mirrored but pin fails → skipped; p2 not mirrored → skipped.
    conn = FakeOffloadConnector(
        mirrored={"p0": True, "p1": True, "p2": False},
        pin_ok={"p0": True, "p1": False})
    scheduler._sslo_offload_conn = conn
    new_pending = [p0, p1, p2]

    scheduler._sslo_apply_offload(
        0.0, _result(deferred=["p0", "p1", "p2"], k_star=0,
                     k_star_unconstrained=2, kv_capped=True),
        b=3, delta=1.0, admitted=[p0, p1, p2], new_pending=new_pending)

    assert "p0" in scheduler._sslo_offloaded
    assert "p1" not in scheduler._sslo_offloaded
    assert "p2" not in scheduler._sslo_offloaded
    assert new_pending == [p1, p2]
    assert scheduler._sslo_vacated_req_ids == {"p0"}
    assert p0.status == RequestStatus.PREEMPTED
    assert p0.num_computed_tokens == 0
    assert conn.pinned == {"p0"}
    assert scheduler._sslo_step.num_vacated == 1
    assert p0.slo_state.num_offload_intervals == 1
    # Minor: the vacate step itself is charged exactly one offloaded iter.
    assert p0.slo_state.chunk_stats._current_offloaded_iters == 1


def test_vacate_skipped_when_not_kv_capped():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    p0 = make_offload_request(
        "p0", make_state(deadline=1000.0), status=RequestStatus.RUNNING)
    p0.slo_state.length_tail_prob = lambda x: 0.0
    scheduler = make_scheduler(pending=[p0], max_num_running_reqs=2, cfg=cfg)
    scheduler._sslo_offload_conn = FakeOffloadConnector(mirrored={"p0": True})
    new_pending = [p0]

    scheduler._sslo_apply_offload(
        0.0, _result(deferred=["p0"], k_star=1, k_star_unconstrained=1,
                     kv_capped=False),
        b=2, delta=1.0, admitted=[p0], new_pending=new_pending)

    assert scheduler._sslo_offloaded == {}
    assert new_pending == [p0]
    assert scheduler._sslo_step.num_vacated == 0


def test_offloaded_request_accounts_step_and_makes_cpu_view():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    req = make_offload_request(
        "c0", make_state(deadline=1000.0), status=RequestStatus.PREEMPTED)
    scheduler = make_scheduler(cfg=cfg)
    scheduler._sslo_offload_conn = FakeOffloadConnector()
    scheduler._sslo_offloaded = {"c0": req}

    scheduler._sslo_offload_prologue(now=0.0)

    # One offloaded step charged to the current chunk window.
    assert req.slo_state.chunk_stats._current_offloaded_iters == 1
    # The scheduler builds a LOC_CPU view for it (R_defer_cpu → E_viol).
    view = scheduler._sslo_offload_view(req, ps.LOC_CPU, 0.0)
    assert view.location == ps.LOC_CPU
    assert view.is_measurable()


def test_promote_forced_insert_off_budget_and_reserves_slot():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    o0 = make_offload_request(
        "o0", make_state(deadline=5.0), status=RequestStatus.PREEMPTED)
    # r_defer_cpu >= 1 forces promotion regardless of eps.
    o0.slo_state.length_tail_prob = lambda x: 1.0
    scheduler = make_scheduler(max_num_running_reqs=2, cfg=cfg)
    scheduler.waiting = FakeWaitingQueue()
    scheduler._sslo_offload_conn = FakeOffloadConnector()
    scheduler._sslo_offloaded = {"o0": o0}
    scheduler._sslo_step.waiting_admission_budget = 1  # k* pre-set by policy

    scheduler._sslo_apply_offload(
        0.0, _result(deferred=[], k_star=1, k_star_unconstrained=1,
                     kv_capped=False),
        b=2, delta=1.0, admitted=[], new_pending=[])

    assert "o0" in scheduler._sslo_promoting
    assert "o0" not in scheduler._sslo_offloaded
    assert list(scheduler.waiting)[0] is o0
    assert scheduler._sslo_step.num_promoted == 1
    # Promote must not consume the k* admission budget.
    assert scheduler._sslo_step.waiting_admission_budget == 1
    # A promoting req reserves a decode slot via its LOC_ONLOADING view.
    view = scheduler._sslo_offload_view(o0, ps.LOC_ONLOADING, 0.0)
    assert view.location == ps.LOC_ONLOADING


def test_onload_completion_fires_offload_exit_and_records_residency():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    req = make_offload_request(
        "o0", make_state(deadline=5.0), status=RequestStatus.RUNNING)
    scheduler = make_scheduler(running=[req], max_num_running_reqs=2, cfg=cfg)
    conn = FakeOffloadConnector()
    scheduler._sslo_offload_conn = conn
    scheduler._sslo_promoting = {"o0": req}
    scheduler._sslo_step_idx = 7
    req.slo_state.on_offload_enter(1.0)  # open an offload interval to close

    scheduler._sslo_offload_prologue(now=3.0)

    assert "o0" not in scheduler._sslo_promoting
    assert scheduler._sslo_last_onload_step["o0"] == 7
    assert conn.unpinned == ["o0"]
    assert req.slo_state.num_onloads == 1


def test_residency_guard_blocks_revacate():
    cfg = SsloConfig(
        method="progress_serve", kv_offload=True,
        kv_offload_min_residency_steps=5)
    p0 = make_offload_request(
        "p0", make_state(deadline=1000.0), status=RequestStatus.RUNNING)
    p0.slo_state.length_tail_prob = lambda x: 0.0
    scheduler = make_scheduler(pending=[p0], max_num_running_reqs=2, cfg=cfg)
    scheduler._sslo_offload_conn = FakeOffloadConnector(mirrored={"p0": True})
    scheduler._sslo_step_idx = 3
    scheduler._sslo_last_onload_step = {"p0": 1}  # 3 - 1 = 2 < 5 → guarded
    new_pending = [p0]

    scheduler._sslo_apply_offload(
        0.0, _result(deferred=["p0"], k_star=0, k_star_unconstrained=1,
                     kv_capped=True),
        b=2, delta=1.0, admitted=[p0], new_pending=new_pending)

    assert scheduler._sslo_offloaded == {}
    assert new_pending == [p0]


def test_finish_cleanup_unpins():
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    req = make_offload_request(
        "c0", make_state(), status=RequestStatus.PREEMPTED)
    scheduler = make_scheduler(cfg=cfg)
    conn = FakeOffloadConnector()
    scheduler._sslo_offload_conn = conn
    scheduler._sslo_offloaded = {"c0": req}

    scheduler._sslo_clear_pending_state(req, now=1.0)

    assert "c0" not in scheduler._sslo_offloaded
    assert conn.unpinned == ["c0"]


# ---------------------------------------------------------------------------
# End-to-end schedule_sslo() waiting-loop coverage. Drives the real
# waiting-admission loop (running=[] to skip the RUNNING loop); the policy and
# output-construction tail are stubbed so the loop's admission / onload
# traversal branches (scheduler.py) are exercised directly.
# ---------------------------------------------------------------------------


def _waiting_request(request_id):
    req = SimpleNamespace(
        request_id=request_id, slo_state=None, has_encoder_inputs=False,
        status=RequestStatus.WAITING, num_computed_tokens=0, num_tokens=1,
        num_prompt_tokens=1, num_output_placeholders=0, prefill_stats=None,
        spec_token_ids=[], lora_request=None)
    return req


def _prep_schedule_sslo(scheduler, *, budget, cur_max, monkeypatch):
    # Attrs the real waiting loop / output construction read but which
    # make_scheduler (bypassing __init__) does not set.
    scheduler.max_num_scheduled_tokens = 1000
    scheduler.scheduler_config = SimpleNamespace(
        async_scheduling=False, long_prefill_token_threshold=0,
        enable_chunked_prefill=True)
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_lookahead_tokens = 0
    scheduler.use_eagle = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.kv_cache_manager.empty_kv_cache_blocks = None

    def stub_policy(now):
        scheduler._sslo_step.waiting_admission_budget = budget
        scheduler._sslo_step.cur_max_num_requests = cur_max
        return {}

    scheduler._apply_sslo_policy = stub_policy
    scheduler._make_cached_request_data = lambda *a, **k: None
    scheduler._record_step_for_next_ema = lambda *a, **k: None
    from vllm.v1.core.sched.output import NewRequestData
    monkeypatch.setattr(
        NewRequestData, "from_request",
        staticmethod(lambda *a, **k: object()))


def test_schedule_sslo_disabled_admits_up_to_budget(monkeypatch):
    # kv_offload disabled: the waiting loop admits exactly k* FCFS requests.
    scheduler = make_scheduler(running=[], max_num_running_reqs=8)
    scheduler.waiting = FakeWaitingQueue(
        [_waiting_request(f"n{i}") for i in range(3)])
    _prep_schedule_sslo(scheduler, budget=2, cur_max=8, monkeypatch=monkeypatch)

    scheduler.schedule_sslo()

    # budget=2 → exactly n0, n1 admitted FCFS; n2 stays in the waiting queue.
    assert request_ids(scheduler.running) == ["n0", "n1"]
    assert request_ids(scheduler.waiting) == ["n2"]


def test_schedule_sslo_promoted_traverses_off_budget(monkeypatch):
    # kv_offload enabled: a promoted onload req (in _sslo_promoting, front of
    # waiting) is admitted WITHOUT spending the k* budget, so with budget=1 both
    # the promoted req and one normal req are admitted.
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    scheduler = make_scheduler(running=[], max_num_running_reqs=8, cfg=cfg)
    promoted = _waiting_request("promoted")
    promoted.status = RequestStatus.PREEMPTED
    normals = [_waiting_request(f"n{i}") for i in range(2)]
    scheduler.waiting = FakeWaitingQueue([promoted, *normals])
    scheduler._sslo_offload_conn = FakeOffloadConnector()
    scheduler._sslo_promoting = {"promoted": promoted}
    _prep_schedule_sslo(scheduler, budget=1, cur_max=8, monkeypatch=monkeypatch)

    scheduler.schedule_sslo()

    admitted = request_ids(scheduler.running)
    assert "promoted" in admitted  # off-budget onload traversal
    assert "n0" in admitted        # one normal admit under budget=1
    assert "n1" not in admitted    # budget spent → not admitted
    assert len(admitted) == 2


def test_promoting_req_excluded_from_waiting_views(monkeypatch):
    # Bug 1: a promoted onload req still queued in self.waiting must NOT be
    # counted as a new admit (waiting_views) — its LOC_ONLOADING view is the
    # sole representative. Otherwise it would occupy two slots / +2 in |A+|.
    cfg = SsloConfig(method="progress_serve", kv_offload=True)
    onloading = make_offload_request(
        "o0", make_state(deadline=5.0), status=RequestStatus.PREEMPTED)
    normal = make_request("w0", make_state(deadline=10.0, expected_len=3.0))
    scheduler = make_scheduler(running=[], max_num_running_reqs=4, cfg=cfg)
    scheduler.waiting = FakeWaitingQueue([onloading, normal])
    scheduler._sslo_offload_conn = FakeOffloadConnector()
    scheduler._sslo_promoting = {"o0": onloading}

    captured = {}
    real_schedule_step = ps.schedule_step

    def spy(views_A, waiting_views, b, delta, kv_feasible, lead=0):
        captured["waiting_ids"] = [v.request_id for v in waiting_views]
        captured["onloading_locs"] = [
            v.location for v in views_A if v.request_id == "o0"]
        return real_schedule_step(
            views_A, waiting_views, b, delta, kv_feasible, lead)

    monkeypatch.setattr(ps, "schedule_step", spy)

    scheduler._apply_sslo_policy(0.0)

    assert "o0" not in captured["waiting_ids"]  # excluded as new admit
    assert "w0" in captured["waiting_ids"]      # normal waiting req still counted
    # The onload req appears exactly once, as its LOC_ONLOADING view.
    assert captured["onloading_locs"] == [ps.LOC_ONLOADING]
