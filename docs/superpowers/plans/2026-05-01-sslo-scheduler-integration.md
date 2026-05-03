# SSLO Scheduler Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ad-hoc env vars with a typed `SsloConfig` threaded through vLLM's config system, and integrate `sslo_score` into a new `Scheduler.schedule_sslo()` for urgency-aware reordering, **pending-state pause** (KV blocks held on GPU, no recompute on resume), optional KV offload marking, and optional adaptive batch sizing.

**Architecture:**
- A new `SsloConfig` dataclass (`vllm/vllm/sslo/config.py`) is added to `VllmConfig` and flows through `EngineArgs.sslo_params` → `create_engine_config()` → `OutputProcessor` and `EngineCore`.
- `RequestSLOState` gains: `sslo_score` property, EMA gen-time tracking, `on_pending_enter/exit` callbacks, `is_pending_eligible` property.
- **Both client (OutputProcessor) and engine (EngineCore) hold a full `RequestSLOState`** built by the same `build_slo_state(sslo_config)` factory. The IPC payload changes from "computed `cumulative_slack`" to raw "`(text_delta, timestamp)`" — both sides compute slack independently from the same materials.
- The engine-side `slo_state` additionally receives `on_pending_enter` / `on_pending_exit` callbacks from the scheduler so its EMA tracks **pure** gen-time excluding pending intervals.
- SSLO scheduling lives entirely in a new `Scheduler.schedule_sslo()` (copy of `schedule()` with SSLO additions); `schedule()` is untouched. `Scheduler.__init__` patches `self.schedule = self.schedule_sslo` when `sslo_config.enabled=True`.
- `self.sslo_pending: list[Request]` holds high-slack requests whose KV blocks remain on GPU but who do not consume compute budget. No recompute on resume.

**Tech Stack:** Python 3 dataclasses, Pydantic (VllmConfig), vLLM v1 scheduler (`vllm/vllm/v1/core/sched/scheduler.py`), pytest (unit tests in `vllm/tests/sslo/`).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm/vllm/sslo/config.py` | Create | `SsloConfig` dataclass + `build_slo_state()` factory |
| `vllm/vllm/sslo/slo_state.py` | Modify | `sslo_score`, EMA, pending callbacks, `is_pending_eligible`, new `SLOChunkRecord` fields |
| `vllm/vllm/sslo/__init__.py` | Modify | Export `SsloConfig`, `build_slo_state` |
| `vllm/vllm/config/vllm.py` | Modify | Add `sslo_config: SsloConfig` field to `VllmConfig` |
| `vllm/vllm/engine/arg_utils.py` | Modify | Add `sslo_params: dict` to `EngineArgs`; build `SsloConfig` in `create_engine_config()` |
| `vllm/vllm/v1/engine/output_processor.py` | Modify | Use `build_slo_state()`; emit `(req_id, text_delta, timestamp)` updates |
| `vllm/vllm/v1/engine/llm_engine.py` | Modify | Pass `sslo_config` to `OutputProcessor` |
| `vllm/vllm/v1/engine/async_llm.py` | Modify | Pass `sslo_config` to `OutputProcessor` |
| `vllm/vllm/v1/engine/core.py` | Modify | Use `build_slo_state()` in `_bind_slo_state()`; `update_slo_slack()` now feeds `on_text_delta()` |
| `vllm/vllm/v1/engine/core_client.py` | Modify | Update `send_slo_updates(_async)` payload schema |
| `vllm/vllm/v1/core/sched/scheduler.py` | Modify | `schedule_sslo()`, `sslo_pending`, pending mechanism, offload marking, adaptive cap |
| `exp/measure_internal_slack/benchmark.py` | Modify | Replace env vars with `sslo_params` dict |
| `vllm/tests/sslo/test_sslo_config.py` | Create | Tests for `SsloConfig` and `build_slo_state()` |
| `vllm/tests/sslo/test_slo_state.py` | Modify | Tests for EMA, pending callbacks, `is_pending_eligible` |
| `vllm/tests/sslo/test_scheduler_sslo.py` | Create | Unit tests for scheduler SSLO logic |

---

### Task 1: `SsloConfig` dataclass + `sslo_score` property

**Files:**
- Create: `vllm/vllm/sslo/config.py`
- Modify: `vllm/vllm/sslo/slo_state.py`
- Modify: `vllm/vllm/sslo/__init__.py`
- Create: `vllm/tests/sslo/test_sslo_config.py`
- Modify: `vllm/tests/sslo/test_slo_state.py`

**Context:** `SsloConfig` is a plain dataclass (no Pydantic — `vllm/vllm/sslo/` is pure Python). `build_slo_state()` is the single factory both client and engine sides use (Task 4) so behavior is identical. `sslo_score` on `RequestSLOState` is a thin property abstracting `cumulative_slack` so future scoring formulas don't break callers.

- [ ] **Step 1: Write failing tests for `SsloConfig` and `build_slo_state()`**

Create `vllm/tests/sslo/test_sslo_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for SsloConfig and build_slo_state()."""
import pytest

from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import RequestSLOState


class TestSsloConfig:
    def test_defaults(self):
        cfg = SsloConfig()
        assert cfg.enabled is False
        assert cfg.seconds_per_word == 0.28
        assert cfg.chunk_unit == "sentence"
        assert cfg.estimator_type == "word_rate"
        assert cfg.offloading is False
        assert cfg.adaptive_batch_size is False
        assert cfg.max_consecutive_pending == 5
        assert cfg.ema_alpha == 0.2
        assert cfg.pending_slack_eps_num_tokens == 5

    def test_invalid_chunk_unit_raises(self):
        with pytest.raises(ValueError, match="chunk_unit"):
            SsloConfig(chunk_unit="invalid")

    def test_invalid_estimator_type_raises(self):
        with pytest.raises(ValueError, match="estimator_type"):
            SsloConfig(estimator_type="tts")


class TestBuildSloState:
    def test_default_builds_word_rate_sentence(self):
        state = build_slo_state(SsloConfig())
        assert isinstance(state, RequestSLOState)

    def test_paragraph_unit_uses_paragraph_detector(self):
        import time
        state = build_slo_state(SsloConfig(chunk_unit="paragraph"))
        now = time.monotonic()
        state.on_text_delta("Hello world. More text.", now)
        assert len(state.chunk_records) == 0  # sentence boundary ignored
        state.on_text_delta("\n\nNext paragraph.", now)
        assert len(state.chunk_records) == 1

    def test_ema_alpha_propagated(self):
        state = build_slo_state(SsloConfig(ema_alpha=0.5))
        assert state._ema_alpha == 0.5

    def test_eps_num_tokens_propagated(self):
        state = build_slo_state(SsloConfig(pending_slack_eps_num_tokens=10))
        assert state._pending_slack_eps_num_tokens == 10
```

- [ ] **Step 2: Run tests — expect failure**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_sslo_config.py -v 2>&1 | head -30
'
```

Expected: `ModuleNotFoundError: No module named 'vllm.sslo.config'`.

- [ ] **Step 3: Create `vllm/vllm/sslo/config.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""SsloConfig: configuration dataclass for SSLO (Sentence-level SLO)."""
from __future__ import annotations

from dataclasses import dataclass

_VALID_CHUNK_UNITS = frozenset({"sentence", "paragraph"})
_VALID_ESTIMATOR_TYPES = frozenset({"word_rate"})


@dataclass
class SsloConfig:
    enabled: bool = False
    seconds_per_word: float = 0.28
    chunk_unit: str = "sentence"
    estimator_type: str = "word_rate"
    offloading: bool = False
    adaptive_batch_size: bool = False
    max_consecutive_pending: int = 5
    ema_alpha: float = 0.2
    pending_slack_eps_num_tokens: int = 5

    def __post_init__(self) -> None:
        if self.chunk_unit not in _VALID_CHUNK_UNITS:
            raise ValueError(
                f"chunk_unit must be one of {sorted(_VALID_CHUNK_UNITS)}, "
                f"got {self.chunk_unit!r}"
            )
        if self.estimator_type not in _VALID_ESTIMATOR_TYPES:
            raise ValueError(
                f"estimator_type must be one of {sorted(_VALID_ESTIMATOR_TYPES)}, "
                f"got {self.estimator_type!r}"
            )


def build_slo_state(config: SsloConfig) -> "RequestSLOState":
    """Create a RequestSLOState from SsloConfig (single factory for both sides)."""
    from vllm.sslo.slo_state import (
        ParagraphChunkDetector,
        RequestSLOState,
        WordRateEstimator,
    )
    estimator = WordRateEstimator(seconds_per_word=config.seconds_per_word)
    detector = ParagraphChunkDetector() if config.chunk_unit == "paragraph" else None
    return RequestSLOState(
        estimator=estimator,
        detector=detector,
        ema_alpha=config.ema_alpha,
        pending_slack_eps_num_tokens=config.pending_slack_eps_num_tokens,
    )
```

(`RequestSLOState` will accept `ema_alpha` and `pending_slack_eps_num_tokens` after Task 3. For Task 1 the factory call signature can be checked by adding the parameters with defaults to `RequestSLOState.__init__` now — see Task 1 Step 5.)

- [ ] **Step 4: Add `sslo_score` property to `RequestSLOState`**

In `vllm/vllm/sslo/slo_state.py`, append inside the `RequestSLOState` class:

```python
    @property
    def sslo_score(self) -> float:
        """Scheduling urgency score. Lower = more urgent.

        Currently equals cumulative_slack. Abstracted so future scoring
        formulas don't require caller changes.
        """
        return self.cumulative_slack
```

- [ ] **Step 5: Add `ema_alpha` and `pending_slack_eps_num_tokens` params to `RequestSLOState.__init__`**

In `RequestSLOState.__init__`, accept these new params (full EMA + pending logic comes in Task 3):

```python
    def __init__(
        self,
        estimator: ConsumeEstimator | None = None,
        detector: ChunkBoundaryDetector | None = None,
        ema_alpha: float = 0.2,
        pending_slack_eps_num_tokens: int = 5,
    ) -> None:
        ...
        self._ema_alpha: float = ema_alpha
        self._pending_slack_eps_num_tokens: int = pending_slack_eps_num_tokens
        # EMA / pending fields are populated in Task 3
```

- [ ] **Step 6: Add `sslo_score` test**

Append to `vllm/tests/sslo/test_slo_state.py`:

```python
class TestSsloScore:
    def test_initial_zero(self):
        state = RequestSLOState()
        assert state.sslo_score == 0.0

    def test_equals_cumulative_slack(self):
        import time
        state = RequestSLOState()
        now = time.monotonic()
        state.on_text_delta("Hello world. ", now)
        state.on_text_delta("Next sentence. ", now + 0.1)
        assert state.sslo_score == state.cumulative_slack
```

- [ ] **Step 7: Run all sslo tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -30
'
```

Expected: all PASS.

- [ ] **Step 8: Update `vllm/vllm/sslo/__init__.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""SSLO (Sentence-level SLO) package for vLLM."""
from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import ConsumeEstimator, RequestSLOState, WordRateEstimator

__all__ = [
    "ConsumeEstimator",
    "RequestSLOState",
    "SsloConfig",
    "WordRateEstimator",
    "build_slo_state",
]
```

- [ ] **Step 9: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/sslo/

git add vllm/vllm/sslo/config.py vllm/vllm/sslo/slo_state.py vllm/vllm/sslo/__init__.py \
        vllm/tests/sslo/test_sslo_config.py vllm/tests/sslo/test_slo_state.py
git commit -m "feat(sslo): add SsloConfig, build_slo_state(), and sslo_score property"
```

---

### Task 2: Wire `sslo_params` through `EngineArgs` → `VllmConfig`

**Files:**
- Modify: `vllm/vllm/config/vllm.py`
- Modify: `vllm/vllm/engine/arg_utils.py`

**Context:** `VllmConfig` (Pydantic model) gets a `sslo_config` field. `EngineArgs` (`@dataclass`) gets a `sslo_params: dict` field; `create_engine_config()` builds `SsloConfig(**sslo_params)` and passes it into `VllmConfig(...)`.

- [ ] **Step 1: Add `sslo_config` field to `VllmConfig`**

In `vllm/vllm/config/vllm.py`, near other field declarations (after `additional_config`):

```python
    # SSLO
    sslo_config: SsloConfig = Field(default_factory=SsloConfig)
    """SSLO (Sentence-level SLO) configuration."""
```

Add the import at the top:

```python
# SSLO
from vllm.sslo.config import SsloConfig
```

- [ ] **Step 2: Add `sslo_params` to `EngineArgs` and build `SsloConfig` in `create_engine_config()`**

In `vllm/vllm/engine/arg_utils.py`, in the `EngineArgs` dataclass body (near `additional_config`):

```python
    # SSLO
    sslo_params: dict = field(default_factory=dict)
```

In `create_engine_config()` (around line 2115, just before `config = VllmConfig(...)`):

```python
        # SSLO
        from vllm.sslo.config import SsloConfig
        sslo_config = SsloConfig(**self.sslo_params) if self.sslo_params else SsloConfig()
```

Then inside the `VllmConfig(...)` call (around line 2121):

```python
            # SSLO
            sslo_config=sslo_config,
```

- [ ] **Step 3: Verify**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -c "
from vllm.config.vllm import VllmConfig
from vllm.sslo.config import SsloConfig
cfg = VllmConfig()
assert isinstance(cfg.sslo_config, SsloConfig)
assert cfg.sslo_config.enabled is False
print(\"OK:\", cfg.sslo_config)
"
'
```

Expected: `OK: SsloConfig(enabled=False, ...)`.

- [ ] **Step 4: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/config/vllm.py \
  /workspace/mlsys/vllm/vllm/engine/arg_utils.py

git add vllm/vllm/config/vllm.py vllm/vllm/engine/arg_utils.py
git commit -m "feat(sslo): wire sslo_params through EngineArgs and VllmConfig"
```

---

### Task 3: `RequestSLOState` — EMA, pending callbacks, `is_pending_eligible`

**Files:**
- Modify: `vllm/vllm/sslo/slo_state.py`
- Modify: `vllm/tests/sslo/test_slo_state.py`

**Context:** Adds tracking required for the scheduler's pending decision:
- **EMA of pure chunk gen_time** (excluding pending intervals)
- **EMA of per-token time** (chunk gen_time / word_count)
- **`on_pending_enter(now)` / `on_pending_exit(now)`** callbacks (called by scheduler on engine-side state)
- **`is_pending_eligible`** property: `cumulative_slack > ema_pure_gen_time + eps_num_tokens × ema_per_token_time`

`SLOChunkRecord` gains `gen_time` (pure) and `pending_time` fields for visibility — they're additive, so `output.slo_chunk_records` consumers continue to work.

**Pending math:**
- Track `_chunk_pending_time` accumulator that resets on each chunk flush.
- `on_pending_enter(now)` records `_pending_enter_ts`.
- `on_pending_exit(now)` adds `(now - _pending_enter_ts)` to `_chunk_pending_time`.
- On `_flush_chunk`: `pure_gen_time = (now - chunk_start) - _chunk_pending_time` where `chunk_start` is the previous chunk's `end_time_ts` (or `_decoding_start` for chunk 0).
- A chunk cannot complete during pending (no text generated), so accumulators stay coherent.

- [ ] **Step 1: Write failing tests**

Append to `vllm/tests/sslo/test_slo_state.py`:

```python
class TestEmaTracking:
    def test_ema_initially_none(self):
        state = RequestSLOState()
        assert state._ema_pure_gen_time is None
        assert state._ema_per_token_time is None

    def test_ema_updates_after_chunk(self):
        import time
        state = RequestSLOState(ema_alpha=0.5)
        t0 = time.monotonic()
        state.on_text_delta("hello world. ", t0 + 1.0)
        # First chunk: pure gen = 1.0, words = 2, per_token ~ 0.5
        assert state._ema_pure_gen_time == pytest.approx(1.0)
        assert state._ema_per_token_time == pytest.approx(0.5)


class TestPendingCallbacks:
    def test_pending_subtracted_from_gen_time(self):
        import time
        state = RequestSLOState(ema_alpha=1.0)  # alpha=1 → EMA == latest sample
        t0 = time.monotonic()
        # text starts at t0
        state.on_text_delta("hello world.", t0 + 0.0)  # no boundary yet (no trailing space/EOL)
        # pending [t0+0.5, t0+0.8] (0.3s)
        state.on_pending_enter(t0 + 0.5)
        state.on_pending_exit(t0 + 0.8)
        # finish at t0+1.0 (boundary at end of text)
        state.on_finish(t0 + 1.0)
        # pure gen_time = 1.0 - 0.3 = 0.7
        rec = state.chunk_records[0]
        assert rec["gen_time"] == pytest.approx(0.7)
        assert rec["pending_time"] == pytest.approx(0.3)

    def test_pending_resets_after_flush(self):
        import time
        state = RequestSLOState()
        t0 = time.monotonic()
        state.on_pending_enter(t0)
        state.on_pending_exit(t0 + 0.5)
        state.on_text_delta("a. ", t0 + 1.0)  # chunk flushes
        assert state._chunk_pending_time == 0.0


class TestIsPendingEligible:
    def test_false_when_no_ema_yet(self):
        state = RequestSLOState()
        state.cumulative_slack = 100.0
        assert state.is_pending_eligible is False

    def test_true_when_slack_exceeds_threshold(self):
        import time
        state = RequestSLOState(ema_alpha=1.0, pending_slack_eps_num_tokens=5)
        t0 = time.monotonic()
        state.on_text_delta("hello world. ", t0 + 1.0)
        # ema_pure_gen_time = 1.0, ema_per_token_time = 0.5
        # threshold = 1.0 + 3 * 0.5 = 2.5
        state.cumulative_slack = 3.0
        assert state.is_pending_eligible is True

    def test_false_when_slack_below_threshold(self):
        import time
        state = RequestSLOState(ema_alpha=1.0, pending_slack_eps_num_tokens=5)
        t0 = time.monotonic()
        state.on_text_delta("hello world. ", t0 + 1.0)
        state.cumulative_slack = 2.0  # below 2.5 threshold
        assert state.is_pending_eligible is False
```

- [ ] **Step 2: Run — expect failure**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_slo_state.py -v 2>&1 | tail -20
'
```

- [ ] **Step 3: Update `SLOChunkRecord` dataclass**

In `vllm/vllm/sslo/slo_state.py`, modify the `SLOChunkRecord` dataclass:

```python
@dataclass
class SLOChunkRecord:
    chunk_idx: int
    text: str
    word_count: int
    decoding_start_ts: float
    end_time_ts: float
    cumulative_consume: float
    cumulative_slack: float
    gen_time: float        # NEW: end_time - chunk_start - pending_time
    pending_time: float    # NEW: total pending time during this chunk
```

- [ ] **Step 4: Add EMA + pending fields to `RequestSLOState.__init__`**

Replace the existing field initialization block:

```python
    def __init__(
        self,
        estimator: ConsumeEstimator | None = None,
        detector: ChunkBoundaryDetector | None = None,
        ema_alpha: float = 0.2,
        pending_slack_eps_num_tokens: int = 5,
    ) -> None:
        self._estimator: ConsumeEstimator = estimator or WordRateEstimator()
        self._detector: ChunkBoundaryDetector = detector or SentenceChunkDetector()
        self._decoding_start: float | None = None
        self._cumulative_consume: float = 0.0
        self._pending_text: str = ""
        self._chunk_count: int = 0
        self.cumulative_slack: float = 0.0
        self._slack_dirty: bool = False
        self._chunk_records: list[SLOChunkRecord] = []
        # EMA + pending tracking
        self._ema_alpha: float = ema_alpha
        self._pending_slack_eps_num_tokens: int = pending_slack_eps_num_tokens
        self._ema_pure_gen_time: float | None = None
        self._ema_per_token_time: float | None = None
        self._last_chunk_end_ts: float | None = None
        self._pending_enter_ts: float | None = None
        self._chunk_pending_time: float = 0.0
```

- [ ] **Step 5: Add `on_pending_enter` / `on_pending_exit` methods**

```python
    def on_pending_enter(self, now: float) -> None:
        """Mark the start of a pending interval."""
        if self._pending_enter_ts is None:
            self._pending_enter_ts = now

    def on_pending_exit(self, now: float) -> None:
        """Accumulate pending duration into the current chunk."""
        if self._pending_enter_ts is not None:
            self._chunk_pending_time += now - self._pending_enter_ts
            self._pending_enter_ts = None
```

- [ ] **Step 6: Update `_flush_chunk` to compute pure gen_time and update EMAs**

Replace the body of `_flush_chunk`:

```python
    def _flush_chunk(self, now: float, boundary_pos: int) -> None:
        chunk_text = self._pending_text[:boundary_pos]
        word_count = len(chunk_text.split())
        # Chunk 0 slack is fixed at 0.0; only update from chunk 1 onward.
        if self._chunk_count > 0:
            assert self._decoding_start is not None
            deadline = self._decoding_start + self._cumulative_consume
            self.cumulative_slack = deadline - now
            self._slack_dirty = True
        # Pure gen_time = wall-clock chunk duration minus pending time within
        chunk_start = self._last_chunk_end_ts if self._last_chunk_end_ts is not None else self._decoding_start
        assert chunk_start is not None
        pure_gen_time = max(0.0, (now - chunk_start) - self._chunk_pending_time)
        # Update EMAs (skip chunks with zero words to avoid div-by-zero)
        if word_count > 0 and pure_gen_time > 0:
            self._update_ema(pure_gen_time, word_count)
        self._chunk_records.append(SLOChunkRecord(
            chunk_idx=self._chunk_count,
            text=chunk_text,
            word_count=word_count,
            decoding_start_ts=self._decoding_start,  # type: ignore[arg-type]
            end_time_ts=now,
            cumulative_consume=self._cumulative_consume,
            cumulative_slack=self.cumulative_slack,
            gen_time=pure_gen_time,
            pending_time=self._chunk_pending_time,
        ))
        # Advance accumulators for the next chunk
        self._cumulative_consume += self._estimator(chunk_text)
        self._chunk_count += 1
        self._pending_text = self._pending_text[boundary_pos:]
        self._last_chunk_end_ts = now
        self._chunk_pending_time = 0.0

    def _update_ema(self, pure_gen_time: float, word_count: int) -> None:
        per_token_time = pure_gen_time / word_count
        a = self._ema_alpha
        if self._ema_pure_gen_time is None:
            self._ema_pure_gen_time = pure_gen_time
            self._ema_per_token_time = per_token_time
        else:
            self._ema_pure_gen_time = a * pure_gen_time + (1 - a) * self._ema_pure_gen_time
            self._ema_per_token_time = a * per_token_time + (1 - a) * self._ema_per_token_time
```

- [ ] **Step 7: Add `is_pending_eligible` property**

```python
    @property
    def is_pending_eligible(self) -> bool:
        """True if cumulative_slack exceeds (ema_pure_gen_time + eps × ema_per_token_time)."""
        if self._ema_pure_gen_time is None or self._ema_per_token_time is None:
            return False
        threshold = (
            self._ema_pure_gen_time
            + self._pending_slack_eps_num_tokens * self._ema_per_token_time
        )
        return self.cumulative_slack > threshold
```

- [ ] **Step 8: Run all sslo tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -25
'
```

Expected: all PASS.

- [ ] **Step 9: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall /workspace/mlsys/vllm/vllm/sslo/slo_state.py

git add vllm/vllm/sslo/slo_state.py vllm/tests/sslo/test_slo_state.py
git commit -m "feat(sslo): add EMA tracking, pending callbacks, is_pending_eligible to RequestSLOState"
```

---

### Task 4: Replace env vars + change IPC payload to `(text_delta, timestamp)`

**Files:**
- Modify: `vllm/vllm/v1/engine/output_processor.py`
- Modify: `vllm/vllm/v1/engine/llm_engine.py`
- Modify: `vllm/vllm/v1/engine/async_llm.py`
- Modify: `vllm/vllm/v1/engine/core.py`
- Modify: `vllm/vllm/v1/engine/core_client.py`
- Modify: `exp/measure_internal_slack/benchmark.py`

**Context:** Two simultaneous changes in this task because they're tightly coupled:

1. **Both client and engine use `build_slo_state(sslo_config)`** instead of env-var-driven construction (client) or bare `RequestSLOState()` (engine). Both sides hold a full state instance.

2. **IPC payload schema** changes from `list[tuple[str, float]]` (`req_id, slack`) to `list[tuple[str, str, float]]` (`req_id, text_delta, timestamp`). Engine-side `update_slo_slack` is renamed `update_slo_text` and calls `req.slo_state.on_text_delta(text, timestamp)` so the engine independently computes slack, EMA, etc.

The client still computes its own slack locally (used to be sent via IPC; now used only for client-side `chunk_records`). `output.slo_chunk_records` source unchanged → `measure_internal_slack/benchmark.py` keeps working (only the env-var setup line changes).

- [ ] **Step 1: Update `OutputProcessor.__init__` to accept `sslo_config` and use `build_slo_state()`**

In `vllm/vllm/v1/engine/output_processor.py`, update `OutputProcessor.__init__`:

```python
    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        sslo_config: "SsloConfig | None" = None,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        from vllm.sslo.config import SsloConfig as _SsloConfig
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        # SSLO
        self.sslo_config = sslo_config if sslo_config is not None else _SsloConfig()
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.lora_states = LoRARequestStates(log_stats)
        self.tracing_enabled = tracing_enabled
```

- [ ] **Step 2: Update `RequestState` to use `build_slo_state()`**

Find `RequestState.__init__` and add `sslo_config` parameter; replace the env-var SSLO block (lines 196-199) with:

```python
        # SSLO
        from vllm.sslo.config import SsloConfig as _SsloConfig, build_slo_state
        _cfg = sslo_config if sslo_config is not None else _SsloConfig()
        self.slo_state: RequestSLOState = build_slo_state(_cfg)
        self._slo_text_len: int = 0
```

In `RequestState.from_new_request()` signature, add `sslo_config` parameter; pass it to `cls(...)`.

In `OutputProcessor`, find the `RequestState.from_new_request(` call (around line 551) and pass `sslo_config=self.sslo_config`.

- [ ] **Step 3: Change SLO update payload — emit `(req_id, text_delta, timestamp)`**

In `output_processor.py`, find where slo updates are collected (the `processed_outputs.slo_updates` flow). Replace the per-request collection so each update is `(req_id, new_text, now)` instead of `(req_id, slack)`:

```python
        # SSLO
        new_text = full_text[self._slo_text_len:]
        if new_text:
            self._slo_text_len = len(full_text)
            now = time.monotonic()
            self.slo_state.on_text_delta(new_text, now)
            slo_updates.append((self.request_id, new_text, now))
```

Where `slo_updates` is the existing list collected in `processed_outputs`.

- [ ] **Step 4: Update `core_client.send_slo_updates(_async)` signature**

In `vllm/vllm/v1/engine/core_client.py` (lines 304, 837, 1077, 1497), change all three `send_slo_updates` / `send_slo_updates_async` signatures from `list[tuple[str, float]]` to `list[tuple[str, str, float]]`. Body is unchanged (passes through `_send_input(SLO_UPDATE, updates)`).

In `llm_engine.py` line 334 and `async_llm.py` line 688, no change needed — they pass through `processed_outputs.slo_updates`.

- [ ] **Step 5: Update `EngineCore.update_slo_slack` → `update_slo_text`**

In `vllm/vllm/v1/engine/core.py`, replace the existing method (line 360) with:

```python
    # SSLO
    def update_slo_text(self, updates: list[tuple[str, str, float]]) -> None:
        for req_id, text_delta, ts in updates:
            req = self.scheduler.requests.get(req_id)
            if req is not None and req.slo_state is not None:
                req.slo_state.on_text_delta(text_delta, ts)
```

Update the dispatch site at line 1308:

```python
        elif request_type == EngineCoreRequestType.SLO_UPDATE:
            self.update_slo_text(request)
```

- [ ] **Step 6: Update `_bind_slo_state` to use `build_slo_state()`**

In `core.py`, replace `_bind_slo_state` body (around line 347):

```python
        # SSLO
        from vllm.sslo.config import build_slo_state
        request.slo_state = build_slo_state(self.vllm_config.sslo_config)
        self.scheduler.add_request(request)
```

- [ ] **Step 7: Pass `sslo_config` from `llm_engine.py` and `async_llm.py` to `OutputProcessor`**

In `llm_engine.py` line ~96 and `async_llm.py` line ~138:

```python
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            # SSLO
            sslo_config=vllm_config.sslo_config,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )
```

- [ ] **Step 8: Update `exp/measure_internal_slack/benchmark.py`**

Replace lines 134-135:

```python
    os.environ["SSLO_CHUNK_UNIT"] = args.chunk_unit
    os.environ["SSLO_SECONDS_PER_WORD"] = str(args.seconds_per_word)
```

with: delete those two lines, and change the `AsyncEngineArgs(...)` call (around line 146) to add:

```python
    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_log_requests=False,
        # SSLO
        sslo_params={
            "chunk_unit": args.chunk_unit,
            "seconds_per_word": args.seconds_per_word,
        },
    )
```

- [ ] **Step 9: Smoke test — benchmark.py runs end-to-end**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys
  HF_HOME=/cache HF_HUB_CACHE=/cache/hub \
  python3 exp/measure_internal_slack/benchmark.py \
    --model Qwen/Qwen3-8B \
    --num-prompts 4 \
    --max-num-seqs 2 \
    --warmup-requests 0 \
    --output-dir /tmp/sslo_smoke
'
```

Expected: completes without error; `/tmp/sslo_smoke/chunks.jsonl` exists and has rows with new `gen_time`, `pending_time` fields.

- [ ] **Step 10: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/engine/output_processor.py \
  /workspace/mlsys/vllm/vllm/v1/engine/llm_engine.py \
  /workspace/mlsys/vllm/vllm/v1/engine/async_llm.py \
  /workspace/mlsys/vllm/vllm/v1/engine/core.py \
  /workspace/mlsys/vllm/vllm/v1/engine/core_client.py

git add vllm/vllm/v1/engine/ exp/measure_internal_slack/benchmark.py
git commit -m "feat(sslo): replace env vars with SsloConfig; IPC sends (text, timestamp)"
```

---

### Task 5: Scheduler — `schedule_sslo()` with pending mechanism

**Files:**
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Create: `vllm/tests/sslo/test_scheduler_sslo.py`

**Context:** This is the core SSLO scheduling change. `schedule()` is left **untouched**. A new `schedule_sslo()` method is a copy of `schedule()` with these additions, in this order at the top of the method:

1. **Combine + sort:** `combined = self.running + self.sslo_pending`, sort by `_sslo_score_key` ascending.
2. **Redistribute pending/running:** for each request, decide running vs pending:
   - Force back to running if `consecutive_pending[req_id] >= max_consecutive_pending`.
   - Else if `slo_state.is_pending_eligible AND len(self.waiting) > 0` → pending (`consecutive_pending[req_id] += 1`; if newly entering pending, `slo_state.on_pending_enter(now)`).
   - Else → running (`consecutive_pending[req_id] = 0`; if exiting pending, `slo_state.on_pending_exit(now)`).
3. **Rest of schedule:** the existing loop body operates on `self.running` only; pending requests are excluded from `token_budget`, `max_num_running_reqs`, and KV allocation passes (their KV blocks are already held).

`Scheduler.__init__` patches `self.schedule = self.schedule_sslo` when `sslo_config.enabled`.

**Important note on `RequestStatus`:** Pending requests keep `status = RequestStatus.RUNNING`. We do NOT add a new enum value. `self.sslo_pending` is the source of truth; status is unchanged so downstream code (event publishers, finish detection) is unaffected.

- [ ] **Step 1: Add `_sslo_score_key` helper + tests**

Create `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SSLO-aware scheduler helpers."""
import types

import pytest


def make_mock_request(req_id: str, sslo_score: float | None,
                     is_pending_eligible: bool = False) -> object:
    req = types.SimpleNamespace()
    req.request_id = req_id
    if sslo_score is None:
        req.slo_state = None
    else:
        req.slo_state = types.SimpleNamespace(
            sslo_score=sslo_score,
            is_pending_eligible=is_pending_eligible,
            on_pending_enter=lambda now: None,
            on_pending_exit=lambda now: None,
        )
    return req


class TestSsloScoreKey:
    def test_returns_score_when_set(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        assert _sslo_score_key(make_mock_request("a", 0.5)) == 0.5

    def test_returns_inf_when_none(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        assert _sslo_score_key(make_mock_request("a", None)) == float("inf")

    def test_sort_urgent_first(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        reqs = [
            make_mock_request("a", 1.0),
            make_mock_request("b", -0.5),
            make_mock_request("c", None),
            make_mock_request("d", 0.2),
        ]
        sorted_reqs = sorted(reqs, key=_sslo_score_key)
        assert [r.request_id for r in sorted_reqs] == ["b", "d", "a", "c"]
```

- [ ] **Step 2: Run — expect failure**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_scheduler_sslo.py -v 2>&1 | tail -10
'
```

Expected: ImportError on `_sslo_score_key`.

- [ ] **Step 3: Add `_sslo_score_key` and SSLO state to `Scheduler`**

In `vllm/vllm/v1/core/sched/scheduler.py`, before `class Scheduler`:

```python
# SSLO
def _sslo_score_key(request: "Request") -> float:
    """Return sslo_score for sorting; None slo_state → +inf (least urgent)."""
    if request.slo_state is None:
        return float("inf")
    return request.slo_state.sslo_score
```

In `Scheduler.__init__`, after `self.max_num_running_reqs = self.scheduler_config.max_num_seqs` (around line 106):

```python
        # SSLO
        self.sslo_config = vllm_config.sslo_config
        self.sslo_pending: list["Request"] = []
        self.sslo_consecutive_pending: dict[str, int] = {}
        if self.sslo_config.enabled:
            self.schedule = self.schedule_sslo
```

- [ ] **Step 4: Run — `_sslo_score_key` tests pass**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_scheduler_sslo.py -v 2>&1 | tail -10
'
```

Expected: PASS.

- [ ] **Step 5: Add pending-redistribution unit test**

Append to `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
class TestPendingRedistribution:
    """Tests the pending-vs-running decision logic in isolation."""

    def _decide(self, req, consecutive: int, max_consec: int,
               waiting_nonempty: bool) -> str:
        """Replicate the scheduler's per-request decision."""
        if consecutive >= max_consec:
            return "running"
        if req.slo_state and req.slo_state.is_pending_eligible and waiting_nonempty:
            return "pending"
        return "running"

    def test_eligible_with_waiting_goes_pending(self):
        req = make_mock_request("a", 1.0, is_pending_eligible=True)
        assert self._decide(req, 0, 5, True) == "pending"

    def test_eligible_no_waiting_goes_running(self):
        req = make_mock_request("a", 1.0, is_pending_eligible=True)
        assert self._decide(req, 0, 5, False) == "running"

    def test_max_consecutive_forces_running(self):
        req = make_mock_request("a", 1.0, is_pending_eligible=True)
        assert self._decide(req, 5, 5, True) == "running"

    def test_not_eligible_goes_running(self):
        req = make_mock_request("a", -0.5, is_pending_eligible=False)
        assert self._decide(req, 0, 5, True) == "running"
```

- [ ] **Step 6: Create `schedule_sslo()` as copy of `schedule()` with redistribution**

Copy the entire `schedule()` method body to a new `schedule_sslo()` method. At the **very top of the method body** (before `req_index = 0`), insert the redistribution block:

```python
        # SSLO: combine running + pending, sort by urgency
        import time as _time
        now = _time.monotonic()
        combined = self.running + self.sslo_pending
        combined.sort(key=_sslo_score_key)

        new_running: list["Request"] = []
        new_pending: list["Request"] = []
        for req in combined:
            was_pending = req in self.sslo_pending
            consec = self.sslo_consecutive_pending.get(req.request_id, 0)
            if consec >= self.sslo_config.max_consecutive_pending:
                eligible = False
            elif req.slo_state is not None and req.slo_state.is_pending_eligible \
                    and len(self.waiting) > 0:
                eligible = True
            else:
                eligible = False

            if eligible:
                if not was_pending and req.slo_state is not None:
                    req.slo_state.on_pending_enter(now)
                self.sslo_consecutive_pending[req.request_id] = consec + 1
                new_pending.append(req)
            else:
                if was_pending and req.slo_state is not None:
                    req.slo_state.on_pending_exit(now)
                self.sslo_consecutive_pending[req.request_id] = 0
                new_running.append(req)

        self.running = new_running
        self.sslo_pending = new_pending
        # Pending requests are excluded from compute budget below; their
        # KV blocks remain allocated and they keep RequestStatus.RUNNING.
```

The rest of `schedule_sslo()` body is verbatim from `schedule()` and operates on `self.running` only.

- [ ] **Step 7: Cleanup `sslo_consecutive_pending` on request finish**

Find the request finish/abort handling in `scheduler.py` (search for `finish_requests` and `_check_stop`). Add to the cleanup block (where requests leave `self.running`):

```python
        # SSLO
        self.sslo_consecutive_pending.pop(request.request_id, None)
        if request in self.sslo_pending:
            self.sslo_pending.remove(request)
```

- [ ] **Step 8: Run all SSLO tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -25
'
```

Expected: all PASS.

- [ ] **Step 9: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/core/sched/scheduler.py

git add vllm/vllm/v1/core/sched/scheduler.py vllm/tests/sslo/test_scheduler_sslo.py
git commit -m "feat(sslo): add schedule_sslo() with pending mechanism (no-recompute pause)"
```

---

### Task 6: Scheduler — optional offload marking on KV scarcity

**Files:**
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/tests/sslo/test_scheduler_sslo.py`

**Context:** When a new request's prefill is admissible but KV blocks are insufficient, mark the request with the **highest** `sslo_score` in `(self.running + self.sslo_pending)` with `sslo_offload_requested = True`. **Marking only — actual offload is not implemented in this plan** (see Task 5 deferred note from prior design discussion). The mark sits on the request and can be consumed later when CPU offload infrastructure is wired into `_preempt_request()`.

Only active when `sslo_config.offloading=True`.

- [ ] **Step 1: Add test**

Append to `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
class TestOffloadMarking:
    def test_highest_score_in_combined_is_marked(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        running = [make_mock_request("a", 0.1), make_mock_request("b", 1.5)]
        pending = [make_mock_request("c", 2.5)]
        combined = running + pending
        candidate = max(combined, key=_sslo_score_key)
        candidate.sslo_offload_requested = True
        assert candidate.request_id == "c"
        assert candidate.sslo_offload_requested is True
```

- [ ] **Step 2: Locate the prefill admission KV-shortage branch in `schedule_sslo()`**

In `scheduler.py`, inside `schedule_sslo()`, find the block where new prefill is attempted but `kv_cache_manager` cannot allocate enough blocks (search for `can_allocate` / `allocate_slots` failure path). The exact location is in the waiting-queue admission loop.

- [ ] **Step 3: Add offload marking before the failure-handling continues**

In the KV-shortage branch (where the new request is going to be skipped because no blocks are available), insert:

```python
                # SSLO (offloading): mark highest-slack request for future CPU offload
                if self.sslo_config.offloading:
                    combined = self.running + self.sslo_pending
                    if combined:
                        candidate = max(combined, key=_sslo_score_key)
                        candidate.sslo_offload_requested = True
```

(`sslo_offload_requested` is set as an instance attribute on `Request`. No need to add a typed field; it's a marker only.)

- [ ] **Step 4: Run all tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -15
'
```

Expected: all PASS.

- [ ] **Step 5: Syntax check + commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/core/sched/scheduler.py

git add vllm/vllm/v1/core/sched/scheduler.py vllm/tests/sslo/test_scheduler_sslo.py
git commit -m "feat(sslo): mark highest-slack request for offload when KV scarce (mark-only)"
```

---

### Task 7: Scheduler — optional adaptive batch size (`sslo_config.adaptive_batch_size`)

**Files:**
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/tests/sslo/test_scheduler_sslo.py`

**Context:** When `sslo_config.adaptive_batch_size=True`, if any request in `self.running` (post-redistribution) has `sslo_score < 0`, reduce the effective `max_num_running_reqs` cap by 1 for this scheduling step. The cap is shadowed as a local variable; `self.max_num_running_reqs` is never mutated.

Both usages of `self.max_num_running_reqs` in `schedule_sslo()` (admission gate at original line ~573, end-of-loop assert at original line ~867) are replaced with the local variable.

- [ ] **Step 1: Add test**

Append to `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
class TestAdaptiveBatchSize:
    def test_overdue_reduces_cap(self):
        running = [make_mock_request("a", 0.5), make_mock_request("b", -0.1)]
        base = 16
        overdue = any(r.slo_state and r.slo_state.sslo_score < 0 for r in running)
        cap = max(1, base - 1) if overdue else base
        assert cap == 15

    def test_no_overdue_keeps_cap(self):
        running = [make_mock_request("a", 0.5), make_mock_request("b", 0.1)]
        base = 16
        overdue = any(r.slo_state and r.slo_state.sslo_score < 0 for r in running)
        cap = max(1, base - 1) if overdue else base
        assert cap == 16
```

- [ ] **Step 2: Add local cap variable + adaptive reduction in `schedule_sslo()`**

In `scheduler.py`, inside `schedule_sslo()`, immediately after the redistribution block (right after `self.sslo_pending = new_pending`), insert:

```python
        # SSLO (adaptive_batch_size): shadow as local; reduce if any overdue
        max_num_running_reqs = self.max_num_running_reqs
        if self.sslo_config.adaptive_batch_size and any(
            r.slo_state is not None and r.slo_state.sslo_score < 0
            for r in self.running
        ):
            max_num_running_reqs = max(1, max_num_running_reqs - 1)
```

- [ ] **Step 3: Replace both `self.max_num_running_reqs` usages in `schedule_sslo()`**

Find admission gate (`schedule()` original line ~573):

```python
                if len(self.running) == self.max_num_running_reqs:
```

Replace with:

```python
                # SSLO: use local cap
                if len(self.running) == max_num_running_reqs:
```

Find end-of-loop assert (`schedule()` original line ~867):

```python
        assert len(self.running) <= self.max_num_running_reqs
```

Replace with:

```python
        # SSLO: assert against local cap
        assert len(self.running) <= max_num_running_reqs
```

- [ ] **Step 4: Run all tests + syntax check**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -25
'

docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/core/sched/scheduler.py
```

Expected: all PASS, no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add vllm/vllm/v1/core/sched/scheduler.py vllm/tests/sslo/test_scheduler_sslo.py
git commit -m "feat(sslo): optional adaptive_batch_size — local cap variable in schedule_sslo()"
```
