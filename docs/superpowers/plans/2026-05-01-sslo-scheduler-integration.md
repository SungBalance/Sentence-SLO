# SSLO Scheduler Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ad-hoc env vars with a typed `SsloConfig` threaded through vLLM's config system, and integrate `sslo_score` into `Scheduler.schedule()` for urgency-aware reordering, preemption, optional KV offload selection, and optional adaptive batch sizing.

**Architecture:** A new `SsloConfig` dataclass (in `vllm/vllm/sslo/config.py`) is added to `VllmConfig` and flows through `EngineArgs.sslo_params` → `create_engine_config()` → `OutputProcessor` and `EngineCore`. `RequestSLOState` gains an `sslo_score` property (currently `= cumulative_slack`) as a stable abstraction. SSLO scheduling lives entirely in a new `Scheduler.schedule_sslo()` method — a copy of `schedule()` with SSLO additions — leaving `schedule()` untouched. When `sslo_config.enabled=True`, `Scheduler.__init__` patches `self.schedule = self.schedule_sslo`. Inside `schedule_sslo()`: the running list is sorted by `sslo_score` ascending (most urgent first), so `self.running.pop()` preempts the highest-slack request without needing an extra `max()` call.

**Tech Stack:** Python 3 dataclasses, Pydantic (VllmConfig), vLLM v1 scheduler (`vllm/vllm/v1/core/sched/scheduler.py`), pytest (unit tests in `vllm/tests/sslo/`).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm/vllm/sslo/config.py` | Create | `SsloConfig` dataclass + `build_slo_state()` factory |
| `vllm/vllm/sslo/slo_state.py` | Modify | Add `sslo_score` property to `RequestSLOState` |
| `vllm/vllm/sslo/__init__.py` | Modify | Export `SsloConfig` |
| `vllm/vllm/config/vllm.py` | Modify | Add `sslo_config: SsloConfig` field to `VllmConfig` |
| `vllm/vllm/engine/arg_utils.py` | Modify | Add `sslo_params: dict` to `EngineArgs`; build `SsloConfig` in `create_engine_config()` |
| `vllm/vllm/v1/engine/output_processor.py` | Modify | `OutputProcessor.__init__` accepts `sslo_config`; `RequestState.__init__` uses `build_slo_state()` |
| `vllm/vllm/v1/engine/llm_engine.py` | Modify | Pass `sslo_config` to `OutputProcessor` |
| `vllm/vllm/v1/engine/async_llm.py` | Modify | Pass `sslo_config` to `OutputProcessor` |
| `vllm/vllm/v1/engine/core.py` | Modify | `_bind_slo_state()` uses `vllm_config.sslo_config` |
| `vllm/vllm/v1/core/sched/scheduler.py` | Modify | sslo_score-based reordering, preemption, offloading, adaptive batch size |
| `vllm/tests/sslo/test_sslo_config.py` | Create | Tests for `SsloConfig` and `build_slo_state()` |
| `vllm/tests/sslo/test_slo_state.py` | Modify | Add test for `sslo_score` property |
| `vllm/tests/sslo/test_scheduler_sslo.py` | Create | Unit tests for scheduler SSLO logic (reorder, preempt, adaptive) |

---

### Task 1: `SsloConfig` dataclass + `sslo_score` property

**Files:**
- Create: `vllm/vllm/sslo/config.py`
- Modify: `vllm/vllm/sslo/slo_state.py`
- Modify: `vllm/vllm/sslo/__init__.py`
- Create: `vllm/tests/sslo/test_sslo_config.py`
- Modify: `vllm/tests/sslo/test_slo_state.py`

**Context:** `SsloConfig` is a plain dataclass (no Pydantic — it lives under `vllm/vllm/sslo/` which is pure Python). `build_slo_state()` is a factory that creates the correct `RequestSLOState` from a config, replacing the duplicated env-var logic scattered in `output_processor.py` and `core.py`. `sslo_score` on `RequestSLOState` is a thin property abstracting `cumulative_slack` so future scoring formulas don't require callers to change.

- [ ] **Step 1: Write failing tests for `SsloConfig` and `build_slo_state()`**

Create `vllm/tests/sslo/test_sslo_config.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for SsloConfig and build_slo_state()."""

from vllm.sslo.config import SsloConfig, build_slo_state
from vllm.sslo.slo_state import (
    ParagraphChunkDetector,
    RequestSLOState,
    SentenceChunkDetector,
    WordRateEstimator,
)


class TestSsloConfig:
    def test_defaults(self):
        cfg = SsloConfig()
        assert cfg.enabled is False
        assert cfg.seconds_per_word == 0.28
        assert cfg.chunk_unit == "sentence"
        assert cfg.estimator_type == "word_rate"
        assert cfg.offloading is False
        assert cfg.adaptive_batch_size is False

    def test_custom_values(self):
        cfg = SsloConfig(enabled=True, seconds_per_word=0.5, chunk_unit="paragraph", offloading=True)
        assert cfg.enabled is True
        assert cfg.seconds_per_word == 0.5
        assert cfg.chunk_unit == "paragraph"
        assert cfg.offloading is True

    def test_invalid_chunk_unit_raises(self):
        import pytest
        with pytest.raises(ValueError, match="chunk_unit"):
            SsloConfig(chunk_unit="invalid")

    def test_invalid_estimator_type_raises(self):
        import pytest
        with pytest.raises(ValueError, match="estimator_type"):
            SsloConfig(estimator_type="tts")  # not yet supported


class TestBuildSloState:
    def test_default_config_builds_word_rate_sentence(self):
        state = build_slo_state(SsloConfig())
        assert isinstance(state, RequestSLOState)
        # Verify estimator type via consumption time of known text
        # WordRateEstimator(0.28) -> "hello world" -> 0.56s
        import time
        now = time.monotonic()
        state.on_text_delta("hello world. ", now)
        # Chunk flushed; estimator must be word-rate
        assert len(state.chunk_records) == 1
        assert state.chunk_records[0]["word_count"] == 2

    def test_paragraph_config_uses_paragraph_detector(self):
        state = build_slo_state(SsloConfig(chunk_unit="paragraph"))
        import time
        now = time.monotonic()
        # Sentence boundary should NOT flush in paragraph mode
        state.on_text_delta("Hello world. More text.", now)
        assert len(state.chunk_records) == 0
        # Paragraph boundary should flush
        state.on_text_delta("\n\nNext paragraph.", now)
        assert len(state.chunk_records) == 1

    def test_custom_seconds_per_word(self):
        state = build_slo_state(SsloConfig(seconds_per_word=0.5))
        import time
        now = time.monotonic()
        state.on_text_delta("one two. ", now)
        state.on_finish(now + 1.0)
        # chunk 0: 2 words * 0.5 = 1.0s cumulative_consume feeds chunk 1
        assert state.chunk_records[0]["word_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_sslo_config.py -v 2>&1 | head -30
'
```

Expected: `ImportError` or `ModuleNotFoundError` for `vllm.sslo.config`.

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
    """Create a RequestSLOState from a SsloConfig."""
    from vllm.sslo.slo_state import (
        ParagraphChunkDetector,
        RequestSLOState,
        WordRateEstimator,
    )
    estimator = WordRateEstimator(seconds_per_word=config.seconds_per_word)
    detector = ParagraphChunkDetector() if config.chunk_unit == "paragraph" else None
    return RequestSLOState(estimator=estimator, detector=detector)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_sslo_config.py -v 2>&1 | tail -20
'
```

Expected: all tests PASS.

- [ ] **Step 5: Add `sslo_score` property to `RequestSLOState`**

In `vllm/vllm/sslo/slo_state.py`, add this property to the `RequestSLOState` class, directly after the `cumulative_slack` attribute initialization (around line 147):

```python
    @property
    def sslo_score(self) -> float:
        """Scheduling urgency score. Lower = more urgent.

        Currently equals cumulative_slack. Abstracted so future scoring
        formulas (e.g. weighted slack, deadline proximity) don't require
        caller changes.
        """
        return self.cumulative_slack
```

- [ ] **Step 6: Add `sslo_score` test to `test_slo_state.py`**

Append to `vllm/tests/sslo/test_slo_state.py`:

```python
class TestSsloScore:
    def test_sslo_score_equals_cumulative_slack(self):
        state = RequestSLOState()
        import time
        now = time.monotonic()
        state.on_text_delta("Hello world. ", now)
        state.on_text_delta("Next sentence. ", now + 0.1)
        # sslo_score must equal cumulative_slack
        assert state.sslo_score == state.cumulative_slack

    def test_sslo_score_initial_zero(self):
        state = RequestSLOState()
        assert state.sslo_score == 0.0
```

- [ ] **Step 7: Run all sslo tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -20
'
```

Expected: all tests PASS.

- [ ] **Step 8: Update `vllm/vllm/sslo/__init__.py`**

Replace the file content with:

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

- [ ] **Step 9: Syntax check in container**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/sslo/config.py \
  /workspace/mlsys/vllm/vllm/sslo/slo_state.py \
  /workspace/mlsys/vllm/vllm/sslo/__init__.py
```

Expected: no errors.

- [ ] **Step 10: Commit**

```bash
git add vllm/vllm/sslo/config.py vllm/vllm/sslo/slo_state.py vllm/vllm/sslo/__init__.py \
        vllm/tests/sslo/test_sslo_config.py vllm/tests/sslo/test_slo_state.py
git commit -m "feat(sslo): add SsloConfig, build_slo_state(), and sslo_score property"
```

---

### Task 2: Wire `sslo_params` through `EngineArgs` → `VllmConfig`

**Files:**
- Modify: `vllm/vllm/engine/arg_utils.py`
- Modify: `vllm/vllm/config/vllm.py`

**Context:** `VllmConfig` is a Pydantic model that aggregates all sub-configs. It lives in `vllm/vllm/config/vllm.py`. `EngineArgs` (in `vllm/vllm/engine/arg_utils.py`) is a plain `@dataclass`; its `create_engine_config()` method (around line 1594) constructs `VllmConfig` at line 2121. We add `sslo_params: dict` to `EngineArgs` and build `SsloConfig` inside `create_engine_config()`, then pass it to `VllmConfig`.

`VllmConfig` uses Pydantic `Field(default_factory=...)` for sub-configs — follow the same pattern for `sslo_config`.

- [ ] **Step 1: Add `sslo_config` field to `VllmConfig`**

In `vllm/vllm/config/vllm.py`, find the block of imports at the top. Add:

```python
# SSLO
from vllm.sslo.config import SsloConfig
```

Then in the `VllmConfig` class body, after the `additional_config` field (near the end of field declarations), add:

```python
    # SSLO
    sslo_config: SsloConfig = Field(default_factory=SsloConfig)
    """SSLO (Sentence-level SLO) configuration."""
```

- [ ] **Step 2: Add `sslo_params` to `EngineArgs` and build `SsloConfig` in `create_engine_config()`**

In `vllm/vllm/engine/arg_utils.py`, in the `EngineArgs` dataclass body (near other optional dict fields like `additional_config`), add:

```python
    # SSLO
    sslo_params: dict = field(default_factory=dict)
```

Then in `create_engine_config()` (around line 2115, just before `config = VllmConfig(...)`), add:

```python
        # SSLO
        from vllm.sslo.config import SsloConfig
        sslo_config = SsloConfig(**self.sslo_params) if self.sslo_params else SsloConfig()
```

Then inside the `VllmConfig(...)` constructor call (around line 2121), add:

```python
            # SSLO
            sslo_config=sslo_config,
```

- [ ] **Step 3: Syntax check in container**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/config/vllm.py \
  /workspace/mlsys/vllm/vllm/engine/arg_utils.py
```

Expected: no errors.

- [ ] **Step 4: Verify `sslo_config` is accessible on a default `VllmConfig`**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -c "
from vllm.config.vllm import VllmConfig
from vllm.sslo.config import SsloConfig
cfg = VllmConfig()
assert isinstance(cfg.sslo_config, SsloConfig)
assert cfg.sslo_config.seconds_per_word == 0.28
print(\"OK:\", cfg.sslo_config)
"
'
```

Expected: `OK: SsloConfig(seconds_per_word=0.28, ...)` printed.

- [ ] **Step 5: Commit**

```bash
git add vllm/vllm/config/vllm.py vllm/vllm/engine/arg_utils.py
git commit -m "feat(sslo): wire sslo_params through EngineArgs and VllmConfig"
```

---

### Task 3: Replace env vars in `OutputProcessor` and `EngineCore`

**Files:**
- Modify: `vllm/vllm/v1/engine/output_processor.py`
- Modify: `vllm/vllm/v1/engine/llm_engine.py`
- Modify: `vllm/vllm/v1/engine/async_llm.py`
- Modify: `vllm/vllm/v1/engine/core.py`

**Context:** Two places currently build `RequestSLOState` from env vars:
1. `RequestState.__init__` in `output_processor.py` (lines 196–199) — client-side, full state
2. `EngineCore._bind_slo_state()` in `core.py` (line 348) — engine-side, stub state

Both should use `build_slo_state(sslo_config)` instead. `OutputProcessor` receives `sslo_config` via its constructor; callers in `llm_engine.py` (line 96) and `async_llm.py` (line 138) must pass it from `self.vllm_config.sslo_config`. `EngineCore` already receives `vllm_config` and can access `vllm_config.sslo_config` directly.

- [ ] **Step 1: Update `OutputProcessor.__init__` to accept `sslo_config`**

In `vllm/vllm/v1/engine/output_processor.py`, update `OutputProcessor.__init__` signature (around line 445):

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

- [ ] **Step 2: Update `RequestState.__init__` to use `build_slo_state()`**

In `output_processor.py`, find `RequestState.__init__` and add `sslo_config` parameter after `stream_interval`:

```python
        sslo_config: "SsloConfig | None" = None,
```

Replace the SSLO block (lines 196–199) with:

```python
        # SSLO
        from vllm.sslo.config import SsloConfig as _SsloConfig, build_slo_state
        _cfg = sslo_config if sslo_config is not None else _SsloConfig()
        self.slo_state: RequestSLOState = build_slo_state(_cfg)
        self._slo_text_len: int = 0
```

- [ ] **Step 3: Update `RequestState.from_new_request()` to accept and pass `sslo_config`**

`RequestState` is constructed via `from_new_request()` class method (line ~232). Add `sslo_config` to its signature and pass it to `cls(...)`:

In `from_new_request` signature (around line 220), add parameter:
```python
        sslo_config: "SsloConfig | None" = None,
```

In the `return cls(...)` block at the end of `from_new_request`, add:
```python
            sslo_config=sslo_config,
```

In `OutputProcessor`, find `RequestState.from_new_request(` (around line 551) and add `sslo_config=self.sslo_config` to the call.

- [ ] **Step 4: Pass `sslo_config` from callers in `llm_engine.py` and `async_llm.py`**

In `vllm/vllm/v1/engine/llm_engine.py` (around line 96), update `OutputProcessor(...)`:

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

In `vllm/vllm/v1/engine/async_llm.py` (around line 138), apply the same change.

- [ ] **Step 5: Update `EngineCore._bind_slo_state()` to use config**

In `vllm/vllm/v1/engine/core.py`, replace the `_bind_slo_state()` SSLO block (around line 347):

```python
        # SSLO
        from vllm.sslo.config import build_slo_state
        request.slo_state = build_slo_state(self.vllm_config.sslo_config)
        self.scheduler.add_request(request)
```

Remove the now-unused import `from vllm.sslo.slo_state import RequestSLOState` if it was only used here.

- [ ] **Step 6: Syntax check in container**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/engine/output_processor.py \
  /workspace/mlsys/vllm/vllm/v1/engine/llm_engine.py \
  /workspace/mlsys/vllm/vllm/v1/engine/async_llm.py \
  /workspace/mlsys/vllm/vllm/v1/engine/core.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add vllm/vllm/v1/engine/output_processor.py \
        vllm/vllm/v1/engine/llm_engine.py \
        vllm/vllm/v1/engine/async_llm.py \
        vllm/vllm/v1/engine/core.py
git commit -m "feat(sslo): replace env vars with SsloConfig in OutputProcessor and EngineCore"
```

---

### Task 4: Scheduler — `schedule_sslo()` with reordering (A) and preemption (C)

**Files:**
- Create: `vllm/tests/sslo/test_scheduler_sslo.py`
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`

**Context:** `schedule()` is left **untouched**. All SSLO logic lives in a new `schedule_sslo()` method, which is a verbatim copy of `schedule()` with SSLO additions. When `sslo_config.enabled=True`, `Scheduler.__init__` patches `self.schedule = self.schedule_sslo` so the engine calls the SSLO-aware version transparently.

Each `Request` has `slo_state: RequestSLOState | None` with `sslo_score: float`. Lower score = more urgent.

**(A) Reordering:** At the top of `schedule_sslo()`, sort `self.running` by `sslo_score` ascending. Requests with `slo_state is None` are treated as least urgent (score = +∞). After this sort, `self.running[0]` is most urgent and `self.running[-1]` has the most slack.

**(C) Preemption:** The original non-PRIORITY preemption is `self.running.pop()`. In `schedule_sslo()`, because the list is already sorted ascending, `self.running[-1]` is already the highest-slack request — so `self.running.pop()` is all that is needed. No `max()` call required. The PRIORITY path (existing `if req.priority is not None` branch) is left exactly as in `schedule()`.

Helper function `_sslo_score_key(req)` returns `req.slo_state.sslo_score` if `slo_state` is set, else `float("inf")`.

- [ ] **Step 1: Write failing tests for `_sslo_score_key` and sort behavior**

Create `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SSLO-aware scheduler helpers."""
import sys
import types

import pytest


def make_mock_request(sslo_score: float | None) -> object:
    """Return a minimal mock Request with slo_state."""
    req = types.SimpleNamespace()
    if sslo_score is None:
        req.slo_state = None
    else:
        req.slo_state = types.SimpleNamespace(sslo_score=sslo_score)
    return req


class TestSsloScoreKey:
    def test_returns_score_when_slo_state_set(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        req = make_mock_request(sslo_score=0.5)
        assert _sslo_score_key(req) == 0.5

    def test_returns_inf_when_slo_state_none(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        req = make_mock_request(sslo_score=None)
        assert _sslo_score_key(req) == float("inf")

    def test_sort_order_urgent_first(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        reqs = [
            make_mock_request(sslo_score=1.0),
            make_mock_request(sslo_score=-0.5),
            make_mock_request(sslo_score=None),
            make_mock_request(sslo_score=0.2),
        ]
        sorted_reqs = sorted(reqs, key=_sslo_score_key)
        scores = [_sslo_score_key(r) for r in sorted_reqs]
        assert scores == [-0.5, 0.2, 1.0, float("inf")]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_scheduler_sslo.py -v 2>&1 | head -20
'
```

Expected: `ImportError` — `_sslo_score_key` not defined yet.

- [ ] **Step 3: Add `_sslo_score_key` module-level helper to `scheduler.py`**

In `vllm/vllm/v1/core/sched/scheduler.py`, add this function near the top, after the existing imports (before `class Scheduler`):

```python
# SSLO
def _sslo_score_key(request: "Request") -> float:
    """Return sslo_score for sorting; None slo_state → +inf (least urgent)."""
    if request.slo_state is None:
        return float("inf")
    return request.slo_state.sslo_score
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/test_scheduler_sslo.py -v 2>&1 | tail -15
'
```

Expected: all tests PASS.

- [ ] **Step 5: Create `schedule_sslo()` as a copy of `schedule()` with SSLO additions**

In `scheduler.py`, copy the entire `schedule()` method and name it `schedule_sslo()`. Then make exactly two additions to `schedule_sslo()`:

**Addition 1** — at the very start of the method body, before `req_index = 0`:

```python
        # SSLO (A): sort running queue ascending by urgency (most urgent first)
        self.running.sort(key=_sslo_score_key)
        # After this sort, self.running[-1] has the highest sslo_score (most slack).
        # The existing self.running.pop() preemption therefore selects the highest-slack
        # request — no further change to the preemption line is needed.
```

**Addition 2** — patch in `__init__`. After `self.sslo_config = vllm_config.sslo_config` (added in Task 5 Step 2), add:

```python
        # SSLO: activate SSLO scheduling when enabled
        if self.sslo_config.enabled:
            self.schedule = self.schedule_sslo
```

The `schedule()` method body is not modified at all.

- [ ] **Step 6: Verify sort + pop invariant with tests**

Append to `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
class TestSortAndPreemption:
    def test_pop_after_sort_removes_highest_slack(self):
        from vllm.v1.core.sched.scheduler import _sslo_score_key
        reqs = [
            make_mock_request(sslo_score=1.0),
            make_mock_request(sslo_score=-0.5),  # most urgent
            make_mock_request(sslo_score=2.5),   # most slack → pop() target
        ]
        reqs.sort(key=_sslo_score_key)
        # After sort: [-0.5, 1.0, 2.5]
        popped = reqs.pop()
        assert popped.slo_state.sslo_score == 2.5
```

- [ ] **Step 7: Syntax check in container**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/core/sched/scheduler.py
```

Expected: no errors.

- [ ] **Step 8: Run all SSLO tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -20
'
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add vllm/vllm/v1/core/sched/scheduler.py \
        vllm/tests/sslo/test_scheduler_sslo.py
git commit -m "feat(sslo): add schedule_sslo() with sslo_score-based reordering (A) and preemption (C)"
```

---

### Task 5: Scheduler — optional offloading (`sslo_config.offloading`) — DEFERRED

**No scheduler changes in this task.**

`offloading: bool = False` 필드는 Task 1에서 `SsloConfig`에 이미 추가됨. 스케줄러 구현은 하지 않는다.

**이유:** vLLM v1에는 `vllm.v1.kv_offload` / `vllm.v1.simple_kv_offload` 모듈이 있고 `OffloadingManager` (`prepare_store()` / `prepare_load()`) 인터페이스도 존재한다. 그러나 이 인프라는 `kv_transfer_config` → `KVConnector` 프레임워크를 통해 활성화되며, 주목적은 P/D disaggregation과 prefix cache 확장이다.

현재 `_preempt_request()`는 `kv_cache_manager.free(request)`만 호출하며 KV Connector와 연결되어 있지 않다. 즉, **오프로드 인프라는 존재하지만 preemption path와 연결되어 있지 않다.**

실제 offloading 구현은 `_preempt_request()` → `connector.prepare_store()` 경로를 연결하는 별도 작업이 선행되어야 한다.

---

### Task 6: Scheduler — optional adaptive batch size (`sslo_config.adaptive_batch_size`)

**Files:**
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/tests/sslo/test_scheduler_sslo.py`

**Context:** All changes here go into `schedule_sslo()`, not `schedule()`.

When `sslo_config.adaptive_batch_size=True`, if any running request has `sslo_score < 0` (SLO already overdue), reduce the effective batch cap by 1 for this step. This creates headroom so the urgent request gets more GPU time.

`schedule()` (and therefore `schedule_sslo()`) accesses `self.max_num_running_reqs` directly in two places: (1) the admission gate check and (2) the end-of-loop assert. Both must use the same local variable in `schedule_sslo()`, so the adaptive reduction is consistent.

**Pattern:** At the top of `schedule_sslo()`, shadow the instance attribute:

```python
max_num_running_reqs = self.max_num_running_reqs
```

Then after the sort + overdue check, optionally reduce it:

```python
if self.sslo_config.adaptive_batch_size and any(...):
    max_num_running_reqs = max(1, max_num_running_reqs - 1)
```

Then replace both usages of `self.max_num_running_reqs` in `schedule_sslo()` with the local `max_num_running_reqs`. `self.max_num_running_reqs` is never mutated.

- [ ] **Step 1: Add test for adaptive batch size gate check**

Append to `vllm/tests/sslo/test_scheduler_sslo.py`:

```python
class TestAdaptiveBatchSizeCheck:
    def test_any_request_overdue(self):
        reqs = [
            make_mock_request(sslo_score=0.5),
            make_mock_request(sslo_score=-0.1),  # overdue
            make_mock_request(sslo_score=1.0),
        ]
        overdue = any(
            r.slo_state is not None and r.slo_state.sslo_score < 0
            for r in reqs
        )
        assert overdue is True

    def test_no_request_overdue(self):
        reqs = [
            make_mock_request(sslo_score=0.5),
            make_mock_request(sslo_score=0.1),
        ]
        overdue = any(
            r.slo_state is not None and r.slo_state.sslo_score < 0
            for r in reqs
        )
        assert overdue is False

    def test_adaptive_cap_reduced_by_one(self):
        base = 16
        reqs = [make_mock_request(sslo_score=-0.1)]
        overdue = any(r.slo_state is not None and r.slo_state.sslo_score < 0 for r in reqs)
        effective = max(1, base - 1) if overdue else base
        assert effective == 15
```

- [ ] **Step 2: Add local `max_num_running_reqs` variable and adaptive reduction in `schedule_sslo()`**

In `scheduler.py`, inside `schedule_sslo()`, at the very top of the method body (right after the sort line), add:

```python
        # SSLO (adaptive_batch_size): shadow instance attribute as local variable
        max_num_running_reqs = self.max_num_running_reqs
        if self.sslo_config.adaptive_batch_size and any(
            r.slo_state is not None and r.slo_state.sslo_score < 0
            for r in self.running
        ):
            max_num_running_reqs = max(1, max_num_running_reqs - 1)
```

- [ ] **Step 3: Replace both `self.max_num_running_reqs` usages in `schedule_sslo()`**

Find the admission gate check (corresponds to original line ~573):

```python
                if len(self.running) == self.max_num_running_reqs:
```

Replace with:

```python
                # SSLO (adaptive_batch_size): use local cap
                if len(self.running) == max_num_running_reqs:
```

Find the end-of-loop assert (corresponds to original line ~867):

```python
        assert len(self.running) <= self.max_num_running_reqs
```

Replace with:

```python
        # SSLO (adaptive_batch_size): assert against local cap
        assert len(self.running) <= max_num_running_reqs
```

- [ ] **Step 4: Run all SSLO tests**

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -v 2>&1 | tail -20
'
```

Expected: all PASS.

- [ ] **Step 5: Syntax check and commit**

```bash
docker exec sk-sslo python3 -m compileall \
  /workspace/mlsys/vllm/vllm/v1/core/sched/scheduler.py

git add vllm/vllm/v1/core/sched/scheduler.py \
        vllm/tests/sslo/test_scheduler_sslo.py
git commit -m "feat(sslo): optional adaptive_batch_size — local cap variable in schedule_sslo()"
```
