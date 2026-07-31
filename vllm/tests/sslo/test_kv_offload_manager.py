# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SSLO KV-offload tier scheduler-side manager API.

Covers ``SimpleCPUOffloadScheduler.is_fully_mirrored`` /
``pin_request_cpu_blocks`` / ``unpin_request_cpu_blocks``. The methods are
invoked unbound against a lightweight stub ``self`` so the real method bodies
run without constructing a full scheduler (which needs a live KVCacheConfig /
coordinator). ``importorskip`` skips the module where vLLM is not installed
(e.g. the CPU-only lint container); it runs on the experiment host.
"""

from types import SimpleNamespace

import pytest

manager_mod = pytest.importorskip("vllm.v1.simple_kv_offload.manager")
Mgr = manager_mod.SimpleCPUOffloadScheduler


class FakeCPUBlockPool:

    def __init__(self, cached):
        self.cached = set(cached)  # block hashes present in the CPU cache
        self.touched = []
        self.freed = []

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        if block_hash in self.cached:
            return [SimpleNamespace(block_id=block_hash)]
        return None

    def touch(self, blocks):
        self.touched.extend(blocks)

    def free_blocks(self, blocks):
        self.freed.extend(list(blocks))


def _stub(*, cached=(), lazy=False, block_size=16):
    return SimpleNamespace(
        _lazy_mode=lazy,
        fa_gidx=0,
        _reqs_to_store={},
        _pinned_cpu_blocks={},
        cpu_kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=block_size))
            ]),
        cpu_block_pool=FakeCPUBlockPool(cached),
    )


def _req(rid="r", computed=48, placeholders=0, block_hashes=("h0", "h1", "h2")):
    return SimpleNamespace(
        request_id=rid, num_computed_tokens=computed,
        num_output_placeholders=placeholders, block_hashes=list(block_hashes))


def _store_state(req, *, num_stored, store_events=()):
    return SimpleNamespace(
        request=req, store_events=set(store_events),
        num_stored_blocks=[num_stored])


# --- is_fully_mirrored ------------------------------------------------------

def test_is_fully_mirrored_lazy_mode_false():
    self = _stub(lazy=True)
    assert Mgr.is_fully_mirrored(self, _req()) is False


def test_is_fully_mirrored_unregistered_false():
    self = _stub()
    assert Mgr.is_fully_mirrored(self, _req()) is False


def test_is_fully_mirrored_inflight_store_false():
    self = _stub()
    req = _req()
    self._reqs_to_store[req.request_id] = _store_state(
        req, num_stored=3, store_events=(5, ))
    assert Mgr.is_fully_mirrored(self, req) is False


def test_is_fully_mirrored_cursor_covers_confirmed_true():
    self = _stub()
    req = _req(computed=48)  # 48 // 16 = 3 confirmed full blocks
    self._reqs_to_store[req.request_id] = _store_state(req, num_stored=3)
    assert Mgr.is_fully_mirrored(self, req) is True


def test_is_fully_mirrored_cursor_behind_false():
    self = _stub()
    req = _req(computed=48)
    self._reqs_to_store[req.request_id] = _store_state(req, num_stored=2)
    assert Mgr.is_fully_mirrored(self, req) is False


def test_is_fully_mirrored_no_full_block_false():
    self = _stub()
    req = _req(computed=0)
    self._reqs_to_store[req.request_id] = _store_state(req, num_stored=0)
    assert Mgr.is_fully_mirrored(self, req) is False


# --- pin_request_cpu_blocks -------------------------------------------------

def test_pin_all_present_touches_and_records():
    self = _stub(cached=("h0", "h1", "h2"))
    req = _req(computed=48)
    assert Mgr.pin_request_cpu_blocks(self, req) is True
    assert len(self.cpu_block_pool.touched) == 3
    assert req.request_id in self._pinned_cpu_blocks


def test_pin_missing_block_pins_nothing():
    self = _stub(cached=("h0", "h1"))  # h2 missing
    req = _req(computed=48)
    assert Mgr.pin_request_cpu_blocks(self, req) is False
    assert self.cpu_block_pool.touched == []
    assert req.request_id not in self._pinned_cpu_blocks


def test_pin_idempotent():
    self = _stub(cached=("h0", "h1", "h2"))
    req = _req(computed=48)
    assert Mgr.pin_request_cpu_blocks(self, req) is True
    assert Mgr.pin_request_cpu_blocks(self, req) is True  # no re-touch
    assert len(self.cpu_block_pool.touched) == 3


def test_pin_no_full_block_false():
    self = _stub(cached=("h0", "h1", "h2"))
    req = _req(computed=0)
    assert Mgr.pin_request_cpu_blocks(self, req) is False


# --- unpin_request_cpu_blocks -----------------------------------------------

def test_unpin_frees_pinned_blocks():
    self = _stub(cached=("h0", "h1", "h2"))
    req = _req(computed=48)
    Mgr.pin_request_cpu_blocks(self, req)
    Mgr.unpin_request_cpu_blocks(self, req)
    assert len(self.cpu_block_pool.freed) == 3
    assert req.request_id not in self._pinned_cpu_blocks


def test_unpin_unpinned_is_noop():
    self = _stub()
    Mgr.unpin_request_cpu_blocks(self, _req())  # no exception
    assert self.cpu_block_pool.freed == []
