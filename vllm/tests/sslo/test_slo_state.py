# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase A v2 SSLO request state."""

import pytest

from vllm.sslo.slo_state import ChunkRecord, Phase, RequestSLOState


def measured_state() -> RequestSLOState:
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(0.0, 1)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=10.0)
    return state


def test_phase_transitions_on_token_and_chunk_boundary():
    state = RequestSLOState(num_warmup_chunks=2)

    assert state.phase == Phase.PREFILL
    state.on_token(1.0, 10)
    assert state.phase == Phase.WARMUP
    assert state.decoding_start_ts == pytest.approx(1.0)

    state.on_chunk_boundary(1.2, word_count=1, chunk_consume_time_s=0.5)
    assert state.phase == Phase.WARMUP
    state.on_token(1.3, 11)
    state.on_chunk_boundary(1.4, word_count=2, chunk_consume_time_s=0.7)
    assert state.phase == Phase.MEASURED
    assert state.chunks_completed == 2


def test_chunk_record_and_diagnostics_append():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(5.0, 1)
    state.on_pending_enter(5.1)
    state.on_pending_exit(5.4)
    state.on_chunk_boundary(5.5, word_count=3, chunk_consume_time_s=0.84)

    assert len(state.chunk_records) == 1
    record = state.chunk_records[0]
    assert isinstance(record, ChunkRecord)
    assert record.chunk_idx == 0
    assert record.deadline_ts == pytest.approx(5.0)
    assert record.gen_finish_ts == pytest.approx(5.5)
    assert record.slack_s == pytest.approx(-0.5)
    assert record.stall_s == pytest.approx(0.5)
    assert record.pending_time_s == pytest.approx(0.3)
    assert state.chunk_stall_time_total == pytest.approx(0.5)
    assert state.total_pending_time_s == pytest.approx(0.3)
    assert state.num_pending_intervals == 1


def test_chunk_expected_len_ema_updates():
    state = RequestSLOState(num_warmup_chunks=1)
    for _ in range(10):
        state.on_token(0.0, 1)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.chunk_expected_len_ema == pytest.approx(10.0)

    for _ in range(20):
        state.on_token(0.2, 1)
    state.on_chunk_boundary(0.3, word_count=4, chunk_consume_time_s=1.0)
    assert state.chunk_expected_len_ema == pytest.approx(12.0)


def test_score_formula_and_deadline_sign():
    state = measured_state()
    for _ in range(4):
        state.on_token(1.0, 1)

    assert state.time_to_deadline(5.1) == pytest.approx(4.9)
    assert state.expected_remaining_len() == pytest.approx(1.0)
    assert state.score(5.1, tpot_s=0.2) == pytest.approx(0.2 / 4.9)
    assert state.score(20.0, tpot_s=0.2) == float("inf")


def test_score_none_during_warmup_or_missing_inputs():
    warmup = RequestSLOState(num_warmup_chunks=4)
    warmup.on_token(0.0, 1)
    warmup.on_chunk_boundary(0.1, word_count=1, chunk_consume_time_s=1.0)
    assert warmup.score(0.2, 0.1) is None

    measured = measured_state()
    assert measured.score(0.2, None) is None
    measured.chunk_expected_len_ema = None
    assert measured.score(0.2, 0.1) is None


def test_offload_lifecycle_counters():
    state = RequestSLOState()
    state.on_offload_enter(2.0)
    assert state.is_offloaded is True
    assert state.num_offload_intervals == 1
    state.on_offload_exit(2.25)
    assert state.is_offloaded is False
    assert state.total_offload_time_s == pytest.approx(0.25)


def test_text_delta_compatibility_flushes_by_chunk_unit():
    sentence = RequestSLOState(chunk_unit="sentence", num_warmup_chunks=1)
    sentence.on_text_delta("Hello world.", 1.0, num_tokens=2)
    assert len(sentence.chunk_records) == 1

    paragraph = RequestSLOState(chunk_unit="paragraph", num_warmup_chunks=1)
    paragraph.on_text_delta("Hello world.", 1.0, num_tokens=2)
    assert len(paragraph.chunk_records) == 0
    paragraph.on_text_delta("\n\nNext", 1.1, num_tokens=1)
    assert len(paragraph.chunk_records) == 1
