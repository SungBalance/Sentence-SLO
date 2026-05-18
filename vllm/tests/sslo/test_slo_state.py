# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO request state."""

import pytest

from vllm.sslo.slo_state import (
    ChunkRecord,
    ChunkSeparator,
    Phase,
    RequestSLOState,
)


def measured_state() -> RequestSLOState:
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=10.0)
    return state


def test_phase_transitions_on_token_and_chunk_boundary():
    state = RequestSLOState(num_warmup_chunks=2)

    assert state.phase == Phase.PREFILL
    state.on_token(1.0)
    assert state.phase == Phase.WARMUP
    assert state.decoding_start_ts == pytest.approx(1.0)

    state.on_chunk_boundary(1.2, word_count=1, chunk_consume_time_s=0.5)
    assert state.phase == Phase.WARMUP
    state.on_token(1.3)
    state.on_chunk_boundary(1.4, word_count=2, chunk_consume_time_s=0.7)
    assert state.phase == Phase.MEASURED
    assert state.chunks_completed == 2


def test_chunk_record_and_diagnostics_append():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(5.0)
    state.on_pending_enter(5.1)
    state.on_pending_exit(5.4)
    state.on_chunk_boundary(5.5, word_count=3, chunk_consume_time_s=0.84)

    assert len(state.chunk_records) == 1
    record = state.chunk_records[0]
    assert isinstance(record, ChunkRecord)
    assert record.chunk_idx == 0
    # Under the new contract d(0) = chunk 0's own finish time, not decoding_start_ts.
    assert record.deadline_ts == pytest.approx(5.5)
    assert record.gen_finish_ts == pytest.approx(5.5)
    assert record.slack_s == 0.0
    assert record.pending_time_s == pytest.approx(0.3)
    assert state.chunk_stall_time_total == 0.0
    assert state.total_pending_time_s == pytest.approx(0.3)
    assert state.num_pending_intervals == 1
    # Chunk 0 is on-time by definition — stall fields must be empty.
    assert record.stall_duration_s == 0.0
    assert record.stall_start_ts is None
    assert record.stall_end_ts is None
    # consume_start_ts is stamped at chunk 0 finish.
    assert state.consume_start_ts == pytest.approx(5.5)


def test_on_step_tracks_total_and_prefill_counts():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_step(decoding_only=True)
    state.on_step(decoding_only=False)
    state.on_step(decoding_only=False)
    state.on_step(decoding_only=True)
    assert state.total_step_count == 4
    assert state.prefill_step_count == 2
    stats = state.compute_stats()
    assert stats.total_step_count == 4
    assert stats.prefill_step_count == 2


def test_chunk1_records_real_slack():
    # Chunk 1+ uses the real slack/stall computation. Stall-aware deadline
    # propagation: after chunk 0 finishes at t=0.5 with consume_time=1.0,
    # the next deadline = max(0.5, 0.5) + 1.0 = 1.5.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(0.5, word_count=2, chunk_consume_time_s=1.0)
    state.on_token(0.5)
    state.on_chunk_boundary(2.0, word_count=2, chunk_consume_time_s=1.0)
    rec = state.chunk_records[1]
    assert rec.chunk_idx == 1
    assert rec.deadline_ts == pytest.approx(1.5)
    assert rec.gen_finish_ts == pytest.approx(2.0)
    assert rec.slack_s == pytest.approx(-0.5)  # missed deadline by 0.5s
    # stall = max(0, -slack); aggregated into chunk_stall_time_total.
    assert state.chunk_stall_time_total == pytest.approx(0.5)
    # Chunk 1 arrived late — stall fields must reflect the overrun.
    assert rec.stall_duration_s == pytest.approx(0.5)
    assert rec.stall_start_ts == pytest.approx(1.5)
    assert rec.stall_end_ts == pytest.approx(2.0)


def test_chunk_expected_len_p90_tracks_history():
    state = RequestSLOState(num_warmup_chunks=1, chunk_len_strategy="p90")
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.chunk_expected_len == pytest.approx(10.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=4, chunk_consume_time_s=1.0)
    # p90 over [10, 20] picks the larger one.
    assert state.chunk_expected_len == pytest.approx(20.0)


def test_chunk_expected_len_ema_strategy_smooths():
    state = RequestSLOState(num_warmup_chunks=1, chunk_len_strategy="ema")
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.chunk_expected_len == pytest.approx(10.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=4, chunk_consume_time_s=1.0)
    # EMA(alpha=0.2): 0.2*20 + 0.8*10 = 12.0
    assert state.chunk_expected_len == pytest.approx(12.0)


def test_chunk_len_strategy_validates():
    with pytest.raises(ValueError):
        RequestSLOState(chunk_len_strategy="median")


def test_expected_remaining_escalates_at_90pct_threshold():
    # Seed predictor with synthetic tiers: p90=10, p95=15, p99=20.
    state = RequestSLOState(
        num_warmup_chunks=1,
        predictor_escalate_threshold=0.9,
        predictor_overshoot_safety_factor=2.5)
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1  # MEASURED phase
    pred = state._chunk_len_predictor
    pred.value = 10.0
    pred.value_mid = 15.0
    pred.value_high = 20.0

    # cur=8 (< 10×0.9=9): use p90 tier → remaining = 10 - 8 = 2.
    state.current_chunk_generated_len = 8
    assert state.expected_remaining_len() == pytest.approx(2.0)
    # cur=9 (>= 9, < 15×0.9=13.5): escalate to p95 → remaining = 15 - 9 = 6.
    state.current_chunk_generated_len = 9
    assert state.expected_remaining_len() == pytest.approx(6.0)
    # cur=14 (>= 13.5, < 20×0.9=18): escalate to p99 → remaining = 20 - 14 = 6.
    state.current_chunk_generated_len = 14
    assert state.expected_remaining_len() == pytest.approx(6.0)
    # cur=18 (>= 18): overshoot branch → (18 - 20) × 2.5 clipped to 1.0.
    state.current_chunk_generated_len = 18
    assert state.expected_remaining_len() == pytest.approx(1.0)


def test_expected_remaining_overshoot_grows_past_topmost():
    # cur way above p99: remaining = (cur - p99) × factor (no 1.0 floor).
    state = RequestSLOState(
        num_warmup_chunks=1,
        predictor_overshoot_safety_factor=2.5)
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1
    pred = state._chunk_len_predictor
    pred.value = 10.0
    pred.value_mid = 15.0
    pred.value_high = 20.0

    state.current_chunk_generated_len = 30
    assert state.expected_remaining_len() == pytest.approx((30 - 20) * 2.5)
    state.current_chunk_generated_len = 100
    assert state.expected_remaining_len() == pytest.approx((100 - 20) * 2.5)


def test_expected_remaining_legacy_threshold_and_factor():
    # threshold=1.0, factor=1.0 → cur > p99 yields (cur - p99) ≈ 1.0 for
    # small overshoots; matches the legacy 1.0 floor closely.
    state = RequestSLOState(
        num_warmup_chunks=1,
        predictor_escalate_threshold=1.0,
        predictor_overshoot_safety_factor=1.0)
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1
    pred = state._chunk_len_predictor
    pred.value = 10.0
    pred.value_mid = 15.0
    pred.value_high = 20.0

    # cur=9 (< 10): legacy uses p90 → remaining = 10 - 9 = 1.
    state.current_chunk_generated_len = 9
    assert state.expected_remaining_len() == pytest.approx(1.0)
    # cur=20 (== high, not strict <): overshoot branch → max(1, 0*1) = 1.
    state.current_chunk_generated_len = 20
    assert state.expected_remaining_len() == pytest.approx(1.0)
    # cur=25: (25 - 20) × 1.0 = 5.
    state.current_chunk_generated_len = 25
    assert state.expected_remaining_len() == pytest.approx(5.0)


def test_predictor_knob_validation():
    with pytest.raises(ValueError):
        RequestSLOState(predictor_escalate_threshold=0.0)
    with pytest.raises(ValueError):
        RequestSLOState(predictor_escalate_threshold=1.5)
    with pytest.raises(ValueError):
        RequestSLOState(predictor_overshoot_safety_factor=-0.1)


def test_score_formula_and_deadline_sign():
    state = measured_state()
    for _ in range(4):
        state.on_token(1.0)

    # measured_state(): chunk 0 finishes at 0.1 with consume=10.0. Under
    # stall-aware propagation: deadline(1) = max(0.1, 0.1) + 10.0 = 10.1.
    # Predictor.value=1.0 after the first chunk (1 token observed). cur=4
    # is past every tier (1.0 × 0.9 escalate threshold), so the overshoot
    # branch fires: remaining = (cur - anchor) * 2.5 = (4 - 1) * 2.5 = 7.5.
    assert state.time_to_deadline(5.1) == pytest.approx(5.0)
    assert state.expected_remaining_len() == pytest.approx(7.5)
    assert state.pressure(5.1, tpot_s=0.2) == pytest.approx(7.5 * 0.2 / 5.0)
    assert state.pressure(20.0, tpot_s=0.2) == float("inf")


def test_score_none_during_warmup_or_missing_inputs():
    warmup = RequestSLOState(num_warmup_chunks=4)
    warmup.on_token(0.0)
    warmup.on_chunk_boundary(0.1, word_count=1, chunk_consume_time_s=1.0)
    assert warmup.pressure(0.2, 0.1) is None

    measured = measured_state()
    assert measured.pressure(0.2, None) is None
    measured._chunk_len_predictor.value = None
    assert measured.pressure(0.2, 0.1) is None


def test_text_delta_compatibility_flushes_by_chunk_unit():
    # min_chunk_tokens=0 disables the merging guard so each boundary flushes.
    sentence = RequestSLOState(
        chunk_unit="sentence", num_warmup_chunks=1, min_chunk_tokens=0)
    sentence.on_text_delta("Hello world.", 1.0, num_tokens=2)
    assert len(sentence.chunk_records) == 1

    paragraph = RequestSLOState(
        chunk_unit="paragraph", num_warmup_chunks=1, min_chunk_tokens=0)
    paragraph.on_text_delta("Hello world.", 1.0, num_tokens=2)
    assert len(paragraph.chunk_records) == 0
    paragraph.on_text_delta("\n\nNext", 1.1, num_tokens=1)
    assert len(paragraph.chunk_records) == 1


def test_short_chunk_below_min_tokens_is_held_back():
    state = RequestSLOState(
        chunk_unit="sentence", num_warmup_chunks=1, min_chunk_tokens=10)
    # First sentence is short (3 tokens) — should NOT flush yet.
    state.on_text_delta("Yes.", 1.0, num_tokens=3)
    assert len(state.chunk_records) == 0
    # Second sentence brings cumulative tokens to 12 (>=10) — flushes the
    # merged chunk at the next boundary.
    state.on_text_delta(" The answer is final.", 1.1, num_tokens=9)
    assert len(state.chunk_records) == 1
    # The merged chunk's word count covers BOTH sentences.
    rec = state.chunk_records[0]
    assert rec.word_count == len("Yes. The answer is final.".split())


def test_on_finish_force_flushes_held_back_text():
    state = RequestSLOState(
        chunk_unit="sentence", num_warmup_chunks=1, min_chunk_tokens=10)
    state.on_text_delta("Short.", 1.0, num_tokens=2)
    assert len(state.chunk_records) == 0
    state.on_finish(now=2.0)
    # Tail flushed even though it's below min_chunk_tokens.
    assert len(state.chunk_records) == 1


def test_min_chunk_tokens_resets_after_flush():
    state = RequestSLOState(
        chunk_unit="sentence", num_warmup_chunks=1, min_chunk_tokens=5)
    # 10 tokens total → flushes at the boundary.
    state.on_text_delta("This is a longer first sentence.", 1.0, num_tokens=10)
    assert len(state.chunk_records) == 1
    # Counter must reset; another 10-token sentence flushes again.
    state.on_text_delta(" Another long sentence here too.", 1.1, num_tokens=10)
    assert len(state.chunk_records) == 2


def test_chunk_separator_newline_is_sentence_boundary():
    # In sentence mode, a single `\n` is a boundary even without ASCII
    # sentence-end punctuation. Each line is 16+ tokens so neither is held back.
    sep = ChunkSeparator(chunk_unit="sentence", min_chunk_tokens=16)
    chunks = list(sep.feed("Line one with enough words to clear min\n", 16))
    chunks += list(sep.feed("Line two also with enough words to clear\n", 16))
    assert len(chunks) == 2
    assert chunks[0].endswith("\n")
    assert chunks[1].endswith("\n")


def test_chunk_separator_newline_under_min_chunk_merges():
    # Short `\n`-separated fragments merge until they hit min_chunk_tokens.
    sep = ChunkSeparator(chunk_unit="sentence", min_chunk_tokens=16)
    out = []
    for _ in range(4):
        out += list(sep.feed("Hi\n", 1))
    assert out == [], "fragments under min_chunk_tokens must not emit"
    # 12 more tokens push us above min_chunk_tokens and the next `\n` flushes.
    out += list(sep.feed(
        "Now the cumulative tokens reach min and we flush.\n", 12))
    assert len(out) == 1


def test_chunk_separator_paragraph_mode_newline_unchanged():
    # Paragraph mode still requires `\n\n`; lone `\n` does not break.
    sep = ChunkSeparator(chunk_unit="paragraph", min_chunk_tokens=0)
    out = list(sep.feed("Single line break only\n", 5))
    assert out == []
    out += list(sep.feed("After paragraph break.\n\n", 5))
    assert len(out) == 1
    assert out[0].endswith("\n\n")


def test_chunk_separator_breaks_degenerate_repetition():
    # `\\boxed{No}\n` repeated — previously grew to ~1000 tokens because no
    # sentence-end punctuation fired. With newline boundary it splits per line.
    sep = ChunkSeparator(chunk_unit="sentence", min_chunk_tokens=16)
    chunks = []
    # 50 lines × 5 tokens = 250 tokens. With min=16 they merge in groups
    # of ~4 lines → ~13 chunks, none above ~30 tokens.
    for _ in range(50):
        chunks += list(sep.feed("\\boxed{No}\n", 5))
    chunks.append(sep.flush() or "")
    # No chunk grew past a few lines' worth of text.
    assert all(len(c) < 120 for c in chunks if c), (
        f"a chunk grew unbounded: max len {max(len(c) for c in chunks if c)}")
    # We DID emit multiple chunks (i.e. the input wasn't one giant chunk).
    assert sum(1 for c in chunks if c) >= 3


def test_chunk_separator_breaks_thai_bullet_list():
    # Non-Latin bullets without `.` — boundary fires on `\n` per line.
    sep = ChunkSeparator(chunk_unit="sentence", min_chunk_tokens=0)
    text = (
        "- ตรวจจับเร็ว\n"
        "- การใช้ที่ปรึกษาทางไกล\n"
        "- การวางแผนระยะยาว\n"
    )
    chunks = list(sep.feed(text, 30))
    assert len(chunks) == 3
    assert all(c.endswith("\n") for c in chunks)


def test_chunk0_slack_zero_by_structure():
    # Chunk 0's deadline = its own finish time, so slack = deadline - now = 0
    # without any special-case branch.
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(3.0)
    state.on_chunk_boundary(4.0, word_count=1, chunk_consume_time_s=1.0)
    rec = state.chunk_records[0]
    assert rec.slack_s == pytest.approx(0.0)
    assert rec.deadline_ts == pytest.approx(rec.gen_finish_ts)


def test_consume_start_ts_set_at_chunk0_end():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(1.0)
    state.on_chunk_boundary(2.5, word_count=2, chunk_consume_time_s=1.0)
    assert state.consume_start_ts == pytest.approx(2.5)
    assert state.consume_start_ts == pytest.approx(
        state.chunk_records[0].gen_finish_ts)


def test_stall_fields_populated_when_late():
    # Chunk 1 arrives after its deadline — stall fields must be set.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(1.0, word_count=1, chunk_consume_time_s=2.0)
    # deadline(1) = max(1.0, 1.0) + 2.0 = 3.0; chunk arrives at 4.0.
    state.on_token(1.0)
    state.on_chunk_boundary(4.0, word_count=1, chunk_consume_time_s=1.0)
    rec = state.chunk_records[1]
    assert rec.stall_duration_s == pytest.approx(1.0)
    assert rec.stall_start_ts == pytest.approx(3.0)
    assert rec.stall_end_ts == pytest.approx(4.0)


def test_stall_fields_none_when_ontime():
    # Chunk 1 arrives before its deadline — stall fields must be empty.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(1.0, word_count=1, chunk_consume_time_s=5.0)
    # deadline(1) = max(1.0, 1.0) + 5.0 = 6.0; chunk arrives at 3.0.
    state.on_token(1.0)
    state.on_chunk_boundary(3.0, word_count=1, chunk_consume_time_s=1.0)
    rec = state.chunk_records[1]
    assert rec.stall_duration_s == 0.0
    assert rec.stall_start_ts is None
    assert rec.stall_end_ts is None


# ---------------------------------------------------------------------------
# Phase 2A tests: ChunkRecord new fields
# ---------------------------------------------------------------------------

def test_chunk_token_indices_monotone():
    # Three chunks; token indices must be contiguous with no gaps.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    # Chunk 0: 5 tokens
    for _ in range(5):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    # Chunk 1: 3 tokens
    for _ in range(3):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=1, chunk_consume_time_s=1.0)
    # Chunk 2: 7 tokens
    for _ in range(7):
        state.on_token(0.4)
    state.on_chunk_boundary(0.5, word_count=3, chunk_consume_time_s=1.0)

    recs = state.chunk_records
    assert len(recs) == 3
    assert recs[0].token_start_idx == 0
    assert recs[0].token_end_idx == 5
    assert recs[1].token_start_idx == 5
    assert recs[1].token_end_idx == 8
    assert recs[2].token_start_idx == 8
    assert recs[2].token_end_idx == 15
    # cumulative_tokens_at_end matches token_end_idx
    for rec in recs:
        assert rec.cumulative_tokens_at_end == rec.token_end_idx


def test_chunk_demand_window():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    # chunk 0: deadline = now = 1.0, consume = 2.0
    state.on_chunk_boundary(1.0, word_count=1, chunk_consume_time_s=2.0)
    rec = state.chunk_records[0]
    assert rec.demand_window_start_ts == pytest.approx(rec.deadline_ts)
    assert rec.demand_window_end_ts == pytest.approx(rec.deadline_ts + rec.chunk_consume_time_s)
    assert rec.chunk_consume_time_s == pytest.approx(2.0)


def test_predictor_source_p90_emits_value_high():
    state = RequestSLOState(
        num_warmup_chunks=1, chunk_len_strategy="p90", min_chunk_tokens=0)
    # Need at least 2 chunks to have a value_high (p99 over history).
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=3, chunk_consume_time_s=1.0)
    # chunk 1 is the first to have a non-None expected_chunk_len_high
    rec = state.chunk_records[1]
    assert rec.predictor_source == "p90"
    assert rec.expected_chunk_len_high is not None


def test_predictor_source_ema_no_value_high():
    state = RequestSLOState(
        num_warmup_chunks=1, chunk_len_strategy="ema", min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    for _ in range(5):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=1, chunk_consume_time_s=1.0)
    rec = state.chunk_records[1]
    assert rec.predictor_source == "ema"
    assert rec.expected_chunk_len_high is None


# ---------------------------------------------------------------------------
# Phase 2B tests: admitted_ts + terminal_outcome on RequestSLOState directly
# ---------------------------------------------------------------------------

def test_admitted_ts_idempotent():
    state = RequestSLOState(num_warmup_chunks=1)
    state.mark_admitted(1.0)
    state.mark_admitted(2.0)  # second call must not overwrite
    assert state.admitted_ts == pytest.approx(1.0)


def test_terminal_outcome_defaults_in_progress():
    state = RequestSLOState(num_warmup_chunks=1)
    assert state.terminal_outcome == "in_progress"
    stats = state.compute_stats()
    assert stats.terminal_outcome == "in_progress"


def test_terminal_outcome_completed_via_mark():
    state = RequestSLOState(num_warmup_chunks=1)
    state.mark_terminal("completed")
    assert state.terminal_outcome == "completed"
    stats = state.compute_stats()
    assert stats.terminal_outcome == "completed"
    assert stats.admitted_ts is None  # not set in this path


def test_admitted_ts_propagates_to_stats():
    state = RequestSLOState(num_warmup_chunks=1)
    state.mark_admitted(5.5)
    stats = state.compute_stats()
    assert stats.admitted_ts == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Phase 3 tests: PressureComponents + pressure_components()
# ---------------------------------------------------------------------------

def test_pressure_components_prefill():
    state = RequestSLOState(num_warmup_chunks=2)
    # Phase is PREFILL (decoding_start_ts is None)
    assert state.phase.name == "PREFILL"
    pc = state.pressure_components(1.0, tpot_s=0.05)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "prefill"
    assert pc.depletion_pressure is None


def test_pressure_components_warmup():
    state = RequestSLOState(num_warmup_chunks=2, min_chunk_tokens=0)
    # Trigger decoding start but stay in WARMUP (chunks_completed < num_warmup_chunks)
    state.on_token(0.0)
    assert state.phase.name == "WARMUP"
    pc = state.pressure_components(0.1, tpot_s=0.05)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "warmup"
    assert pc.depletion_pressure is None


def test_pressure_components_no_tpot():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    # Advance to MEASURED phase: complete warmup chunk
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.phase.name == "MEASURED"
    pc = state.pressure_components(0.2, tpot_s=None)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "no_tpot"
    assert pc.depletion_pressure is None


def test_pressure_components_no_predictor():
    state = RequestSLOState(
        num_warmup_chunks=1, chunk_len_strategy="past-future", min_chunk_tokens=0)
    # "past-future" strategy never sets predictor.value
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.phase.name == "MEASURED"
    assert state._chunk_len_predictor.value is None
    pc = state.pressure_components(0.2, tpot_s=0.05)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "no_predictor"
    assert pc.depletion_pressure is None


def test_pressure_components_available_finite():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    assert state.phase.name == "MEASURED"
    now = 0.5
    tpot = 0.04
    pc = state.pressure_components(now, tpot_s=tpot)
    assert pc.pressure_available
    assert pc.pressure_missing_reason is None
    assert pc.remaining_tokens is not None
    assert pc.estimated_refill_time_s is not None
    assert pc.buffer_slack_s is not None
    assert pc.refill_slack_s is not None
    assert pc.depletion_pressure is not None
    # depletion_pressure >= 1 iff refill_slack_s <= 0
    if pc.refill_slack_s <= 0:
        assert pc.depletion_pressure >= 1.0
    else:
        assert pc.depletion_pressure < 1.0


def test_pressure_components_deadline_passed():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    # chunk 0 completes very late (now=100.0), consume=1.0 → next deadline ≈ 101.0
    state.on_chunk_boundary(100.0, word_count=2, chunk_consume_time_s=1.0)
    assert state.phase.name == "MEASURED"
    # Query well after the deadline (now=200.0)
    pc = state.pressure_components(200.0, tpot_s=0.05)
    assert pc.pressure_available
    assert pc.buffer_slack_s is not None and pc.buffer_slack_s <= 0
    assert pc.depletion_pressure == float("inf")


def test_pressure_equals_depletion_pressure():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, chunk_consume_time_s=1.0)
    now = 0.5
    tpot = 0.04
    p = state.pressure(now, tpot)
    pc = state.pressure_components(now, tpot_s=tpot)
    assert p == pc.depletion_pressure


# ---------------------------------------------------------------------------
# multi_level_pressure_components() tests (policy="multi_level_pressure")
# ---------------------------------------------------------------------------

def _measured_state_with_history(now_chunk_finish: float = 0.1):
    """Build a MEASURED-phase state with predictor populated.

    Mirrors the pattern used by other pressure_components tests: 10
    tokens accumulated, one warmup chunk closed at now_chunk_finish so
    chunks_completed becomes 1 and phase advances to MEASURED.
    """
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(
        now_chunk_finish, word_count=2, chunk_consume_time_s=1.0)
    assert state.phase.name == "MEASURED"
    return state


def test_mlp_components_prefill():
    state = RequestSLOState(num_warmup_chunks=2)
    pc = state.multi_level_pressure_components(1.0, tpot_s=0.05, epoch_s=0.05)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "prefill"
    assert pc.serve_pressure is None
    assert pc.defer_pressure is None


def test_mlp_components_warmup():
    state = RequestSLOState(num_warmup_chunks=2, min_chunk_tokens=0)
    state.on_token(0.0)
    assert state.phase.name == "WARMUP"
    pc = state.multi_level_pressure_components(
        0.1, tpot_s=0.05, epoch_s=0.05)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "warmup"
    assert pc.serve_pressure is None
    assert pc.defer_pressure is None


def test_mlp_components_no_tpot():
    state = _measured_state_with_history()
    pc = state.multi_level_pressure_components(
        0.2, tpot_s=None, epoch_s=None)
    assert not pc.pressure_available
    assert pc.pressure_missing_reason == "no_tpot"
    assert pc.serve_pressure is None
    assert pc.defer_pressure is None


def test_mlp_components_no_epoch_defaults_to_serve():
    state = _measured_state_with_history()
    pc = state.multi_level_pressure_components(
        0.2, tpot_s=0.04, epoch_s=None)
    assert pc.pressure_available
    assert pc.serve_pressure is not None
    # epoch_s=None collapses defer onto serve.
    assert pc.defer_pressure == pc.serve_pressure
    assert pc.defer_buffer_slack_s is None


def test_mlp_components_deadline_passed():
    state = _measured_state_with_history(now_chunk_finish=100.0)
    pc = state.multi_level_pressure_components(
        200.0, tpot_s=0.05, epoch_s=0.05)
    assert pc.pressure_available
    assert pc.buffer_slack_s is not None and pc.buffer_slack_s <= 0
    # ttd <= 0 → both serve and defer saturate to +inf.
    assert pc.serve_pressure == float("inf")
    assert pc.defer_pressure == float("inf")


def test_mlp_components_defer_blows_when_epoch_exceeds_slack():
    state = _measured_state_with_history()
    # State sets deadline at ~1.1s (now_chunk_finish=0.1 + consume=1.0).
    # Query at now=1.0 so ttd ≈ 0.1, then pick epoch_s > 0.1 so
    # defer_buffer goes non-positive while serve stays finite.
    pc = state.multi_level_pressure_components(
        1.0, tpot_s=0.04, epoch_s=0.2)
    assert pc.pressure_available
    assert pc.buffer_slack_s is not None and pc.buffer_slack_s > 0
    assert pc.serve_pressure is not None
    assert pc.serve_pressure != float("inf")
    # defer_buffer = ttd - epoch_s ≤ 0 → defer saturates to +inf.
    assert pc.defer_buffer_slack_s is not None
    assert pc.defer_buffer_slack_s <= 0
    assert pc.defer_pressure == float("inf")
