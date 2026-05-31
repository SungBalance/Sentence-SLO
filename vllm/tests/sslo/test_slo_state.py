# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO request state."""

import pytest

from vllm.sslo.slo_state import (
    ChunkConsumeEstimator,
    ChunkLengthPredictor,
    ChunkRecord,
    ChunkSeparator,
    Phase,
    RequestSLOState,
)


class WordScaledTtsEstimator(ChunkConsumeEstimator):

    def estimate(
        self,
        chunk_text: str,
        word_count: int,
    ) -> tuple[float, float | None]:
        del chunk_text
        return 0.3 * word_count, 0.1 * word_count

    def predict_conversion(self, word_count: int) -> float:
        return 0.1 * word_count


def measured_state() -> RequestSLOState:
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=10.0)
    return state


def test_phase_transitions_on_token_and_chunk_boundary():
    # num_warmup_chunks no longer gates phase; it only controls when the
    # hybrid predictor switches off the global fallback.
    state = RequestSLOState(num_warmup_chunks=2)

    assert state.phase == Phase.PREFILL
    state.on_token(1.0)
    assert state.phase == Phase.WARMUP  # decode started, first chunk not done
    assert state.decoding_start_ts == pytest.approx(1.0)

    state.on_chunk_boundary(1.2, word_count=1, consume_duration=0.5)
    # First chunk completed → MEASURED immediately (regardless of num_warmup_chunks).
    assert state.phase == Phase.MEASURED
    assert state.chunks_completed == 1
    state.on_token(1.3)
    state.on_chunk_boundary(1.4, word_count=2, consume_duration=0.7)
    assert state.phase == Phase.MEASURED
    assert state.chunks_completed == 2


def test_chunk_record_and_diagnostics_append():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(5.0)
    state.on_pending_enter(5.1)
    state.on_pending_exit(5.4)
    state.on_chunk_boundary(5.5, word_count=3, consume_duration=0.84)

    assert len(state.chunk_records) == 1
    record = state.chunk_records[0]
    assert isinstance(record, ChunkRecord)
    assert record.unit_index == 0
    # Under the new contract d(0) = chunk 0's own finish time, not decoding_start_ts.
    assert record.deadline == pytest.approx(5.5)
    assert record.text_generation_end_time == pytest.approx(5.5)
    assert record.consumer_ready_time == pytest.approx(5.5)
    assert record.pending_time_s == pytest.approx(0.3)
    assert state.chunk_stall_time_total == 0.0
    assert state.total_pending_time_s == pytest.approx(0.3)
    assert state.num_pending_intervals == 1
    # Chunk 0 is on-time by definition.
    assert record.unit_deadline_miss_s == 0.0
    assert record.unit_deadline_missed == 0
    # consume_start_time is stamped at chunk 0 finish.
    assert state.consume_start_time == pytest.approx(5.5)


def test_tts_chunk0_deadline_is_consumer_side():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(9.0)
    state.on_chunk_boundary(
        10.0, word_count=2, consume_duration=5.0, conversion_time=1.0)

    record = state.chunk_records[0]
    assert record.deadline == pytest.approx(11.0)
    assert record.consumer_ready_time == pytest.approx(11.0)
    assert record.unit_deadline_miss_s == pytest.approx(0.0)
    assert state.consume_start_time == pytest.approx(11.0)
    # next_deadline = max(deadline, now) + conv + consume
    #              = max(11.0, 10.0) + 1.0 + 5.0 = 17.0.
    assert state.next_deadline_ts == pytest.approx(17.0)


def test_tts_deadline_miss_subtracts_current_conversion():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(
        1.0, word_count=1, consume_duration=2.0, conversion_time=0.2)
    state.on_token(1.0)
    state.on_chunk_boundary(
        3.0, word_count=2, consume_duration=1.0, conversion_time=0.5)

    # chunk 0 deadline(consumer) = 1.0 + 0.2 = 1.2;
    # next_deadline = max(1.2, 1.0) + 0.2 + 2.0 = 3.4 → chunk 1's deadline.
    rec = state.chunk_records[1]
    assert rec.deadline == pytest.approx(3.4)
    # deadline_text = 3.4 - conv(0.5) = 2.9; text arrived at 3.0 → miss 0.1.
    assert rec.unit_deadline_miss_s == pytest.approx(0.1)
    assert rec.consumer_ready_time == pytest.approx(3.5)
    # next_deadline = max(3.4, 3.0) + 0.5 + 1.0 = 4.9.
    assert state.next_deadline_ts == pytest.approx(4.9)


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


def test_chunk1_records_real_deadline_miss():
    # Chunk 1+ uses the real deadline miss computation. Stall-aware deadline
    # propagation: after chunk 0 finishes at t=0.5 with consume_time=1.0,
    # the next deadline = max(0.5, 0.5) + 1.0 = 1.5.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(0.5, word_count=2, consume_duration=1.0)
    state.on_token(0.5)
    state.on_chunk_boundary(2.0, word_count=2, consume_duration=1.0)
    rec = state.chunk_records[1]
    assert rec.unit_index == 1
    assert rec.deadline == pytest.approx(1.5)
    assert rec.text_generation_end_time == pytest.approx(2.0)
    assert rec.consumer_ready_time - rec.deadline == pytest.approx(0.5)
    assert state.chunk_stall_time_total == pytest.approx(0.5)
    # Chunk 1 arrived late.
    assert rec.unit_deadline_miss_s == pytest.approx(0.5)
    assert rec.unit_deadline_missed == 1


def test_chunk_expected_len_p90_tracks_history():
    state = RequestSLOState(num_warmup_chunks=1, chunk_len_strategy="p90")
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=1.0)
    assert state.chunk_expected_len == pytest.approx(10.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=4, consume_duration=1.0)
    # p90 over [10, 20] picks the larger one.
    assert state.chunk_expected_len == pytest.approx(20.0)


def test_chunk_expected_len_ema_strategy_smooths():
    state = RequestSLOState(num_warmup_chunks=1, chunk_len_strategy="ema")
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=1.0)
    assert state.chunk_expected_len == pytest.approx(10.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=4, consume_duration=1.0)
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
    pred.tier_values = [10.0, 15.0, 20.0]

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
    pred.tier_values = [10.0, 15.0, 20.0]

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
    pred.tier_values = [10.0, 15.0, 20.0]

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


def test_predictor_uses_global_when_per_req_below_16_samples():
    # Global predictor warmed up beyond cold-start threshold (p90=50).
    glob = ChunkLengthPredictor(strategy="p90")
    for v in [50] * 128:
        glob.update(v)
    assert glob.value == 50.0

    # num_warmup_chunks now controls the hybrid switch threshold.
    state = RequestSLOState(
        num_warmup_chunks=16,
        global_chunk_len_predictor=glob,
    )
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1
    # Per-req sample_count == 0 < 16 → use global. cur=0, p90=50.
    assert state.expected_remaining_len() == pytest.approx(50.0)
    # Even after 1 per-req sample, still below 16 → still global.
    state._chunk_len_predictor.update(10)
    assert state.expected_remaining_len() == pytest.approx(50.0)


def test_predictor_switches_to_per_req_at_warmup_threshold():
    glob = ChunkLengthPredictor(strategy="p90")
    for _ in range(128):
        glob.update(50)
    state = RequestSLOState(
        num_warmup_chunks=16,
        global_chunk_len_predictor=glob,
    )
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1
    # Feed 16 per-req samples — all 10s → per-req p90=10.
    for _ in range(16):
        state._chunk_len_predictor.update(10)
    # sample_count == num_warmup_chunks → use per-req (p90=10).
    assert state.expected_remaining_len() == pytest.approx(10.0)


def test_cold_start_returns_max_remaining_when_global_under_threshold():
    # Fresh global predictor (sample_count = 0 < default 128) → cold
    # start fallback returns cold_start_max_remaining_tokens - cur.
    glob = ChunkLengthPredictor(strategy="p90")
    state = RequestSLOState(
        num_warmup_chunks=16,
        global_chunk_len_predictor=glob,
        cold_start_max_remaining_tokens=2048,
        global_warmup_predictor_samples=128,
    )
    state.decoding_start_ts = 0.0
    state.next_deadline_ts = 100.0
    state.chunks_completed = 1
    state.current_chunk_generated_len = 32
    assert state.expected_remaining_len() == pytest.approx(2048 - 32)


def test_on_chunk_boundary_updates_global_predictor():
    glob = ChunkLengthPredictor(strategy="p90")
    state = RequestSLOState(
        num_warmup_chunks=1,
        global_chunk_len_predictor=glob,
    )
    state.on_token(0.0)
    state.on_chunk_boundary(1.0, word_count=2, consume_duration=1.0)
    # Both per-req and global should have one sample now.
    assert state._chunk_len_predictor.sample_count == 1
    assert glob.sample_count == 1
    assert glob.value == 1.0  # only one sample of value 1 (single on_token)


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


def test_tts_time_to_deadline_subtracts_current_conversion_estimate():
    state = RequestSLOState(
        num_warmup_chunks=1,
        min_chunk_tokens=0,
        consume_estimator=WordScaledTtsEstimator(),
    )
    for _ in range(8):
        state.on_token(9.0)
    consume_s, conversion_s = state.consume_estimator.estimate("", 4)
    assert conversion_s is not None
    state.on_chunk_boundary(
        10.0,
        word_count=4,
        consume_duration=consume_s,
        conversion_time=conversion_s,
    )

    assert state.last_num_token == 8
    assert state.last_word_count == 4
    # No in-progress chunk after the boundary (token counter reset to 0),
    # so the current-conversion estimate is 0 and time_to_deadline reduces
    # to deadline - now. deadline = next_deadline_ts after chunk 0:
    #   chunk 0 deadline(consumer) = 10.0 + conv(0.4) = 10.4;
    #   next_deadline = max(10.4, 10.0) + conv(0.4) + consume(1.2) = 12.0.
    assert state._predict_current_conversion() == pytest.approx(0.0)
    assert state.time_to_deadline(10.5) == pytest.approx(12.0 - 10.5)


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


def test_chunk0_deadline_miss_zero_by_structure():
    # Chunk 0's deadline = its own finish time, so it is on-time without any
    # special-case branch.
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(3.0)
    state.on_chunk_boundary(4.0, word_count=1, consume_duration=1.0)
    rec = state.chunk_records[0]
    assert rec.unit_deadline_miss_s == pytest.approx(0.0)
    assert rec.consumer_ready_time == pytest.approx(4.0)
    assert rec.deadline == pytest.approx(rec.text_generation_end_time)


def test_consume_start_time_set_at_chunk0_end():
    state = RequestSLOState(num_warmup_chunks=1)
    state.on_token(1.0)
    state.on_chunk_boundary(2.5, word_count=2, consume_duration=1.0)
    assert state.consume_start_time == pytest.approx(2.5)
    assert state.consume_start_time == pytest.approx(
        state.chunk_records[0].text_generation_end_time)


def test_unit_deadline_miss_populated_when_late():
    # Chunk 1 arrives after its deadline.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(1.0, word_count=1, consume_duration=2.0)
    # deadline(1) = max(1.0, 1.0) + 2.0 = 3.0; chunk arrives at 4.0.
    state.on_token(1.0)
    state.on_chunk_boundary(4.0, word_count=1, consume_duration=1.0)
    rec = state.chunk_records[1]
    assert rec.unit_deadline_miss_s == pytest.approx(1.0)
    assert rec.unit_deadline_missed == 1
    assert rec.deadline == pytest.approx(3.0)
    assert rec.consumer_ready_time == pytest.approx(4.0)


def test_unit_deadline_miss_zero_when_ontime():
    # Chunk 1 arrives before its deadline.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    state.on_chunk_boundary(1.0, word_count=1, consume_duration=5.0)
    # deadline(1) = max(1.0, 1.0) + 5.0 = 6.0; chunk arrives at 3.0.
    state.on_token(1.0)
    state.on_chunk_boundary(3.0, word_count=1, consume_duration=1.0)
    rec = state.chunk_records[1]
    assert rec.unit_deadline_miss_s == 0.0
    assert rec.unit_deadline_missed == 0


# ---------------------------------------------------------------------------
# Phase 2A tests: ChunkRecord new fields
# ---------------------------------------------------------------------------

def test_chunk_token_indices_monotone():
    # Three chunks; token indices must be contiguous with no gaps.
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    # Chunk 0: 5 tokens
    for _ in range(5):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=1.0)
    # Chunk 1: 3 tokens
    for _ in range(3):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=1, consume_duration=1.0)
    # Chunk 2: 7 tokens
    for _ in range(7):
        state.on_token(0.4)
    state.on_chunk_boundary(0.5, word_count=3, consume_duration=1.0)

    recs = state.chunk_records
    assert len(recs) == 3
    assert recs[0].token_start == 0
    assert recs[0].token_end == 5
    assert recs[1].token_start == 5
    assert recs[1].token_end == 8
    assert recs[2].token_start == 8
    assert recs[2].token_end == 15
    # token_boundary matches token_end.
    for rec in recs:
        assert rec.token_boundary == rec.token_end


def test_chunk_demand_window_derives_from_deadline_and_consume_duration():
    state = RequestSLOState(num_warmup_chunks=1, min_chunk_tokens=0)
    state.on_token(0.0)
    # chunk 0: deadline = now = 1.0, consume = 2.0
    state.on_chunk_boundary(1.0, word_count=1, consume_duration=2.0)
    rec = state.chunk_records[0]
    assert rec.deadline == pytest.approx(1.0)
    assert rec.deadline + rec.consume_duration == pytest.approx(3.0)
    assert rec.consume_duration == pytest.approx(2.0)


def test_predictor_source_p90_emits_value_high():
    state = RequestSLOState(
        num_warmup_chunks=1, chunk_len_strategy="p90", min_chunk_tokens=0)
    # Need at least 2 chunks to have a value_high (p99 over history).
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=1.0)
    for _ in range(20):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=3, consume_duration=1.0)
    # chunk 1 is the first to have a non-None expected_chunk_len_high
    rec = state.chunk_records[1]
    assert rec.predictor_source == "p90"
    assert rec.expected_chunk_len_high is not None


def test_predictor_source_ema_no_value_high():
    state = RequestSLOState(
        num_warmup_chunks=1, chunk_len_strategy="ema", min_chunk_tokens=0)
    for _ in range(10):
        state.on_token(0.0)
    state.on_chunk_boundary(0.1, word_count=2, consume_duration=1.0)
    for _ in range(5):
        state.on_token(0.2)
    state.on_chunk_boundary(0.3, word_count=1, consume_duration=1.0)
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
# Empirical tail API on ChunkLengthPredictor (ProgressServe primitives)
# ---------------------------------------------------------------------------

def test_predictor_empirical_tail_api():
    pred = ChunkLengthPredictor(strategy="p90")
    for v in [10, 20, 30, 40, 50]:
        pred.update(v)
    # |{h > 25}| = {30, 40, 50} = 3.
    assert pred.sample_count_above(25) == 3
    # P(L > 25 | L > 5) = |{>25}| / |{>5}| = 3 / 5 = 0.6.
    assert pred.tail_prob(25, 5) == pytest.approx(0.6)
    # Non-increasing in x.
    probs = [pred.tail_prob(x, 5) for x in (5, 15, 25, 35, 45, 55)]
    assert all(b <= a for a, b in zip(probs, probs[1:]))
    # Conditioning event {L > 95} is empty → denominator 0 → None.
    assert pred.tail_prob(100, 95) is None
