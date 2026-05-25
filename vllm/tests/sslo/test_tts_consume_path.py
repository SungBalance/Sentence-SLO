# SPDX-License-Identifier: Apache-2.0
"""Tests for SSLO reader and TTS consume timing paths."""

import pytest

from vllm.sslo.config import SsloConfig
from vllm.sslo.slo_state import ChunkConsumeEstimator, RequestSLOState


class WordScaledTtsEstimator(ChunkConsumeEstimator):

    def estimate(
        self,
        chunk_text: str,
        word_count: int,
    ) -> tuple[float, float | None]:
        del chunk_text
        return 0.3 * word_count, 0.1 * word_count


def _add_one_token_and_close_chunk(
    state: RequestSLOState,
    *,
    now: float,
    word_count: int,
    consume_duration: float,
    conversion_time: float | None = None,
) -> None:
    state.on_token(now)
    state.on_chunk_boundary(
        now=now,
        word_count=word_count,
        consume_duration=consume_duration,
        conversion_time=conversion_time,
    )


def test_reader_path_uses_generation_finish_for_slack_and_deadline():
    state = RequestSLOState(num_warmup_chunks=1)
    chunks = [(1.0, 2, 0.6), (1.4, 3, 0.9), (2.7, 1, 0.3)]

    for now, word_count, consume_s in chunks:
        _add_one_token_and_close_chunk(
            state,
            now=now,
            word_count=word_count,
            consume_duration=consume_s,
            conversion_time=None,
        )
        record = state.chunk_records[-1]
        assert record.consumer_ready_time == pytest.approx(
            record.text_generation_end_time)
        assert record.unit_deadline_miss_s == pytest.approx(
            max(0.0, record.consumer_ready_time - record.deadline))
        assert state.next_deadline_ts == pytest.approx(
            max(record.deadline, record.consumer_ready_time) + consume_s)
        assert record.conversion_time == pytest.approx(0.0)


def test_tts_path_uses_audio_ready_time_for_slack_and_deadline():
    state = RequestSLOState(
        num_warmup_chunks=1,
        consume_estimator=WordScaledTtsEstimator(),
    )
    chunks = [(1.0, 2), (1.4, 3), (2.7, 1)]

    for now, word_count in chunks:
        consume_s, conversion_s = state.consume_estimator.estimate(
            "", word_count)
        assert conversion_s is not None
        _add_one_token_and_close_chunk(
            state,
            now=now,
            word_count=word_count,
            consume_duration=consume_s,
            conversion_time=conversion_s,
        )
        record = state.chunk_records[-1]
        consumer_ready_time = record.text_generation_end_time + 0.1 * word_count
        assert record.conversion_time == pytest.approx(0.1 * word_count)
        assert record.consumer_ready_time == pytest.approx(consumer_ready_time)
        # Text-side miss: text_generation_end vs text-side deadline.
        # (Equivalent to consumer-side miss; the two differ only by a
        # fixed conv shift on the time axis.)
        assert record.unit_deadline_miss_s == pytest.approx(
            max(0.0, record.text_generation_end_time - record.deadline))
        assert state.next_deadline_ts == pytest.approx(
            max(record.deadline, record.text_generation_end_time)
            + 0.3 * word_count)


def test_tts_chunk0_consume_start_waits_for_audio_ready_time():
    state = RequestSLOState(num_warmup_chunks=1)

    _add_one_token_and_close_chunk(
        state,
        now=5.0,
        word_count=4,
        consume_duration=1.2,
        conversion_time=0.4,
    )

    assert state.consume_start_time == pytest.approx(5.4)


def test_consume_mode_tts_requires_profile_path():
    with pytest.raises(ValueError):
        SsloConfig(consume_mode="tts", tts_profile_path=None)


def test_consume_mode_read_rejects_dead_tts_profile_path():
    with pytest.raises(ValueError):
        SsloConfig(consume_mode="read", tts_profile_path="x.csv")


def test_consume_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        SsloConfig(consume_mode="bad")


def test_tts_profile_estimator_loads_both_models():
    import csv
    from pathlib import Path

    from vllm.sslo.slo_state import TtsProfileConsumeEstimator

    profile_path = (
        Path(__file__).resolve().parents[2]
        / "exp"
        / "measure_tts_duration"
        / "output"
        / "profile_per_wc"
        / "word_count_duration_stats.csv")
    if not profile_path.exists():
        profile_path = (
            Path(__file__).resolve().parents[3]
            / "exp"
            / "measure_tts_duration"
            / "output"
            / "profile_per_wc"
            / "word_count_duration_stats.csv")
    if not profile_path.exists():
        pytest.skip(f"TTS profile CSV not found: {profile_path}")

    rows_by_key = {}
    with profile_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows_by_key[(row["model"], int(row["word_count_low"]))] = row

    for model in ("hexgrad/Kokoro-82M", "Supertone/supertonic-3"):
        estimator = TtsProfileConsumeEstimator(str(profile_path), model)
        for word_count in (1, 5, 10, 20):
            row = rows_by_key[(model, word_count)]
            assert estimator.estimate("", word_count) == (
                float(row["audio_duration_s_mean"]),
                float(row["conversion_time_s_mean"]),
            )


def test_tts_profile_estimator_nearest_neighbor_and_unknown_model(tmp_path):
    from vllm.sslo.slo_state import TtsProfileConsumeEstimator

    profile_path = tmp_path / "tts_profile.csv"
    profile_path.write_text(
        "model,word_count_low,conversion_time_s_mean,"
        "audio_duration_s_mean\n"
        "known-model,1,0.1,1.0\n"
        "known-model,5,0.5,5.0\n"
        "known-model,9,0.9,9.0\n"
        "other-model,1,0.2,2.0\n")

    estimator = TtsProfileConsumeEstimator(str(profile_path), "known-model")
    assert estimator.estimate("", -10) == (1.0, 0.1)
    assert estimator.estimate("", 0) == (1.0, 0.1)
    assert estimator.estimate("", 99) == (9.0, 0.9)
    assert estimator.estimate("", 3) == (1.0, 0.1)
    assert estimator.estimate("", 7) == (5.0, 0.5)

    with pytest.raises(ValueError, match="missing-model"):
        TtsProfileConsumeEstimator(str(profile_path), "missing-model")
