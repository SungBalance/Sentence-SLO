# TTS profiles (consumed by SSLO)

`run_sweep.sh` and `run_smoke.sh` default to `TTS_PROFILE_PATH=exp/run_sslo/profiles/word_count_duration_stats.csv`. Drop the canonical profile CSV (produced by `exp/measure_tts_duration/summarize_word_stats.py`) here so SSLO is decoupled from the measurement pipeline's transient outputs.

Required CSV columns (other columns are ignored by `TtsProfileConsumeEstimator`):
- `model` — HF model id (e.g. `hexgrad/Kokoro-82M`, `Supertone/supertonic-3`)
- `word_count_low` — bin lower bound (1-unit binning recommended)
- `conversion_time_s_mean` — TTS synthesis wall-time per chunk (seconds)
- `audio_duration_s_mean` — output audio playback duration per chunk (seconds)

When running TTS sweeps, the launcher passes `--tts-model <model>` to filter the profile; if no row matches, `TtsProfileConsumeEstimator.__init__` raises with the list of models present in the file. Bundle profiles for all TTS models (currently Kokoro + Supertone) into one CSV.
