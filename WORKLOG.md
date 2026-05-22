# Work Log

## 2026-05-22 (session 3)

- Modified: `exp/run_sslo/run_test.py` — switched measurement window from time-based (`MEASUREMENT_WINDOW_S=180s` fixed) to **completion-gated** workflow per user request:
  - Pool size 4000 → **4096** (wildchat2048+lmsys2048).
  - `warmup_target = max_num_seqs * 2`, `measurement_target = max_num_seqs * 4`.
  - Window timestamps written INLINE by the gate-flipping task (race-free; capturing in watcher after `await event.wait()` would race with other completions in between and produce a near-zero window).
  - Records both wall (`time.time()`) and monotonic (`time.monotonic()`) bounds. Wall for request `in_window` (vs `completion_wall_ts`), monotonic for scheduler_stats trim (matches vLLM's stat clock).
  - `in_window = completed in [window_start, window_end]` (per user spec "측정 시작 이후 완료된 request 수집"). New field `completion_wall_ts` on each request row.
  - No cooldown: on measurement_done_event, `engine.abort(in_flight_ids)` + `task.cancel()` + `asyncio.gather(return_exceptions=True)`.
  - `chunks.jsonl` filtered to in_window reqs (no warmup/post-window partials).
  - `scheduler_stats.jsonl` post-trimmed in-place to monotonic-window bounds.
  - Warmup + measurement safety timeouts (= `MEASUREMENT_WINDOW_S`, now default 900s) prevent hang if throughput too low.
- Modified: `exp/run_sslo/analyze.py` — request filter prefers per-row `in_window` flag when present (new flow), falls back to `injection_ts ∈ [mw0, mw1)` for legacy runs. Scheduler-stats filter uses `measurement_window_start_mono_ts` when present.
- Modified: `exp/run_sslo/run_test.sh` — `NUM_PROMPTS` default 4000 → 4096, `MEASUREMENT_WINDOW_S` 180 → 900 (now repurposed as safety timeout).
- Verification: Smoke 9B cap=128 rate=8 sslo_mlp — `in_window=513` (target 512, +1 boundary task), `measurement_completed=514`, window=86.3s, no `engine.abort` errors. Race-free window timestamp recording confirmed by window matching the gate semantics (vs prior buggy 0).

## 2026-05-22 (session 2)

- Modified: `exp/tools/lm_datasets.py` — added `_max_response_chunk_chars()` helper that runs SSLO's `ChunkSeparator` (sentence mode, `min_chunk_tokens=0`) on a response and returns the longest chunk length. Added `max_response_chunk_chars: int | None` parameter to `load_prompts`, `_load_wildchat`, `_load_lmsys`, `_load_combine`. Filter drops rows whose first assistant response would produce a chunk longer than the threshold under runtime SSLO boundary rules.
- Modified: `exp/run_sslo/run_test.py` — added `--max-response-chunk-chars` CLI flag (int, default 0 = off); `_build_pool` forwards as `None` when 0.
- Modified: `exp/run_sslo/run_test.sh` — added `MAX_RESPONSE_CHUNK_CHARS` env var, default `1000`. New sweeps automatically drop dump-style source responses.
- Verification: WildChat 500 English first-asst-response sample → p99=373 chars, max=502 (well below 1000). Three known v12 mega-chunk offender shapes → 1725 / 2707 / 3961 chars (caught). 30K-row scan per source → drop rate 0.04% (2 rows each), false positive 0.

## 2026-05-22
- Modified: `exp/run_sslo/run_test.py` — added `--english-only` CLI flag; `_build_pool` wildchat/lmsys calls forward `args.english_only`.
- Modified: `exp/run_sslo/run_test.sh` — added `ENGLISH_ONLY` env var (default `1`) → `--english-only` flag. Default ON: all new runs are English-only unless explicitly disabled.
- Verification: `python3 -m compileall` + `bash -n` pass. Smoke test (`combine`, 50 prompts): all English. Comparison (wildchat 200 prompts): heavy non-ASCII 14 (off) → 7 (on); remaining 7 are English with Unicode quotes / em-dashes / TAB, no non-English content. Partial pool scan (250K rows): English ratio 46.4%, conv+nocode pass 17.5%. Extrapolated combine English pool ≈ 1.1 M (well above 4 K sweep budget).

## 2026-05-02 (session 2)

- Modified: `vllm/vllm/sslo/slo_state.py` — added `SsloRequestStats` dataclass (8 fields), added `_num_pending_intervals`, `_cur_consecutive_pending`, `_max_consecutive_pending` tracking to `RequestSLOState.__init__`, updated `on_pending_enter/exit` to count intervals and track consecutive, added `compute_stats()` method.
- Modified: `vllm/vllm/outputs.py` — added `sslo_metrics: "SsloRequestStats | None" = None` kwarg and `self.sslo_metrics` assignment to `RequestOutput`.
- Modified: `vllm/vllm/v1/engine/output_processor.py` — wired `sslo_metrics=self.slo_state.compute_stats() if finished else None` in `_new_request_output`.
- Verification: All three files passed `python3 -m compileall` in `sk-sslo-vllm`.

## 2026-05-02

- Modified: Added env-var-gated SSLO scheduler iteration stats logging in `vllm/vllm/v1/core/sched/scheduler.py` after pending/running redistribution.
- Added: Created `exp/sslo_test/` with `run_test.py`, `run_test.sh`, `analyze.py`, and `README.md` for baseline-vs-SSLO TTFT measurement and H1/H2 summary output.
- Debugging/verification: In `sk-sslo`, reinstalled the editable vLLM checkout, ran `compileall` for `exp/sslo_test/run_test.py` and `exp/sslo_test/analyze.py`, and ran `bash -n exp/sslo_test/run_test.sh`. The 16-request smoke completed with H1 FAIL and H2 FAIL. The full baseline completed; the full SSLO run crashed in `schedule_sslo()` on `assert len(self.running) <= max_num_running_reqs` after logging `max_running_plus_pending=71`, so H1 PASS evidence was captured but H2 could not be measured for the full run.

## 2026-04-30

- Added: Created `exp/measure_KV_overhead/` experiment for profiling GPU↔CPU KV cache block transfer overhead.
  - `measure_kv_overhead.py`: reads HF model config (no weights) to derive KV shape (`num_layers`, `num_kv_heads`, `head_dim`), allocates GPU tensor + pinned CPU tensor of shape `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]`, sweeps `num_blocks` values, measures offload/onload wall-time with CUDA events, writes `kv_overhead.csv` and `kv_overhead.png`.
  - `run_experiment.sh`: host-side `docker exec` launcher for `sk-sslo`, defaults to Qwen/Qwen3-8B, block_size=16, bfloat16, 12-point sweep (1–2048 blocks).
  - `README.md`: documents experiment purpose, usage, output layout, and all CLI flags.
- Debugging/verification: Smoke-tested inside `sk-sslo` with Qwen/Qwen3-8B (`num_layers=36`, `num_kv_heads=8`, `head_dim=128`), `--num-blocks 1 2 4`, produced correct CSV with header `num_blocks,total_bytes,offload_ms,onload_ms,offload_bandwidth_GBs,onload_bandwidth_GBs` and PNG plot. Measured ~34–53 GB/s PCIe bandwidth (expected range).

## 2026-04-20

- Modified: Added vLLM editable install instructions to root `README.md`, root `AGENTS.md`, and `vllm/AGENTS.md`, including `sk-sslo`, `/workspace/mlsys/vllm`, git safe-directory, precompiled editable install, and build-helper prerequisites.
- Added: Documented the `HF_HOME=/cache` and `HF_HUB_CACHE=/cache/hub` cache convention in the README install/run flow.
- Debugging/verification: Confirmed `vllm` is installed editable in `sk-sslo` at `/workspace/mlsys/vllm` with version `0.0.0+sslo`, and `import vllm` succeeds.
- Modified: Updated root `AGENTS.md` and `vllm/AGENTS.md` with SSLO module placement, experiment directory, container execution, Hugging Face cache, and work-log rules.
- Added: Created root `CLAUDE.md` earlier to point Claude Code at `AGENTS.md`; created `WORKLOG.md` for ongoing session summaries.
- Debugging/verification: Checked `run_docker.sh` to confirm `sk-sslo` mounts this repo at `/workspace/mlsys` and host `/data` at container `/cache`.
- Modified: Added the rule that experiment `.sh` launch scripts should live in the same folder as their Python experiment script.
- Added: Created `exp/run_slack_bench.py` for prebuilt-vLLM slack timeline measurement without modifying or editable-installing vLLM; created `exp/run_slack_bench.sh` next to it with `HF_HOME=/cache` and `HF_HUB_CACHE=/cache/hub`.
- Debugging/verification: Confirmed the fresh `sk-sslo` container uses prebuilt vLLM `0.17.1+a03ca76a.nv26.03.46967107` with torch `2.11.0a0` CUDA `13.2`; installed the missing `datasets` benchmark dependency; ran `python3 -m compileall exp/run_slack_bench.py`; completed a smoke run on the real HF dataset `Aeala/ShareGPT_Vicuna_unfiltered` with no fixed input/output length and produced `exp/slack_results.json` plus `exp/slack_results.jsonl`.
- Modified: Added `--apply-chat-template` to `exp/run_slack_bench.py` for single-turn chat-template wrapping of sampled text prompts.
- Debugging/verification: Ran `python3 -m compileall exp/run_slack_bench.py` and completed a one-request smoke run with `--apply-chat-template`, producing `exp/slack_chat_results.json` and `exp/slack_chat_results.jsonl` with `apply_chat_template: true`.
- Modified: Updated `exp/run_slack_bench.sh` to run 1000 requests for the requested benchmark.
- Debugging/verification: Ran `exp/run_slack_bench.sh` inside `sk-sslo` using the real HF dataset `Aeala/ShareGPT_Vicuna_unfiltered`; produced `./slack_results.json` and `./slack_results.jsonl` with 1000 JSONL records, 8704 chunks, 170.69 requests/s, 80552.92 total tokens/s, and 37181.97 output tokens/s.
- Added: Created `exp/postprocess_slack.py` and `exp/postprocess_slack.sh` to compute per-chunk consume time, current deadline/slack, and cumulative deadline/slack from slack timeline JSONL.
- Debugging/verification: Ran `python3 -m compileall exp/postprocess_slack.py` and `exp/postprocess_slack.sh` inside `sk-sslo`; produced `exp/slack_chunks_postprocessed.csv` with 8704 chunk rows and the requested output columns.
- Modified: Updated slack post-processing to emit both `exp/slack_chunks_postprocessed.jsonl` and `exp/slack_chunks_postprocessed.csv`; CSV export uses pandas when available and falls back to the standard CSV writer.
- Debugging/verification: Re-ran `exp/postprocess_slack.sh` inside `sk-sslo`; verified 8704 JSONL lines and 8704 parsed CSV records.
- Modified: Merged the post-processing launch step into `exp/run_slack_bench.sh`, removed the separate `exp/postprocess_slack.sh`, and moved benchmark/postprocessed outputs under `exp/outputs/`.
- Debugging/verification: Ran `bash -n exp/run_slack_bench.sh` and `python3 -m compileall exp/run_slack_bench.py exp/postprocess_slack.py` inside `sk-sslo`.
- Modified: Updated `exp/run_slack_bench.sh` to keep experiment options as constants and iterate over `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.5-27B` with a shell `for` loop; added the same run-option convention to `AGENTS.md`.
- Debugging/verification: Ran `exp/run_slack_bench.sh` inside `sk-sslo` for 1024 requests per model. Produced per-model outputs under `exp/outputs/Qwen__Qwen3.5-35B-A3B/` and `exp/outputs/Qwen__Qwen3.5-27B/`; 35B-A3B produced 22285 postprocessed chunk rows at 5.99 requests/s and 2805.32 total tokens/s, while 27B produced 23649 rows at 3.47 requests/s and 1624.51 total tokens/s.
- Added: Created `exp/add_audio_duration_slack.py`, `exp/plot_slack_distributions.py`, and `exp/run_audio_slack.sh` to synthesize each chunk with `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, cache WAV durations, add audio-duration slack columns, and plot human-vs-audio slack distributions.
- Modified: Updated `exp/run_audio_slack.sh` to keep TTS/postprocess options as shell constants and run the two LLM model result sets in parallel across `cuda:0` and `cuda:1`.
- Debugging/verification: Installed `qwen-tts`, `sox`, and `matplotlib` in `sk-sslo`; found that public `torchaudio` is ABI-incompatible with the NVIDIA torch wheel, so `add_audio_duration_slack.py` stubs the unused `torchaudio.compliance.kaldi` import. Found that Qwen TTS CUDA vocoder needs `torch.backends.cudnn.enabled = False`; smoke tests produced WAV files and plot PNGs successfully. Started the full audio-duration run; it is resumable through per-model `audio_duration_cache.jsonl` files and writes outputs under `exp/outputs/<model>/`.
- Modified: Reworked `exp/plot_slack_distributions.py` to use seaborn `displot` histograms for model-by-model human/audio slack distributions, and updated `exp/run_audio_slack.sh` to install seaborn when missing.
- Debugging/verification: Installed seaborn in `sk-sslo`, regenerated partial distribution plots from the current audio-duration cache under `exp/outputs/plots/`, and restarted the full audio-duration run in the background with logs at `exp/outputs/run_audio_slack.log`.
- Added: Created `exp/analyze_slack_positive.py` to summarize positive slack by model, chunk position, cumulative/current mode, and previous-chunk gap/consume statistics.
- Modified: Fixed `exp/run_slack_bench.py` sentence boundary handling so whitespace streamed after a sentence-ending token is not flushed as a zero-word chunk.
- Modified: Updated `exp/add_audio_duration_slack.py` audio cache keys to include a text hash so stale durations from old chunk boundaries are not reused after re-running the benchmark.
- Debugging/verification: Stopped the in-progress audio TTS run because it was processing the old zero-word chunk artifacts. Verified with the positive-slack analysis that roughly 89-91% of non-first positive human current slack cases in the old outputs followed a zero-word previous chunk, and compile/smoke-tested the chunk collector/cache fixes in `sk-sslo`.
- Modified: Changed human and audio slack columns to use `deadline - actual_chunk_end`, updated the seaborn plot x-axis label, and moved fresh rerun outputs under `exp/outputs/deadline_minus_actual/`.
- Modified: Added `model`, `dataset_name`, and `dataset_path` to the slack benchmark summary JSON and removed a duplicate `--async-engine` flag from `exp/run_slack_bench.sh`.
- Debugging/verification: Restored `transformers==4.57.5` for vLLM after the qwen-tts install had downgraded it, reran the full 1024-request benchmark for both Qwen models from scratch, and verified the new postprocessed outputs have zero zero-word chunks. Started the full audio-duration rerun in the background; logs are at `exp/outputs/deadline_minus_actual/run_audio_slack.log`.
- Modified: Updated `exp/run_slack_bench.py` to build `SamplingParams` from the model generation config via `model_config.get_diff_sampling_param()` and `SamplingParams.from_optional()`, while using `GENERATION_MAX_TOKENS=1024` only as a safety cap when `max_new_tokens` is absent.
- Debugging/verification: Stopped stale experiment processes, cleaned `exp/outputs`, reran the full 1024-request benchmark from scratch with generation-config sampling, and confirmed vLLM applied `top_k=20`, `top_p=0.95`, plus `temperature=0.6` for `Qwen/Qwen3.5-27B`. New outputs under `exp/outputs/deadline_minus_actual/` have zero zero-word chunks; audio-duration TTS postprocessing is running in the background.
- Added: Created `exp/vllm_omni_tts_duration_smoke.py`, `exp/run_vllm_omni_tts_smoke.sh`, `exp/qwen3_tts_omni_batch_gpu0.yaml`, and `exp/omni_torchaudio_stub/sitecustomize.py` to test Qwen3-TTS duration generation through vLLM-Omni with batched decoding.
- Debugging/verification: Installed `vllm-omni==0.18.0` in a temporary venv and verified it conflicts with the container's prebuilt vLLM `0.17.1` API; installing `vllm==0.18.0 --no-deps` in that venv then failed against the NVIDIA torch build. Also tested `vllm-omni==0.16.0`, which failed against the same prebuilt vLLM due older `OmniEngineArgs` API expectations. Removed the temporary root-level venvs after testing. Conclusion: vLLM-Omni needs a matching vLLM/Torch stack or purpose-built container before it can replace the current Qwen TTS path.

## 2026-04-22

- Modified: Updated `run_docker.sh` to support launching the original `sk-sslo` vLLM container, a `sk-sslo-omni` container from `vllm/vllm-omni:v0.18.0`, or a `sk-sslo-omni-016` container from `vllm/vllm-omni:v0.16.0`.
- Debugging/verification: Ran `bash -n run_docker.sh`; the script preserves the same workspace mount, `/cache` mount, GPU, IPC, privileged, and host-network options for all container variants.
- Added: Created `exp/qwen3_tts_omni_batch_gpu1.yaml`, `exp/qwen3_tts_omni_batch_tp2.yaml`, `exp/add_audio_duration_slack_omni.py`, and `exp/run_audio_slack_omni.sh` to compute audio-duration slack with vLLM-Omni Qwen3-TTS and produce JSONL/CSV plus seaborn distribution plots.
- Modified: Updated the vLLM-Omni audio slack launcher to run the full chunk set with TP=2 over GPUs 0 and 1, no partial chunk limit, and outputs under `exp/outputs/deadline_minus_actual/`.
- Debugging/verification: Installed seaborn in `sk-sslo-omni`, compiled the relevant experiment scripts, verified vLLM-Omni smoke generation, ran a 256-chunk TP=2 speed test at about 1.60 TTS segments/s, cleaned previous partial audio outputs, and started the full audio-duration slack run in the background with logs at `exp/outputs/deadline_minus_actual/audio_omni_full.log`.

## 2026-04-28

- Modified: Updated repo-root `AGENTS.md` to require all project code execution, tests, syntax checks, installs, benchmarks, and experiment launch scripts to run inside the appropriate Docker container, with host shell usage limited to file inspection/editing and git metadata.
- Modified: Updated `exp/run_experiment.sh` to write the new singular output layout under `exp/output/{model_slug}/{dataset_slug}/{slack_mode}/`, with `text_outputs`, `audio_durations`, and `results` subdirectories.
- Modified: Split the three Python stages cleanly: `exp/benchmark.py` writes request and chunk text outputs, `exp/audio_duration.py` writes only TTS duration rows/cache, and `exp/analyze_results.py` combines chunks plus durations to compute either `previous_chunk` or `cumulative` slack.
- Modified: Added mode-based result normalization so both slack methods write `human_slack_seconds` and `audio_slack_seconds` in `results/slack_rows.*`, while the selected mode only changes the source column mapping.
- Modified: Updated `exp/README.md` to document the new single launcher, three-stage pipeline, mode handling, slugged dataset/model folders, and output filenames.
- Debugging/verification: Inside `sk-sslo`, ran `python3 -m py_compile` for the three stage scripts and common utilities, `bash -n exp/run_experiment.sh`, and a synthetic `analyze_results.py` smoke test for both `previous_chunk` and `cumulative` outputs. Inside `sk-sslo-omni`, ran `python3 -m py_compile` for TTS/analysis/common scripts and `bash -n exp/run_experiment.sh`.
- Modified: Collapsed `exp/` to one experiment entrypoint: `exp/run_experiment.sh`, `exp/benchmark.py`, `exp/audio_duration.py`, `exp/analyze_results.py`, `exp/common/slack_utils.py`, one TP=2 vLLM-Omni config, and the torchaudio stub.
- Modified: Merged human slack post-processing into `exp/benchmark.py`, so the benchmark now writes both per-request timelines and per-chunk human slack rows without a separate postprocess script.
- Modified: Removed legacy/direct/smoke-only experiment code from `exp/`: direct Qwen TTS audio slack scripts, positive-slack diagnostic script, vLLM-Omni smoke script/launcher, extra GPU-specific Omni configs, and the previous multi-launcher folder structure.
- Added: Rewrote `exp/README.md` around the single remaining experiment flow and the human/audio analysis outputs.
- Debugging/verification: Ran `python3 -m py_compile` for `exp/benchmark.py`, `exp/audio_duration.py`, `exp/analyze_results.py`, and `exp/common/slack_utils.py`, plus `bash -n exp/run_experiment.sh`, inside both `sk-sslo` and `sk-sslo-omni` where relevant. Ran a small `build_slack_rows` smoke test inside `sk-sslo`.
- Modified: Reorganized `exp/` into workflow folders: `common/` for shared helpers, `slack_bench/` for chunk timeline benchmark/postprocess code, `audio_slack/` for direct Qwen TTS slack analysis/plots, and `omni_tts/` for vLLM-Omni TTS code/configs/stubs.
- Modified: Updated experiment launch scripts to keep outputs under `exp/outputs/` after the folder move and to call Python scripts from their new colocated workflow directories.
- Added: Created `exp/README.md` documenting the experiment folder structure and run order.
- Debugging/verification: Ran `python3 -m py_compile` for the reorganized Python scripts, `bash -n` for the launch scripts, and a small common-helper smoke test inside the `sk-sslo` container. Root-owned old `exp/__pycache__` files could not be removed from the host.
- Modified: Refactored slack experiment post-processing scripts to share JSONL/CSV writing, percentile/stat helpers, TTS text splitting, audio cache keys, torchaudio stubbing, and audio slack row construction through `exp/common/slack_utils.py`.
- Modified: Simplified the vLLM-Omni TTS smoke/production scripts by sharing Omni TTS input construction and removed unused smoke CLI options (`--log-dir`, `--batch-timeout`, `--shm-threshold-bytes`) that were not consumed by the code paths shown in logs.
- Added: Created `exp/common/slack_utils.py` as the common SSLO experiment utility module.
- Debugging/verification: Confirmed `exp/outputs/deadline_minus_actual/audio_omni_full.log` completed the vLLM-Omni audio slack run while the older direct Qwen TTS log ended with `Terminated`. Ran `PYTHONPYCACHEPREFIX=/tmp/sslo_pycache python3 -m py_compile` on the refactored experiment scripts and a small `slack_utils` smoke test. `ruff` was unavailable in the local shell.
- Modified: Removed the legacy `exp/outputs/` artifact tree through the container so the refactored pipeline only writes new runs under singular `exp/output/`.
- Debugging/verification: Re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`, plus a synthetic `analyze_results.py` schema smoke test for both `previous_chunk` and `cumulative`.
- Modified: Consolidated repeated human/audio deadline-slack timeline math into `exp/common/slack_utils.py:add_deadline_slack_columns`, moved shared mean/stat helpers there, reused common JSON writers in `exp/benchmark.py`, and made the TTS `sitecustomize` path call the same runtime patch helper as `exp/audio_duration.py`.
- Modified: Added short role comments to the experiment launcher and Python stages so the config blocks, three stages, cache handling, mode mapping, and plotting sections are easier to scan.
- Debugging/verification: Re-ran container-only `py_compile`/`bash -n` checks in `sk-sslo` and `sk-sslo-omni`, a synthetic two-mode analysis smoke test in `sk-sslo`, and a `PYTHONPATH=exp/torchaudio_stub` sitecustomize smoke test.
- Modified: Moved canonical stage filenames out of `exp/run_experiment.sh` and into Python path helpers in `exp/common/slack_utils.py`; the launcher now passes only stage directories via `--output-dir`, `--input-dir`, `--text-output-dir`, and `--audio-duration-dir`.
- Debugging/verification: Re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`, verified the new directory-based analysis smoke path, and checked the updated `audio_duration.py`/`analyze_results.py` CLI help in `sk-sslo-omni`.
- Modified: Removed unused LoRA support from `exp/benchmark.py`, including `lora_requests`, the LoRA import, sample kwargs, CLI flags, and `llm.generate(..., lora_request=...)`.
- Debugging/verification: Confirmed `rg` finds no remaining LoRA references under `exp/`, then re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`.
- Modified: Added measured-stage GPU warmup support to `exp/benchmark.py` via `--warmup-requests`, and added request arrival pacing via `--request-rate` plus `--request-burstiness`; `exp/run_experiment.sh` exposes these as top-level constants.
- Modified: Recorded warmup/request-rate settings in benchmark `summary.json` and documented the new stage-1 behavior in `exp/README.md`.
- Debugging/verification: Re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`, checked the new benchmark CLI options, and ran a small request-arrival helper smoke test in `sk-sslo`.
- Modified: Kept `exp/benchmark.py` independent of SSLO-specific vLLM chunk timing fields by using the local streamed `RequestOutput` chunk collector, while preserving `request_submit_ts`, warmup, and request-rate pacing.
- Modified: Updated `exp/README.md` to state that chunk timestamps come from the benchmark stream receive path and do not require custom vLLM fields such as `chunk_timings`.
- Debugging/verification: Re-ran container-only `py_compile`/`bash -n` checks in `sk-sslo` and `sk-sslo-omni`, and ran a helper-level smoke test for request-arrival delays, stream chunk collection, and `request_submit_ts` schema.
- Modified: Moved fixed benchmark options into `exp/benchmark.py`: vLLM backend, async engine mode, and chat-template application are now set internally instead of passed from `exp/run_experiment.sh`.
- Modified: Simplified the stage-1 launcher command and documented the fixed benchmark options in `exp/README.md`.
- Debugging/verification: Re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`, checked `exp/run_experiment.sh` with `bash -n`, and verified `benchmark.py --help` no longer exposes the removed fixed-option flags.
- Modified: Added `MAX_CHUNKS_PER_REQUEST=48` to `exp/run_experiment.sh` and `--max-chunks-per-request` handling in `exp/benchmark.py`, so each measured request writes at most 48 sentence chunks to downstream text outputs.
- Modified: Recorded chunk cap metadata in request timelines (`num_chunks_before_cap`, `max_chunks_per_request`, `chunks_truncated`) and documented the cap in `exp/README.md`.
- Debugging/verification: Re-ran container-only syntax checks in `sk-sslo` and `sk-sslo-omni`, ran a helper smoke test proving 60 collected chunks are capped to 48, removed pycache, and stopped leftover vLLM-Omni processes from the interrupted full run.
- Modified: Refactored `exp/audio_duration.py` prompt-length handling into a `PromptLengthEstimator` object that owns the Qwen3-TTS tokenizer/config once per TTS run, removing the ambiguous in-memory prompt-length cache dict.
- Modified: Renamed the resumable TTS result cache variable to `duration_cache` so it is clearly separate from prompt-length estimation.
- Debugging/verification: Stopped the restarted benchmark/TTS launcher and cleaned remaining vLLM processes, then ran `python3 -m py_compile exp/audio_duration.py` inside `sk-sslo-omni`.
- Debugging/verification: Stopped the interrupted `sk-sslo-omni` TTS run, killed the remaining orphan `VLLM::EngineCore`, and confirmed both GPUs returned to 0 MiB used memory.
- Modified: Converted `exp/run_experiment_read.sh` into a human-reading-only launcher by removing the vLLM-Omni container, TTS config rendering, audio duration generation, and audio output directories.
- Modified: Added `--analysis-target human` to `exp/analyze_results.py` so read-only runs can create per-mode human slack rows, summaries, and figures directly from benchmark text outputs.
- Debugging/verification: In `sk-sslo`, ran `python3 -m py_compile` for the touched Python files, `bash -n exp/run_experiment_read.sh`, and a human-only `analyze_results.py` smoke run using existing text outputs.
- Modified: Added optional `tqdm` progress bars to `exp/benchmark.py` for warmup requests and measured request completion in the async vLLM benchmark loop.
- Debugging/verification: In `sk-sslo`, ran `python3 -m py_compile exp/benchmark.py` and `bash -n exp/run_experiment_read.sh`.
- Modified: Added sentence/paragraph chunk grouping to `exp/benchmark.py`, including multi-output chunk groups from a single inference pass.
- Modified: Updated `exp/run_experiment_read.sh` to write human-read outputs under `exp/output/{model_slug}/{dataset_slug}/{chunk_unit}/{slack_mode}/...` for both `sentence` and `paragraph`.
- Debugging/verification: In `sk-sslo`, reran `py_compile`, `bash -n exp/run_experiment_read.sh`, and a small collector smoke test for sentence and paragraph boundaries.
- Modified: Set the read-only launcher to `NUM_PROMPTS=256`, `MAX_MODEL_LEN=8192`, `GENERATION_MAX_TOKENS=8192`, `TENSOR_PARALLEL_SIZE=2`, and `CUDA_VISIBLE_DEVICES=0,1`.
- Modified: Added benchmark summary metadata for max model length, tensor parallel size, and GPU memory utilization.
- Debugging/verification: Stopped the interrupted no-cap run, cleared the model/dataset output folders, ran the 256-request human-read experiment to completion in `sk-sslo`, backfilled the generated benchmark summaries with the 8K/TP=2 metadata, and confirmed both GPUs were idle afterward.
- Modified: Updated `exp/analyze_results.py` so final slack result rows, summaries, and figures exclude each request's first decoding chunk while preserving it for subsequent deadline calculation.
- Modified: Documented the first-decoding-chunk exclusion in `exp/README.md`.
- Debugging/verification: Re-ran container-only `py_compile`/`bash -n`, regenerated the human-read analysis outputs for the completed 256-request run, and verified every result CSV starts at `chunk_idx=1` with 256 excluded rows per model/chunk unit/slack mode.
- Added: Created `exp/eval_datasets.py` to load and clean the supported evaluation dataset, `HuggingFaceH4/Koala-test-set`, with deterministic prompt repetition when `NUM_PROMPTS` exceeds the dataset size.
- Modified: Removed vLLM benchmark dataset sampling from `exp/benchmark.py`; the benchmark now receives cleaned eval prompts, applies the model chat template, and records `dataset_item_id` in request timelines.
- Modified: Updated `exp/run_experiment.sh`, `exp/run_experiment_read.sh`, and `exp/README.md` for the Koala dataset flow and the new `--dataset-split` option.
- Debugging/verification: In `sk-sslo`, ran Python compile checks, `bash -n` for both launchers, a Koala loader smoke test for 185 prompts, and checked `benchmark.py --help` for the simplified dataset CLI.
- Debugging/verification: Cleared the regenerated Koala human-read result folders, reran `exp/run_experiment_read.sh` in `sk-sslo` for 256 requests across Qwen3.5 35B-A3B and 27B, and verified sentence/paragraph `previous_chunk` and `cumulative` summaries were recreated with first decoding chunks excluded.
- Modified: Changed `exp/eval_datasets.py` prompt selection so `NUM_PROMPTS` larger than the dataset size is clamped to the available rows instead of repeating prompts.
- Modified: Updated the `exp/benchmark.py --num-prompts` help text to describe the new clamp behavior.
- Debugging/verification: In `sk-sslo`, ran `python3 -m py_compile exp/eval_datasets.py exp/benchmark.py` and verified Koala `num_prompts=256` returns 180 unique dataset rows with no `__r` repeat ids.
- Modified: Removed mode-level duplication of shared experiment artifacts from `exp/run_experiment.sh` and `exp/run_experiment_read.sh`; text outputs and audio durations now live once outside `previous_chunk`/`cumulative`, while only analysis results are mode-specific.
- Modified: Updated `exp/README.md` to document the deduplicated output layout for full and human-read-only runs.
- Debugging/verification: In `sk-sslo`, ran `bash -n exp/run_experiment.sh && bash -n exp/run_experiment_read.sh` and checked that the old copy helper/source-mode paths are gone from the launchers.
- Modified: Added read-only batch sweep support around `max_num_seqs` in `exp/run_experiment_read.sh`, propagated `max_num_seqs` into benchmark and analysis summaries, and added `exp/compare_read_batch_results.py` for batch-size summary tables and comparison figures.
- Modified: Updated `exp/README.md` for the new `batch_{max_num_seqs}` read-only output layout and batch comparison outputs.
- Debugging/verification: In `sk-sslo`, ran `py_compile` for `exp/benchmark.py`, `exp/analyze_results.py`, `exp/common/slack_utils.py`, and `exp/compare_read_batch_results.py`, `bash -n exp/run_experiment_read.sh`, a synthetic human-only batch comparison smoke test, and a partial clean rerun where `batch_16` and `batch_32` completed before an operator interrupt during `batch_64`.
- Modified: Updated repo-root `AGENTS.md` so `exp/` is treated as a collection of experiment-specific folders rather than a flat script bucket.
- Modified: Added the rule that every experiment folder under `exp/` must include a concise `README.md` describing the experiment, main scripts, and output layout.
- Debugging/verification: Inspected the current `exp/` tree and confirmed the existing `exp/slack_dist/README.md` already satisfies the new documentation rule.
- Added: Created the new `exp/measure_tts_duration/` experiment folder with its own `README.md`, shared helpers, dataset chunk preparation, per-model TTS duration measurement, word-count summary aggregation, and a container-first launcher.
- Modified: Scoped the first version of the experiment to `hexgrad/Kokoro-82M` and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` in `sk-sslo-omni`, leaving `microsoft/VibeVoice-Realtime-0.5B` out for now.
- Modified: Set the default Qwen batch size for `exp/measure_tts_duration/run_experiment.sh` to `64`.
- Modified: Improved `exp/measure_tts_duration/measure_audio_duration.py` so Kokoro writes resumable cache rows incrementally and shows progress while running.
- Debugging/verification: Re-ran `bash -n exp/measure_tts_duration/run_experiment.sh`, container-side `py_compile`, and relaunched the experiment detached after clearing stray duplicate TTS processes.
- Added: Created `exp/measure_tts_duration/fit_duration_regression.py` to fit `word_count -> duration_seconds` regressions from raw duration rows and save per-group summaries plus figures.
- Modified: Documented the regression stage in `exp/measure_tts_duration/README.md`.
- Debugging/verification: In `sk-sslo-omni`, ran `python3 -m py_compile exp/measure_tts_duration/fit_duration_regression.py` and fit a linear regression for the existing Kokoro sentence-duration CSV.
- Modified: Extended `fit_duration_regression.py` with reviewer-facing quantile regression outputs and figures using `p10/p50/p90/p95` by default.
- Modified: Updated `exp/measure_tts_duration/README.md` to mention the quantile-regression artifact.
- Debugging/verification: In `sk-sslo-omni`, re-ran `py_compile` and generated the new quantile-regression outputs for the existing Kokoro sentence-duration CSV.
- Added: Created repo-root `.gitignore` entries for editor noise, local assistant/tool settings, Python caches/build artifacts, virtual environments, scratch files, experiment output/runtime directories, and local model/data artifacts.
- Debugging/verification: Verified representative paths with `git check-ignore`, including `.claude/`, nested `exp/.claude/`, `__pycache__/`, `exp/slack_dist/output/`, and experiment log files.
- Added: Created `vllm/vllm/sslo/config.py` with `SsloConfig` and `build_slo_state()`, plus SSLO config tests.
- Modified: Added `RequestSLOState.sslo_score`, stored Task 3 constructor parameters, and exported the new SSLO config factory symbols.
- Debugging/verification: In `sk-sslo`, installed the declared `tblib` test dependency after pytest conftest import failed, then ran SSLO pytest and compile checks successfully.
- Modified: Wired `SsloConfig` through `VllmConfig` and `EngineArgs`, including `sslo_params` construction in `create_engine_config()`.
- Debugging/verification: Reinstalled the local vLLM checkout editable in `sk-sslo`, ran the requested `VllmConfig().sslo_config` smoke check, and ran compileall for the two touched vLLM files.
- Modified: Added Task 3 `RequestSLOState` EMA tracking, pending enter/exit callbacks, pure generation/pending chunk record fields, and `is_pending_eligible` threshold logic.
- Added: Added SSLO state tests for EMA initialization/update, pending-time subtraction/reset, and pending eligibility.
- Debugging/verification: Confirmed the new tests failed before implementation, then ran `tests/sslo/test_slo_state.py`, all `tests/sslo/`, and `compileall` for `vllm/vllm/sslo/slo_state.py` inside `sk-sslo`; a separate verification pass found no issues.
- Modified: Replaced SSLO scheduler IPC updates from `(request_id, slack)` to `(request_id, text_delta, engine_timestamp)` and moved MP scheduler-side SLO updates to replay text deltas into the core-owned `RequestSLOState`.
- Modified: Wired `sslo_config` into `OutputProcessor`/`RequestState`, removed env-var SLO construction from the internal-slack benchmark, and passed benchmark `chunk_unit`/`seconds_per_word` through `sslo_params`.
- Added: Added `build_slo_state()` to `vllm/vllm/sslo/slo_state.py` for shared request-state construction from SSLO config.
- Debugging/verification: In `sk-sslo`, ran compileall for modified vLLM engine files plus benchmark and `slo_state.py`, ran `tests/sslo/` (`41 passed`), and completed the Qwen3-8B smoke after warming the compile cache and bypassing the container FlashInfer version-check mismatch.
- Modified: Applied Task 4 SSLO scheduler review fixes by removing the duplicate `build_slo_state()` from `vllm/vllm/sslo/slo_state.py`, switching engine imports to `vllm.sslo.config`, coalescing missing SSLO config to `SsloConfig()`, and reverting out-of-plan `slo_timestamp` threading.
- Debugging/verification: In `sk-sslo`, ran the requested compileall successfully and `tests/sslo/` (`41 passed, 16 warnings`); the exact Qwen3-8B smoke was blocked by the container FlashInfer/JIT cache version mismatch, then completed with `FLASHINFER_DISABLE_VERSION_CHECK=1` and produced `chunks.jsonl` rows with `gen_time` and `pending_time`.
- Modified: Added Task 5 SSLO scheduler wiring, including `_sslo_score_key`, SSLO scheduler state initialization, `schedule_sslo()` with pending redistribution ahead of the copied scheduler body, and SSLO cleanup when requests leave running state or are freed.
- Added: Created `vllm/tests/sslo/test_scheduler_sslo.py` covering SSLO score ordering and pending redistribution cases for no waiting work, eligible pending, and max-consecutive pending fallback.
- Debugging/verification: In `sk-sslo`, ran scheduler `compileall`, `tests/sslo/` (`47 passed, 16 warnings`), verified `schedule()` still matches `HEAD` byte-for-byte, and confirmed the requested removed-line grep produced no output.
- Modified: Added Task 6 SSLO offload marking in `scheduler.py` inside `schedule_sslo()` when KV allocation returns no blocks.
- Added: Appended `TestOffloadMarking` to `vllm/tests/sslo/test_scheduler_sslo.py`.
- Debugging/verification: In `sk-sslo`, ran `tests/sslo/` (`48 passed, 16 warnings`) and `compileall` for `scheduler.py` successfully.
- Session date: 2026-05-01
- Task: Task 7 — adaptive_batch_size in schedule_sslo()
- Modified: `vllm/vllm/v1/core/sched/scheduler.py` (`schedule_sslo()` only: added local cap, replaced 2 usages)
- Added: `TestAdaptiveBatchSize` class in `vllm/tests/sslo/test_scheduler_sslo.py`
- Verification: pytest tail, compileall clean, diff confirms `schedule()` lines 589/883 untouched

## 2026-05-02 (continued)

- Task: SSLO scheduler E2E test bug fix and full validation
- Fixed: `schedule_sslo()` redistribution could push `len(self.running) > max_num_seqs` when `max_consecutive_pending` forced pending requests back, causing `InputBatch` `assert new_req_index < self.max_num_reqs` AssertionError mid-run. Added strict cap enforcement: after redistribution, if `len(new_running) > max_num_running_reqs`, the highest-slack overflow is bumped to pending (cap takes priority over starvation prevention). Adaptive cap moved up so cap enforcement uses the reduced limit. Admission gate `==` → `>=` (defensive). End-of-loop assert restored.
- Modified: `exp/sslo_test/run_test.py` adds GPU cleanup (`del engine`, `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`) in `finally`, and 15s sleep between baseline/SSLO subprocesses to let CUDA driver release memory.
- Modified: `exp/sslo_test/run_test.sh` lowered `GPU_MEMORY_UTILIZATION` 0.95 → 0.85 to give headroom across the back-to-back subprocess runs.
- Verification: pytest 50 PASS. Full E2E run on Qwen3-8B / 256 prompts / max_num_seqs=64. **H1 PASS**: max(running+pending)=256, 2008 iterations above cap, max_pending=192. **H2 PASS**: post-cap-arrival cohort TTFT p50 18.35s → 2.45s (-86.66%), p90 27.04s → 4.49s.
- Modified: Added queue stall extraction from `RequestOutput.metrics` in `exp/sslo_test/run_test.py`, added queue stall p50/p90 summary columns and absent-metrics warning in `exp/sslo_test/analyze.py`, added SSLO scheduler early-return when slack is at or below EMA generation time, and changed `pending_slack_eps_num_tokens` default/docs/tests from 3 to 5.
- Added: Added `TestPendingEarlyReturn` coverage for slack below/above EMA eligibility decisions.
- Debugging/verification: `RequestOutput` class-level probe in `sk-sslo` did not expose dataclass fields or `metrics`; pytest `tests/sslo/` passed (`52 passed, 16 warnings`). Full sweep completed and wrote `exp/sslo_test/output/sweep_summary.json`; queue stall metrics were unavailable (`n/a` columns). H3 still failed for seqs 32, 64, and 128 with residual neg-slack-ratio diffs of +0.0014874, +0.0004916, and +0.0001316 respectively.

## 2026-05-02 (continued)

- Modified: Updated `exp/sslo_test/run_test.py` queue stall extraction to use `metrics.arrival_time` and guarded `metrics.scheduled_ts > 0`; updated `schedule_sslo()` early return to use real-time slack against `now`; replaced stale cumulative-slack early-return tests with `TestRealtimeSlackEarlyReturn`.
- Added: No new files.
- Debugging/verification: Confirmed `arrival_time` and `scheduled_ts` in `vllm/v1/metrics/stats.py`; `python3 -m pytest tests/sslo/ -v` passed (`52 passed, 16 warnings`); scheduler `compileall` passed. Benchmark sweep for `max_num_seqs=64` completed. H3 verdict: FAIL, SSLO neg_slack_ratio 0.004479713298348906 vs baseline 0.004182509505703422. Queue stall available with baseline p50/p90 -1776583702.4933395/-1776583687.3193479 and SSLO p50/p90 -1776583712.4382787/-1776583709.8867314. TPOT p50 changed 0.017062328668145388 -> 0.06119234455086896 (+258.6400528382292%). TTFT H2 p50 changed 18.376535241375677 -> 2.458701277966611 (-86.62043064336348%).

## 2026-05-02 (continued)

- Modified: Updated `exp/sslo_test/run_test.py` queue stall extraction to use monotonic `metrics.queued_ts` and `metrics.scheduled_ts`, and added `decoding_start_ts` from `metrics.first_token_ts` to TTFT rows.
- Modified: Changed `RequestSLOState.is_pending_eligible` from a property to `is_pending_eligible(now)` using realtime slack, updated `schedule_sslo()` to call it with `now`, and updated SSLO tests for the method contract.
- Added: No new files.
- Debugging/verification: Per-file `py_compile` passed in `sk-sslo`; `python3 -m pytest tests/sslo/ -v` passed (`52 passed, 16 warnings`). The requested `max_num_seqs=64` sweep completed with H3 FAIL: baseline neg_slack_ratio 0.004182509505703422 vs SSLO 0.006238064926798218. Queue stall is now sane positive seconds: baseline p50/p90 11.704976434004493/26.834332884056494, SSLO p50/p90 1.3076576631283388/2.806667191442102. TPOT p50 changed 0.017063264845307267 -> 0.062441564169156485 (+265.9414815115476%). H2 TTFT p50 changed 18.352578241028823 -> 1.7862871129764244 (-90.26683286938388%).

## 2026-05-02 (continued)

- Modified: Reverted `RequestSLOState.is_pending_eligible` to a property using stale `cumulative_slack`, and updated scheduler/test call sites to property access while preserving the scheduler realtime-slack early-return check.
- Added: No new files.
- Debugging/verification: In `sk-sslo`, confirmed no remaining `is_pending_eligible(now)` call sites in the requested files; `python3 -m pytest tests/sslo/ -v 2>&1 | tail -20` passed (`52 passed, 16 warnings`).

## 2026-05-03

- Modified: Refactored SSLO chunk-generation timing from private EMA fields into `ChunkGenerationEstimator`, wired scheduler redistribution through the estimator, and configured `exp/sslo_test/run_test.py` to use p99 with window 100.
- Added: Added EMA and percentile chunk-generation estimators plus SSLO config fields/tests for estimator selection.
- Debugging/verification: In `sk-sslo`, `python3 -m pytest tests/sslo/ -v 2>&1 | tail -40` passed (`63 passed, 16 warnings`) and `compileall` passed. The `max_num_seqs=64` sweep completed: H1 PASS, H2 PASS, H3 FAIL; TTFT p50 18.51s -> 1.85s, negative-slack chunks baseline 33/7890 vs SSLO 50/7654.

## 2026-05-03 (continued)

- Modified: Refactored `exp/sslo_test/run_test.py` per-request collection to use engine-internal `RequestOutput.metrics` for TTFT, TPOT, queue stall, token count, and decoding start; removed client-side wall-clock timing fields. Updated `exp/sslo_test/analyze.py`, `exp/sslo_test/analysis/analyze_negslack.py`, `exp/sslo_test/README.md`, and `exp/sslo_test/run_test.sh`.
- Added: Shared `exp/sslo_test/jsonl_utils.py` JSONL reader and `exp/sslo_test/analysis/README.md` documenting final-output `slo_chunk_records` and `sslo_metrics` (`SsloRequestStats`) flow.
- Debugging/verification: In `sk-sslo`, compileall passed for changed experiment Python files; `python3 -m pytest tests/sslo/ 2>&1 | tail -5` passed (`68 passed, 16 warnings`). Smoke baseline run wrote `/tmp/sslo_smoke/baseline_ttft.jsonl` with populated engine metrics (`ttft=0.0661`, `tpot=0.01213`, `queue_stall=0.000009` for request 0). Full sweep completed for max_num_seqs 32/64/128/256 with `run_complete=true`; neg-slack chunk deltas were +4, -1, +0, +0 respectively.

## 2026-05-03 (continued)

- Refactored: pending in/out decision encapsulated inside `RequestSLOState.should_enter_pending(now)` / `should_exit_pending(now)`. Scheduler now only handles system-level guards (max_consec, waiting-empty, cap enforcement) and delegates per-request slack/EMA logic to the state object. Hysteresis factors (`pending_enter_factor=2.5`, `pending_exit_factor=2.0`) and warmup (`pending_warmup_chunks=5`) configurable via `SsloConfig`.
- Added: `ChunkGenerationEstimator` Protocol with `EmaChunkGenerationEstimator` and `PercentileChunkGenerationEstimator` (p99). Selectable via `SsloConfig.chunk_gen_estimator` ("ema"/"p99"). Estimator exposes `n_samples` for warmup checks.
- Modified: `exp/sslo_test/run_test.py` now reads engine-internal metrics (`output.metrics.first_token_latency` for TTFT, derived from `last_token_ts - first_token_ts` for TPOT) instead of client wall-clock measurements. Removed `t_submit`/`t_first_token`/`t_finish` fields. Added `exp/sslo_test/jsonl_utils.py` shared helper. Added `exp/sslo_test/analysis/` folder for ad-hoc post-hoc scripts.
- Verification: pytest 68 PASS. Full sweep (`max_num_seqs ∈ {32,64,128,256}`) on Qwen3-8B / 256 koala prompts:
  - 32: TTFT 28.6→11.9s (-58%), neg slack 25→29 (+4); H3 FAIL
  - 64: TTFT 18.2→4.7s (-74%), neg slack 33→32 (-1); **H1/H2/H3 ALL PASS**
  - 128: TTFT 11.5→2.9s (-74%), neg slack 40→40 (0 absolute); H3 ratio FAIL because SSLO produced fewer total chunks
  - 256 (control): SSLO ≈ baseline (no waiting-queue pressure)

## 2026-05-03 (later)

- Added: `sslo_adaptive` run mode in `exp/sslo_test/run_test.py` (sets `adaptive_batch_size=True`); `--run-kind` choices now `[all, baseline, sslo, sslo_adaptive]` and `run_all` runs all three sequentially with GPU cleanup between.
- Added: `analyze.py` 3-way comparison (separate metrics per SSLO variant).
- Modified: `run_single.sh` reordered positional args to match script section order (vLLM → Common → SSLO); pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1` to avoid contention with other containers.
- Modified: `run_test.py` `wait_for_gpu_memory_ready` polls the GPU index from `CUDA_VISIBLE_DEVICES`.
- Fixed: scheduler `adaptive_batch_size` branch had `max_num_running_reqs / 2` (float) which broke list slicing in the cap-overflow path. Changed to `// 2`.
- Verification: pytest 68 PASS (in `sk-sslo`). Smoke + full max_num_seqs=64 / 256 prompts on Qwen3-8B (`sk-sslo-vllm`):
  - TTFT p50: baseline 18.92s → sslo 4.79s → sslo+adaptive 4.59s
  - Queue stall p50: 11.71s → 3.34s → 3.14s
  - TPOT p50: 17.4ms → 58.1ms → 61.0ms
  - Neg slack chunks: 35/7770 → 34/7678 → **31/7656** (sslo+adaptive best)

## 2026-05-03 (3-run aggregate)

- Added: `exp/sslo_test/run_repeat.sh` — runs `run_single.sh` N times into `${output_root}/run_{i}/` for noise estimation.
- Added: `exp/sslo_test/analysis/aggregate_repeats.py` — reads N per-run summaries, prints mean ± stddev for TTFT/queue_stall/TPOT/neg_slack across baseline / sslo / sslo_adaptive.
- 3-run aggregate on Qwen3-8B / max_num_seqs=64 / 256 koala prompts:
  - TTFT p50: baseline 18.634±0.007 → sslo 4.709±0.002 → sslo+adaptive **4.580±0.003**
  - Queue stall p50: 11.831±0.003 → 3.324±0.003 → **3.139±0.001**
  - TPOT p50 (ms): 17.28±0.00 → 57.97±0.02 → 60.95±0.05
  - Neg slack: [34,35,34] → [34,34,34] → **[31,31,31]** (sslo+adaptive 100% reproducible at 31)
  - Standard deviations are 1-2 orders of magnitude below the inter-mode gaps → diffs not within error term.

## 2026-05-03 (Poisson arrival)

- Added `--request-rate` (reqs/sec, Poisson arrivals) and `--request-rate-seed` to `run_test.py`.
  - 0 (default) submits all prompts at t=0 (current behavior).
  - Positive value samples inter-arrival gaps from `Exp(rate)`.
  - Each prompt's task `await asyncio.sleep(arrival_offset)` before calling `engine.generate()`.
- `run_single.sh` accepts `request_rate` and `request_rate_seed` as positional args 6 and 7.
- `run_repeat.sh` accepts `request_rate` (positional 7) and `base_seed` (8); each run uses `seed = base_seed + i` so the N runs sample different arrival patterns at the same rate.
- Within one config, all 3 modes (baseline / sslo / sslo_adaptive) share the same seed → identical arrival schedule, apples-to-apples comparison.
- Smoke verified at rate=4 reqs/s (16 prompts, max_num_seqs=8): all three modes show identical decoding_start_ts spread (~2.3s span, same seed) → arrival timing reproducible.

## 2026-05-03 (systemic metrics)

- Modified: Extended `exp/sslo_test/analyze.py` summaries with TTFT and queue-stall p99/max, SLO compliance, negative-slack magnitude, running/combined scheduler occupancy, pending duration/intervals, and inter-chunk delay metrics.
- Modified: `exp/sslo_test/run_test.py` now carries chunk `end_time_ts` and per-request `sslo_metrics` pending fields (`total_pending_time_s`, `num_pending_intervals`, `max_consecutive_pending`) into JSONL rows.
- Modified: `aggregate_repeats.py` and `aggregate_sweep.py` print the new metrics and recompute raw-file-derived fallback fields for old summaries.
- Debugging/verification: In containers, compileall passed for changed experiment Python files; `python3 -m pytest vllm/tests/sslo/ -x -q` passed (`65 passed, 16 warnings`). GPU smoke at `/tmp/sslo_metric_smoke` completed and summary keys were non-null for SSLO. Re-aggregated sweep output now includes new TTFT/queue-stall tails plus SLO compliance, slack magnitude, running/combined, pending, and inter-chunk sections.

## 2026-05-03 (systemic metrics)

Added 7 new metric categories to exp/sslo_test/ for systemic-effect analysis (in addition to existing TTFT p50/p90, TPOT p50, queue_stall p50, neg_slack_ratio):

1. **TTFT/queue_stall p99 + max** — tail behaviour, not just median.
2. **Per-request SLO compliance rate** — `slo_compliance_rate_{mode}`: fraction of 256 requests where ALL chunks made deadline. Direct policy-evaluation metric (vs the chunk-level ratio which conflates short and long requests).
3. **Neg slack magnitude p50/p90/p99/max** — "how late were the late chunks", not just count.
4. **`len(running)` time-series mean/p50/p99** (from `${mode}_stats.jsonl`) — actual cap utilization.
5. **Effective batch (running + pending) mean/p50/p99** — total in-flight (KV-occupying) requests.
6. **Per-request pending duration p50/p90/p99 + interval count p50/p90** — extracted from `RequestOutput.sslo_metrics` (already exposed by `RequestSLOState.compute_stats()`); now captured in `run_test.py` per-request rows.
7. **Inter-chunk delay p50/p90/p99/max** — gap between consecutive chunk `end_time_ts` per request, useful for streaming smoothness (TTS / human reading rate).

`analyze.py` writes all these to `summary.json`; `aggregate_repeats.py` and `aggregate_sweep.py` extend their METRICS tuples to print per-mode mean ± stddev.

Existing 36-cell sweep re-aggregated successfully — all metrics populate from existing chunks/stats jsonl except #6 (which needs a fresh run since `total_pending_time_s` is per-request and was added to row schema in this change).

## 2026-05-06 (stall-aware slack + seqs=64 sweep)

- Modified: `vllm/vllm/sslo/slo_state.py` — `cumulative_consume_time` now updated stall-aware (`max(end_offset, current) + audio`) so a late chunk extends only the next deadline by its overage; previous formula carried debt forward and inflated downstream `위반률`. Added `ChunkConsumeEstimator` class (legacy `word_count × seconds_per_word` default) so TTS-derived durations can be injected later without touching `RequestSLOState`. Tests adjusted (`test_chunk1_records_real_slack`, `test_score_formula_and_deadline_sign`).
- Modified: `exp/sslo_test/run_sweep.sh` — defaults set to `MAX_NUM_SEQS_VALUES="64"`, `REQUEST_RATES="0 4 16 32"`, `CHUNK_UNITS="sentence paragraph"`. `GPU_RATE_ASSIGNMENTS` now auto-derived round-robin from `REQUEST_RATES` across `NUM_PARALLEL_GPUS=4`, removing the second hand-edited rate list.
- Verification: 58 SSLO tests pass inside `sk-sslo-vllm`. 24-cell sweep ran end-to-end (T+258m): TTFT p99 baseline ~278s → sslo/sslo_adaptive ~145–155s (~1.9× speedup); chunk_violation% ≤0.1% across all modes (formula B effect — no carryover); SLO compliance 97.6–98.6%; `summary.csv` written with 72 rows.

## 2026-05-09 (run_sslo schema overhaul)

- Modified: `vllm/vllm/sslo/slo_state.py` — `ChunkRecord` extended with `start_time_ts`, `num_token`, `num_iters`, `num_running_iters`, `num_pending_iters_per_chunk`. `ChunkStatCollector` gains `accumulate_running_step()` / `accumulate_pending_step()`; counts are stamped onto each ChunkRecord at the next chunk boundary and reset.
- Modified: `vllm/vllm/v1/core/sched/scheduler.py` — new `_account_per_chunk_step(running, pending)` helper, called at the end of both `_apply_sslo_policy` and `_apply_sslo_policy_v2`, walks `self.running` / `self.sslo_pending` and increments per-chunk iter buckets.
- Modified: `exp/run_sslo/run_test.py` — emits new chunks.jsonl schema (`chunk_slack`, `num_words`, `num_token`, `num_iters`, `num_running_iters`, `num_pending_iters_per_chunk`, `start_time_ts`, `request_idx`); requests.jsonl gains `ttfc` (queued_ts → first chunk end), `num_chunks`, and `num_pending_iters_per_request` (rename of `num_pending_intervals`). Per-mode `sslo_config_<mode>.json` sidecar written for analyze.py.
- Modified: `exp/run_sslo/_consolidate_mode_outputs.py` — folds per-mode `sslo_config_<mode>.json` into a single `sslo_config.json` keyed by mode, then removes the per-mode sidecar.
- Modified: `exp/run_sslo/analyze.py` — reads renamed chunk fields, emits `metrics.ttfc`, `metrics.slack.<mode>.violated-magnitude` (rename of `magnitude`), adds `min` to `metrics.scheduler.<mode>.combined`, and embeds `sslo_config` in `summary.config`. New `distribution_stats(include_min=True)` flag.
- Modified: `exp/run_sslo/metrics_utils.py` — DISPLAY_GROUPS reflect new `ttfc` rows, `violated-magnitude`, and `combined min/max`.
- Modified: `exp/run_sslo/analysis/sweep_analysis.py` — `csv` subcommand emits `sslo_*` config columns, `ttfc_s_*`, `violated_magnitude_s_*`, `combined_{min,max}`, and `slo_request_violation_rate` (= 1 - compliance_rate). One row per (cell, mode), 61 columns total.
- Verification: `pytest vllm/tests/sslo/` — 65 passed. End-to-end smoke with synthetic jsonl confirmed analyze.py + sweep_analysis.py csv produce all required fields.

## 2026-05-09 (token timeline plot)

- Added: `exp/run_sslo/analysis/plot_token_timeline.py` — plots cumulative-token timelines (generated vs consumed) for one request or averaged across all requests in a run. Generated curve uses chunk-TPS (= num_token / chunk_gen_duration) for token-level interpolation within each chunk window. Consumed curve uses (num_token / (num_words × seconds_per_word)) within the consumption window, with explicit idle segments when the reader was waiting for a chunk to arrive.
- Verification: smoke-tested with a 2-request synthetic chunks.jsonl in `sk-sslo-vllm`; produced single-request and average PNGs whose curve geometry matches the expected piecewise-linear segments and the chunk-arrival idle shape.

## 2026-05-11 (ProgressServe metric coverage + derivation module)

- Added: `/home/cheezestick/.claude/plans/metric-memoized-volcano.md` — ProgressServe metric handoff (§1–14) coverage analysis vs. existing `requests.jsonl` / `chunks.jsonl` / `scheduler_stats.jsonl`. Confirmed user-resolved definitions: deadline-based consumer model (chunk i consume window = [deadline_{i-1}, deadline_i]); pending_wait := queue_stall; consume_origin_ts := chunk_idx==0 end_time_ts; demand_duration := final_deadline - consume_origin; stall_fraction := total_stall / demand_duration; CP_SLO_violation := max_stall_time > tau (per-tau analysis param); measurement_window := first finish ~ (N - max_num_seqs)-th finish; request_class := canonical reference-run q33/q67 split with fixed request_id→class mapping. E7 ablation deferred. Critical gaps documented for future PRs: run_metadata sidecar, GPU sampler, drop/timeout (finish_reason), per-step scheduler decisions, reference-run class mapping.
- Modified: `exp/run_sslo/run_test.py` — `collect_request` extracts `num_prompt_tokens` from vLLM `RequestStateStats` (fallback: `len(prompt_token_ids)`); added to both early-return and main return dicts → flows into `requests.jsonl`. `run_one` prints `workload mean num_prompt_tokens=…, num_output_tokens=…` after the existing completion log line.
- Added: `exp/run_sslo/analysis/progress_metrics.py` — pure derivation module. Functions: `per_request_progress` (arrival_ts / completion_ts / completion_latency / consume_origin_ts / final_deadline_ts / demand_duration / total/max/stall_intervals/stall_fraction), `measurement_window` (first finish ~ (N − max_num_seqs)-th finish; returns (None,None) when N ≤ cap), `handling_users_stats` (time-weighted mean + p50/p95/p99/max within window), `throughput_stats` (tokens/s, completed_req/s within window), `cp_slo_violation_rates` (per-tau rate keyed `tau_{tau:g}`). `DEFAULT_TAUS = (0.5, 1.0, 2.0, 5.0)`.
- Modified: `exp/run_sslo/analyze.py` — imports `progress_metrics as pm`; `summary['metrics']` gains 6 new top-level groups: `progress_request`, `workload`, `throughput`, `cp_slo_violation`, `measurement_window`, `handling_users` (last is sslo* only). Existing keys, `passes` block, and `stall_time` group untouched.
- Modified: `exp/run_sslo/metrics_utils.py` — appended 6 new `DISPLAY_GROUPS` entries (Stall (request), Latency (request), Workload, Throughput, Handling users, CP-SLO violation) so `sweep_analysis.py csv` automatically emits new columns. Total groups: 7 → 13.
- Verification: `python3 -m compileall` on the three modified files inside `sk-sslo-vllm` — clean. Synthetic-data smoke test of `progress_metrics` produces the spec-expected derivations (1 request, 3 chunks; total_stall=0.5, max_stall=0.5, num_stall_intervals=1, demand_duration=2.5, stall_fraction=0.2). No existing `output_sweep` cell available for full end-to-end check; deferred to next sweep run.

## 2026-05-11 (smoke: sslo single-cell metric verification)

- Verification: ran `run_test.sh sslo 128 Qwen/Qwen3-8B` (NUM_PROMPTS=256, GEN=512, sentence, GPU 1) inside `sk-sslo-vllm`. 256 reqs in 46.4s, 4674 chunks. workload print line emitted: `mean num_prompt_tokens=163.5, num_output_tokens=510.1`.
- Verification: `analyze.py` emits all 6 new top-level keys in `summary.json.metrics` (`progress_request`, `workload`, `throughput`, `cp_slo_violation`, `measurement_window`, `handling_users`); existing keys unchanged. Sample sslo cell results — completion_latency p95=45.8s; total_stall_time max=4.65s; max_stall_time max=1.23s; CP-SLO viol rate {tau=0.5: 0.39%, tau=1: 0.39%, tau=2: 0%, tau=5: 0%} (consistent with max_stall=1.23s); throughput duration=9.84s, tokens/s=6610, completed_req/s=13.0; handling_users time_avg=210 (over 591 step samples); measurement_window N=256, cap=128 → 128 requests fall inside window (= N - max_num_seqs).

## 2026-05-11 (sweep_analysis csv extension + baseline comparison verification)

- Modified: `exp/run_sslo/analyze.py` — `sched_by_mode` now buckets for `ALL_MODES` (baseline included), and the `handling_users` loop iterates `ALL_MODES`. Existing `scheduler` SSLO-diagnostic block stays scoped to `SSLO_MODES`. Reason: baseline writes scheduler_stats.jsonl too (1294 rows for 256-prompt run); without this fix `handling_users.baseline` stays empty and capacity_gain comparison against baseline is impossible.
- Modified: `exp/run_sslo/analysis/sweep_analysis.py` — `DIST_METRICS` extended with 9 entries (total_stall_s, max_stall_s, num_stall_intervals, stall_fraction_pct (×100), completion_latency_s, demand_duration_s, prompt_tokens, output_tokens, chunks_per_req). `SCALAR_METRICS` extended with 10 entries (tokens_per_second, completed_req_per_s, measurement_duration_s, handling_users_{time_avg,p95,max}, cp_slo_viol_tau_{0_5,1,2,5}). `DIST_EXTRA_STATS` adds p95/max for stall and latency metrics. `agg-sweep` / `agg-repeat` already iterate `DISPLAY_GROUPS` so no change needed there.
- Verification: ran baseline (sslo) at `seqs_128/rate_0` (256 prompts, GEN=512, sentence). After analyze.py + sweep_analysis.py csv: 107-column summary.csv with 46 new metric columns. agg-sweep (with run_0 → run_1 symlink) prints all 6 new DISPLAY_GROUPS groups (Stall (request), Latency (request), Workload, Throughput, Handling users, CP-SLO violation) for both modes.
- Single-cell comparison snapshot: handling_users_time_avg baseline=116 vs sslo=210 (1.8× more concurrent users via pending state); max_stall_s_max baseline=6.53s vs sslo=1.23s; cp_slo_viol_tau_2 baseline=0.39% vs sslo=0% (tau=5s same). tokens/s & completed_req/s look very different across modes (113K vs 6.6K) because the `measurement_window` collapses to 0.58s on baseline (256 prompts finish in two dense 128-batches at request_rate=0); window definition is consistent — absolute throughput numbers are only comparable across cells, not policies, at request_rate=0.
- Real `capacity_gain` requires sweep across (max_num_seqs, request_rate) cells with `supported_users(tau)` evaluated per cell — single-cell proxy here just confirms the data needed for the calculation is now emitted by both modes.

## 2026-05-11 (summary_warmup.csv sidecar)

- Modified: `exp/run_sslo/analysis/sweep_analysis.py` `cmd_csv` — after writing the existing `summary.csv` (unchanged), additionally writes `summary_warmup.csv` next to it. One row per (cell, mode, run): `CONTEXT_COLUMNS` + `measurement_start_ts`, `measurement_end_ts`, `measurement_duration_s`. Window definition matches `progress_metrics.measurement_window`: first request finish → (num_requests − max_num_seqs)-th request finish. Source: `summary.json.metrics.measurement_window.<mode>`. No `run_test.py` change required (data already flows through chunks.jsonl → analyze.py → summary.json).
- Verification: re-ran `sweep_analysis.py csv` on the smoke sweep — `summary.csv` shape unchanged (107 cols), `summary_warmup.csv` emitted with the expected 13 columns and the correct ts values (baseline 0.58s window, sslo 9.84s window).

## 2026-05-11 (R1/R2 metric audit + DRY refactor loop)

- Round 1 audit (Explore agent): all derivations in `progress_metrics.py` matched plan's Confirmed Definitions / Derivation Recipe. Three DRY violations flagged: (a) `metrics_utils.lookup` vs `sweep_analysis._metric_node` duplicate nested-dict traversal; (b) 10-key context row inline-built in both `_emit_rows` and the warmup-csv block; (c) `pm.measurement_window` called twice per mode in `analyze.py`.
- DRY refactor (codex): extended `metrics_utils.lookup(summary, path, field, mode)` so `field=None` returns the node itself; deleted `_metric_node` and replaced call sites with `lookup(..., None, ...)`; added `_context_row(summary, path_label, run_idx, mode)` helper shared by `_emit_rows` and `cmd_csv` warmup block; precomputed `windows[mode]` dict in the first ALL_MODES loop of `analyze.py` and reused in the handling_users loop. Net delta: -12 lines across the 3 files.
- Verification (byte-identical): re-ran analyze + sweep_analysis csv on existing smoke cells; `summary.json` (baseline + sslo), `summary.csv` (107 cols), `summary_warmup.csv` (13 cols) all `diff`-SAME against pre-refactor snapshots.
- Round 2 audit (Explore agent): re-confirmed all refactors preserve functional equivalence; verified end-to-end values against plan (manual recompute of total_stall_time, completion_latency, measurement_window endpoints, cp_slo_violation rate at tau=0.5s, handling_users time_avg for baseline). No new DRY violations or orphan imports.

## 2026-05-12 — CP-SLO metric refactor (7 phases, commits 4fc2301..3c4812c)

Multi-agent planning + dev-verify loop per phase per AGENTS.md.
Spec at `/tmp/cpslo_todo.md`; canonical reference in `metric.md`.

- Phase 1 (`4fc2301`): timing contract. d(0) = consume_start_ts = chunk 0
  gen_finish_ts; removed special-case `slack=0` branch; new ChunkRecord
  fields `stall_start_ts / stall_end_ts / stall_duration_s`.
- Phase 2 (`1a2eecb`): schema expansion. 8 new ChunkRecord fields
  (token indices, demand window, predictor source/high). admitted_ts +
  terminal_outcome on RequestSLOState; scheduler stamps in
  schedule_sslo. F2 request_class via fixed token buckets
  (`xs/s/m/l`) on observed `num_generation_tokens`. New module
  `exp/run_sslo/analysis/cpslo_names.py` for legacy↔canonical alias.
  run_test.py dual-writes 15 chunk + 9 request canonical keys.
- Phase 3 (`74d7034`): pressure components + missing preservation
  (D1 split). New `PressureComponents` dataclass and
  `state.pressure_components(now, tpot)`; `pressure()` rerouted as
  thin wrapper. Scheduler `_state_pressure` renamed to
  `_policy_score_with_fallback` — keeps the 1.0 normalize on the
  policy path (cold-start critical guard depends on it) while
  decision logs see the raw None.
- Phase 4 (`f0fccab`): per-step × per-admitted-request decision log.
  Config knobs `decision_log_mode` (off / step / tier_changes /
  admit_only, default tier_changes) and `decision_heartbeat_steps`
  (200). Buffered JSONL writer to `decisions.jsonl`. Schema includes
  PressureComponents + admission/backfill/preemption_reason slots
  (None for now — populated in a follow-up).
- Phase 5 (`25d8f7a`): analysis migration. New
  `progress_metrics.compute_stall_intervals` merges contiguous
  stalls; `cp_slo_violation_rates` now thresholds on
  `max_stall_interval_s` (not per-chunk max). analyze.py adds
  `metrics["cpslo"]` section with max_stall_interval_distribution,
  num_stall_intervals_merged_distribution, cp_slo_violation_rates_by_tau,
  mean_handling_users_time_weighted.
- Phase 6 (`3c4812c`): validity + no-harm + run-level + F3. New
  `exp/run_sslo/analysis/validity.py` with pre-declared thresholds
  and `validate_run`. `run_meta.json` sidecar with run_id, variant,
  seed, trace_id, workload_id, N, M, measurement_start/end_ts,
  gpu_memory_peak_bytes, num_preemptions_total, completion counters.
  `_consolidate_mode_outputs.consolidate_validity_csv` writes
  `validity_checks.csv`.

Tests: 83 SSLO engine + 14 analysis = 97 total, all pass.

CP-SLO contract semantics that BREAK comparability with pre-refactor
sweep outputs:
- chunk 0 `deadline_ts` shifts from `decoding_start_ts` to
  `gen_finish_ts(0)` (chunk 0 itself).
- `mean_handling_users` headline switches from sample-mean to
  time-weighted-mean — analyze.py dual-writes for one transition release.
- `queue_stall_s` semantics shift from `scheduled_ts - queued_ts` to
  `admitted_ts - queued_ts` (admit-time-based).
- `cp_slo_violation` basis switches from per-chunk max to merged
  contiguous interval max.

Bump `summary.json` schema_version when dropping legacy alias keys
(planned follow-up).

## 2026-05-13 — throughput basis switched to full-run wall-time

`progress_metrics.throughput_stats` previously used the "first N-M
completions" window. That window stretched for SSLO modes whose
admission pattern is wave-shaped (admit → run → admit next wave),
producing apparent 30-50% throughput regression that was a
measurement artifact, not a real effect.

Canonical basis is now:
  duration_s = run_meta.measurement_end_ts - run_meta.measurement_start_ts
  tokens    = Σ num_output_tokens over ALL requests
  tok/s     = tokens / duration_s

`run_meta` is loaded once at the top of `analyze.analyze()` and
passed through to throughput_stats. Fallback chain: run_meta →
derived min/max of per_req arrival/completion → legacy window arg.
The result dict gains a `basis` field ("full_run" / "unknown").

Verified on cpslo_smoke_128 sweep (re-running analyze.py on the
existing outputs):

| mode/rate | old tok/s | new tok/s |
|---|---|---|
| baseline 16 | 6168 | 4454 |
| sslo 16     | 4363 | 4840 |
| sslo_adapt16| 5537 | 4340 |
| baseline 32 | 12605 | 5060 |
| sslo 32     |  5972 | 5338 |
| sslo_adapt32|  5863 | 5307 |

Under the new basis SSLO modes are within ±10% of baseline at this
load, matching the per-step Δts evidence (mode-independent ~21 ms).

## 2026-05-17 — MLP cap-shrink fix (system_factor) + Qwen3.5 chat-template plumbing

### Modified
- `vllm/vllm/v1/core/sched/scheduler.py`: `_compute_serve_defer_pair`에
  optional `cap_n` 파라미터 추가. `_mlp_pick_adaptive_n` 후보 평가 루프
  (line 2160)에서 `cap_n=n` 전달 → 각 후보 cap의 contention scale을
  `admitted/n`으로 산출. 기존엔 항상 `admitted/base_n`이라 작은 n 후보가
  shrink해도 effective throughput 손실이 pressure에 반영 안 됨.
- `vllm/tests/sslo/test_scheduler_sslo.py`: `_mlp_pick_adaptive_n` cap_n
  scaling 양방향 테스트 2개 추가 (`test_mlp_pick_n_system_scale_blocks_
  pointless_shrink`, `..._allows_meaningful_shrink`).
- `exp/run_sslo/run_test.py`: `--enable-thinking` / `--no-thinking` 플래그 +
  `apply_chat_template_to_prompts` helper. Qwen3.5 unified Instruct+Thinking
  모델에서 chat template `enable_thinking=False` 적용. Sampling override
  CLI args (`--temperature`, `--top-p`, `--top-k`, `--min-p`,
  `--presence-penalty`, `--repetition-penalty`) 추가.
- `exp/run_sslo/run_test.sh`: `ENABLE_THINKING`, `TEMPERATURE`/`TOP_P`/
  `TOP_K`/`MIN_P`/`PRESENCE_PENALTY`/`REPETITION_PENALTY` env vars →
  CLI args 전달.
- `exp/run_sslo/run_sweep.sh`: `START_RUN_INDEX` env var (default 1) — 기존
  runs를 덮지 않고 추가 run만 실행 가능.

### Added
- `exp/run_sslo/_diag_worst.py` — worst-case 셀 진단 (cap shrink rate,
  critical mode 비율, violating chunks 텍스트).
- `exp/run_sslo/_diag_tpot.py` — `decisions.jsonl`에서 batch별 step time
  추출해 memory-vs-compute-bound 판정.

### Findings (cap=64 r=128 sslo_mlp on Qwen3.5-27B)
- Before fix: tput 654 tok/s (-28.5% vs baseline 916), viol@τ=1 7.54%,
  cap_p50=24, has_critical=81% — death spiral.
- After fix: tput 929 tok/s (+1.4% vs baseline), viol@τ=1 0.79%,
  cap_p50=64, has_critical=59% (감지는 함, shrink만 자연 reject).
- 9B control (cap=128 r=64): tput 3705 vs baseline 3162 (+17%),
  viol 0% — fix이 compute-bound 케이스 무회귀.

### Verification
- `pytest tests/sslo/` 109/109 pass (이전 107 + 신규 2).
- Smoke 27B + 9B 모두 합격 기준 만족.

## 2026-05-18 — MLP backfill 후 stats single-dump + predictor escalate/overshoot fix

### Modified (`vllm/vllm/v1/core/sched/scheduler.py`)
- `SsloStepState`: 새 필드 `defer_base` — non-critical에서 stash해 post-waiting
  backfill ranking에 사용.
- `_apply_sslo_multi_level_pressure` non-critical (line 2320~): backfill 제거.
  `new_running = forced + warmup` 만 commit. `_sslo_step.defer_base = defer_base`
  로 snapshot.
- `schedule()` 메인 loop (line ~2980): waiting admit 후 새 backfill block 추가
  — `self.sslo_pending`에서 `defer_base` DESC 순으로 cap까지 채움. 각 backfilled
  req에 대해 KV alloc + decode token 배정 → 해당 step에 실제 실행됨.
  - `request.has_encoder_inputs` 케이스 skip (다음 step 처리).
- `_sslo_commit_step` (line 1228~): `_sslo_dump_step_stats` 호출 제거. docstring
  업데이트.
- `schedule()` 메인 loop 끝 (backfill 후): `_sslo_dump_step_stats(
  scheduled_timestamp)` 단일 호출 추가 → stats가 최종 running 반영.

### Modified (`vllm/vllm/sslo/config.py`, `slo_state.py`)
- `SsloConfig`: 새 knob 2개.
  - `mlp_predictor_escalate_threshold: float = 0.9` — 현재 tier 예측치의 90%에
    도달하면 다음 tier(p90 → p95 → p99)로 진입.
  - `mlp_predictor_overshoot_safety_factor: float = 2.5` — `cur > topmost tier`
    이후 `remaining = (cur - anchor) × factor` 산출 (1.0 saturate 제거).
- `ChunkLengthPredictor.__init__` + `RequestSLOState.expected_remaining_len`
  재작성: 위 두 knob 적용.
- `RequestSLOState.from_config`에 knobs 전달.

### Modified (`vllm/tests/sslo/`)
- `test_slo_state.py`: 신규 4 (`test_expected_remaining_escalates_at_90pct_threshold`,
  `_overshoot_grows_past_topmost`, `_legacy_threshold_and_factor`,
  `test_predictor_knob_validation`). `test_score_formula_and_deadline_sign`
  예상치 갱신 (1.0 floor → 7.5 overshoot).
- `test_scheduler_sslo.py`: `test_mlp_non_critical_admission_budget_uses_measured_only`
  업데이트 — backfill이 main loop으로 이동했으므로 MLP가 commit하는
  `new_running == 1` (warmup only), pending 3 + defer_base 검증.

### Verification

**Unit tests**: `pytest tests/sslo/` 113/113 pass.

**Worst-case smoke** (35B-A3B cap=128 r=128 sslo_mlp × 3 runs vs v3):
| metric | v3 (3-run avg) | v4 (3-run avg) | Δ |
|---|---|---|---|
| tput | 1620 | 1651 | +1.9% |
| TTFC mean | 64.0s | 60.0s | -6.3% |
| **viol@τ=1** | **1.40%** | **0.53%** | **-62%** |
| crit% | 21% | 12% | -43% |

**Stats dump fix** (v5, post-backfill dump):
| 지표 | v4 (pre-backfill dump) | v5 (post-backfill dump) |
|---|---|---|
| running p75 | 6 | 120 |
| cap=128 step 비율 | 11% | 24% |

→ stats가 backfill 후 실제 실행 시점을 정확히 반영.

**Controls (1 run)**:
- 27B cap=64 r=128 mlp: tput 972 (v3 932, +4%), viol 0.79% (v3 0.6%, 동등).
- 9B cap=128 r=64 mlp: tput 3685 (v3 3660, +0.7%), viol 0% (v3 0.2%, 동등).

### Added — backfill diagnostics
- `SsloStepState`에 `bf_skip_reason / bf_slack / bf_promoted / bf_kv_full /
  bf_too_few_tokens / bf_encoder_skipped / bf_budget_exhausted` 필드 추가.
- `schedule()` 메인 loop의 backfill block을 instrumented (skip 이유, KV alloc
  실패 카운터 등) — scheduler_stats.jsonl에 dump.

### Diag finding (35B-A3B cap=128 r=128)
Backfill loop이 86.2% step에서 진입했지만 그 중 **48.2%가 promoted=0**.
원인: **KV cache full** — `kv_cache_manager.allocate_slots()`가 6026 events
에서 None 반환. sslo_pending 260 reqs 분량 KV가 살아있어 추가 decode 슬롯
없음. → 후속 fix: pending pool size 제한 + KV 압력 기반 eviction.

## 2026-05-18 — hybrid predictor (global + per-request)

### Modified
- `vllm/vllm/sslo/slo_state.py`:
  - `_CHUNK_LEN_HISTORY_MAX`: 64 → 4096 (per-req history cap 완화).
  - `_GLOBAL_PREDICTOR_WARMUP_SAMPLES = 16` (per-req sample이 16 미만이면
    global predictor 사용).
  - `ChunkLengthPredictor`: `sample_count` property + 모든 strategy에서
    `_sample_count` 추적.
  - `RequestSLOState`: 새 InitVar `global_chunk_len_predictor`. 
    `on_chunk_boundary`에서 per-req + global 둘 다 update.
  - `expected_remaining_len()`: per-req sample_count < 16 이면 global,
    그 외엔 per-req (hard switch).
  - `from_config`에 `global_chunk_len_predictor` kwarg 추가.
- `vllm/vllm/sslo/config.py`: `chunk_len_predictor_history_max: int = 4096`
  knob + validation.
- `vllm/vllm/v1/core/sched/scheduler.py`: `Scheduler.__init__`이
  `_sslo_global_chunk_len_predictor` 인스턴스 생성 (EngineCore subprocess
  안, single thread → lock 불필요).
- `vllm/vllm/v1/engine/core.py`: `add_request`가 scheduler의 global
  predictor 참조를 `from_config`에 전달.
- `vllm/tests/sslo/test_slo_state.py`: 신규 3 tests (warmup 사용, 16-sample
  switchover, on_chunk_boundary가 global update).

### Verification
- `pytest tests/sslo/` 116/116 pass.
- Worst-case smoke (35B-A3B cap=128 r=128 sslo_mlp × 3 runs vs B1):
  tput 1814 → 1842 (+1.5%), viol@τ=1 0.80% → 0.60% (-25%).

### Notes (다른 시도)
- Phase A (backfill을 waiting admit **앞**으로 reorder) 시도 → trade-off
  불리. bf_kv_full=0 / tput 동등 / **TTFC 54s → 84s** (waiting 정체).
  Codex 검토 후 revert.

## 2026-05-18 (cont.) — dataset code-gen 필터 + dataset 선택

### Modified
- `exp/tools/lm_datasets.py`:
  - `_CODE_PROMPT_PATTERN` regex + `_is_code_request(prompt, response)`
    helper.
  - `load_prompts(..., exclude_code=False)` 새 kwarg.
  - 모든 3개 loader (`_load_koala`, `_load_wildchat`, `_load_lmsys`)에
    `exclude_code` 파라미터 + filter 적용.
  - WildChat/LMSYS는 `_first_user_and_assistant` helper로 첫 user prompt +
    첫 assistant response 추출 → response 코드 블럭(```) 확인.
  - Koala는 응답 없음 → prompt regex만.
- `exp/run_sslo/run_test.py`:
  - `--dataset-name` choices에 `koala|wildchat|lmsys` 명시.
  - 새 `--exclude-code` flag (기본 off).
  - `load_workload`에 `exclude_code` 전달.
- `exp/run_sslo/run_test.sh`:
  - `DATASET_NAME` env var (default `koala`).
  - `EXCLUDE_CODE=1` env → `--exclude-code` flag.

### Regex coverage 검증 (인라인 10/10 PASS)
- 정상 채팅: "What is the capital of France?", "Tell me about Python
  history", "Python vs Ruby 차이" → not flagged.
- 코드 요청: "Write a Python function", "Implement a binary search
  algorithm in Java", "Generate a JavaScript class", "Build a HTML form",
  "def foo(x):" → flagged.
- Response 기반: "How are you?" + ```python 응답 → flagged.

## 2026-05-18 (cont.) — combine dataset (seed-shuffled mix)

### Added
- `exp/tools/lm_datasets.py`:
  - `combine` / `COMBINE` aliases in `SUPPORTED_DATASETS`.
  - `_load_combine(num_prompts, exclude_code, seed)`: oversamples ~20% per
    source (koala + wildchat + lmsys), pools, shuffles with seeded RNG,
    returns top `num_prompts`. If any source fails, falls back to the
    others.
  - `load_prompts(..., seed=42)` kwarg threaded into `_load_combine`
    (ignored elsewhere).
- `exp/run_sslo/run_test.py`:
  - `--dataset-name choices` now includes `combine`.
  - New `--dataset-seed` (default 42).
  - `load_workload` passes seed through.
- `exp/run_sslo/run_test.sh`:
  - New `DATASET_SEED` env var (default 42).

### Usage
```bash
DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 \
  bash exp/run_sslo/run_test.sh sslo_mlp 128 Qwen/Qwen3.5-35B-A3B
```
