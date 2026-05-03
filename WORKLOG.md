# Work Log

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
