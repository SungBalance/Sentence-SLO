# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Project Architecture

The project measures and enforces *sentence-level output SLOs (SSLO)*: each LLM request is modeled as a stream of sentence/paragraph chunks ("consumable units"), each chunk gets a wall-clock deadline derived from consumer speed (human reading rate or a measured TTS profile), and a modified vLLM scheduler (the **ProgressServe** policy) spends the resulting slack on extra admissions to raise GPU occupancy without breaking per-chunk deadlines.

### SSLO vLLM package (`vllm/vllm/sslo/`)

- `config.py` — `SsloConfig` dataclass: `method ∈ {"baseline","progress_serve"}` (`baseline` records metrics but skips SSLO placement), chunk unit (`sentence`/`paragraph`), chunk-length predictor strategy (`ema`/`p90`/`p99`), consume mode (`read` fixed seconds-per-word / `tts` profile CSV), KV-aware admission cap (`kv_blocks_per_new_admit`), `adaptive_batching`, KV offload tier (`kv_offload` + `kv_onload_lead_iters` / `kv_offload_risk_eps` / `kv_offload_share_includes_parked` / `kv_offload_min_residency_steps`), decision-log knobs.
- `progress_serve.py` — pure-function ProgressServe math (no engine imports, unit-testable): per-request run/defer violation risks from posterior length-tail probabilities, admission search for the largest `k*` with expected violations `E_viol < 1` (`schedule_step`), and deadline-driven decode-batch shrink (`pick_adaptive_batch`).
- `slo_state.py` — `RequestSLOState` lifecycle (`PREFILL → WARMUP → MEASURED`), deadline recurrence, `ChunkSeparator` (streaming sentence/paragraph boundary detection with `min_chunk_tokens` merging), `ChunkLengthPredictor` (per-request + shared-global hybrid, empirical tail for ProgressServe), `ChunkConsumeEstimator`/`TtsProfileConsumeEstimator`, `ChunkRecord` diagnostics.

Integration points outside `vllm/vllm/sslo/` are marked with `# SSLO` line comments (`grep -rn "# SSLO" vllm/vllm/`); the densest is `vllm/v1/core/sched/scheduler.py` (`schedule_sslo()` replaces vanilla `schedule()` when SSLO is enabled). Tests live in `vllm/tests/sslo/`.

### Experiment map (`exp/`)

Each experiment folder has its own `README.md` — consult it before running. There is no make/pytest build system; the `.sh` launchers are self-contained with options as constants at the top.

| Folder | Purpose |
|--------|---------|
| `run_sslo/` | **Main end-to-end sweep**: `baseline` vs `progress_serve` vs `progress_serve_adaptive` vs `progress_serve_offload` (KV offload tier) over `max_num_seqs × request_rate × chunk_unit × consume_mode`. `run_test.py` (single-config runner; sweeps rates in one engine via `reset_sslo_state`), `run_test.sh` (env wrapper), `run_sweep.sh` (multi-GPU orchestration + `summary.csv`), `analyze.py` (per-cell `summary.json`), `analysis/` (aggregators, diagnostics, plots), `profiles/` (canonical TTS consume profile CSV). |
| `measure_batch/` | Baseline vLLM batch-size knee (where throughput gain per doubling drops below 10%). |
| `measure_internal_slack/` | Cumulative slack distribution vs human reading speed (model size × chunk index × sentence/paragraph). |
| `measure_KV_overhead/` | KV block GPU↔CPU offload/onload transfer time and bandwidth profile. |
| `measure_tts_duration/` | word_count → TTS audio duration regression; source of the TTS consume profile. |
| `chunk_length_study/` | Chunk-length vs request-length variability; oracle vs online posterior predictor replay. |
| `plots/` | Paper Figures 5/6/7 (`figures/` = script tree + data contract README; `figures_adaptive/`, `figures_progress_serve/` = per-policy renders). |
| `tools/` | Shared dataset loaders (wildchat/lmsys, code-request classifier) and dataset cache. |

### Running the main sweep

Inside the `sk-sslo` container, from `/workspace/mlsys`:

```bash
# Single mode / single config
OUTPUT_DIR=exp/run_sslo/output/test \
bash exp/run_sslo/run_test.sh progress_serve 64 Qwen/Qwen3-8B

# Full sweep, 3 repeats, 4-GPU parallel
PARALLEL=4 bash exp/run_sslo/run_sweep.sh 3
```

Consume mode is selected via env: `CONSUME_MODE=read` (default) or `CONSUME_MODE=tts` (requires `TTS_PROFILE_PATH` and `TTS_MODEL`).

### Output layout

```
exp/run_sslo/output_sweep/{chunk_unit}/seqs_{N}/rate_{r}/run_{i}/
  requests.jsonl          ← per-request latency rows (mode column prepended)
  chunks.jsonl            ← per-chunk records (deadline, slack, predictor)
  scheduler_stats.jsonl   ← per-step scheduler stats (sslo modes)
  summary.json            ← analyze.py aggregate for the cell
  run_status.json
```


# Work Guideline

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
