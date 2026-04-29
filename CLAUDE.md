# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Experiment Pipeline Architecture

The project measures *sentence-level output (SSLO) slack* — whether an LLM produces text fast enough for a human reader or a TTS engine to consume it chunk-by-chunk without waiting. It runs as a three-stage pipeline entirely inside Docker containers.

### Stage overview

| Stage | Script | Container | What it does |
|-------|--------|-----------|--------------|
| 1 | `exp/benchmark.py` | `sk-sslo` | Runs vLLM async inference; records request metrics and per-chunk timestamps from the streamed output |
| 2 | `exp/audio_duration.py` | `sk-sslo-omni` | Calls Qwen3-TTS on each text chunk and records audio duration |
| 3 | `exp/analyze_results.py` | `sk-sslo` | Joins chunk timelines with audio durations, computes slack for one mode, writes result rows and plot |

The launcher `exp/run_experiment.sh` drives all three stages in sequence, iterating over `MODELS` and `SLACK_MODES` arrays defined near the top of the file. There is no make/pytest build system — the `.sh` files are self-contained.

For human-reading-only runs (skips TTS), use `exp/run_experiment_read.sh` instead; it attaches both `sentence` and `paragraph` chunk collectors to the same inference pass.

### Running the full experiment (from the host)

```bash
bash exp/run_experiment.sh
```

Both containers (`sk-sslo`, `sk-sslo-omni`) must be running before executing the launcher. The launcher calls `docker exec` internally.

### Running a single pipeline stage manually

```bash
# Stage 1 — LLM inference (inside sk-sslo)
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys
  export HF_HOME=/cache HF_HUB_CACHE=/cache/hub
  python3 exp/benchmark.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --dataset-name hf --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
    --num-prompts 48 --output-dir exp/output/test/sentence/text_outputs
'
```

### Key source files

- `exp/benchmark.py` — Stage 1; defines `StreamingChunkCollector` (sentence/paragraph boundary detection from the stream receive path), request arrival pacing, and timeline record schema.
- `exp/audio_duration.py` — Stage 2; runs vLLM-Omni TTS with a per-chunk cache (`duration_cache.jsonl`) so re-runs are incremental.
- `exp/analyze_results.py` — Stage 3; reads text + audio paths, applies slack math via `add_deadline_slack_columns`, plots `slack_distribution.png`.
- `exp/common/slack_utils.py` — Shared helpers: output path constructors (`text_output_paths`, `audio_duration_paths`, `result_output_paths`), slack math (`add_deadline_slack_columns`, `build_slack_rows`, `attach_audio_timeline`), and I/O utilities.
- `exp/configs/qwen3_tts_omni_batch.yaml` — Template TTS stage config rendered at runtime into `exp/output/_runtime/qwen3_tts_omni.yaml`.

### SSLO vLLM package (`vllm/vllm/sslo/`)

`chunk_timer.py` implements an alternative in-engine chunk timing path:
- `ChunkTimingCollector` — per-output-stream collector, supports both CUMULATIVE and DELTA `RequestOutputKind`.
- `ChunkTimer` — request-level helper that wraps collectors for all output indices; attaches `chunk_timings` and `generation_start_time` to `RequestOutput` objects.
- `get_detector(spec)` — factory for `SentenceChunkDetector`, `ParagraphChunkDetector`, `TokenCountChunkDetector`, or any custom `ChunkBoundaryDetector`.

**Note:** `benchmark.py` uses its own `StreamingChunkCollector` (timestamps from the benchmark's stream receive path, not from inside vLLM), so `ChunkTimer` is available for vLLM-internal use but is not wired into the current experiment launcher.

### Output layout

```
exp/output/{model_slug}/{dataset_slug}/{slack_mode}/
  text_outputs/         ← Stage 1 writes here
  audio_durations/      ← Stage 2 writes here
  results/              ← Stage 3 writes here
```

Slugs are produced by `slugify()` in the launcher: `/` → `__`, unsafe chars → `_`.

### Slack modes

Two modes select different columns from `add_deadline_slack_columns`:
- `previous_chunk` — deadline for chunk N is set by the end of chunk N-1 plus its consumption time.
- `cumulative` — deadline is the sum of all prior chunk consumption times from decoding start.


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
