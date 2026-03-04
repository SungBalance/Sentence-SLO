# Sentence-SLO Project Content

Last Updated: 2026-03-04 (UTC)

## Project Card
- Project Name: Sentence-SLO
- Core Objective: Maximize serving goodput while satisfying sentence-level jitter SLOs for interactive TTS-style workloads.
- Primary Method: Sentence-EDF scheduling with quantile next-sentence length prediction and selective KV cache offloading.
- Current Scope:
- Model: Qwen3.5-14B-base (single-model phase first)
- Serving Engine Baseline: vLLM default scheduler (latest stable pinned to v0.16.0)
- Hardware: 4x RTX PRO 6000 (96GB each), no NVLink
- TTS Profiling Model: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

## Step-by-Step Decision Log
- Step 1:
- Problem framing fixed: token/sequence latency metrics miss sentence-gap jitter that dominates interactive UX.
- Cognitive Slack Time introduced as schedulable slack during sentence consumption.
- Step 2:
- System direction fixed to Option 2: Predictor-driven Sentence-EDF + selective KV offloading.
- Step 3:
- SLO evaluation fixed to dual reporting: strict delta=0 and practical delta>0.
- Step 4:
- Baseline fixed: vLLM default scheduler.
- Sentence-EDF(no predictor/no offload) moved to ablation (not baseline).
- Step 5:
- Model scope fixed to Qwen3.5-14B first.
- Hardware constraints fixed: no NVLink, therefore offload overhead analysis is mandatory.
- Step 6:
- Next-sentence predictor design fixed to quantile classification (not regression).
- Step 7:
- Sentence consume duration source fixed to real TTS profiling (not synthetic only).
- TTS model fixed: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice.
- Step 8:
- Checkpoint fixed: Qwen3.5-14B-base.
- English abbreviation exception list fixed at current v1 set (`Mr.`, `Mrs.`, `Dr.`, `e.g.`, `i.e.`).
- Step 9:
- TTS duration profiling policy fixed: use word-count-proportional profiling model.
- Fallback rate target tightened from <=0.5% to 0% for headline runs.
- Predictor bin policy fixed: remaining-to-sentence-end token bins at {8, 16, 32, 64, 128, 256, 512, >512}.
- Language tags fixed to two classes: `en`, `ko`.
- Step 10:
- TTS duration estimator fixed to piecewise bucket means (not strict linear).
- `word_count` definition fixed: whitespace-delimited token count after normalization.

## Facts vs Assumptions vs Speculation vs Open Questions
### Facts
- Interactive workloads can be sensitive to inter-sentence gaps.
- No NVLink can increase KV offload/reload transfer cost.
- TTS profiling code is currently missing and must be implemented before main experiment runs.

### Assumptions (to verify experimentally)
- Sentence-level scheduling can improve goodput at equal SLO violation rate.
- Quantile classification can reduce deadline misses versus naive estimates.

### Speculation
- Offloading benefit may be limited under PCIe-only topology at low memory pressure.

### Open Questions
- Exact sentence boundary and normalization policy for profiling and runtime consistency.
- Fallback policy quality when TTS duration lookup fails.
- Scope of zero-fallback policy (headline-only vs all reported tables).

## Experiment Plan Snapshot
- E1 (locked candidate): Compare full method vs vLLM default baseline under SLO grid.
- SLO grid: delta_ms in {0, 50, 100}, eps_target in {0.01, 0.05}.
- Repeats: >=5 seeds per config with paired trace replay and bootstrap CIs.
- Predictor bins: remaining tokens to sentence end in {<=8, 9-16, 17-32, 33-64, 65-128, 129-256, 257-512, >512}.

## Deterministic Sentence Segmentation Rules (v1)
- Rule S1: Normalize whitespace to single spaces and trim both ends.
- Rule S2: Split on terminal punctuation boundaries: `.`, `?`, `!`, `…`, or newline.
- Rule S3: Do not split decimal numbers (example: `3.14`) or known English abbreviations (fixed set: `Mr.`, `Mrs.`, `Dr.`, `e.g.`, `i.e.`).
- Rule S4: Keep trailing closing quotes/brackets attached to the sentence they close.
- Rule S5: Merge extremely short fragments (length < 6 chars, punctuation-only, or emoji-only) into the previous sentence.
- Rule S6: Apply the same segmentation function to both profiling-time text and serving-time text to avoid alignment drift.

## Profiling-Time vs Runtime Segmentation (clarification)
- Profiling-time segmentation:
- Offline step on reference text to create lookup keys and `consume_duration_ms` in `tts_profile`.
- Usually runs on normalized full text with deterministic batching.
- Runtime segmentation:
- Online step on generated stream to determine sentence boundaries and scheduler deadlines.
- Operates on partial/live outputs where spacing, punctuation timing, or normalization can differ.
- Why mismatch happens:
- Same content can produce different sentence keys if normalization or boundary handling differs even slightly.
- This is not token-vs-word granularity; both are sentence-level, but the boundary and normalization decisions can diverge.

## Fallback Duration Definition
- Fallback duration is used only when a sentence has no reliable profiled TTS duration due to mismatch or profiling miss.
- Headline policy (current): fallback disabled; target fallback_rate = 0%.
- Headline means primary claim numbers shown in Abstract, Introduction, and main result figures/tables (for example SLO-goodput frontier and main comparison table).
- Diagnostic means secondary analysis tables/appendix used for debugging or failure analysis.
- Trigger conditions (diagnostic only):
- F1: sentence key not found in `tts_profile`.
- F2: profiled audio invalid or duration <= 0.
- F3: segmentation mismatch detected for the request.
- Run validity contract:
- Any fallback event marks the run invalid for primary tables/figures.
- `duration_source` must be logged and expected to be always `profiled` in valid headline runs.

## TTS Duration Profiling Policy (v2)
- Goal: derive sentence `consume_duration_ms` proportional to word count, conditioned by language tag.
- Language tags: fixed to `en` and `ko`.
- Estimation unit: sentence-level duration with auxiliary feature `word_count`.
- `word_count` definition: count whitespace-delimited tokens on normalized sentence text.
- Recommended stored fields in `tts_profile`:
- `request_id, sentence_idx, sentence_text_norm, language_tag, word_count, consume_duration_ms, duration_source, profiling_version`.
- Modeling note:
- Use piecewise word-count buckets with per-bucket mean duration per language.
- Suggested buckets: `{1-3, 4-6, 7-10, 11-15, 16-24, 25-40, 41+}`.
- Keep mapping deterministic and versioned (`profiling_version`) so runtime lookup is reproducible.

## Next Actions
- Build P0 profiling pipeline and emit `tts_profile.parquet`.
- Freeze E1 run matrix and logging schema.
- Run baseline + ablations + full method on Qwen3.5-14B.
