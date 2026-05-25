# Numba JIT Probe — `_compute_serve_defer_pair`

## Fields that matter for the arithmetic

From `RequestSLOState`:
- `expected_remaining_len()` → derived from `_chunk_len_predictor.value` and `current_chunk_generated_len` (scalar float)
- `time_to_deadline(now)` → `next_deadline_ts - now` (scalar float, None if deadline unset)

From `SsloConfig` (via `_compute_serve_defer_pair`):
- `max_num_running_reqs` → denominator for scale factor `n_admitted / max(1, denom_cap)`

Hot-path arithmetic per request (all in `multi_level_pressure_components`):
```
refill      = remaining * tpot_s
serve       = refill / max(ttd, ε)          # ε = 1e-9
defer_buf   = ttd - epoch_s
defer       = refill / max(defer_buf, ε)   # inf if defer_buf ≤ 0
scaled      = pressure * (n_admitted / denom_cap)
```

`PressureComponents` fields used downstream: `serve_pressure` (scaled), `defer_pressure` (scaled). The rest (`remaining_tokens`, `buffer_slack_s`, etc.) are logging only.

## Is `np.percentile` on the hot path?

No. `np.percentile` lives in `ChunkLengthPredictor._update_percentiles()`, which is called only inside `on_chunk_boundary()` — a hook fired when a sentence boundary is detected, not every scheduler step. `multi_level_pressure_components` calls only `expected_remaining_len()`, which reads already-computed `pred.value` / `pred.value_mid` / `pred.value_high` scalars with no percentile recomputation. The hot path is pure scalar arithmetic.

## Benchmark results (10k iters, container: sk-sslo-vllm, numba 0.65.0)

| N   | Python (µs) | Numba (µs) | Speedup |
|-----|-------------|------------|---------|
| 8   | 4.43        | 0.87       | 5.1x    |
| 32  | 17.15       | 0.89       | 19.3x   |
| 64  | 33.59       | 0.92       | 36.7x   |
| 128 | 59.76       | 0.93       | 64.1x   |
| 256 | 124.29      | 1.01       | 123.0x  |

At N=64: **36.7x speedup** (33.6 µs → 0.92 µs). Numba time is nearly flat across N, indicating the JIT loop cost is dominated by fixed overhead (~0.9 µs), not per-element work.

## Is the function hot enough to bother?

From `scheduler_stats.jsonl` (final_qwen3_5_27b / sentence / seqs_48 / rate_128 / sslo_mlp / run_1):
- 11231 total steps, **7519 critical steps (67%)**
- Median admitted = 48 requests per critical step
- `_compute_serve_defer_pair` is called at least **2×/critical step** = **≥15,038 calls** per run
- At N≈48 Python baseline ≈ 25 µs/call → **~376 ms total** per run in just this function

15k calls × 25 µs = 376 ms Python; with numba ≈ 15k × 0.9 µs = 14 ms. Savings: ~360 ms per run. For a sweep run with dozens of seeds/rates that adds up, but it's not a latency-per-step blocker (scheduler step is ~10s per batch decode, this is µs-scale overhead).

## Verdict

**Speedup is real but integration complexity is non-trivial.** Worth integrating if the sweep is run frequently at scale.

Integration shape:
1. Scheduler builds two float64 arrays per step (shape `[n_admitted]`) — `remaining` and `ttd` — before the `_compute_serve_defer_pair` call. These are already computed one at a time per-request; batch them instead.
2. Call `nb_pressure_pair(remaining, ttd, tpot_s, epoch_s, n_admitted, denom_cap, serve_buf, defer_buf)` where `serve_buf`/`defer_buf` are step-level pre-allocated buffers (avoids alloc cost).
3. Re-materialise the `dict[str, float]` result from the output arrays using a zip over `request_ids`.

The bottleneck for integration is step 1 (refactoring `expected_remaining_len` and `time_to_deadline` to be vectorisable) and step 3 (dict materialisation still incurs Python overhead, but that's O(N) string lookups, not arithmetic). If the caller only needs the partition (serve ≥ threshold), step 3 can be skipped and the threshold check done in the kernel — further reducing overhead.

**Non-trivial parts:** `expected_remaining_len` contains branching over `value_mid`/`value_high` escalation tiers; these must be pre-extracted to arrays before the kernel call.
