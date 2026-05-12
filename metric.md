# CP-SLO Metrics 정의

> 코드 내부 prefix는 `sslo_*`로 남기고, paper / figure / table export는 **CP-SLO** 용어를 쓴다. 이 문서는 canonical (export) 이름 기준으로 정의를 정리한다. Internal legacy 이름은 코드 옆에 alias로 dual-write된다 (`exp/run_sslo/analysis/cpslo_names.py`).

---

## 1. 시간/타임스탬프 기본량 (per request)

| 기호 | export name | internal | 의미 |
|---|---|---|---|
| $t_\text{first}$ | `first_token_ts` | `decoding_start_ts` | 첫 토큰이 생성된 wall-clock |
| $t_\text{consume\_start}$ | `consume_start_ts` | `consume_start_ts` | chunk 0이 끝난 wall-clock = $t_\text{end}(0)$. CP-SLO continuity가 여기서 시작 |
| $t_\text{queued}$ | `queued_ts` | (engine `metrics.queued_ts`) | request가 engine 큐에 들어온 wall-clock |
| $t_\text{admitted}$ | `admitted_ts` | `admitted_ts` | request가 처음으로 `running` 상태에 들어간 wall-clock |
| $t_\text{end}(k)$ | `chunk_generation_end_ts` | `gen_finish_ts` | chunk $k$의 마지막 토큰 wall-clock |
| $\tau(k)$ | `chunk_generation_start_ts` | `start_time_ts` | chunk $k$가 시작된 wall-clock |
| $d(k)$ | `chunk_deadline_ts` | `deadline_ts` | consumer가 chunk $k$를 요구하는 시각 |
| $c(k)$ | `chunk_consume_time_s` | `chunk_consume_time_s` | chunk $k$의 소비 시간. 기본 = $\text{word\_count}(k) \cdot \text{seconds\_per\_word}$ |

### Chunk 시작 시각

$$
\tau(k) = \begin{cases}
t_\text{first} & (k = 0) \\
t_\text{end}(k{-}1) & (k \ge 1)
\end{cases}
$$

### Stall-aware deadline recurrence

$$
\boxed{\;d(0) = t_\text{consume\_start} = t_\text{end}(0)\;}
$$

$$
d(k{+}1) = \max\bigl(d(k),\, t_\text{end}(k)\bigr) + c(k)
$$

- **$d(0) = t_\text{end}(0)$** (CP-SLO 새 contract). 첫 chunk 생성 지연(TTFC)은 CP-SLO stall에 포함되지 않음 — TTFC는 별도 latency guardrail.
- 조기 도착은 buffer로 쌓이지 않고, 지각 도착은 다음 deadline을 정확히 그 지각분만큼 미룸 (carryover 누적 차단).

### Latency guardrails (CP-SLO stall이 아님)

| 컬럼 | 정의 |
|---|---|
| `TTFT` | $t_\text{first} - t_\text{queued}$ — first-token latency |
| `TTFC` | $t_\text{consume\_start} - t_\text{queued}$ — time to first consumable chunk |
| `TPOT` | $(t_\text{last} - t_\text{first}) / (\text{num\_gen} - 1)$ |
| `queue_stall_s` | $t_\text{admitted} - t_\text{queued}$ — admission wait |

---

## 2. Chunk-level metrics (`ChunkRecord`)

`slo_chunk_records[]`로 chunks.jsonl에 emit. 양쪽 (legacy + canonical) 키 dual-write.

| Export 이름 | 정의 / 수식 |
|---|---|
| `chunk_idx` | 0-based index |
| `token_start_idx`, `token_end_idx` | 이번 chunk의 token range (half-open) |
| `cumulative_tokens_at_end` | = `token_end_idx` |
| `chunk_generation_start_ts` | $\tau(k)$ |
| `chunk_generation_end_ts` | $t_\text{end}(k)$ |
| `chunk_deadline_ts` | $d(k)$ |
| `chunk_consume_time_s` | $c(k)$ |
| `demand_window_start_ts` | $= d(k)$ |
| `demand_window_end_ts` | $= d(k) + c(k)$ |
| `chunk_deadline_margin_s` (legacy `slack_s`) | $d(k) - t_\text{end}(k)$. 양수면 여유, 음수면 지각 |
| `stall_start_ts` | $d(k)$ if late, else `None` |
| `stall_end_ts` | $t_\text{end}(k)$ if late, else `None` |
| `stall_duration_s` | $\max(0,\, t_\text{end}(k) - d(k))$ |
| `pending_time_s` | chunk window 동안 pending 풀에 있던 wall-time |
| `word_count`, `num_token` | chunk 텍스트 단어 수 / 토큰 수 |
| `num_iters`, `num_running_iters`, `num_pending_iters` | chunk window 내 scheduler step counters |
| `expected_chunk_len`, `expected_chunk_len_high` | predictor 값 (p90 strategy면 p99 companion도 함께) |
| `predictor_source` | `"ema"` / `"p90"` / `"p99"` / `"past-future"` |

### Chunk 0 특수성

- $d(0) = t_\text{end}(0)$ → $\text{chunk\_deadline\_margin\_s}(0) = 0$ **구조적으로** (special-case branch 없음).
- `stall_*` 필드는 모두 `None` / `0.0`.

---

## 3. Pressure components

`PressureComponents` dataclass — `RequestSLOState.pressure_components(now, tpot)`로 산출.

```text
remaining_tokens
  = max(1, expected_chunk_len - current_chunk_generated_len)

estimated_refill_time_s
  = remaining_tokens * estimated_step_time_s

buffer_slack_s
  = time_to_deadline = chunk_deadline_ts(k) - now

refill_slack_s
  = buffer_slack_s - estimated_refill_time_s

depletion_pressure
  = estimated_refill_time_s / buffer_slack_s
  (= +inf when buffer_slack_s <= 0)
```

### Critical 등가식

$$
\text{depletion\_pressure} \ge 1 \;\Longleftrightarrow\; \text{refill\_slack\_s} \le 0
$$

### Missing 처리 (log vs policy split)

| 상황 | `pressure_available` | `pressure_missing_reason` | `depletion_pressure` (log) | Policy fallback (`_policy_score_with_fallback`) |
|---|---|---|---|---|
| `phase == PREFILL` | False | `"prefill"` | `None` | 1.0 |
| `phase == WARMUP` | False | `"warmup"` | `None` | 1.0 |
| `tpot is None` | False | `"no_tpot"` | `None` | 1.0 |
| predictor.value None | False | `"no_predictor"` | `None` | 1.0 |
| Deadline 지남 ($ttd \le 0$) | True | `None` | `inf` | `inf` |
| 정상 | True | `None` | finite | depletion_pressure |

**Policy는 1.0 fallback을 유지** (cold-start critical guard 보존). **Log는 raw `None`을 보존** (missing vs critical-boundary 구분).

---

## 4. Per-step decision log

`decision_log_mode` config로 emit 제어. 기본 `tier_changes` (transition + heartbeat 200 step).

`decisions.jsonl` 파일 (옆에 `scheduler_stats.jsonl`), 한 row per `(step_idx, request_id)`:

| 필드 | 의미 |
|---|---|
| `ts`, `step_idx`, `request_id`, `phase`, `tier` | 식별 |
| `was_candidate`, `was_selected` | admitted 풀에 있었는지 / 이번 step에 running 선택됐는지 |
| `priority_rank` | sort 순위 (Phase 6 후속) |
| `buffer_slack_s`, `estimated_refill_time_s`, `refill_slack_s`, `depletion_pressure` | PressureComponents raw |
| `pressure_available`, `pressure_missing_reason`, `fallback_score_used` | Missing 진단 |
| `policy_score` | 정책이 본 점수 (1.0 fallback 적용됨) |
| `estimated_step_time_s`, `tpot_lookup_source` | EMA context |
| `batch_size`, `num_prefills` | step 구성 |
| `cur_max_num_requests` | 이 step의 cap |
| `admission_reason`, `backfill_reason`, `preemption_reason` | Phase 6 placeholder (현재 `None`) |

모드 4종:
- `off` — 미emit
- `step` — 모든 step × 모든 admitted (~1M rows / 5min, M5 deep dive용)
- `tier_changes` (기본) — tier 변경 + admit/preempt + heartbeat
- `admit_only` — admission/preemption 이벤트만 (M6 minimal)

---

## 5. Phase / tier

`Phase` enum:
- `PREFILL` — `decoding_start_ts is None`
- `WARMUP` — `decoding_start_ts != None ∧ chunks_completed < num_warmup_chunks`
- `MEASURED` — `chunks_completed >= num_warmup_chunks`

Tier (`_classify_tier`):
- 0 (critical) — `phase == MEASURED ∧ policy_score >= critical_threshold` (1.0)
- 1 (warmup) — `phase == WARMUP`
- 2 (measured-OK) — 그 외

Cold-start guard: `has_critical` 판정은 `base_tpot is not None`일 때만 발동 → tpot 없는 cold-start에서 normalize된 1.0이 critical로 잘못 분류되는 것을 차단.

---

## 6. Request-level metrics (`SsloRequestStats`)

| Export 이름 | 정의 |
|---|---|
| `total_pending_time_s` | 요청 생애 동안 pending 풀에 있던 누적 wall-time |
| `num_pending_intervals` | pending 풀 진입 횟수 |
| `chunks_completed` | 완료 chunk 수 |
| `final_chunk_expected_len` | 종료 시점 predictor `value` |
| `total_step_count`, `prefill_step_count` | scheduler step counters |
| `admitted_ts`, `consume_start_ts` | timing |
| `terminal_outcome` | `"in_progress"` / `"completed"` (abort/timeout split은 follow-up) |

run_test.py가 derive하는 per-request 컬럼:

| 컬럼 | 정의 |
|---|---|
| `ttft`, `TTFT` | first_token_latency |
| `ttfc`, `TTFC` | `consume_start_ts - queued_ts` |
| `tpot` | 평균 token 간격 |
| `queue_stall_s` | `admitted_ts - queued_ts` (legacy `queue_stall`은 `scheduled_ts - queued_ts`) |
| `num_chunks` | `len(slo_chunk_records)` |
| `num_pending_intervals` | `SsloRequestStats.num_pending_intervals` (legacy alias `num_pending_iters_per_request`) |
| `num_preemptions` | engine metrics |
| `reference_output_tokens` | observed `num_generation_tokens` (F2 option C — post-hoc) |
| `request_class` | bucket: `[1,100)="xs"` / `[100,500)="s"` / `[500,2000)="m"` / `[2000,∞)="l"` / else `"unknown"` |

---

## 7. CP-SLO violation

### Per-chunk stall

$$
\text{stall\_duration\_s}(k) = \max\bigl(0,\, t_\text{end}(k) - d(k)\bigr)
$$

`stall_start_ts(k) = d(k)`, `stall_end_ts(k) = t_\text{end}(k)` (둘 다 stall 있을 때만; 아니면 `None`).

### Contiguous interval merge (post-hoc)

`progress_metrics.compute_stall_intervals(chunks)` — chunk N의 `stall_end_ts == ` chunk N+1의 `stall_start_ts`이면 합쳐서 한 interval. 그렇지 않으면 별개. 일반적으로 `chunk_consume_time_s > 0`이면 consumer가 중간에 chunk N을 소비하므로 stall이 끊어져 별도 intervals.

$$
\text{stall\_intervals} = [(s_1, e_1), (s_2, e_2), \dots]
$$

$$
\boxed{\;\text{max\_stall\_interval\_s} = \max_i (e_i - s_i)\;}
$$

### Violation

$$
\text{CP\_SLO\_violation}_i(\tau) = \mathbb{1}[\text{max\_stall\_interval\_s}_i > \tau]
$$

Default τ sweep: `{0.5, 1.0, 2.0, 5.0}` 초. Headline: $\tau = 1.0\text{s}$.

---

## 8. Run-level metrics + validity

### `run_meta.json` sidecar (run당 1개)

| 필드 | 의미 |
|---|---|
| `run_id` | `{policy}_{seqs}_{rate}_{seed}_{ts}` unique key |
| `policy` | `args.run_kind` (baseline / sslo / sslo_adaptive …) |
| `variant` | `{run_kind}/abatch={on/off}` for M6 ablation |
| `seed`, `trace_id`, `workload_id` | reproducibility |
| `N` | request 총 수 |
| `M` | `max_num_seqs` |
| `measurement_start_ts`, `measurement_end_ts` | run window |
| `gpu_memory_peak_bytes` | `torch.cuda.max_memory_allocated()` |
| `num_preemptions_total` | request num_preemptions 합 |
| `num_requests_completed`, `num_requests_total` | completion gate |

### Time-weighted handling-users

$$
\overline{U} = \mathbb{E}_t[|running| + |pending|] = \frac{\sum_i (|running_i|+|pending_i|) \cdot \Delta t_i}{\sum_i \Delta t_i}
$$

M3 frontier x축. `progress_metrics.handling_users_stats(...).time_avg`.

### No-harm thresholds (pre-declared)

| Threshold | Default |
|---|---|
| `DROP_RATE_MAX` | 0.01 |
| `TIMEOUT_RATE_MAX` | 0.01 |
| `THROUGHPUT_REGRESSION_MAX` | 0.10 (i.e. ≥ 90% of baseline tokens/s) |
| `QUEUE_STALL_P99_MAX_S` | 5.0 |
| `PENDING_TIME_P99_MAX_S` | 5.0 |

`ValidityReport(validity_pass, no_harm_pass, invalid_reason)`. Fail한 run은 frontier에서 제외되지만 `validity_checks.csv`에 `invalid_reason`과 함께 남음.

### Capacity ratio

$$
\text{capacity\_ratio}(\tau) = \frac{U_\text{sslo}(\tau)}{U_\text{baseline}(\tau)}
$$

Paired baseline ↔ sslo run (`_consolidate_mode_outputs.consolidate_validity_csv`)에서 계산.

---

## 핵심 관계식 한 줄

$$
\text{depletion\_pressure} = \frac{(\text{predicted\_chunk\_len} - \text{generated\_so\_far}) \cdot \overline{t_\text{step}}}{d(k) - t_\text{now}}
$$

`refill_slack_s = (분모) - (분자)`. 1.0 부근이 admission / preemption 임계.

CP-SLO violation은 chunk-level stall이 아니라 **contiguous interval의 최대 duration**이 τ를 넘는지로 판정.
