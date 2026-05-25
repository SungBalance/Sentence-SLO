# SSLO Multi-Level Pressure 스케줄링 알고리즘

## 목적

`multi_level_pressure`는 문장 단위 SLO를 지키기 위해, 현재 admit된 요청들을 매 scheduler step마다 `running`과 `sslo_pending`으로 재배치한다.

- `running`: 이번 step에서 실제 decoding 후보가 되는 요청
- `sslo_pending`: 이미 admit되어 KV cache를 가진 상태지만, 잠시 decoding에서 제외되는 요청
- `waiting`: 아직 admit되지 않은 새 요청 큐

핵심 아이디어는 다음과 같다.

> 마감 위험이 큰 요청은 `running`에 유지하고, 안전하게 미룰 수 있는 요청은 `sslo_pending`에 주차하며, 남는 여유만큼만 `waiting` 요청을 새로 admit한다.

## 입력

- `R = running ∪ sslo_pending`: 현재 admit된 요청 집합
- `W`: waiting queue
- `M`: 기본 최대 running 요청 수, `max_num_running_reqs`
- `τ`: 최근 scheduler step wall-time EMA
- `θ_c`: critical serve threshold, 기본값 `1.0`
- `θ_d`: defer constraint, 기본값 `1.0`
- `B_free`: 현재 사용 가능한 KV block 수
- `K`: 새 요청 1개 admit에 필요한 예상 KV block 수

## 요청별 압력 계산

각 measured 요청 `r`에 대해:

```text
remaining_r = 현재 chunk에서 남은 예상 토큰 수
ttd_r       = deadline_r - now
refill_r    = remaining_r × τ
raw_serve_r = refill_r / max(ttd_r, ε)
```

현재 scheduler 구현은 여기에 contention scaling을 적용한다.

```text
scale   = |R| / cap
serve_r = raw_serve_r × scale
defer_r = serve_r
```

주의: `RequestSLOState`에는 원래 의미의 `defer = refill / max(ttd - epoch, ε)` 계산이 존재하지만, 현재 scheduler placement 경로의 `_compute_serve_defer_pair()`에서는 `serve`와 `defer`가 같은 contention-scaled pressure로 계산된다.

## 메인 알고리즘

```text
Algorithm MultiLevelPressureSchedule(now)

1. R ← running ∪ sslo_pending
2. τ ← 현재 R에 맞는 step wall-time EMA 조회

3. if τ is None:
       현재 running / sslo_pending 배치를 유지한다.
       waiting_budget ← min(|W|, M - |running|)
       return

4. cap = M 기준으로 모든 r ∈ R에 대해 serve_r, defer_r 계산

5. critical ← 다음 조건을 만족하는 요청이 하나라도 있는지 검사:
       r.phase = MEASURED and serve_r ≥ θ_c

6. if critical:
       if adaptive_batching enabled:
            cap, running, sslo_pending ← PickAdaptiveCap(R)
       else:
            cap ← M
            running, sslo_pending ← Partition(R, serve, defer, cap)

       if allow_admit_critical = false:
            waiting_budget ← 0
       else:
            slack ← cap - |running|
            kv_cap ← floor(B_free / K)
            waiting_budget ← min(|W|, slack, kv_cap)

       return

7. else:
       forced_running ← ∅
       warmup_running ← ∅
       eligible_pending ← ∅

       for each r ∈ R:
            if r.phase ≠ MEASURED:
                 warmup_running.add(r)
            else if defer_r is None or defer_r = ∞ or defer_r ≥ θ_d:
                 forced_running.add(r)
            else:
                 eligible_pending.add(r)

       running ← forced_running ∪ warmup_running
       sslo_pending ← eligible_pending

       if |running| > M:
            초과분 warmup 요청을 sslo_pending으로 이동

       load ← sum serve_r over measured running
             + sum defer_r over measured pending
             + count unmeasured running/pending

       admission_capacity ← floor(M - load)
       slack ← M - |running|
       kv_cap ← floor(B_free / K)

       waiting_budget ← min(|W|, admission_capacity, slack, kv_cap)

       defer pressure snapshot을 저장한다.
       return
```

## Partition 서브루틴

```text
Partition(R, serve, defer, cap)

1. forced ← MEASURED 요청 중 defer ≥ θ_d
2. high   ← MEASURED 요청 중 serve ≥ θ_c
3. warmup ← PREFILL / WARMUP / unmeasured 요청
4. low    ← 나머지 MEASURED 요청

5. high를 serve 내림차순으로 정렬
6. low를 serve 내림차순으로 정렬

7. ordered ← forced + high + warmup + low
8. running ← ordered의 앞 cap개 요청
9. pending ← 나머지 요청
```

## Adaptive Cap 선택

```text
PickAdaptiveCap(R)

1. candidates ← {M} ∪ profiling된 CUDA graph batch size 중 M보다 작은 값들

2. for each candidate cap n:
       τ_n ← batch size n의 wall EMA
       scale ← |R| / n
       serve_n, defer_n 계산
       running_n, pending_n ← Partition(R, serve_n, defer_n, n)

       feasible 조건:
           pending_n 안에 defer ≥ θ_d인 MEASURED 요청이 없어야 한다.

       objective_n ← Σ serve over running_n + Σ defer over pending_n

3. feasible candidate가 있으면:
       objective_n이 최소인 n 선택

4. feasible candidate가 없으면:
       다음 값을 최소화하는 n 선택:
           (# pending measured defer violators, objective_n)

5. 선택된 n과 partition 반환
```

## Policy 이후 실제 스케줄링

`multi_level_pressure` policy는 `running`, `sslo_pending`, `waiting_budget`을 정할 뿐이고, 실제 token scheduling은 `schedule_sslo()` 후반에서 마무리된다.

```text
1. Phase 0:
       waiting_budget - 1개까지 waiting에서 running으로 admit

2. Backfill:
       남은 token budget으로 sslo_pending 요청을 running으로 복귀
       정렬 기준은 defer pressure 내림차순

3. Phase 1:
       마지막으로 waiting에서 1개를 추가 admit할 수 있으면 admit
```

이 2-phase 구조는 pending backfill이 실제로 KV와 token budget을 사용할 기회를 먼저 주고, 그 뒤에 남는 자원이 있을 때 waiting 요청을 하나 더 admit하기 위한 장치다.

## 요약

`multi_level_pressure`는 다음 원칙으로 동작한다.

1. 아직 step latency EMA가 없으면 SSLO 판단을 보류하고 현재 배치를 유지한다.
2. measured 요청 중 `serve ≥ 1.0`인 요청이 있으면 critical mode로 들어간다.
3. critical mode에서는 urgent 요청을 우선 running에 배치하고, 기본적으로 waiting admit을 막는다.
4. non-critical mode에서는 `defer ≥ 1.0`인 요청과 warmup 요청을 running에 둔다.
5. 안전하게 미룰 수 있는 measured 요청은 `sslo_pending`에 둔다.
6. 남은 capacity, pressure load, KV 여유를 기준으로 waiting admit budget을 계산한다.
7. 이후 scheduler가 waiting admit, pending backfill, 추가 admit을 수행해 실제 실행 배치를 완성한다.
