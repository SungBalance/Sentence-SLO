# `Scheduler.schedule_sslo()` — 스케줄링 규칙

## 매 스텝마다 유지되는 상태

- `self.running` — 실제로 디코딩 중인 요청들
- `self.sslo_pending` — 입원은 되어 있지만(KV 유지) 더 급한 요청에 슬랙 예산을 양보하기 위해 일시 대기시킨 요청들
- `self.sslo_consecutive_pending[req_id]` — 한 요청이 연속으로 pending 상태에 머문 스텝 수 (기아 방지 카운터)

## 우선순위 키

`_sslo_score_key(req) = req.slo_state.cumulative_slack` — **값이 작을수록 더 급함**. `slo_state`가 없으면 `inf` (가장 낮은 우선순위).

`cumulative_slack`(`slo_state.py`) = `decoding_start + Σ consume_time[0..i-1] − chunk[i].end_time`. 음수면 deadline을 넘긴 상태(overdue).

## 1단계 — `running ∪ sslo_pending` 재분배 (`scheduler.py:978-1043`)

슬랙 오름차순(급한 순)으로 정렬한 뒤, 각 요청의 pending 가능 여부를 결정.

**스케줄러 측 가드 3가지** — 슬랙과 무관하게 `eligible=False`로 강제(즉, running 유지):
1. `consec >= max_consecutive_pending` — 기아 방지 한도 도달, 무조건 running으로
2. `slo_state is None` — SSLO 데이터 없음, running으로
3. `len(self.waiting) == 0` — 대기 중인 요청이 없으면 demote할 이유 없음

가드를 통과하면 요청별 판정:
- 직전 스텝에 pending이었던 경우 → `eligible = not should_exit_pending(now, prev_pending_count)`
- 직전 스텝에 running이었던 경우 → `eligible = should_enter_pending(now, prev_pending_count)`

요청별 enter/exit 로직 (`slo_state.py:423-473`):
- **Pending 진입 조건**: 샘플 수 ≥ `pending_warmup_chunks` **그리고** `realtime_slack > (enter_factor + λ·pending_count) · gen_time`
- **Pending 탈출 조건** (셋 중 하나):
  - 저점 기준: `slack ≤ (effective_enter − hysteresis_gap) · gen_time`
  - 예측 가드: 현재 진행 중인 청크의 남은 생성 시간이 슬랙을 초과
  - 추정기에 데이터가 없는 경우(예외 케이스)

**압력(pressure) 효과**: pending 수가 많아질수록 effective 임계값이 커져 새로 pending에 보내기 어려워지고(λ), hysteresis_gap만큼 enter/exit 사이에 갭을 둠으로써 진동을 방지.

## 2단계 — 적응형 배치 크기 (`scheduler.py:1018-1023`)

`sslo_config.adaptive_batch_size`가 켜져 있고, new_running 중 하나라도 `is_overdue_post_warmup()`(워밍업 이후 슬랙<0)이면 이번 스텝의 `max_num_running_reqs`를 절반으로 축소.

## 3단계 — 하드 캡 강제 (`scheduler.py:1030-1040`)

`len(new_running) > max_num_running_reqs`이면 슬랙 기준으로 정렬해 **슬랙이 가장 큰(=덜 급한) 초과분을 pending으로 강등**. 캡이 기아 방지보다 우선 — vLLM의 `InputBatch`가 `max_num_seqs`로 사이징되어 있어 절대 초과 불가.

## 4단계 — Waiting 큐 admission 게이트 (`scheduler.py:1281-1294`)

`waiting`/`skipped_waiting`에서 새 요청을 받을 때:
- `len(running) >= max_num_running_reqs`(축소된 로컬 캡)이면 중단
- **pending 중에 `should_exit_pending(...)`인 요청이 하나라도 있으면 중단** — 곧 자리를 돌려줘야 할 요청이 있는 동안 새 요청을 받지 않음

## 5단계 — 오프로드 우선 선점 (`scheduler.py:1497-1539`, `sslo_config.offloading`일 때)

KV 할당 실패 시 FCFS tail이나 PRIORITY 최하위 대신 `running ∪ sslo_pending` 전체에서 **슬랙이 가장 큰 요청을 victim으로 선택**(`max(_sslo_score_key)`), 선점 후 `on_offload_enter`. 커넥터가 KV를 CPU로 저장하고, 재개 시 GPU로 로드.

## 라이프사이클 훅

요청 완료/중단/free 시 (`scheduler.py:2261, 2582, 2627`) `sslo_pending`과 `sslo_consecutive_pending`에서 제거.

## 한 줄 요약

**슬랙 기준 정렬 → 가장 여유 있는 요청을 pending으로 demote(단, `max_consecutive_pending`·`waiting>0`·요청별 EMA 히스테리시스 조건 충족 시) → overdue가 있으면 배치 축소 → 하드 캡 초과분은 슬랙 큰 순으로 추가 demote → KV 압박 시에는 슬랙 큰 요청부터 오프로드.**
