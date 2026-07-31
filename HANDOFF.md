# HANDOFF — KV Offload Tier 실험 머신 검증 (2026-07-31)

> 이 문서는 실험용 GPU 머신의 Claude(Fable) 세션이 받아서 그대로 실행하기 위한
> 핸드오프다. 위에서부터 순서대로 진행하고, 각 단계의 결과를 `WORKLOG.md`에
> 리포 규약(수정/추가/검증 3분류)으로 기록해라. AGENTS.md의 규칙(컨테이너 실행,
> HF 캐시, `# SSLO` 마커 등)이 전부 적용된다.

## 배경

`progress_serve_offload` 모드(KV offload tier)가 개발 머신에서 구현 완료됐다.
slack이 깊은 deferred 요청의 KV를 CPU로 비우고(vacate) deadline 전에
복원(promote)해 KV-capped 상황에서 admission headroom을 만든다. 설계·구현
내역은 `vllm/vllm/sslo/README.md` §"KV Offload Tier"와 `WORKLOG.md`
2026-07-31 항목 참조.

개발 머신에는 `sk-sslo` 컨테이너가 없어 **vLLM 설치가 필요한 테스트 22개가
미실행 상태**다 (순수 계층 101개는 통과). 이 머신에서의 목표는 ① 미실행 테스트
실행, ② 스모크 런으로 vacate→promote 왕복 실증, ③ kv_capped regime 사전
분석이다.

## Step 0 — 환경 준비

`sk-sslo` 컨테이너가 떠 있고 리포가 `/workspace/mlsys`로 마운트되어 있어야
한다. vLLM 로컬 체크아웃이 이 브랜치 기준으로 설치되어 있지 않다면
(AGENTS.md 절차):

```bash
docker exec sk-sslo bash -lc '
  git config --global --add safe.directory /workspace/mlsys
  cd /workspace/mlsys/vllm
  VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.0.0+sslo pip install -e . --no-build-isolation
'
```

Python-only 변경이므로 이미 editable 설치가 되어 있으면 재설치 불필요.

## Step 1 — 테스트 실행 (필수)

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys/vllm
  python3 -m pytest tests/sslo/ -q
'
```

- 기대: 전부 pass. 총 ~123개 (순수 계층 101 + scheduler 신규 10 +
  `test_kv_offload_manager.py` 12).
- 개발 머신에서 **실행된 적 없는** 것: `test_scheduler_sslo.py`의 KV-offload
  테스트 10개(unit 8 + `schedule_sslo()` e2e 2)와
  `test_kv_offload_manager.py` 12개. e2e 2개는 Fake 기반이지만 실제
  waiting-loop를 구동하므로, 실패 시 가장 유력한 원인은 Fake가 안 채운
  scheduler attribute 누락이다 — 테스트 쪽을 고치되, 프로덕션 코드 결함이
  드러나면 `# SSLO` 마커 규칙을 지켜 수정해라.

## Step 2 — 스모크 런 (필수)

```bash
docker exec sk-sslo bash -lc '
  cd /workspace/mlsys
  OUTPUT_DIR=exp/run_sslo/output/offload_smoke \
  CPU_OFFLOAD_GB=16 \
  bash exp/run_sslo/run_test.sh progress_serve_offload 64 Qwen/Qwen3-8B
'
```

(모델은 캐시에 있는 것으로 대체 가능. 비교 기준이 필요하면 같은 설정으로
`progress_serve`도 한 번 돌려라.)

확인 항목 — `${OUTPUT_DIR}` 하위:

1. `scheduler_stats.jsonl`: per-step `kv_capped` / `k_star_unconstrained` /
   `num_vacated` / `num_promoted` / `num_offloaded` 필드 존재.
2. `decisions.jsonl`: `{"kind":"vacate"}` / `{"kind":"promote"}` row.
3. `requests.jsonl`: offload 행에 `total_offloaded_time_s` /
   `num_offload_intervals` / `num_onloads` (0이 아닌 값이 나오는 요청 존재).
4. `summary.json`: `metrics.offload.progress_serve_offload` 집계 존재.
5. 크래시·행 없이 정상 종료 (특히 vacate→promote 왕복 후 요청이 정상 완료
   되는지 — `terminal_outcome`).

**vacate가 한 번도 안 일어나는 경우**: `kv_capped`가 True인 스텝이 없다는
뜻이고, 이는 KV가 admission을 안 묶는 워크로드에서 **정상**이다. 그 경우
`GPU_MEMORY_UTILIZATION`을 낮춰(예: 0.5 → 0.35 …) KV 풀을 인위적으로 줄여
kv_capped를 유도한 뒤 위 1–5를 재확인해라. 유도에 성공한 설정값을 WORKLOG에
기록해라.

알려진 경계 (스모크 해석 시 참고):

- cold-start(글로벌 예측기 웜업 전) analytic tail에서 lead=2일 때
  M_cpu≈0.00098로 기본 `kv_offload_risk_eps`(1e-3) 바로 아래에 걸린다.
  eps/lead를 sweep할 때 이 경계를 의식할 것.
- chunk-level `num_offloaded_iters`는 멀티프로세스 경로에서 0으로 남는다
  (기존 `num_pending_iters`와 동일한 IPC 한계). request-level 3필드만 유효.
- onload 완료 스텝은 offloaded로 집계된다 (신규 admit의 첫 스텝 미집계
  관례와 대칭 — 버그 아님).

## Step 3 — kv_capped regime 사전 분석 (실험 설계 근거)

기존 sweep 산출물(`exp/run_sslo/output_sweep/` 또는 이 머신의 보관 위치)의
`scheduler_stats.jsonl`은 KV offload 이전 버전이라 `kv_capped` 필드가 없다.
대신 `kv_blocks_used`/`kv_blocks_total`(기존 필드)로 KV 포화 근접 스텝 비율을
셀별로 집계해서, **어느 (seqs, rate, 모델) 셀에서 KV가 binding constraint에
근접하는지** 표로 정리해라. 결과가 "어디서도 안 묶임"이면 long-context 축
(긴 프롬프트 데이터셋 또는 `GPU_MEMORY_UTILIZATION` 축소)이 실험 설계에
필수라는 근거가 된다. 이 표는 사용자와 long-context 워크로드 축을 논의할 때
쓸 것이므로 WORKLOG에 붙여라.

## Step 4 — 이후 (사용자 확인 후 진행)

Step 1–3 결과를 사용자에게 보고하고 방향을 확인한 뒤:

- long-context 워크로드 축 설계 (데이터셋 선정, `run_sweep.sh` 셀 구성),
- `kv_onload_lead_iters` / `kv_offload_risk_eps` /
  `kv_offload_min_residency_steps` 튜닝 sweep,
- baseline / progress_serve / progress_serve_offload 본 실험.

## 완료 조건

- [ ] Step 1: `tests/sslo/` 전부 pass (실패 시 수정 내역 WORKLOG 기록)
- [ ] Step 2: vacate/promote 왕복이 로그로 실증됨 (유도 설정 포함 기록)
- [ ] Step 3: 셀별 KV 포화 근접 표 작성
- [ ] WORKLOG.md 갱신
- [ ] 완료 후 이 파일(HANDOFF.md)의 체크박스를 갱신하고, 전부 끝나면 파일
      삭제 여부를 사용자에게 확인
