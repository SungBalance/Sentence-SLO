# Sweep v2 Plan (TI1, 2026-08-04 확정)

hybrid 비호환 발견(WORKLOG 2026-08-04, `kv_offload_model_compat.md`)으로 sweep을 처음부터 재설계·재실행한다. 이 문서는 실행 전 설계 기록이다.

## 확정 사항 (사용자 결정)

- **모델: Qwen3-32B 단일** (dense full-attention, offload 호환). `/cache/hub`로 다운로드.
- **GENERATION_MAX_TOKENS 축: 2048 / 4096 / 8192** (신규 sweep 축)
- **모드: baseline, progress_serve_adaptive, progress_serve_offload** (plain progress_serve 제외)
- TI1 GPU 0,1만 사용 → `NUM_GPUS=2`
- kv_capped 유도는 GMU 축소 금지 — GMU 0.95 고정, cap·gen-length로 유도

## 고정 세팅

| 항목 | 값 |
|---|---|
| GPU_MEMORY_UTILIZATION | 0.95 (기본) |
| 데이터셋 | koala, EXCLUDE_CODE=1, CONVERSATION_ONLY=1, ENGLISH_ONLY=1 |
| chunk unit | sentence |
| offload 셀 | CPU_OFFLOAD_GB=16, prefix caching ON (커넥터 전제), spec-decode 없음 |
| MAX_MODEL_LEN | 0 (auto) |
| repeat | 3 |

KV 규모 감: Qwen3-32B ≈ 256KB/token (64L × 8KV × 128d, BF16), 가중치 ~65GB → KV 풀 ~21GB ≈ 85k tokens. GEN 8K 요청 하나가 최대 ~2.1GB → cap 축과 결합해 KV-bound가 강하게 걸림.

## Sweep table

| 축 | 값 | 개수 |
|---|---|---|
| model | Qwen3-32B | 1 |
| GENERATION_MAX_TOKENS | 2048, 4096, 8192 | 3 |
| cap (CAPS) | 128, 256, 512 | 3 |
| mode (MODES) | baseline, progress_serve_adaptive, progress_serve_offload | 3 |
| rate (RATES, 엔진 공유 내부 sweep) | 4, 8, 16, 24, 32 | 5 |
| consume (CONSUME_CELLS) | read, tts:Kokoro-82M, tts:supertonic-3 | 3 |
| repeat | 3 | 3 |

총 3×3×3×3×3 = **243 엔진-런** (각 5 rate). 셀당 ~40-60분(GEN 8K 셀은 그 이상) → 전체 ~180 GPU-h ≈ **PARALLEL 2로 ~4일**. 단계 실행:

| Phase | 내용 | 엔진-런 | 비고 |
|---|---|---|---|
| 1 | read, GEN 2048, run_1 (3 cap × 3 mode) | 9 | 4-way 비교 첫 그림, ~6h |
| 2 | read, GEN 4096/8192, run_1 | 18 | gen 축 효과 확인 |
| 3 | read run_2-3 | 54 | 반복 |
| 4 | tts 2종 전체 | 162 | 마지막 |

## 실행 전 필요한 조정 (미구현, 기록만)

1. **run_sweep.sh 단일 모델 지원**: 현재 `MODEL_SPECS`는 2개 필수(Phase 1-4가 [0]/[1] 인덱스). 1개만 받도록 소폭 수정 필요.
2. **GEN 축**: run_sweep.sh에 GENERATION_MAX_TOKENS sweep이 없음. 대안 (a) `OUTPUT_ROOT=output_sweep_v2/gen{2048,4096,8192}`로 3회 호출 (코드 무수정), (b) 스크립트에 gen 축 추가(출력 경로에 `gen<N>` 세그먼트). 구현 시 결정.
3. cap 512 초과로 올릴 경우에만 `--warmup-target` 오버라이드 필요 (512까지는 NUM_PROMPTS 4000으로 커버).

## 결정 사항 2차 (2026-08-04, 실행 승인)

- **cap 축: B안 확정** — 하네스 준비 후 gen별 1셀 probe 런(offload 모드, cap 256, ~30분×3)으로 kv_capped 비율·프롬프트/출력 길이 분포를 실측하고 CAPS를 확정한다.
- **multi-turn 축: 채택** — 워크로드를 wildchat multi-turn 대화 prefill 주입으로 전환 (koala single-turn 대체). run_test.py에 dialogue 주입 경로 추가 필요. 프롬프트가 수천 토큰이 되어 prefill 쪽 KV 압박이 현실적으로 걸림.
- **repeat = 1** (multi-turn 채택의 비용 상쇄). 셀 수: 3 gen × |CAPS| × 3 mode × 3 consume × 1.
- 출력 길이 vs GEN cap의 EOS 이슈는 probe에서 실측으로 확인 (`ENABLE_THINKING` 도입 여부는 probe 결과 후 재논의).

## 하네스 확장 (구현 완료, 2026-08-04)

1. `run_test.py` — `--dialogue-prompts`: `load_dialogues()`(wildchat/lmsys/combine)로 대화를 받아, 마지막 user 턴까지의 메시지 프리픽스에 chat template을 적용해 프롬프트 생성 (기존 single-turn 경로 불변). `--max-prompt-tokens` 필터, `--no-chat-template` 병용 시 명시적 에러, run_meta `pool_source=dialogue_<dataset>` 정정.
2. `run_test.sh` — `DIALOGUE_PROMPTS` / `MAX_PROMPT_TOKENS` env 패스스루.
3. `run_sweep.sh` — `MODEL_SPECS` 단일 항목 허용 (1개면 Phase 2/4 스킵).
4. **dialogue 풀 캐시** — `dataset_cache.py`에 raw 대화 JSONL 캐시 추가 (셀마다 HF 스트리밍 반복 제거; miss 37s → hit 0.0s). 캐시 웜업 완료: `dialogues_wildchat_conv-en-nocode.jsonl` 4,000 대화 30MB. 컨테이너에 `datasets` 5.0.1 설치 (의존성 다운그레이드 없음 확인).
5. GEN 축은 스크립트 수정 없이 `OUTPUT_ROOT=output_sweep_v2/gen{2048,4096,8192}` 3회 호출로 처리 (분석 경로 파서와의 호환 유지).
6. 개발(sslo-developer) → 검증(sslo-verifier) 2라운드 수행; 라운드 2 지적 (combine seed 캐시 고정 Important, 로그 구분자 Minor) 수정 반영.

## 워크로드 실측 통계 (wildchat 4,000 대화, Qwen3-32B 토크나이저)

- user 턴: 평균 4.3, p50 3, p90 9, 최대 37
- 프롬프트 토큰: 평균 1,301 / p50 704 / p90 3,683 / p99 7,664 / 최대 9,186 (koala ~140의 ~10배)
- >8K 토큰 프롬프트 0.3% → `MAX_PROMPT_TOKENS=0`(필터 없음)으로 확정. 최대 프롬프트 9.2K + GEN 8K = 17.4K < MAX_MODEL_LEN auto(32K) ✓
- KV 함의: 평균 재적 컨텍스트 ~1.7K 토큰(GEN 2K 기준) → KV 풀 ~85K 토큰이 지탱하는 동시 재적 ~50 → **cap 64~128부터 KV-bound 예상**. cap 사다리를 기존 {128,256,512}에서 아래로 이동해야 하며, probe로 확정.

## Probe 결과 (2026-08-04 실행 완료: gen별 1셀, offload, cap 128, rate 16, wildchat dialogue)

| gen | 출력 p50/p90/p99 | cap 도달 | kv_capped 스텝 | KV p50/p90 | running p50/p90 | vac/prom |
|---|---|---|---|---|---|---|
| 2048 | 705/1,465/2,048 | 3.4% | 12.5% | 0.91/0.99 | 46/68 | 6/6 |
| 4096 | 713/1,460/3,596 | 0.8% | 13.1% | 0.92/0.99 | 41/65 | 7/7 |
| 8192 | 700/1,454/4,024 | 0.5% | 6.7% | 0.34/0.98 | 5/61 | 7/7 |

산출물: `exp/run_sslo/output/probe_v2/gen{2048,4096,8192}/progress_serve_offload/`.

**확정 — CAPS = {32, 64, 128}**: running이 p90에서도 61-68로 cap 128에 닿지 않음 = KV가 ~45-68 재적에서 먼저 binding. cap 256/512는 128과 동일 셀이라 제외. 32 = KV-slack 대조군, 64 = 전이, 128 = KV-bound.

**확정 — GEN 축 (iv)안 (2026-08-05 사용자 결정)**: GEN=8192 고정으로 축 제거 + `ENABLE_THINKING=1` 셀 별도 추가. thinking 셀도 동일 CAPS {32,64,128} 사용 (비교 가능성 우선; thinking은 출력이 길어 세 cap 모두 KV-bound일 수 있으나 cap 32가 최소-부하 대조 역할 — probe 없이 수용한 리스크로 기록).

## 실행 상태

- [x] Phase N (구 rate 사다리 {4..32}): **2026-08-05 사용자 지시로 중단** —
  ~65/135 rate-run 산출물 보존 (`output_sweep_v2/gen8192/`; cap128 전체 +
  cap64 대부분). 전 rate가 서비스 용량(~0.64 req/s)의 6~50배 과포화라
  "overload regime" 슬라이스로만 유효. TTFC는 전 구간 12~30분(순수 큐 대기).
- [x] **Phase N′ 완료 (2026-08-06, 48/48)** — 결과 요약: offload 단독이
  cap64/128에서 승자 (viol 0.9–2.0% vs base 5.8–6.5%, tput 동률); 결합 모드
  전면 열세, cap32에서 파국 (viol 5.8–8.1%, tok/s ~190, offload 최대 490회
  — sub-saturation에서!); adaptive는 cap32에서 tput 241–268로 붕괴.
  cap32 offload 계열의 offload 폭주가 share 회계 낙관(parked 제외 → M_cpu
  과소평가) 가설과 부합 → share 포함 수정(A1) 착수.
- Ablation 확정 (2026-08-06 사용자 결정): **A2 생략, A1 → A3 순서**.
  A1 = share 포함(`kv_offload_share_includes_parked=True`, 기본값) — 실행 중
  (`.../a1_share`, offl/comb 6셀). A3 = A1 + **admission 델타 기준**
  (`admission_delta_criterion`, 기본 False·env `SSLO_ADMISSION_DELTA_CRITERION`
  으로 옵트인): 판정을 `E_viol(k) − E_viol(0) < 1`로 — 매몰비용을 예산에서
  제외하고 admission이 유발한 증가분에만 기존 예산 적용 (신규 knob 없음,
  저부하에서 현행과 동일). 구현·검증 완료 (153 pass, "이슈 없음").
- **A1 결과 (2026-08-07, 24/24)**: ① offload 단독 — cap32 저하 셀 교정
  (r0.5 vac 8→1·tput 337→368, r1 viol 1.1→0.6%·tput 250→314, r4
  1.6→1.2%·251→284), cap64/128은 동등 (일부 셀 ±는 seed-less 샘플링 노이즈
  범위). ② **결합 모드는 구제 실패** — cap32 offload 여전히 272~498회,
  viol 3.5~7.2%, 저부하(cap64 r0.5)는 오히려 악화(1.4→2.6%, vac 0→59).
  결론: 결합 병인은 share 회계가 아니라 **adaptive 축소 나선(②축)** 자체.
  A3(델타 기준)는 ③축(admission 잠금) 대상이므로 결합 구제 가능성 낮음 —
  A3 결과에 따라 "결합 폐기, offload 단독 채택" 권고 예정.
- **A3 결과 (2026-08-07, 10/24에서 조기 중단)**: **델타 기준 기각.**
  offload 단독까지 악화 — cap128/offl viol A1 1.3/1.5/2.0% → A3
  3.7/4.5/7.0% (r1/r2/r4), cap64/offl 1.8→3.4~6.3%, comb/cap128 r0.5
  1.2→7.7%(vac 100). 원인은 per-step `e_viol` 최대가 16~41까지 누적 폭주
  (A1은 ~1 유지) — 절대 임계 `E_viol<1`이 매몰비용 잠금의 원인인 동시에
  **시스템 총위험의 유일한 브레이크**였다. rate 단조 악화로 노이즈 아님.
  잔여 14 rate-run은 실익 없어 중단 (사용자 결정). 산출물은
  `.../a3_share_delta`에 보존(부분).
- **현재 권고**: `admission_delta_criterion`은 기본 False 유지(기각된 옵션
  으로 코드에 보존), `kv_offload_share_includes_parked=True` 채택,
  본 실험 모드는 **offload 단독**. 결합(offload+adaptive)은 A1/A3 모두에서
  열세 — adaptive 축소 나선(②축)이 병인.
## 2026-08-09 adaptive 재평가 → prefill budget 제어 (신규 방향)

로그 분석 결과 현행 adaptive(`max_num_seqs` 축소)는 축이 틀렸다:
- decode-only 스텝 Δ는 p50 74ms / p90 78ms로 매우 안정적. **prefill이 섞인
  스텝은 6%인데 벽시계의 18.4%를 먹고 Δ p90 463ms** (6배) — 마감을 깨는 건
  prefill spike인데 `max_num_seqs` 축소는 여기에 영향이 없다.
- 선택 cap이 재적보다 큰(비구속) 스텝이 cap128 55.7% / cap32 73.8%. 같은
  재적에서 cap 128→96의 실제 Δ 이득은 1.9%뿐(74.3→72.9ms)인데 모델은 이걸
  이득으로 계산 → 축소가 "공짜 점심"으로 보임.
- 결과: adaptive는 스텝의 45%를 running<10으로 보냈고 그때 **대기열 2,691명**,
  선택 cap p10=1. 목적함수 E_viol이 admit된 요청만 보고 큐를 못 보기 때문
  (admission 자기잠금과 같은 뿌리). baseline/offload는 같은 조건에서 붕괴 11%.

**신규 제어 (구현·검증 완료)**: 스텝당 **총 prefill 토큰** 예산
`P* = clamp((γ·t_min − Δ_decode)/κ, floor, base)` — κ는 온라인 EMA, t_min은
MEASURED in-flight의 최소 잔여시간. running 캐리오버 + waiting 신규 admit이
단일 카운터를 공유하고 decode는 무제한. 동시성을 안 줄이므로 처리량 중립이고
service share s를 안 건드려 offload와 간섭이 없다(결합 재시도 가능).
신규 run kind: `progress_serve_prefill_budget`,
`progress_serve_offload_prefill_budget`. config:
`prefill_budget_control`(기본 False) / `_floor`(512) / `_gamma`(0.5).

- [x] **Phase P (4-way, 2026-08-10, 37/48에서 종료)**: baseline / offload /
  prefill_budget / offload+prefill_budget, read, CAPS {32,64,128},
  RATES {0.5,1,2,4}. cap128·cap64는 4모드 완주, cap32 일부 미완.
  결과: cap128에서 offload 1.4~1.6%·pbud 0.7~1.5%·결합 1.0~1.7%(tput 500
  고정) vs baseline 5.8~6.2%; cap64에서 pbud가 전 rate 1.1~1.2%로 최안정;
  cap32는 baseline이 이미 0.9~1.4%라 개입 불필요(offload 계열은 처리량만
  손실). 처리량은 전 SSLO 모드가 baseline의 89~99% (request-level adaptive의
  -20~40%와 대조).
- **미해결**: cap64 결합의 재적 붕괴(19.5 vs 단독 30.8/37.8, tput -15%).
  초기 가설(“prefill 예산이 admission을 조여 저점유 → 불필요 offload”)은
  **반증됨** — pbud 단독이 클램프를 더 많이(32.9% vs 22.1%) 하고도 재적
  37.8을 유지. offloaded 인구도 평균 0.9명뿐이라 11명 감소를 설명 못 함.
  규명하려면 스텝별 “k* 대비 실제 admit 수”를 별도 계측해야 함(현재 미기록).
- **적용 범위 (사용자 지적, 2026-08-10)**: 신규 제어는 **token-level**
  (스텝당 prefill 토큰 예산)이라 **prefill 작업이 없는 스텝에서는 무동작**.
  클램프 발화 22~54%가 곧 “prefill 대기 작업이 있는 국면”의 비율이며,
  프롬프트가 짧은 워크로드에서는 지렛대가 작아진다.

## request-level adaptive 폐기 (2026-08-10 사용자 결정)

`progress_serve_adaptive` / `progress_serve_offload_adaptive`(구 결합)의 **실험
산출물 전량 삭제** (nprime·a1_share·a3_share_delta의 해당 셀 10개 디렉토리,
23GB→10GB). 폐기 근거는 위 "2026-08-09 adaptive 재평가" 항목의 진단(잘못된
축 + 큐-블라인드 붕괴)이며, 그 진단 자체는 유효하므로 기록은 유지한다.
현재 유효한 모드는 baseline / offload / **token_budget** /
offload+token_budget 4종 (2026-08-10 TB* 개명 — 구 prefill_budget 명칭은
phaseP 산출물 경로 호환용 deprecated alias로만 잔존. TB*는 P축+D축 통합,
`paper/scheduling.md` §3⑤ 참조).

- 미해결 (차기 설계 후보): ③축 잠금의 안전한 해제 —
  델타+상한 하이브리드(`ΔE_viol<1` AND `E_viol<C`), 유휴 시에만 델타 적용,
  또는 M≈0(회생 불능) 요청만 예산에서 제외하는 국소 수정.
- **TODO — prefill 비용 오프라인 프로파일 (2026-08-10 추가)**: 온라인 κ
  추정을 시작 시 프로파일로 대체/보강. 실측상 κ는 모드·cap 불문
  0.15~0.16으로 사실상 상수라 오프라인화하면 추정기 병리 클래스(폴백 오염·
  소분모 발산·콜드스타트·피드백 루프) 전체가 소멸. 단 토큰당 비용이 청크
  크기에 의존(P≤512에서 0.23~0.30 vs 2048에서 0.16 — per-chunk 오버헤드
  상각)하므로 스칼라가 아니라 **비용 곡선 cost(P)** 로 프로파일하고, P*는
  `Δ_dec(b)+cost(P) ≤ γ·T_min`을 곡선에서 직접 역산(κ 매개변수 제거).
  기존 CUDA-graph decode 프로파일의 하이브리드 패턴(오프라인 shape ×
  라이브 anchor 스케일)을 재사용해 런타임 드리프트 흡수. 시작 비용은
  P ∈ {256..4096} 프로파일 런 수 초.
- **TODO — 통합 스텝-시간 예산 (2026-08-10, Δ-내생 defer의 일반화)**:
  request cap도 원시 토큰 합도 아닌 **가중 토큰 합 = 예측 스텝 시간**을
  단일 예산으로: `Δ_0 + κ_d·D + κ_p·P ≤ γ·T_min`. 원시 P+D 캡은 두 토큰의
  단가가 다를 수 있어(decode는 컨텍스트 KV를 쿼리 혼자 읽고 prefill 청크는
  상각 — 단, compute-bound면 ~1:1) 기각. **주의: κ_d는 관측 데이터로 추정
  불가** — 요청수와 KV사용량이 런 내 공선(컨텍스트 ≈ 상수×요청수)이라
  횡단면 회귀가 무효(절편 −104ms로 실증, 2026-08-10). κ_d는 반드시
  (배치 × 컨텍스트 길이) 그리드의 통제 프로파일로 측정할 것. 현행 P*는
  D 고정 특수해(stage 1, 완료); decode 축(defer로 D 축소)은 stage 2 —
  큐 공백+at-risk 국면에서 별도 게이트 없이 제약이 자동 강제. 주의:
  decode 축은 정수(요청 단위)이며, 부하 심층에서 축소 나선 방지를 위해
  **D 하한(at-risk+forced) + 기아 가드** 필수 (P_floor 문법 재사용).
  κ_d·κ_p는 오프라인 프로파일 TODO가 커버. 구현 전 TTS 축 실험에서
  해당 국면 발생량 측정 선행 (waiting==0 ∧ T_min<τ 스텝 비율).
- [x] Phase N′ 설계: read만,
  RATES="0.5 1 2 4" (0.25는 셀당 ~85분이라 시간 절감 위해 제외 —
  sub-saturation은 0.5 < 용량 0.64 req/s로 커버; 2026-08-05 사용자 요청),
  모드 4종(baseline, adaptive, offload, **offload_adaptive 결합**),
  CAPS {32,64,128} = 12 엔진-런(48 rate-run), `OUTPUT_ROOT=.../nprime` —
  **2026-08-05 재시작** (구 gen8192 산출물 4.8GB 삭제, 예상 ~7h).
  완료 후 admission 한계 기준(ΔE_viol) 수정 → 재측정으로 ablation.
- [ ] Phase T′ (thinking): 재보정 후 재설계
- [ ] sweep-level 분석

## 2026-08-05 발견/결정

- **ProgressServe admission 자기잠금** (구 사다리 offload/adaptive 셀 tput
  저하의 원인): 대기 요청은 설계대로 E_viol에 미포함(무deadline)이나,
  admission이 service share/decode 슬롯을 줄여 기존 MEASURED 요청 위험을
  올리는 간접 경로로 E_viol 절대 임계(<1)가 소진되면 k*≈0으로 잠김 —
  저하 셀에서 65% 스텝이 running≈5·KV 20%·e_viol=0.99. 원인은 (a) 절대
  임계(한계 기준 아님) + (b) M≈0(회생 불능) 요청의 매몰비용 미분리.
  개선 후보: 한계 기준 ΔE_viol, M≈0 예산 제외, 유휴 시 k* 하한 —
  **방향 미결정 (재보정 sweep 전 수정 여부 포함)**.
- 메트릭 분리: `num_handling_users`(offloaded 포함, 주 지표) /
  `num_handling_users_online`(제외, 비교용) — 구현 완료. 구버전 분석
  스크립트(`analysis/capacity.py` 등)의 구 의미 소비는 구 산출물과 함께
  폐기 예정이라 미조치 (2026-08-05 사용자 확인).
- 결합 모드 `progress_serve_offload_adaptive` run kind 추가 — 구현 완료.

## 사전 작업 상태

- [x] Qwen3-32B 다운로드 완료 (2026-08-04, `/cache/hub`, 17/17 샤드 62GB
      검증). sweep에서는 `MODEL_SPECS="Qwen3-32B:Qwen/Qwen3-32B"`로 참조
      (run_test.sh가 `HF_HUB_CACHE=/cache/hub`를 설정하므로 HF id로 캐시 히트)
- [ ] run_sweep.sh 단일 모델/gen 축 조정
- flashinfer jit-cache 이슈는 dense 모델이라 무관 (MoE 추가 시에만 `pip uninstall flashinfer-jit-cache`)
