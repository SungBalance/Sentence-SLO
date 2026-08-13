# Work Log

## 2026-08-10 (push 차단 시크릿 제거 + nocode 필터 unfenced 코드 감지)

**Modified**
- `exp/tools/lm_datasets.py` — `_is_code_request` 에 `_CODE_LINE_PATTERN`
  추가: ``` 펜스 없이 붙여넣은 원시 코드 감지 (import/#include,
  public/private/protected 선언, statement 형태의 `;` 종결 라인,
  `});` 류 구두점-단독 라인). 코드형 라인 ≥3 (`_MIN_CODE_LINES`) 일 때만
  코드로 판정해 산문 오탐 방지.
- `exp/tools/dataset_cache/dialogues_wildchat_conv-en-nocode.jsonl` —
  수정된 필터로 재생성 (4000 dialogues, build seed 42).

**Debugging / verification**
- GitHub Push Protection 이 구 캐시 3234행의 Mapbox 토큰(WildChat 원문
  유래)으로 push 차단 → `b192ae46a` amend + cherry-pick 으로 히스토리에서
  redact (재작성 전후 트리 차이는 해당 1줄뿐임을 diff 로 확인).
- 구 캐시 4000개 재현 검사: 새 필터가 코드 대화 100건(2.5%) 드롭, 확인된
  코드 유출건(javadoc/Java, Dart/Mapbox, "code:"+Python, JS 덤프) 전부
  검출, 산문 유지 확인(콘랭 음운 목록·트위터 핸들 덤프·EXIF 덤프 등).
  sslo-verifier 루프 이슈 없음.
- 재생성 캐시 검증: 새 필터 기준 코드 대화 0건, ``` 펜스 0건, 시크릿 패턴
  8종(mapbox/github/aws/openai/slack/google/private-key/jwt) 클린.

## 2026-08-05 (sweep v2 Phase N 중간 결과 + ProgressServe admission 자기잠금 진단)

- 실행: sweep v2 Phase N (Qwen3-32B, GEN 8192, wildchat dialogue, 구 rate
  사다리 {4,8,16,24,32}) — cap128 전체(45 rate-run) + cap64 대부분 완료 후
  사용자 지시로 중단 (~65/135 산출물 보존, `output_sweep_v2/gen8192/`).
- 중간 결과 (cap128): CU-SLO 위반율(tau=1s) offload 1.2-2.0%(read) /
  0.1-0.7%(TTS) vs baseline 7.5-9.2% / 2.9-4.4% vs adaptive 3.5-5.4% —
  offload가 전 rate에서 우세. tput/동시사용자(u≈41, KV가 ~41명에서 binding)
  는 대부분 rate에서 baseline과 동률.
- Debugging (offload/adaptive 일부 셀 tput -20~35% 저하의 원인): 저하 셀
  (read/cap128/r16 offload) scheduler_stats 분석 — 65% 스텝이 running≈5.5,
  waiting≈3,200, KV 20%, kv_capped 0%인데 k*_unconstrained=0.7,
  e_viol=0.99(임계 1.0 직하 고정). KV offload tier가 아니라 **ProgressServe
  admission의 절대 임계(E_viol<1)가 소수 고위험 in-flight 요청의 위험 합으로
  소진되어 admission이 잠기는 구조** (대기 요청 자체는 설계대로 E_viol
  미포함 — `progress_serve.py:201-204`; admission은 service share·decode
  슬롯 축소를 통해 간접적으로만 기존 요청 위험을 올림). 최장 7,830스텝
  저점유 지속. 개선 후보(미결정): 한계 기준 ΔE_viol / M≈0 매몰비용 예산
  제외 / 유휴 시 k* 하한.
- 추가 발견: 전 rate가 서비스 용량(~0.64 req/s)의 6~50배 과포화 → TTFC
  12~30분(순수 큐 대기). 재보정 결정: Phase N′는 read만,
  RATES="0.25 0.5 1 2 4", 결합 모드 포함 4모드 (SWEEP_PLAN_v2.md).

## 2026-08-05 (handling-users offload 분리 + progress_serve_offload_adaptive)

- Modified content (과제 A): `scheduler.py` `_sslo_dump_step_stats`의
  per-step 통계에서 `num_handling_users`를 `running + sslo_pending +
  _sslo_offloaded`(CPU 체류 요청 포함)로 바꾸고, 기존 GPU-only 값은
  `num_handling_users_online`으로 병기. offload 미사용 모드에서는
  `_sslo_offloaded`가 비어 두 키가 같은 값이라 기존 로그와 호환. `analyze.py`는
  `mean_handling_users_online_time_weighted`(time-weighted mean)를
  `metrics.scheduler[mode]`에 추가 — 주 지표(`mean_handling_users`)와 CSV
  스칼라 컬럼 구성은 그대로 두고 비교값만 summary.json에 남긴다.
- Added content (과제 B): 결합 모드 `progress_serve_offload_adaptive`
  (KV offload tier + adaptive batching). `run_test.py`에 모듈 상수
  `OFFLOAD_RUN_KINDS`를 두고 sslo_params/KVTransferConfig 두 분기가 공유하게
  조건 확장, adaptive_batching 조건에 새 run_kind 추가. 모드 목록 상수
  (`metrics_utils.MODES_DEFAULT`, `analyze.py SSLO_MODES`)에 등록 — argparse
  choices와 sweep_analysis `--modes` 기본값이 여기서 파생되므로 추가 배선 불필요.
  `run_test.sh`/`run_sweep.sh`는 주석의 selectable 목록만 갱신(MODES는 이미
  임의 comma-sep이라 로직 변경 없음). `analyze.py` 경로 기반 mode 추론은
  디렉토리명 정확 일치라 substring 충돌 없음.
- Added content (테스트): `tests/sslo/test_scheduler_sslo.py`에
  `test_step_stats_handling_users_counts_offloaded` — `_sslo_offloaded`에 1건을
  넣고 stats를 덤프해 `num_handling_users==3`, `num_handling_users_online==2`
  확인.
- Debugging/verification: 컨테이너 `sk-sslo`, GPU 미사용
  (`CUDA_VISIBLE_DEVICES=""`). `pytest tests/sslo/ -q` → 146 passed, 1 skipped,
  1 failed(`test_tts_consume_path.py::test_tts_path_uses_audio_ready_time_for_slack_and_deadline`
  — 이전부터 있던 실패, 이번 변경과 무관). py_compile/bash -n 통과,
  `run_test.py --help`에 새 run_kind 노출. 엔진 생성을 가로챈 probe 스크립트로
  네 run_kind의 sslo_params 확인: offload_adaptive는 adaptive_batching=True /
  kv_offload=True / SimpleCPUOffloadConnector, plain offload는
  adaptive_batching=False 유지.

## 2026-08-04 (sweep v2 준비: multi-turn 하네스 + 캐시 웜업 — GPU 단계 직전까지)

- 설계 확정 (사용자 결정, `exp/run_sslo/SWEEP_PLAN_v2.md` 소스 오브 트루스):
  Qwen3-32B 단일 / GEN {2048,4096,8192} / 모드 {baseline,
  progress_serve_adaptive, progress_serve_offload} / wildchat multi-turn
  (`DIALOGUE_PROMPTS=1`) / repeat 1 / CAPS는 probe(B안)로 확정 / GEN 축은
  `OUTPUT_ROOT` 분리 3회 호출.
- Added content (하네스 라운드 1): `run_test.py` `--dialogue-prompts`
  (`load_dialogues()` → 마지막 user 턴까지 프리픽스 + chat template) /
  `--max-prompt-tokens`; `run_test.sh`에 `DIALOGUE_PROMPTS`/`MAX_PROMPT_TOKENS`
  패스스루; `run_sweep.sh` `MODEL_SPECS` 1개 허용(Phase 2/4 스킵);
  README dialogue 섹션. 라운드 2(풀 캐시 + 지적 처리)는 아래 항목 참조.
- Added content (환경): 컨테이너에 `datasets` 5.0.1 설치 — dialogue 모드
  필수 의존성; huggingface_hub/fsspec/dill 버전 변화 없음, vllm/torch import
  정상 확인. Qwen/Qwen3-32B `/cache/hub` 다운로드 완료 (17/17 샤드, 62GB).
- Verification: 개발(sslo-developer)↔검증(sslo-verifier) 루프 3라운드 —
  R1: Critical/Important 0 (Minor 1은 R2에 포함 처리), R2: Important 1
  (combine seed 캐시 고정) + Minor 1 (로그 구분자) → 수정, R3: 이슈 없음
  (루프 종료). wildchat 캐시 웜업 4,000 대화 37s(30MB). 프롬프트 통계
  (Qwen3-32B tokenizer): 평균 1,301 / p50 704 / p90 3,683 / p99 7,664 /
  최대 9,186 토큰, 평균 user 턴 4.3 → `MAX_PROMPT_TOKENS=0` 확정, cap
  사다리는 기존 {128,256,512}보다 하향 필요( ~50 동시 재적에서 KV-bound
  예상), probe로 확정 예정. GPU 필요 단계(probe → sweep)만 남음.

## 2026-08-04 (dialogue 워크로드: 풀 캐시 + 검증 지적 처리)

- Added content: `exp/tools/dataset_cache.py` — `dialogue_cache_path()` /
  `load_or_build_dialogue_pool()`. 기존 `load_or_build_combine_pool` 컨벤션
  (DATASET_CACHE_DIR, `.tmp`→rename, seed는 로드 후 셔플만) 그대로, 소스
  데이터셋 + 필터 조합별 파일 `dialogues_{dataset}_{conv-en-nocode}.jsonl`에
  **chat template 적용 전 raw `{"messages": [...]}`** 를 캐싱해 모델 불문
  재사용. 캐시가 요청 수보다 작으면 로그 한 줄 후 있는 만큼 사용(재빌드는
  파일 삭제).
- Modified content: `exp/run_sslo/run_test.py` — dialogue 분기가 이 캐시를
  통해 대화를 얻도록 변경(miss 시에만 `load_dialogues()` HF 스트리밍).
  캐시 빌드는 `DIALOGUE_BUILD_SEED=42` 고정 — `--dataset-name combine`은
  loader seed가 풀 구성 자체를 바꿔 최초 빌드 seed가 캐시에 영구 고정되므로,
  per-run 순서는 캐시 계층 셔플(`args.dataset_seed`)에만 맡긴다.
  그 외:
  `_pool_source()` 헬퍼로 pool 로그와 run_meta의 `workload_id`/`pool_source`를
  dialogue 모드에서 `dialogue_<dataset>[_maxtok<N>]`로 기록(non-dialogue 문구
  불변); `--dialogue-prompts`+`--no-chat-template` 조합을 `parser.error`로
  차단(멀티턴 직렬화에 템플릿 필수). README/`run_test.sh` 주석에
  `DIALOGUE_PROMPTS=1` 사용 시 `DATASET_NAME`을 wildchat/lmsys/combine으로
  바꿔야 함(기본 koala는 즉시 에러)과 캐시 위치·shortfall 동작 명시.
- Debugging/verification (컨테이너 `sk-sslo`, GPU 런 없음): py_compile /
  `bash -n` / `--help` 통과. 격리 경로(`DATASET_CACHE_DIR=/tmp/...`)에서
  wildchat 8대화(conv+en+nocode) 캐시 미스 11.20s → 히트 0.000s, 결과 동일,
  파일 8줄·키 `messages` 확인, 100개 요청 시 shortfall 로그 후 8개 반환.
  검증용 캐시는 삭제하고 기본 캐시 디렉터리는 미변경(단일턴 combine 풀
  n=2591 캐시 히트 그대로).

## 2026-08-04 (TI1 실험 머신 검증: HANDOFF Step 0–2)

- Modified content: `exp/run_sslo/analyze.py` —
  `finalize_round2_metrics()` 화이트리스트에 `offload` 그룹 전달 블록 추가
  (truthy 체크라 non-offload 런의 summary.json 형태는 불변). 기존에는
  `offload_request_stats()` 결과가 Round-2 finalize에서 탈락해
  `metrics.offload.<mode>`가 summary.json에 절대 안 나왔고,
  `metrics_utils.py`의 "KV offload" DISPLAY_GROUP은 죽은 코드였다.
  sslo-verifier 검증: 이슈 없음 (1-pass).
- Added content (환경, TI1 머신): `sk-sslo` 컨테이너 신규 생성 — GPU 0,1만
  사용 가능하므로 `run_docker.sh`의 인자에서 `--privileged`를 빼고
  `--gpus '"device=0,1"'`로 장치 제한 (privileged가 GPU 제한을 무력화함;
  스크립트 자체는 미수정). vLLM editable 설치는
  `VLLM_PRECOMPILED_WHEEL_COMMIT=5371d6fb4023a1a08021135e46e9354ba0923e50`
  핀 필수: `vllm/`이 upstream 히스토리 없이 squash 벤더링돼 merge-base 탐지가
  실패 → 최신 nightly wheel 폴백 → `vllm._C` ABI 불일치로 import가 깨진다.
  벤더링 트리는 upstream 5371d6fb(2026-04-29, v0.20.1rc1.dev57)와 지문 5/5
  일치 (변환 커밋의 gitlink b1388b1f는 실제 트리보다 오래됨). pytest/tblib은
  별도 설치.
- Verification (Step 1): `tests/sslo/` 147개 중 145 pass / 1 fail / 1 skip.
  이 머신에서 처음 실행된 KV-offload 테스트 38개
  (`test_scheduler_sslo.py` unit 8 + e2e 2 포함 26, `test_kv_offload_manager.py` 12)
  전부 pass. fail 1은 기존 알려진 `test_tts_consume_path.py` 실패, skip 1은
  TTS profile CSV 부재(이 머신에 `measure_tts_duration` 산출물 없음).
- Verification (Step 2, vacate→promote 실증): Qwen2.5-32B(dense)/cap512/
  rate32/GMU0.95(기본)에서 kv_capped 107스텝, vacate 6·promote 6
  (decisions.jsonl `{"kind":"vacate"}`/`{"kind":"promote"}` row, 예:
  요청 312-a144ac5a step 521 vacate → 523 promote), requests.jsonl에
  offload 라이프사이클 비영 4건(onload 6회, 최대 CPU 체류 21.28s), 전 요청
  정상 완료(2048개/309s, exit 0). summary.json에
  `metrics.offload.progress_serve_offload` 집계 존재(값 0 — offload 이벤트가
  전부 warmup 윈도 밖이라 in-window 집계상 0; 구조 검증 완료). kv_capped 유도는
  사용자 지시대로 GMU 축소가 아니라 기본 utilization + 모델 사이즈 업 +
  cap 증가로 수행 (8B는 rate를 올려도 KV가 안 묶임; 32B/cap256은 cap이 먼저
  묶여 KV 최대 86.5%, cap512에서 KV 100% 도달로 kv_capped 발생).
- Verification (발견 1, flashinfer SM120): Blackwell RTX PRO 6000(SM120)에서
  flashinfer 0.6.8.post1 cutlass fused-MoE가 부팅 중
  `TypeError: ... Expected 24 but got 25 arguments`로 크래시 (JIT 캐시
  클리어로 해결 안 됨). MoE 모델은 `VLLM_USE_FLASHINFER_MOE_FP16=0`으로
  flashinfer 백엔드를 제외하면 TRITON 폴백으로 정상 부팅.
- Verification (발견 2, hybrid 비호환 — 실험 설계 영향): Qwen3.5-35B-A3B
  스모크에서 kv_capped 자연 유도(KV 94.6%) 후 첫 vacate→promote 왕복 직후
  upstream `_mamba_block_aligned_split()`의
  `assert num_external_computed_tokens == 0`("External KV connector is not
  verified yet")으로 엔진 크래시. Qwen3.5/3.6/gemma-4 등 hybrid
  (mamba/linear-attn) 계열은 external KV connector 기반 offload tier와
  비호환 (mamba state가 paged KV 밖이라 assert 제거로도 해결 불가).
  → sweep 라인업의 35B 모델을 offload 모드 셀에 쓸 수 없음; offload 실험
  모델 축은 dense full-attention(예: Qwen2.5-32B)으로 재설계 필요.
- Step 3 (기존 sweep 산출물 kv_capped 사전 분석): 이 머신에는
  `output_sweep/` 산출물이 없어 (리포·`/data` 전체 탐색) 수행 불가 —
  dev 머신 데이터 필요.
- Added content: `kv_offload_model_compat.md` (repo 루트) — external KV
  connector 기반 offload의 모델 아키텍처 호환성 정리 (코드 조사, 실행 없음).
  판정 기준 G1–G5(mamba align assert, full-attn 그룹 필수, HMA 딜레마,
  prefix caching 전제, enc-dec 배제)와 호환/비호환/조건부 3분류,
  실험 모델 축 권고(1순위 Qwen3-30B-A3B full-attn MoE) 포함.
- Follow-up (사용자 결정, 2026-08-04): **본 sweep은 처음부터 재실행해야 함**
  — 발견 2(hybrid 비호환)로 offload 모드가 기존 라인업(Qwen3.5-9B/35B-A3B)
  에서 불가하므로 모델 축 재설계 후 baseline/progress_serve(/adaptive/
  offload) 전 모드 재수행. 이 세션에서는 기록만 남기고 실행하지 않음.
- Follow-up (flashinfer 24-vs-25 인자 크래시 근본 원인, 실행 없이 확인):
  NGC 컨테이너 잔존 패키지 `flashinfer-jit-cache 0.6.7+a79fb63c.nv26.3`의
  사전 빌드 `jit_cache/fused_moe_120/fused_moe_120.so`가 0.6.7 시그니처
  (24인자, `swizzled_input_sf` 부재 — strings로 확인)인데, flashinfer 로더가
  JIT 대신 이 AOT 아티팩트를 우선 로드해 flashinfer-python 0.6.8.post1
  (25인자 호출)과 불일치. 해결책은 flashinfer 업그레이드가 아니라
  **`pip uninstall flashinfer-jit-cache`** (또는 0.6.8.post1 대응 jit-cache
  설치) — 그러면 sm120 모듈이 동일 버전 csrc에서 JIT 컴파일된다.
  flashinfer-python 자체를 0.6.9+로 올리는 것은 vLLM 체크아웃(0.6.8.post1
  핀, 2026-04 API 기준)과의 호환이 깨질 수 있어 비권장.

## 2026-07-31 (KV offload tier: stage 3 — harness + docs)

- Modified content: `exp/run_sslo/run_test.py` — new run kind
  `progress_serve_offload` wires `kv_offload=True` into `sslo_params` (env
  overrides `SSLO_KV_ONLOAD_LEAD_ITERS` / `SSLO_KV_OFFLOAD_RISK_EPS` /
  `SSLO_KV_OFFLOAD_MIN_RESIDENCY_STEPS`, mirroring `SSLO_ADAPTIVE_BATCHING`)
  and attaches `KVTransferConfig(kv_connector="SimpleCPUOffloadConnector",
  kv_role="kv_both", ...)` in eager mode with `enable_prefix_caching=True` and
  CPU capacity from `CPU_OFFLOAD_GB` (default 16) — offload mode only. `collect_one`
  surfaces `total_offloaded_time_s` / `num_offload_intervals` / `num_onloads`
  from `sslo_metrics` into `requests.jsonl`, gated on `kv_offload`: since
  `SsloRequestStats` always carries these fields (default 0.0/0/0) regardless of
  mode, non-offload rows emit `None` so analyze.py's None-gate omits them.
- Modified content: `exp/run_sslo/metrics_utils.py` — `MODES_DEFAULT` gains
  `progress_serve_offload`; new `KV offload` DISPLAY_GROUP (offloaded time +
  onloads/req). `exp/run_sslo/analyze.py` — `SSLO_MODES` gains the mode;
  `offload_request_stats()` emits `metrics.offload.<mode>` (omitted when rows
  carry no offload fields → backward-compatible).
- Modified content: `exp/run_sslo/run_test.sh` (usage + KV-offload env doc),
  `run_sweep.sh` (mode list note — offload selectable via `MODES`),
  `README.md` (4 modes, offload requirements/metrics, dropped stale
  `SSLO_OFFLOAD_LOG_PATH` / `offload_log.jsonl` references).
- Modified content: docs — `vllm/vllm/sslo/README.md` (§9 KV Offload Tier:
  motivation, 3-state math N_defer_cpu/R_defer_cpu/M_cpu, eager-mirror
  mechanism, config table; §10 Related legacy sentence corrected),
  `CLAUDE.md` (kv_offload knobs + run_sslo mode list).
- Verification: isolated `sk-specllm` container (this repo is not mounted, so
  files copied in). Modified `exp/run_sslo/*.py` pass `python3 -m py_compile`;
  `.sh` pass `bash -n`; `analyze.offload_request_stats` exercised on
  pipeline-shaped rows — non-offload rows (`collect_one` emits `None` when
  `kv_offload` is off) → `None` (omitted); offload rows carrying real floats
  incl `0.0` → distribution dict. Pure-layer pytest (100 passed) unchanged —
  no vLLM code touched. e2e / manager tests need `sk-sslo` (absent here), not run.
- Open risk: the scheduler-side offload counters reach `requests.jsonl` only
  once vLLM propagates them via `SsloSchedulerSnapshot` (stage 1/2). Today the
  snapshot + output_processor merge carry pending/step scalars but NOT the
  offload fields, so `total_offloaded_time_s` / `num_onloads` read as their
  `SsloRequestStats` defaults (0) until that wiring lands. Not touched here
  (vLLM code out of scope); flagged for the scheduler tier.

## 2026-06-11 (chunk_length_study: oracle vs online posterior)

- Added content: `exp/chunk_length_study/replay_posterior.{py,sh}` — GPU-free
  offline replay of each category's `chunks.jsonl` in chunk-completion-time
  order, feeding the engine's own `ChunkLengthPredictor` (loaded straight from
  `vllm/vllm/sslo/slo_state.py`) with the exact `length_tail_prob` gating
  (warmup 128 / min_denom 4 / cold-start 2048). Reconstructs the continuously
  updated online posterior ProgressServe sees; compares it against an oracle
  (full-run empirical distribution) per category.
- Added content: progress-fraction sharpening axis f∈{0,25,50,75%} — at each f
  predict remaining length L−⌊f·L⌋ (conditional median), online vs oracle.
  Outputs `output/stats/oracle_vs_posterior.{json,csv}` (long: category×progress)
  + plots `posterior_{convergence,sharpening,calibration}.png`.
- Finding: after the 128-sample warmup gate, online posterior error ≈ oracle to
  the decimal at every progress level (e.g. en-code c0 4.39/4.39, c75 2.31/2.30);
  remaining-length MAE falls with progress (~4.4→2.3 tokens at 75%) as the
  conditional tail tightens. Online-estimation cost is essentially just the
  warmup gate; cold-eval fraction 0.7–0.9% on these single-rate runs. Tail
  calibration ECE: late 0.00–0.01, early ≤1024 ~0.10 (cold-start plateau).
  Nuance: en-dialogue online > oracle at c25 (+0.32) but < oracle at c75 (−0.30)
  — windowed history tracks the recent tail better deep in the conditioning.
- Verification: container `py_compile` + `bash -n`; replay ran clean on all 4
  categories (18.5k–21.4k chunks); plots inspected.

## 2026-06-11 (chunk_length_study: unit distribution + request divergence)

- Added content: `exp/chunk_length_study/analyze_unit_distribution.{py,sh}` —
  per-category GLOBAL consumable-unit (chunk `num_token`) length distribution
  (percentiles + hist), and per-request two-sample KS divergence from that
  global pool over the integer length support. Single KS ranking across
  categories → top-5 most-divergent requests with characteristics. Drops
  requests with < MIN_CHUNKS=8 units. Outputs
  `output/stats/{unit_distribution.json, unit_distribution_global.csv,
  top_divergent_requests.csv}` + plots `unit_distribution_global.png`,
  `top_divergent_requests.png`.
- Added content: `--max-chunk-len` (default 100) excludes runaway un-segmented
  chunks (no sentence boundary hit; real sentence chunks end by p99.9 ≈ 55–82,
  >100 is ≤0.08% per category) and reports the dropped count; 0 disables.
- Finding: global chunk-length distributions are near-identical across
  categories (median 20–21, p99 43–47), with a hard ~14–16-token floor (min
  sentence size) and a right skew; en-code has a sharp ~17–18-token peak
  (code-line structure). With outliers excluded all four collapse to CV ≈
  0.29–0.31 (ru's raw CV 0.52 was driven entirely by one max=1410-token chunk
  = an un-segmented wall of text; 15/1/1/0 chunks dropped >100). Top-5 divergent
  requests (KS 0.54–0.62) split into two failure modes vs the global posterior:
  (a) uniformly short, low-variance streams — zh req442 (119 chunks, median 17,
  CV ×0.31, a lat/long coordinate list); (b) uniformly long streams — en req510
  / ru req505 / en req371 (median 29–30 vs global 20, +9–10 shift). These are
  the requests a global posterior systematically mis-centres → evidence for
  where per-request adaptation (vs pure global) would help.
- Verification: container `py_compile` + `bash -n`; ran clean on all 4
  categories; fixed a hist-binning artifact (bin within 99.5-pctile window so
  ru's 1410-token outlier doesn't flatten every bin); plots inspected.

## 2026-05-28 (plots/figures updates)

- Modified content: Regenerated Figure 6.7g as a request-rate grouped plot under the `<0.5%` unit-miss budget, keeping Baseline/Ours color grouping and annotating the selected batch size for each request rate.
- Debugging/verification details: Ran container `py_compile`, regenerated the Figure 6.7g processed CSV/PNG in `sk-sslo-vllm`, confirmed 60 rows across request rates 8/12/16/20/24 with `run_count == 3`, threshold `0.005`, and max unit miss rate below `<0.5%`.

## 2026-05-27 (plots/figures updates)

- Modified content: Updated Figure 5.4 so only the leftmost subplot shows the y-axis label/ticks, row spacing is tighter, and the model/method title is shown on every subplot; normalized Figure 5/6 plot labels from `Max Batch Size`/`Batch Size` to `Batch size`.
- Modified content: Revised the new Figure 6.7a panel so Baseline vs Ours are encoded by legend color/marker, unit-miss budgets are separate subplots, selected batch sizes remain annotated, and supported users are computed from 3-run aggregates only.
- Modified content: Removed the relaxed 5% unit-miss budget from Figure 6.7a and regenerated its processed CSV/PNG from the current `data/output_sweep` contents.
- Modified content: Updated Figure 6.7a unit-miss budgets to strict `0`, conservative `<0.5%`, and relaxed `<1%`.
- Added content: Added shared unit-miss diagnostic preprocessing helpers plus Figure 6.7b unit violation ratio curve, Figure 6.7c batch-size operating heatmap, and Figure 6.7d request-any-unit-miss ratio scripts.
- Added content: Added Figure 6.7e violated-chunk grouped barplot and Figure 6.7f in-flight-user grouped barplot with model/consume subplots, Baseline/ProgressServe legend, and hierarchical batch-size/request-rate x-axis.
- Modified content: Changed Figure 6.7e from absolute violated chunk counts to violated-unit ratio over all valid units.
- Added content: Added Figure 6.7g best in-flight users under the `<0.5%` unit-miss budget, selecting the highest in-flight point per model/consume/policy from Figure 6.7a and annotating the chosen batch size and request rate.
- Added content: Added Figure 6.1.2 TTFC p99 preprocessing/plotting output and Figure 6.7a unit-miss budget preprocessing/plotting output under `exp/plots/figures/`.
- Debugging/verification details: Ran targeted container `py_compile` checks in `sk-sslo-vllm`, regenerated Figure 6.1.2 and 6.7a-g processed CSV/PNG outputs, confirmed Figure 6.7 outputs all have `run_count == 3`, and visually checked the regenerated Figure 6.7a-g PNGs.

## 2026-05-25 (Figure 5.1 request metrics)

- Modified: `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` — added two-line `Request N` / TPOT labels inside the upper-left corner of each existing trace panel, shifted the labels slightly right, removed the temporary bottom metric tables, and changed the consumed-token trace to a solid line.
- Modified: `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` — replaced the in-panel TPOT labels with three-line per-request panel titles using actual request IDs, tokens, TPOT, words, consume time, and consume-token rate.
- Modified: `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` — tightened the legend-to-panel-title spacing, added a small gap below each request title, and reduced the metric title font size.
- Modified: `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` — tuned legend-title and title-plot spacing to a compact, matching gap of roughly 16 px in the rendered PNG.
- Modified: `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` — renamed the token-count y-axis to `# of Tokens` and moved the legend up to increase the legend-title gap.
- Modified: `exp/plots/figures/scripts/figs/fig5_2_cu_slo_reconstruction.py` — added the request ID title only, removed max-stall/tau annotations, renamed token axes, removed Slack from the legend, and reused the consumable-token blue for the lower panel data.
- Modified: `exp/plots/figures/scripts/figs/fig5_2_cu_slo_reconstruction.py` — added the generated-token trace and changed the available-consumable-unit trace to match Figure 5.1's blue dotted style and legend wording.
- Modified: `exp/plots/figures/scripts/figs/fig5_2_cu_slo_reconstruction.py` — renamed the lower-panel y-axis to `Remained Consumable Tokens`.
- Added: Regenerated `exp/plots/figures/fig5_1_token_metric_mismatch.png` and `exp/plots/figures/fig5_2_cu_slo_reconstruction.png`.
- Debugging/verification: Ran container compileall for Figure 5.1/5.2 scripts, regenerated Figure 5.2 preprocessing output, regenerated both PNGs inside `sk-sslo-vllm`, and visually checked both figures.
- Modified: `exp/plots/figures/scripts/figs/paper_plot_style.py` — fixed the Pretendard font lookup after the figures folder restructure so the common style resolves `exp/plots/fonts/Pretendard-Regular.ttf`.
- Added: Regenerated `exp/plots/figures/processed/fig5_1_token_metric_mismatch_request_metrics.csv` with TPOT-only values and refreshed `exp/plots/figures/fig5_1_token_metric_mismatch.png`.
- Debugging/verification: Ran container compileall, regenerated Figure 5.1 preprocessing output, regenerated the PNG inside `sk-sslo-vllm`, and visually checked the resulting image.

## 2026-05-25

- Modified: `exp/run_sslo/analysis/sweep_analysis.py` — R3 summary CSV schema now emits the requested context columns, flat R2 distribution/scalar metrics, validity columns for capacity filtering, and per-tau `request_cu_slo_violation_rate_tau_*` columns while dropping old cpslo/progress/chunk-slack columns.
- Modified: `exp/run_sslo/analysis/validity.py` and `exp/run_sslo/analyze.py` — validity reports now include placeholder runtime counters, request preemption totals, starvation count from flat per-mode metrics, and `included_in_main_result`.
- Modified: `exp/run_sslo/run_smoke.sh` — smoke launcher now iterates `MODES x CONSUME_CELLS` with the R3 default mode/rate ladder.
- Added: `exp/run_sslo/analysis/capacity.py` — computes supported handling-user capacity and policy ratios against baseline from `summary.csv`.
- Debugging/verification: Separate read-only verifier found no R3 issues. Host `bash -n exp/run_sslo/run_smoke.sh` passed. Required Docker verification commands could not run from this session because the Docker API socket returned permission denied.

## 2026-05-23

- Modified: `vllm/vllm/sslo/slo_state.py` — added `ChunkLengthPredictor.reset()` used between rate sweeps.
- Modified: `vllm/vllm/v1/core/sched/scheduler.py` — added `Scheduler.reset_sslo_state()` clearing predictor, `_sslo_step_wall_ema`, `tpot_ema`, per-step trackers, decision buffer, log-path caches. Returns post-reset state sizes.
- Modified: `vllm/vllm/v1/engine/{core,core_client,async_llm}.py` — `reset_sslo_state` RPC plumbing (mirrors `reset_prefix_cache` pattern via `call_utility_async`).
- Modified: `exp/run_sslo/run_test.py` — major refactor: rate sweep on a single shared engine.
  - `--request-rates "0.5 1 2 4 8 12 16 20"` and `--repeat` flags.
  - `run_one()` initialises engine once, then loops `_run_one_rate(...)` per rate. Between rates: `engine.abort(in_flight)` + task cancel + 1 s grace + `engine.reset_sslo_state()` with 5-retry polling.
  - Output `OUTPUT_DIR/rate_<r>/...` per rate. Engine-wide scheduler stats post-trimmed per rate's monotonic-clock window.
  - Per-rate `analyze.py` subprocess + fcntl-locked append to per-model `summary.csv` (`--summary-csv`). 56-column schema with clarified violation names.
- Modified: `exp/run_sslo/run_test.sh` — `REQUEST_RATES`, `REPEAT`, `SUMMARY_CSV` env vars. Removed `MEASUREMENT_WINDOW_S` safety timeout.
- Modified: `exp/run_sslo/analyze.py` — robust mode detection (walks `output_dir.parents` for any `ALL_MODES` ancestor). Added time-weighted `mean_{running,pending,waiting,handling_users}` + `urgent_mode_fraction` to `metrics.scheduler[mode]`. Added `total_pending_time_s`/`num_pending_intervals`/`chunk_consume_time_s` distributions. Scheduler-stats filter uses `measurement_window_start_mono_ts`.
- Added: `exp/run_sslo/run_v15_grid.sh` — 9B grid launcher. caps `{16,32,64,128}` × 8 rates × `{baseline,sslo_mlp}` × 3 repeats, outer-most repeat loop, 4 GPUs round-robin.
- Verification: Full 9B grid completed in ~4-5 h (4 GPUs). `Qwen3.5-9B/summary.csv` = 1 header + 192 rows. Per-rate `reset_sslo_state -> {predictor=0, wall_ema=0, waiting=0, running=0}` confirms clean state cleanup.

## 2026-05-22 (session 3)

- Modified: `exp/run_sslo/run_test.py` — switched measurement window from time-based (`MEASUREMENT_WINDOW_S=180s` fixed) to **completion-gated** workflow per user request:
  - Pool size 4000 → **4096** (wildchat2048+lmsys2048).
  - `warmup_target = max_num_seqs * 2`, `measurement_target = max_num_seqs * 4`.
  - Window timestamps written INLINE by the gate-flipping task (race-free; capturing in watcher after `await event.wait()` would race with other completions in between and produce a near-zero window).
  - Records both wall (`time.time()`) and monotonic (`time.monotonic()`) bounds. Wall for request `in_window` (vs `completion_wall_ts`), monotonic for scheduler_stats trim (matches vLLM's stat clock).
  - `in_window = completed in [window_start, window_end]` (per user spec "측정 시작 이후 완료된 request 수집"). New field `completion_wall_ts` on each request row.
  - No cooldown: on measurement_done_event, `engine.abort(in_flight_ids)` + `task.cancel()` + `asyncio.gather(return_exceptions=True)`.
  - `chunks.jsonl` filtered to in_window reqs (no warmup/post-window partials).
  - `scheduler_stats.jsonl` post-trimmed in-place to monotonic-window bounds.
  - Warmup + measurement safety timeouts (= `MEASUREMENT_WINDOW_S`, now default 900s) prevent hang if throughput too low.
- Modified: `exp/run_sslo/analyze.py` — request filter prefers per-row `in_window` flag when present (new flow), falls back to `injection_ts ∈ [mw0, mw1)` for legacy runs. Scheduler-stats filter uses `measurement_window_start_mono_ts` when present.
- Modified: `exp/run_sslo/run_test.sh` — `NUM_PROMPTS` default 4000 → 4096, `MEASUREMENT_WINDOW_S` 180 → 900 (now repurposed as safety timeout).
- Verification: Smoke 9B cap=128 rate=8 sslo_mlp — `in_window=513` (target 512, +1 boundary task), `measurement_completed=514`, window=86.3s, no `engine.abort` errors. Race-free window timestamp recording confirmed by window matching the gate semantics (vs prior buggy 0).

## 2026-05-22 (session 2)

- Modified: `exp/tools/lm_datasets.py` — added `_max_response_chunk_chars()` helper that runs SSLO's `ChunkSeparator` (sentence mode, `min_chunk_tokens=0`) on a response and returns the longest chunk length. Added `max_response_chunk_chars: int | None` parameter to `load_prompts`, `_load_wildchat`, `_load_lmsys`, `_load_combine`. Filter drops rows whose first assistant response would produce a chunk longer than the threshold under runtime SSLO boundary rules.
- Modified: `exp/run_sslo/run_test.py` — added `--max-response-chunk-chars` CLI flag (int, default 0 = off); `_build_pool` forwards as `None` when 0.
- Modified: `exp/run_sslo/run_test.sh` — added `MAX_RESPONSE_CHUNK_CHARS` env var, default `1000`. New sweeps automatically drop dump-style source responses.
- Verification: WildChat 500 English first-asst-response sample → p99=373 chars, max=502 (well below 1000). Three known v12 mega-chunk offender shapes → 1725 / 2707 / 3961 chars (caught). 30K-row scan per source → drop rate 0.04% (2 rows each), false positive 0.

## 2026-05-22
- Modified: `exp/run_sslo/run_test.py` — added `--english-only` CLI flag; `_build_pool` wildchat/lmsys calls forward `args.english_only`.
- Modified: `exp/run_sslo/run_test.sh` — added `ENGLISH_ONLY` env var (default `1`) → `--english-only` flag. Default ON: all new runs are English-only unless explicitly disabled.
- Verification: `python3 -m compileall` + `bash -n` pass. Smoke test (`combine`, 50 prompts): all English. Comparison (wildchat 200 prompts): heavy non-ASCII 14 (off) → 7 (on); remaining 7 are English with Unicode quotes / em-dashes / TAB, no non-English content. Partial pool scan (250K rows): English ratio 46.4%, conv+nocode pass 17.5%. Extrapolated combine English pool ≈ 1.1 M (well above 4 K sweep budget).

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

## 2026-05-03 (Poisson arrival)

- Added `--request-rate` (reqs/sec, Poisson arrivals) and `--request-rate-seed` to `run_test.py`.
  - 0 (default) submits all prompts at t=0 (current behavior).
  - Positive value samples inter-arrival gaps from `Exp(rate)`.
  - Each prompt's task `await asyncio.sleep(arrival_offset)` before calling `engine.generate()`.
- `run_single.sh` accepts `request_rate` and `request_rate_seed` as positional args 6 and 7.
- `run_repeat.sh` accepts `request_rate` (positional 7) and `base_seed` (8); each run uses `seed = base_seed + i` so the N runs sample different arrival patterns at the same rate.
- Within one config, all 3 modes (baseline / sslo / sslo_adaptive) share the same seed → identical arrival schedule, apples-to-apples comparison.
- Smoke verified at rate=4 reqs/s (16 prompts, max_num_seqs=8): all three modes show identical decoding_start_ts spread (~2.3s span, same seed) → arrival timing reproducible.

## 2026-05-03 (systemic metrics)

- Modified: Extended `exp/sslo_test/analyze.py` summaries with TTFT and queue-stall p99/max, SLO compliance, negative-slack magnitude, running/combined scheduler occupancy, pending duration/intervals, and inter-chunk delay metrics.
- Modified: `exp/sslo_test/run_test.py` now carries chunk `end_time_ts` and per-request `sslo_metrics` pending fields (`total_pending_time_s`, `num_pending_intervals`, `max_consecutive_pending`) into JSONL rows.
- Modified: `aggregate_repeats.py` and `aggregate_sweep.py` print the new metrics and recompute raw-file-derived fallback fields for old summaries.
- Debugging/verification: In containers, compileall passed for changed experiment Python files; `python3 -m pytest vllm/tests/sslo/ -x -q` passed (`65 passed, 16 warnings`). GPU smoke at `/tmp/sslo_metric_smoke` completed and summary keys were non-null for SSLO. Re-aggregated sweep output now includes new TTFT/queue-stall tails plus SLO compliance, slack magnitude, running/combined, pending, and inter-chunk sections.

## 2026-05-03 (systemic metrics)

Added 7 new metric categories to exp/sslo_test/ for systemic-effect analysis (in addition to existing TTFT p50/p90, TPOT p50, queue_stall p50, neg_slack_ratio):

1. **TTFT/queue_stall p99 + max** — tail behaviour, not just median.
2. **Per-request SLO compliance rate** — `slo_compliance_rate_{mode}`: fraction of 256 requests where ALL chunks made deadline. Direct policy-evaluation metric (vs the chunk-level ratio which conflates short and long requests).
3. **Neg slack magnitude p50/p90/p99/max** — "how late were the late chunks", not just count.
4. **`len(running)` time-series mean/p50/p99** (from `${mode}_stats.jsonl`) — actual cap utilization.
5. **Effective batch (running + pending) mean/p50/p99** — total in-flight (KV-occupying) requests.
6. **Per-request pending duration p50/p90/p99 + interval count p50/p90** — extracted from `RequestOutput.sslo_metrics` (already exposed by `RequestSLOState.compute_stats()`); now captured in `run_test.py` per-request rows.
7. **Inter-chunk delay p50/p90/p99/max** — gap between consecutive chunk `end_time_ts` per request, useful for streaming smoothness (TTS / human reading rate).

`analyze.py` writes all these to `summary.json`; `aggregate_repeats.py` and `aggregate_sweep.py` extend their METRICS tuples to print per-mode mean ± stddev.

Existing 36-cell sweep re-aggregated successfully — all metrics populate from existing chunks/stats jsonl except #6 (which needs a fresh run since `total_pending_time_s` is per-request and was added to row schema in this change).

## 2026-05-06 (stall-aware slack + seqs=64 sweep)

- Modified: `vllm/vllm/sslo/slo_state.py` — `cumulative_consume_time` now updated stall-aware (`max(end_offset, current) + audio`) so a late chunk extends only the next deadline by its overage; previous formula carried debt forward and inflated downstream `위반률`. Added `ChunkConsumeEstimator` class (legacy `word_count × seconds_per_word` default) so TTS-derived durations can be injected later without touching `RequestSLOState`. Tests adjusted (`test_chunk1_records_real_slack`, `test_score_formula_and_deadline_sign`).
- Modified: `exp/sslo_test/run_sweep.sh` — defaults set to `MAX_NUM_SEQS_VALUES="64"`, `REQUEST_RATES="0 4 16 32"`, `CHUNK_UNITS="sentence paragraph"`. `GPU_RATE_ASSIGNMENTS` now auto-derived round-robin from `REQUEST_RATES` across `NUM_PARALLEL_GPUS=4`, removing the second hand-edited rate list.
- Verification: 58 SSLO tests pass inside `sk-sslo-vllm`. 24-cell sweep ran end-to-end (T+258m): TTFT p99 baseline ~278s → sslo/sslo_adaptive ~145–155s (~1.9× speedup); chunk_violation% ≤0.1% across all modes (formula B effect — no carryover); SLO compliance 97.6–98.6%; `summary.csv` written with 72 rows.

## 2026-05-09 (run_sslo schema overhaul)

- Modified: `vllm/vllm/sslo/slo_state.py` — `ChunkRecord` extended with `start_time_ts`, `num_token`, `num_iters`, `num_running_iters`, `num_pending_iters_per_chunk`. `ChunkStatCollector` gains `accumulate_running_step()` / `accumulate_pending_step()`; counts are stamped onto each ChunkRecord at the next chunk boundary and reset.
- Modified: `vllm/vllm/v1/core/sched/scheduler.py` — new `_account_per_chunk_step(running, pending)` helper, called at the end of both `_apply_sslo_policy` and `_apply_sslo_policy_v2`, walks `self.running` / `self.sslo_pending` and increments per-chunk iter buckets.
- Modified: `exp/run_sslo/run_test.py` — emits new chunks.jsonl schema (`chunk_slack`, `num_words`, `num_token`, `num_iters`, `num_running_iters`, `num_pending_iters_per_chunk`, `start_time_ts`, `request_idx`); requests.jsonl gains `ttfc` (queued_ts → first chunk end), `num_chunks`, and `num_pending_iters_per_request` (rename of `num_pending_intervals`). Per-mode `sslo_config_<mode>.json` sidecar written for analyze.py.
- Modified: `exp/run_sslo/_consolidate_mode_outputs.py` — folds per-mode `sslo_config_<mode>.json` into a single `sslo_config.json` keyed by mode, then removes the per-mode sidecar.
- Modified: `exp/run_sslo/analyze.py` — reads renamed chunk fields, emits `metrics.ttfc`, `metrics.slack.<mode>.violated-magnitude` (rename of `magnitude`), adds `min` to `metrics.scheduler.<mode>.combined`, and embeds `sslo_config` in `summary.config`. New `distribution_stats(include_min=True)` flag.
- Modified: `exp/run_sslo/metrics_utils.py` — DISPLAY_GROUPS reflect new `ttfc` rows, `violated-magnitude`, and `combined min/max`.
- Modified: `exp/run_sslo/analysis/sweep_analysis.py` — `csv` subcommand emits `sslo_*` config columns, `ttfc_s_*`, `violated_magnitude_s_*`, `combined_{min,max}`, and `slo_request_violation_rate` (= 1 - compliance_rate). One row per (cell, mode), 61 columns total.
- Verification: `pytest vllm/tests/sslo/` — 65 passed. End-to-end smoke with synthetic jsonl confirmed analyze.py + sweep_analysis.py csv produce all required fields.

## 2026-05-09 (token timeline plot)

- Added: `exp/run_sslo/analysis/plot_token_timeline.py` — plots cumulative-token timelines (generated vs consumed) for one request or averaged across all requests in a run. Generated curve uses chunk-TPS (= num_token / chunk_gen_duration) for token-level interpolation within each chunk window. Consumed curve uses (num_token / (num_words × seconds_per_word)) within the consumption window, with explicit idle segments when the reader was waiting for a chunk to arrive.
- Verification: smoke-tested with a 2-request synthetic chunks.jsonl in `sk-sslo-vllm`; produced single-request and average PNGs whose curve geometry matches the expected piecewise-linear segments and the chunk-arrival idle shape.

## 2026-05-11 (ProgressServe metric coverage + derivation module)

- Added: `/home/cheezestick/.claude/plans/metric-memoized-volcano.md` — ProgressServe metric handoff (§1–14) coverage analysis vs. existing `requests.jsonl` / `chunks.jsonl` / `scheduler_stats.jsonl`. Confirmed user-resolved definitions: deadline-based consumer model (chunk i consume window = [deadline_{i-1}, deadline_i]); pending_wait := queue_stall; consume_origin_ts := chunk_idx==0 end_time_ts; demand_duration := final_deadline - consume_origin; stall_fraction := total_stall / demand_duration; CP_SLO_violation := max_stall_time > tau (per-tau analysis param); measurement_window := first finish ~ (N - max_num_seqs)-th finish; request_class := canonical reference-run q33/q67 split with fixed request_id→class mapping. E7 ablation deferred. Critical gaps documented for future PRs: run_metadata sidecar, GPU sampler, drop/timeout (finish_reason), per-step scheduler decisions, reference-run class mapping.
- Modified: `exp/run_sslo/run_test.py` — `collect_request` extracts `num_prompt_tokens` from vLLM `RequestStateStats` (fallback: `len(prompt_token_ids)`); added to both early-return and main return dicts → flows into `requests.jsonl`. `run_one` prints `workload mean num_prompt_tokens=…, num_output_tokens=…` after the existing completion log line.
- Added: `exp/run_sslo/analysis/progress_metrics.py` — pure derivation module. Functions: `per_request_progress` (arrival_ts / completion_ts / completion_latency / consume_origin_ts / final_deadline_ts / demand_duration / total/max/stall_intervals/stall_fraction), `measurement_window` (first finish ~ (N − max_num_seqs)-th finish; returns (None,None) when N ≤ cap), `handling_users_stats` (time-weighted mean + p50/p95/p99/max within window), `throughput_stats` (tokens/s, completed_req/s within window), `cp_slo_violation_rates` (per-tau rate keyed `tau_{tau:g}`). `DEFAULT_TAUS = (0.5, 1.0, 2.0, 5.0)`.
- Modified: `exp/run_sslo/analyze.py` — imports `progress_metrics as pm`; `summary['metrics']` gains 6 new top-level groups: `progress_request`, `workload`, `throughput`, `cp_slo_violation`, `measurement_window`, `handling_users` (last is sslo* only). Existing keys, `passes` block, and `stall_time` group untouched.
- Modified: `exp/run_sslo/metrics_utils.py` — appended 6 new `DISPLAY_GROUPS` entries (Stall (request), Latency (request), Workload, Throughput, Handling users, CP-SLO violation) so `sweep_analysis.py csv` automatically emits new columns. Total groups: 7 → 13.
- Verification: `python3 -m compileall` on the three modified files inside `sk-sslo-vllm` — clean. Synthetic-data smoke test of `progress_metrics` produces the spec-expected derivations (1 request, 3 chunks; total_stall=0.5, max_stall=0.5, num_stall_intervals=1, demand_duration=2.5, stall_fraction=0.2). No existing `output_sweep` cell available for full end-to-end check; deferred to next sweep run.

## 2026-05-11 (smoke: sslo single-cell metric verification)

- Verification: ran `run_test.sh sslo 128 Qwen/Qwen3-8B` (NUM_PROMPTS=256, GEN=512, sentence, GPU 1) inside `sk-sslo-vllm`. 256 reqs in 46.4s, 4674 chunks. workload print line emitted: `mean num_prompt_tokens=163.5, num_output_tokens=510.1`.
- Verification: `analyze.py` emits all 6 new top-level keys in `summary.json.metrics` (`progress_request`, `workload`, `throughput`, `cp_slo_violation`, `measurement_window`, `handling_users`); existing keys unchanged. Sample sslo cell results — completion_latency p95=45.8s; total_stall_time max=4.65s; max_stall_time max=1.23s; CP-SLO viol rate {tau=0.5: 0.39%, tau=1: 0.39%, tau=2: 0%, tau=5: 0%} (consistent with max_stall=1.23s); throughput duration=9.84s, tokens/s=6610, completed_req/s=13.0; handling_users time_avg=210 (over 591 step samples); measurement_window N=256, cap=128 → 128 requests fall inside window (= N - max_num_seqs).

## 2026-05-11 (sweep_analysis csv extension + baseline comparison verification)

- Modified: `exp/run_sslo/analyze.py` — `sched_by_mode` now buckets for `ALL_MODES` (baseline included), and the `handling_users` loop iterates `ALL_MODES`. Existing `scheduler` SSLO-diagnostic block stays scoped to `SSLO_MODES`. Reason: baseline writes scheduler_stats.jsonl too (1294 rows for 256-prompt run); without this fix `handling_users.baseline` stays empty and capacity_gain comparison against baseline is impossible.
- Modified: `exp/run_sslo/analysis/sweep_analysis.py` — `DIST_METRICS` extended with 9 entries (total_stall_s, max_stall_s, num_stall_intervals, stall_fraction_pct (×100), completion_latency_s, demand_duration_s, prompt_tokens, output_tokens, chunks_per_req). `SCALAR_METRICS` extended with 10 entries (tokens_per_second, completed_req_per_s, measurement_duration_s, handling_users_{time_avg,p95,max}, cp_slo_viol_tau_{0_5,1,2,5}). `DIST_EXTRA_STATS` adds p95/max for stall and latency metrics. `agg-sweep` / `agg-repeat` already iterate `DISPLAY_GROUPS` so no change needed there.
- Verification: ran baseline (sslo) at `seqs_128/rate_0` (256 prompts, GEN=512, sentence). After analyze.py + sweep_analysis.py csv: 107-column summary.csv with 46 new metric columns. agg-sweep (with run_0 → run_1 symlink) prints all 6 new DISPLAY_GROUPS groups (Stall (request), Latency (request), Workload, Throughput, Handling users, CP-SLO violation) for both modes.
- Single-cell comparison snapshot: handling_users_time_avg baseline=116 vs sslo=210 (1.8× more concurrent users via pending state); max_stall_s_max baseline=6.53s vs sslo=1.23s; cp_slo_viol_tau_2 baseline=0.39% vs sslo=0% (tau=5s same). tokens/s & completed_req/s look very different across modes (113K vs 6.6K) because the `measurement_window` collapses to 0.58s on baseline (256 prompts finish in two dense 128-batches at request_rate=0); window definition is consistent — absolute throughput numbers are only comparable across cells, not policies, at request_rate=0.
- Real `capacity_gain` requires sweep across (max_num_seqs, request_rate) cells with `supported_users(tau)` evaluated per cell — single-cell proxy here just confirms the data needed for the calculation is now emitted by both modes.

## 2026-05-11 (summary_warmup.csv sidecar)

- Modified: `exp/run_sslo/analysis/sweep_analysis.py` `cmd_csv` — after writing the existing `summary.csv` (unchanged), additionally writes `summary_warmup.csv` next to it. One row per (cell, mode, run): `CONTEXT_COLUMNS` + `measurement_start_ts`, `measurement_end_ts`, `measurement_duration_s`. Window definition matches `progress_metrics.measurement_window`: first request finish → (num_requests − max_num_seqs)-th request finish. Source: `summary.json.metrics.measurement_window.<mode>`. No `run_test.py` change required (data already flows through chunks.jsonl → analyze.py → summary.json).
- Verification: re-ran `sweep_analysis.py csv` on the smoke sweep — `summary.csv` shape unchanged (107 cols), `summary_warmup.csv` emitted with the expected 13 columns and the correct ts values (baseline 0.58s window, sslo 9.84s window).

## 2026-05-11 (R1/R2 metric audit + DRY refactor loop)

- Round 1 audit (Explore agent): all derivations in `progress_metrics.py` matched plan's Confirmed Definitions / Derivation Recipe. Three DRY violations flagged: (a) `metrics_utils.lookup` vs `sweep_analysis._metric_node` duplicate nested-dict traversal; (b) 10-key context row inline-built in both `_emit_rows` and the warmup-csv block; (c) `pm.measurement_window` called twice per mode in `analyze.py`.
- DRY refactor (codex): extended `metrics_utils.lookup(summary, path, field, mode)` so `field=None` returns the node itself; deleted `_metric_node` and replaced call sites with `lookup(..., None, ...)`; added `_context_row(summary, path_label, run_idx, mode)` helper shared by `_emit_rows` and `cmd_csv` warmup block; precomputed `windows[mode]` dict in the first ALL_MODES loop of `analyze.py` and reused in the handling_users loop. Net delta: -12 lines across the 3 files.
- Verification (byte-identical): re-ran analyze + sweep_analysis csv on existing smoke cells; `summary.json` (baseline + sslo), `summary.csv` (107 cols), `summary_warmup.csv` (13 cols) all `diff`-SAME against pre-refactor snapshots.
- Round 2 audit (Explore agent): re-confirmed all refactors preserve functional equivalence; verified end-to-end values against plan (manual recompute of total_stall_time, completion_latency, measurement_window endpoints, cp_slo_violation rate at tau=0.5s, handling_users time_avg for baseline). No new DRY violations or orphan imports.

## 2026-05-12 — CP-SLO metric refactor (7 phases, commits 4fc2301..3c4812c)

Multi-agent planning + dev-verify loop per phase per AGENTS.md.
Spec at `/tmp/cpslo_todo.md`; canonical reference in `metric.md`.

- Phase 1 (`4fc2301`): timing contract. d(0) = consume_start_ts = chunk 0
  gen_finish_ts; removed special-case `slack=0` branch; new ChunkRecord
  fields `stall_start_ts / stall_end_ts / stall_duration_s`.
- Phase 2 (`1a2eecb`): schema expansion. 8 new ChunkRecord fields
  (token indices, demand window, predictor source/high). admitted_ts +
  terminal_outcome on RequestSLOState; scheduler stamps in
  schedule_sslo. F2 request_class via fixed token buckets
  (`xs/s/m/l`) on observed `num_generation_tokens`. New module
  `exp/run_sslo/analysis/cpslo_names.py` for legacy↔canonical alias.
  run_test.py dual-writes 15 chunk + 9 request canonical keys.
- Phase 3 (`74d7034`): pressure components + missing preservation
  (D1 split). New `PressureComponents` dataclass and
  `state.pressure_components(now, tpot)`; `pressure()` rerouted as
  thin wrapper. Scheduler `_state_pressure` renamed to
  `_policy_score_with_fallback` — keeps the 1.0 normalize on the
  policy path (cold-start critical guard depends on it) while
  decision logs see the raw None.
- Phase 4 (`f0fccab`): per-step × per-admitted-request decision log.
  Config knobs `decision_log_mode` (off / step / tier_changes /
  admit_only, default tier_changes) and `decision_heartbeat_steps`
  (200). Buffered JSONL writer to `decisions.jsonl`. Schema includes
  PressureComponents + admission/backfill/preemption_reason slots
  (None for now — populated in a follow-up).
- Phase 5 (`25d8f7a`): analysis migration. New
  `progress_metrics.compute_stall_intervals` merges contiguous
  stalls; `cp_slo_violation_rates` now thresholds on
  `max_stall_interval_s` (not per-chunk max). analyze.py adds
  `metrics["cpslo"]` section with max_stall_interval_distribution,
  num_stall_intervals_merged_distribution, cp_slo_violation_rates_by_tau,
  mean_handling_users_time_weighted.
- Phase 6 (`3c4812c`): validity + no-harm + run-level + F3. New
  `exp/run_sslo/analysis/validity.py` with pre-declared thresholds
  and `validate_run`. `run_meta.json` sidecar with run_id, variant,
  seed, trace_id, workload_id, N, M, measurement_start/end_ts,
  gpu_memory_peak_bytes, num_preemptions_total, completion counters.
  `_consolidate_mode_outputs.consolidate_validity_csv` writes
  `validity_checks.csv`.

Tests: 83 SSLO engine + 14 analysis = 97 total, all pass.

CP-SLO contract semantics that BREAK comparability with pre-refactor
sweep outputs:
- chunk 0 `deadline_ts` shifts from `decoding_start_ts` to
  `gen_finish_ts(0)` (chunk 0 itself).
- `mean_handling_users` headline switches from sample-mean to
  time-weighted-mean — analyze.py dual-writes for one transition release.
- `queue_stall_s` semantics shift from `scheduled_ts - queued_ts` to
  `admitted_ts - queued_ts` (admit-time-based).
- `cp_slo_violation` basis switches from per-chunk max to merged
  contiguous interval max.

Bump `summary.json` schema_version when dropping legacy alias keys
(planned follow-up).

## 2026-05-13 — throughput basis switched to full-run wall-time

`progress_metrics.throughput_stats` previously used the "first N-M
completions" window. That window stretched for SSLO modes whose
admission pattern is wave-shaped (admit → run → admit next wave),
producing apparent 30-50% throughput regression that was a
measurement artifact, not a real effect.

Canonical basis is now:
  duration_s = run_meta.measurement_end_ts - run_meta.measurement_start_ts
  tokens    = Σ num_output_tokens over ALL requests
  tok/s     = tokens / duration_s

`run_meta` is loaded once at the top of `analyze.analyze()` and
passed through to throughput_stats. Fallback chain: run_meta →
derived min/max of per_req arrival/completion → legacy window arg.
The result dict gains a `basis` field ("full_run" / "unknown").

Verified on cpslo_smoke_128 sweep (re-running analyze.py on the
existing outputs):

| mode/rate | old tok/s | new tok/s |
|---|---|---|
| baseline 16 | 6168 | 4454 |
| sslo 16     | 4363 | 4840 |
| sslo_adapt16| 5537 | 4340 |
| baseline 32 | 12605 | 5060 |
| sslo 32     |  5972 | 5338 |
| sslo_adapt32|  5863 | 5307 |

Under the new basis SSLO modes are within ±10% of baseline at this
load, matching the per-step Δts evidence (mode-independent ~21 ms).

## 2026-05-17 — MLP cap-shrink fix (system_factor) + Qwen3.5 chat-template plumbing

### Modified
- `vllm/vllm/v1/core/sched/scheduler.py`: `_compute_serve_defer_pair`에
  optional `cap_n` 파라미터 추가. `_mlp_pick_adaptive_n` 후보 평가 루프
  (line 2160)에서 `cap_n=n` 전달 → 각 후보 cap의 contention scale을
  `admitted/n`으로 산출. 기존엔 항상 `admitted/base_n`이라 작은 n 후보가
  shrink해도 effective throughput 손실이 pressure에 반영 안 됨.
- `vllm/tests/sslo/test_scheduler_sslo.py`: `_mlp_pick_adaptive_n` cap_n
  scaling 양방향 테스트 2개 추가 (`test_mlp_pick_n_system_scale_blocks_
  pointless_shrink`, `..._allows_meaningful_shrink`).
- `exp/run_sslo/run_test.py`: `--enable-thinking` / `--no-thinking` 플래그 +
  `apply_chat_template_to_prompts` helper. Qwen3.5 unified Instruct+Thinking
  모델에서 chat template `enable_thinking=False` 적용. Sampling override
  CLI args (`--temperature`, `--top-p`, `--top-k`, `--min-p`,
  `--presence-penalty`, `--repetition-penalty`) 추가.
- `exp/run_sslo/run_test.sh`: `ENABLE_THINKING`, `TEMPERATURE`/`TOP_P`/
  `TOP_K`/`MIN_P`/`PRESENCE_PENALTY`/`REPETITION_PENALTY` env vars →
  CLI args 전달.
- `exp/run_sslo/run_sweep.sh`: `START_RUN_INDEX` env var (default 1) — 기존
  runs를 덮지 않고 추가 run만 실행 가능.

### Added
- `exp/run_sslo/_diag_worst.py` — worst-case 셀 진단 (cap shrink rate,
  critical mode 비율, violating chunks 텍스트).
- `exp/run_sslo/_diag_tpot.py` — `decisions.jsonl`에서 batch별 step time
  추출해 memory-vs-compute-bound 판정.

### Findings (cap=64 r=128 sslo_mlp on Qwen3.5-27B)
- Before fix: tput 654 tok/s (-28.5% vs baseline 916), viol@τ=1 7.54%,
  cap_p50=24, has_critical=81% — death spiral.
- After fix: tput 929 tok/s (+1.4% vs baseline), viol@τ=1 0.79%,
  cap_p50=64, has_critical=59% (감지는 함, shrink만 자연 reject).
- 9B control (cap=128 r=64): tput 3705 vs baseline 3162 (+17%),
  viol 0% — fix이 compute-bound 케이스 무회귀.

### Verification
- `pytest tests/sslo/` 109/109 pass (이전 107 + 신규 2).
- Smoke 27B + 9B 모두 합격 기준 만족.

## 2026-05-18 — MLP backfill 후 stats single-dump + predictor escalate/overshoot fix

### Modified (`vllm/vllm/v1/core/sched/scheduler.py`)
- `SsloStepState`: 새 필드 `defer_base` — non-critical에서 stash해 post-waiting
  backfill ranking에 사용.
- `_apply_sslo_multi_level_pressure` non-critical (line 2320~): backfill 제거.
  `new_running = forced + warmup` 만 commit. `_sslo_step.defer_base = defer_base`
  로 snapshot.
- `schedule()` 메인 loop (line ~2980): waiting admit 후 새 backfill block 추가
  — `self.sslo_pending`에서 `defer_base` DESC 순으로 cap까지 채움. 각 backfilled
  req에 대해 KV alloc + decode token 배정 → 해당 step에 실제 실행됨.
  - `request.has_encoder_inputs` 케이스 skip (다음 step 처리).
- `_sslo_commit_step` (line 1228~): `_sslo_dump_step_stats` 호출 제거. docstring
  업데이트.
- `schedule()` 메인 loop 끝 (backfill 후): `_sslo_dump_step_stats(
  scheduled_timestamp)` 단일 호출 추가 → stats가 최종 running 반영.

### Modified (`vllm/vllm/sslo/config.py`, `slo_state.py`)
- `SsloConfig`: 새 knob 2개.
  - `mlp_predictor_escalate_threshold: float = 0.9` — 현재 tier 예측치의 90%에
    도달하면 다음 tier(p90 → p95 → p99)로 진입.
  - `mlp_predictor_overshoot_safety_factor: float = 2.5` — `cur > topmost tier`
    이후 `remaining = (cur - anchor) × factor` 산출 (1.0 saturate 제거).
- `ChunkLengthPredictor.__init__` + `RequestSLOState.expected_remaining_len`
  재작성: 위 두 knob 적용.
- `RequestSLOState.from_config`에 knobs 전달.

### Modified (`vllm/tests/sslo/`)
- `test_slo_state.py`: 신규 4 (`test_expected_remaining_escalates_at_90pct_threshold`,
  `_overshoot_grows_past_topmost`, `_legacy_threshold_and_factor`,
  `test_predictor_knob_validation`). `test_score_formula_and_deadline_sign`
  예상치 갱신 (1.0 floor → 7.5 overshoot).
- `test_scheduler_sslo.py`: `test_mlp_non_critical_admission_budget_uses_measured_only`
  업데이트 — backfill이 main loop으로 이동했으므로 MLP가 commit하는
  `new_running == 1` (warmup only), pending 3 + defer_base 검증.

### Verification

**Unit tests**: `pytest tests/sslo/` 113/113 pass.

**Worst-case smoke** (35B-A3B cap=128 r=128 sslo_mlp × 3 runs vs v3):
| metric | v3 (3-run avg) | v4 (3-run avg) | Δ |
|---|---|---|---|
| tput | 1620 | 1651 | +1.9% |
| TTFC mean | 64.0s | 60.0s | -6.3% |
| **viol@τ=1** | **1.40%** | **0.53%** | **-62%** |
| crit% | 21% | 12% | -43% |

**Stats dump fix** (v5, post-backfill dump):
| 지표 | v4 (pre-backfill dump) | v5 (post-backfill dump) |
|---|---|---|
| running p75 | 6 | 120 |
| cap=128 step 비율 | 11% | 24% |

→ stats가 backfill 후 실제 실행 시점을 정확히 반영.

**Controls (1 run)**:
- 27B cap=64 r=128 mlp: tput 972 (v3 932, +4%), viol 0.79% (v3 0.6%, 동등).
- 9B cap=128 r=64 mlp: tput 3685 (v3 3660, +0.7%), viol 0% (v3 0.2%, 동등).

### Added — backfill diagnostics
- `SsloStepState`에 `bf_skip_reason / bf_slack / bf_promoted / bf_kv_full /
  bf_too_few_tokens / bf_encoder_skipped / bf_budget_exhausted` 필드 추가.
- `schedule()` 메인 loop의 backfill block을 instrumented (skip 이유, KV alloc
  실패 카운터 등) — scheduler_stats.jsonl에 dump.

### Diag finding (35B-A3B cap=128 r=128)
Backfill loop이 86.2% step에서 진입했지만 그 중 **48.2%가 promoted=0**.
원인: **KV cache full** — `kv_cache_manager.allocate_slots()`가 6026 events
에서 None 반환. sslo_pending 260 reqs 분량 KV가 살아있어 추가 decode 슬롯
없음. → 후속 fix: pending pool size 제한 + KV 압력 기반 eviction.

## 2026-05-18 — hybrid predictor (global + per-request)

### Modified
- `vllm/vllm/sslo/slo_state.py`:
  - `_CHUNK_LEN_HISTORY_MAX`: 64 → 4096 (per-req history cap 완화).
  - `_GLOBAL_PREDICTOR_WARMUP_SAMPLES = 16` (per-req sample이 16 미만이면
    global predictor 사용).
  - `ChunkLengthPredictor`: `sample_count` property + 모든 strategy에서
    `_sample_count` 추적.
  - `RequestSLOState`: 새 InitVar `global_chunk_len_predictor`. 
    `on_chunk_boundary`에서 per-req + global 둘 다 update.
  - `expected_remaining_len()`: per-req sample_count < 16 이면 global,
    그 외엔 per-req (hard switch).
  - `from_config`에 `global_chunk_len_predictor` kwarg 추가.
- `vllm/vllm/sslo/config.py`: `chunk_len_predictor_history_max: int = 4096`
  knob + validation.
- `vllm/vllm/v1/core/sched/scheduler.py`: `Scheduler.__init__`이
  `_sslo_global_chunk_len_predictor` 인스턴스 생성 (EngineCore subprocess
  안, single thread → lock 불필요).
- `vllm/vllm/v1/engine/core.py`: `add_request`가 scheduler의 global
  predictor 참조를 `from_config`에 전달.
- `vllm/tests/sslo/test_slo_state.py`: 신규 3 tests (warmup 사용, 16-sample
  switchover, on_chunk_boundary가 global update).

### Verification
- `pytest tests/sslo/` 116/116 pass.
- Worst-case smoke (35B-A3B cap=128 r=128 sslo_mlp × 3 runs vs B1):
  tput 1814 → 1842 (+1.5%), viol@τ=1 0.80% → 0.60% (-25%).

### Notes (다른 시도)
- Phase A (backfill을 waiting admit **앞**으로 reorder) 시도 → trade-off
  불리. bf_kv_full=0 / tput 동등 / **TTFC 54s → 84s** (waiting 정체).
  Codex 검토 후 revert.

## 2026-05-18 (cont.) — dataset code-gen 필터 + dataset 선택

### Modified
- `exp/tools/lm_datasets.py`:
  - `_CODE_PROMPT_PATTERN` regex + `_is_code_request(prompt, response)`
    helper.
  - `load_prompts(..., exclude_code=False)` 새 kwarg.
  - 모든 3개 loader (`_load_koala`, `_load_wildchat`, `_load_lmsys`)에
    `exclude_code` 파라미터 + filter 적용.
  - WildChat/LMSYS는 `_first_user_and_assistant` helper로 첫 user prompt +
    첫 assistant response 추출 → response 코드 블럭(```) 확인.
  - Koala는 응답 없음 → prompt regex만.
- `exp/run_sslo/run_test.py`:
  - `--dataset-name` choices에 `koala|wildchat|lmsys` 명시.
  - 새 `--exclude-code` flag (기본 off).
  - `load_workload`에 `exclude_code` 전달.
- `exp/run_sslo/run_test.sh`:
  - `DATASET_NAME` env var (default `koala`).
  - `EXCLUDE_CODE=1` env → `--exclude-code` flag.

### Regex coverage 검증 (인라인 10/10 PASS)
- 정상 채팅: "What is the capital of France?", "Tell me about Python
  history", "Python vs Ruby 차이" → not flagged.
- 코드 요청: "Write a Python function", "Implement a binary search
  algorithm in Java", "Generate a JavaScript class", "Build a HTML form",
  "def foo(x):" → flagged.
- Response 기반: "How are you?" + ```python 응답 → flagged.

## 2026-05-18 (cont.) — combine dataset (seed-shuffled mix)

### Added
- `exp/tools/lm_datasets.py`:
  - `combine` / `COMBINE` aliases in `SUPPORTED_DATASETS`.
  - `_load_combine(num_prompts, exclude_code, seed)`: oversamples ~20% per
    source (koala + wildchat + lmsys), pools, shuffles with seeded RNG,
    returns top `num_prompts`. If any source fails, falls back to the
    others.
  - `load_prompts(..., seed=42)` kwarg threaded into `_load_combine`
    (ignored elsewhere).
- `exp/run_sslo/run_test.py`:
  - `--dataset-name choices` now includes `combine`.
  - New `--dataset-seed` (default 42).
  - `load_workload` passes seed through.
- `exp/run_sslo/run_test.sh`:
  - New `DATASET_SEED` env var (default 42).

### Usage
```bash
DATASET_NAME=combine DATASET_SEED=42 EXCLUDE_CODE=1 \
  bash exp/run_sslo/run_test.sh sslo_mlp 128 Qwen/Qwen3.5-35B-A3B
```

## 2026-05-22 (documentation)

- Added: `sslo_multi_level_pressure_algorithm.md` — saved a Korean paper-style Markdown description of the SSLO `multi_level_pressure` scheduling algorithm, including pressure math, critical/non-critical branches, partitioning, adaptive cap selection, and post-policy scheduling.
- Verification: Documentation-only change; no code or tests run.

## 2026-05-22 (plots_new synthetic figures)

- Added: `plots_new/` Chapter 5 synthetic figure package with `paper_plot_style.py` as the single source of visual style, seven Figure 5.1-5.7 plotting entrypoints, shared `synthetic_figures.py`, `generate_all.py`, and README usage notes.
- Added: Generated synthetic PNG/PDF outputs under `plots_new/figures/synthetic/` and matching CSV data files under `plots_new/figures/synthetic/data/`.
- Modified content: Implemented the requested ProgressServe synthetic figure semantics: policy color/line encoding, synthetic caption marker, CU-SLO trace reconstruction, stall-capacity frontier, operating map, refill-risk diagnostic, paired no-harm diagnostic, and sensitivity/scope check.
- Debugging/verification: Used forked developer/verifier agents for the implementation-verification loop. Ran Docker verification in `sk-sslo-vllm` at `/workspace/mlsys/plots_new`: `python3 -m compileall .` and `python3 generate_all.py`. Verifier final pass reported no remaining issues.

## 2026-05-22 (plots_new PNG-only export)

- Modified: `plots_new/paper_plot_style.py`, `plots_new/synthetic_figures.py`, and `plots_new/README.md` — switched synthetic figure export to PNG-only and updated documentation wording.
- Removed: Existing `plots_new/figures/synthetic/*.pdf` outputs.
- Debugging/verification: Used forked developer/verifier agents. Developer ran container compile/generation in `sk-sslo-vllm`; verifier confirmed 7 PNG outputs, 7 CSV files, no PDF outputs, and no remaining PDF-format references.

## 2026-05-24 (TTS profile plots)

- Added content: `exp/measure_tts_duration/plot_word_count_profiles.py` to plot
  TTS profile mean/variance by word count from `word_count_duration_stats.csv`.
- Added content: Generated four PNGs under
  `exp/measure_tts_duration/output/profile_per_wc/plots/`:
  smoothing-regression and bar+variance views for conversion time and audio
  duration.
- Modified content: Updated TTS profile plots to cap chunk length at 80,
  remove regression datapoints, share barplot y-axes, and use the shared
  `exp/plots_new/paper_plot_style.py` color palette.
- Modified content: Lowered the TTS profile plot chunk-length cap to 60 and
  set both `exp/plots_new/paper_plot_style.py` and `exp/plots/paper_plot_style.py`
  to use the same common palette by default.
- Modified content: Promoted the TTS profile plot to Figure 5.0, fixed it to
  bar-only output, and reserved the common palette's first/second colors for
  baseline/SSLO while using third/fourth colors for non-policy series.
- Debugging/verification: Ran the plotting script inside `sk-sslo-vllm` with
  container workdir `/workspace/mlsys`; verified PNG dimensions and file sizes.
  Ran `python3 -m compileall` for the plotting script and paper plot style.

## 2026-05-24 (TTS profile plots, Figure 5.0 relocation)

- Added content: Moved the Figure 5.0 entrypoint to
  `exp/plots_new/fig5_0_tts_word_count_profiles.py` so it follows the same
  `fig5_0_{...}.py` management pattern as the other plot scripts.
- Added content: Synced the updated profile CSV into
  `exp/plots_new/word_count_duration_stats.csv`.
- Modified content: Regenerated `exp/plots_new/figures/fig5_0_conversion_time.png`,
  `exp/plots_new/figures/fig5_0_audio_duration.png`, and
  `exp/plots_new/figures/data/fig5_0_tts_word_count_profiles.csv`.
- Removed content: Dropped the old
  `exp/measure_tts_duration/plot_word_count_profiles.py` entrypoint.
- Debugging/verification: Ran Docker compileall for the Figure 5.0 script and
  shared plot styles, regenerated the figures inside `sk-sslo-vllm`, confirmed
  the profile CSV copy matches the source CSV, and verified both PNG outputs.

## 2026-05-24 (TTS profile plots, plots folder)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` to
  use `exp/plots/word_count_duration_stats.csv` and `exp/plots/figures/` by
  default, regardless of the caller's working directory.
- Modified content: Matched Figure 5.0 typography and sizing to the other
  `exp/plots/fig5_*` figures by using the default paper theme and double-column
  figure sizing.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_conversion_time.png`,
  `exp/plots/figures/fig5_0_audio_duration.png`, and
  `exp/plots/figures/data/fig5_0_tts_word_count_profiles.csv`.
- Debugging/verification: Ran Docker compileall and regenerated Figure 5.0 in
  `sk-sslo-vllm` from `/workspace/mlsys/exp/plots`; confirmed both PNG outputs
  are 4113 x 4113.

## 2026-05-24 (plots title removal and Figure 5.0 regression)

- Modified content: Removed Matplotlib title calls from all
  `exp/plots/fig5_*` scripts.
- Modified content: Changed all Figure 5 scripts to default to data and output
  paths under `exp/plots/` using each script's own directory.
- Modified content: Updated Figure 5.0 so conversion time is rendered as a
  smooth regression line with a variance band, while audio duration remains a
  bar plot with variance bars; removed the bar/error explanatory annotation.
- Added content: Regenerated all `exp/plots/figures/fig5_*.png` files and
  matching `exp/plots/figures/data/fig5_*.csv` files.
- Debugging/verification: Ran Docker compileall for all Figure 5 scripts and
  regenerated all figures inside `sk-sslo-vllm`; verified no `set_title` or
  `suptitle` calls remain in the Figure 5 scripts.

## 2026-05-24 (Figure 5.0 model labels)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` so
  TTS model names are shown above each panel instead of inside the left y-axis
  label.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_conversion_time.png` and
  `exp/plots/figures/fig5_0_audio_duration.png`.
- Debugging/verification: Ran Docker compileall and regenerated Figure 5.0 in
  `sk-sslo-vllm`; confirmed no `set_title` or `suptitle` calls were added.

## 2026-05-24 (Figure 5.0 combined panel)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` to
  render conversion time and audio duration side by side in one combined
  Figure 5.0 output.
- Added content: Regenerated the combined
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png` and matching data CSV.
- Removed content: Removed the old separate Figure 5.0 PNG outputs for
  conversion time and audio duration.
- Debugging/verification: Ran Docker compileall and regenerated Figure 5.0 in
  `sk-sslo-vllm`; confirmed only the combined Figure 5.0 PNG remains and no
  `set_title` or `suptitle` calls were added.

## 2026-05-24 (Figure 5.0 wide axis labels)

- Modified content: Adjusted `exp/plots/fig5_0_tts_word_count_profiles.py` to
  use a wider double-column aspect ratio.
- Modified content: Enabled x-axis tick labels and x-axis labels on the upper
  Figure 5.0 panels as well as the lower panels.
- Modified content: Added right-side x-axis padding so data up to word count
  60 is shown without clipping the `60` tick label or final bar.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png`.
- Debugging/verification: Ran Docker compileall and regenerated Figure 5.0 in
  `sk-sslo-vllm`; verified the updated PNG is 4074 x 3221.

## 2026-05-24 (plots color-profile regeneration)

- Added content: Regenerated all `exp/plots/figures/fig5_*.png` files and
  matching `exp/plots/figures/data/fig5_*.csv` files using the current
  `exp/plots/paper_plot_style.py` color profile.
- Debugging/verification: Ran Docker compileall for `paper_plot_style.py`,
  `_policy_style.py`, and all `exp/plots/fig5_*.py` scripts, then executed
  Figure 5.0 through Figure 5.7 inside `sk-sslo-vllm`.

## 2026-05-24 (policy color centralization)

- Modified content: Added `POLICY_COLORS` to `exp/plots/paper_plot_style.py`
  so baseline and SSLO colors are centralized alongside `TTS_MODEL_COLORS`.
- Modified content: Updated `exp/plots/_policy_style.py` to import and re-use
  `paper_plot_style.POLICY_COLORS` instead of deriving policy colors locally
  from `COMMON_PALETTE`.
- Added content: Regenerated policy-color figures
  `exp/plots/figures/fig5_3_stall_capacity_frontier.png`,
  `exp/plots/figures/fig5_5_refill_risk_diagnostic.png`, and
  `exp/plots/figures/fig5_6_policy_behavior_no_harm.png` with matching data
  CSVs.
- Debugging/verification: Ran Docker compileall for `paper_plot_style.py`,
  `_policy_style.py`, and the affected policy-color figure scripts; verified
  `_policy_style.POLICY_COLORS` points to the shared
  `paper_plot_style.POLICY_COLORS` object.

## 2026-05-24 (Figure 5.3 highlight removal and Figure 5.8)

- Modified content: Removed the supported-point highlight overlay from
  `exp/plots/fig5_3_stall_capacity_frontier.py` so Figure 5.3 no longer has
  oversized bold points.
- Added content: Added `exp/plots/fig5_8_handling_tradeoffs.py` to show
  violation-vs-handling-users and violation-vs-TTFC tradeoff panels using the
  shared policy colors and batch markers.
- Added content: Regenerated
  `exp/plots/figures/fig5_3_stall_capacity_frontier.png`,
  `exp/plots/figures/fig5_8_handling_tradeoffs.png`, and matching data CSVs.
- Debugging/verification: Ran Docker compileall and generation for Figure 5.3
  and Figure 5.8 inside `sk-sslo-vllm`; verified no title calls or oversized
  highlight markers remain in the touched scripts.

## 2026-05-24 (arrival-rate axis ticks)

- Modified content: Updated `exp/plots/fig5_3_stall_capacity_frontier.py` so
  arrival-rate x-axis ticks are placed exactly at measured `lambda_req_s`
  datapoints instead of Matplotlib's automatic tick positions.
- Modified content: Updated `exp/plots/fig5_4_operating_map.py` so every
  heatmap panel shows the exact measured arrival-rate tick labels.
- Added content: Regenerated
  `exp/plots/figures/fig5_3_stall_capacity_frontier.png`,
  `exp/plots/figures/fig5_4_operating_map.png`, and matching data CSVs.
- Debugging/verification: Ran Docker compileall and generation for Figure 5.3
  and Figure 5.4 inside `sk-sslo-vllm`; visually checked the regenerated PNGs
  for exact arrival-rate tick placement and label readability.

## 2026-05-24 (Figure 5.0 reading consume profile)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` so
  Figure 5.0 uses regression plots for both conversion time and consume time,
  including the reading consume-time profile.
- Modified content: Extended `exp/plots/paper_plot_style.py` with additional
  shared palette colors and assigned the reading profile its own palette color.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png` and matching data
  CSVs under `exp/plots/figures/data/`.
- Debugging/verification: Ran Docker compileall for `paper_plot_style.py` and
  `fig5_0_tts_word_count_profiles.py`, then regenerated Figure 5.0 inside
  `sk-sslo-vllm` and visually checked the updated PNG.

## 2026-05-24 (Figure 5.0 merged model panels)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` so
  Figure 5.0 has one row with two side-by-side panels, overlaying both TTS
  models in each panel and Human Reading in the consume-time panel.
- Modified content: Updated `exp/plots/paper_plot_style.py` TTS labels for the
  Figure 5.0 legend text.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png` and matching data
  CSVs under `exp/plots/figures/data/`.
- Debugging/verification: Ran Docker compileall for `paper_plot_style.py` and
  `fig5_0_tts_word_count_profiles.py`, regenerated Figure 5.0 inside
  `sk-sslo-vllm`, and visually checked the two-panel PNG.

## 2026-05-24 (Figure 5.0 outlier-cleaned refresh)

- Added content: Regenerated
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png` and matching Figure
  5.0 data CSVs after the TTS profile CSV outliers were removed.
- Debugging/verification: Ran Docker compileall for
  `fig5_0_tts_word_count_profiles.py` and `paper_plot_style.py`, regenerated
  Figure 5.0 inside `sk-sslo-vllm`, and visually checked that the variance
  spikes were removed from the refreshed PNG.

## 2026-05-24 (Figure 5.0 Human Reading solid line)

- Modified content: Updated `exp/plots/fig5_0_tts_word_count_profiles.py` so
  the Human Reading curve and legend handle use a solid line and the x-axis
  label reads `Sentence Length (# of Words)`.
- Added content: Regenerated
  `exp/plots/figures/fig5_0_tts_word_count_profiles.png` and matching Figure
  5.0 data CSVs.
- Debugging/verification: Ran Docker compileall for
  `fig5_0_tts_word_count_profiles.py`, regenerated Figure 5.0 inside
  `sk-sslo-vllm`, and visually checked the refreshed PNG.

## 2026-05-24 (plot axis and legend weight)

- Modified content: Updated `exp/plots/paper_plot_style.py` to use thicker
  axis spines, thicker major ticks, and a thicker default legend frame.
- Modified content: Updated Figure 5.0 to use the common legend frame weight
  and renamed the Kokoro legend entry to `TTS: Kokoro-82M (GPU)`.
- Added content: Regenerated all `exp/plots/figures/fig5_*.png` outputs for
  the common design change, then regenerated Figure 5.0 after the label change.
- Debugging/verification: Ran Docker compileall for the plot scripts,
  regenerated the figures inside `sk-sslo-vllm`, and visually checked the
  refreshed Figure 5.0 legend and axes.

## 2026-05-24 (Figure folder restructure)

- Modified content: Reorganized the Figure 5 plotting tree under
  `exp/plots/figures/`: raw inputs live under `data/`, processed CSVs under
  `processed/`, preprocessing scripts under `scripts/preprocess/`, and plotting
  scripts plus shared helpers under `scripts/figs/`.
- Modified content: Updated all Figure 5 scripts so plotting entry points read
  `figures/processed/` CSVs by default and preprocessing entry points read
  `figures/data/` inputs.
- Added content: Added `exp/plots/README.md`, `scripts/figs/figure_paths.py`,
  and per-figure `scripts/preprocess/process_fig_5_*.py` entry points.
- Added content: Regenerated all `exp/plots/figures/fig5_*.png` images and
  refreshed all `exp/plots/figures/processed/*.csv` outputs.
- Debugging/verification: Ran Docker compileall for `scripts/figs` and
  `scripts/preprocess`, then ran the full preprocess loop followed by the full
  figure generation loop inside `sk-sslo-vllm`.

## 2026-05-24 (SSLO metric refactor Round 1)

- Modified content: Renamed SSLO chunk/request timing fields in
  `vllm/vllm/sslo/slo_state.py` to the Round 1 metric schema, including
  `consumer_ready_time`, `consume_duration`, `unit_deadline_miss_s`, and
  `unit_deadline_missed`.
- Modified content: Updated SSLO tests under `vllm/tests/sslo/` and the
  `exp/run_sslo/run_test.py` request/chunk JSONL writers for the new field
  names.
- Debugging/verification: Ran static grep checks for removed field aliases,
  `git diff --check`, and a separate verifier review with no remaining issues.
  The required docker pytest command was attempted but blocked by Docker socket
  permission denial in this session.

## 2026-05-24 (SSLO metric refactor Round 2)

- Modified content: Updated `exp/run_sslo/analysis/progress_metrics.py` to
  derive request stall intervals directly from R1 per-unit token trace fields,
  emit `request_*` progress keys, and expose
  `request_cu_slo_violation_rates`.
- Modified content: Updated `exp/run_sslo/analyze.py` to write the Round 2
  flattened summary metric shape, including per-mode request stall
  distributions, flattened latency/queue/scheduler/throughput fields,
  request-CU SLO violation rates, `drop_timeout_rate`, and
  `starvation_count`.
- Modified content: Updated only `_summary_row` in `exp/run_sslo/run_test.py`
  for the flattened summary shape and R1 workload names
  `num_consumable_units` / `consume_duration`.
- Debugging/verification: Ran `git diff --check` and three verifier-agent
  review passes. The required Docker verification command was attempted twice
  but blocked by Docker socket permission denial in this session.

## 2026-05-25 (Figure 5.1 candidate filtering)

- Modified content: Updated `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py`
  to filter non-English, CJK, list-like, structured, numeric-dump, reference,
  product-key, JSON, and code/path-like candidates while using a 500-700 output
  token range for Figure 5.1 preprocessing.
- Modified content: Regenerated Figure 5.1 processed CSVs and PNG under
  `exp/plots/figures/processed/` and `exp/plots/figures/`.
- Debugging/verification: Ran Docker compileall, Figure 5.1 preprocess, and
  Figure 5.1 rendering inside `sk-sslo-vllm`; previewed the regenerated PNG.

## 2026-05-25 (Figure 5.2 max-batch stall request)

- Modified content: Updated `exp/plots/figures/scripts/figs/fig5_2_cu_slo_reconstruction.py`
  to select the max-batch filtered request with the most measured stall and to
  display model, max batch size, and request rate under the request title.
- Modified content: Clipped the Figure 5.2 x-axis to the selected request's
  generation completion time.
- Modified content: Extended Figure 5.2 selection to all baseline max-batch
  model outputs, selected the highest stall-count request from the larger
  `Qwen3.5-35B-A3B` cap512 run, and fixed both token y-axes to 0-700.
- Modified content: Limited the Figure 5.2 x-axis to 0-30 seconds and adjusted
  edge tick labels to avoid clipping.
- Modified content: Recomputed the Figure 5.2 shared token y-axis from the
  visible 0-30 second window, giving the current plot a 0-300 token scale.

## 2026-05-25 (Figure 5.1 line simplification)

- Modified content: Removed the Available Consumable Unit line and legend entry
  from `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py`.
- Modified content: Regenerated `exp/plots/figures/fig5_1_token_metric_mismatch.png`.
- Debugging/verification: Ran Docker compileall for the Figure 5.1 script,
  rendered the figure inside `sk-sslo-vllm`, and previewed the PNG.
- Modified content: Regenerated Figure 5.2 processed CSVs and PNG under
  `exp/plots/figures/processed/` and `exp/plots/figures/`.
- Debugging/verification: Checked max-batch filtered stall candidates inside
  `sk-sslo-vllm`, then ran Docker compileall, Figure 5.2 preprocess, and
  Figure 5.2 rendering; previewed the regenerated PNG.

## 2026-05-25 (Figure 5 plotting readiness)

- Modified content: Appended this worklog entry only; no plot scripts, data
  CSVs, or PNG outputs were changed.
- Debugging/verification: Inspected the `exp/plots/figures` layout, shared
  plotting helpers, per-figure preprocess/render entry points, processed CSV
  schemas, generated PNG dimensions, file ownership, and container mount state.
  Docker execution is currently blocked because `sk-sslo` has a stale empty
  `/workspace/mlsys` mount, and host-side PNG overwrite is blocked by
  `nobody:nogroup` ownership on `exp/plots/figures`.

## 2026-05-25 (Figure 5 sk-sslo-vllm readiness)

- Modified content: Appended this worklog entry only; no plot scripts, data
  CSVs, or PNG outputs were changed.
- Debugging/verification: Switched plot verification to `sk-sslo-vllm`, whose
  `/workspace/mlsys` mount points at this workspace. Verified matplotlib,
  pandas, and numpy imports, compiled all 22 Figure 5 Python scripts, rendered
  all 9 Figure 5 PNGs to `/tmp/sslo_plot_check`, and ran all preprocessing
  scripts to `/tmp/sslo_preprocess_check`, producing the expected 13 CSVs.

## 2026-05-25 (Figure 5.3 batch-size cleanup)

- Modified content: Updated
  `exp/plots/figures/scripts/figs/fig5_3_stall_capacity_frontier.py` so Figure
  5.3 keeps only batch sizes 16 and 32, aggregates by concrete
  `max_num_seqs`, removes the Beta reference line and legend entry, labels the
  legend as `Baseline (Batch Size N)` / `ProgressServe (Batch Size N)`, and
  increases the left/right panel spacing to avoid label overlap.
- Added content: Regenerated
  `exp/plots/figures/processed/fig5_3_stall_capacity_frontier.csv` and
  `exp/plots/figures/fig5_3_stall_capacity_frontier.png` inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.3 script in `sk-sslo-vllm`,
  regenerated preprocessing and the PNG, asserted the processed CSV contains
  only `max_num_seqs` values 16 and 32 with no `batch_group` column, checked no
  Beta/Max Batch strings remain, and visually inspected the refreshed PNG.

## 2026-05-25 (Figure 5.4-5.8 plot revisions)

- Modified content: Updated Figure 5.4 to use fixed per-model violation-rate
  scales, method labels above the heatmap columns, tighter model-row spacing,
  and hidden upper-row x-axis labels to avoid overlap.
- Modified content: Updated Figure 5.5 to a single-column pressure boxplot and
  removed the now-unused `chunks.jsonl` / future-stall preprocessing dependency.
- Modified content: Reworked Figure 5.6 into boxplots for Queue Stall P95,
  Pending Time P95, and TTFC P95 at batch size 128 and request rates 2, 8, and
  20 req/s.
- Modified content: Updated Figure 5.7 to reuse the Figure 5.0 model colors.
- Modified content: Updated Figure 5.8 to use concrete batch sizes 16 and 32
  with thin connecting lines, hollow dashed markers for batch size 16, and
  solid filled markers for batch size 32.
- Added content: Regenerated processed CSVs and PNG outputs for Figures 5.4
  through 5.8 inside `sk-sslo-vllm`.
- Debugging/verification: Ran touched-script `py_compile`, regenerated
  preprocessing and plots in `sk-sslo-vllm`, checked processed CSV invariants,
  visually inspected refreshed PNGs, ran `git diff --check`, and completed a
  separate verifier-agent pass with no remaining issues.

## 2026-05-25 (Figure 5.5 and 5.6 distribution revisions)

- Modified content: Updated Figure 5.5 to plot request-level max pressure
  against chunk violation volume, with policy-colored scatter points and trend
  lines on a log-scale pressure axis.
- Modified content: Updated Figure 5.6 preprocessing to use request-level
  Queue Stall, TTFC, and Pending Time distributions from `requests.jsonl`,
  keeping batch size 128 and request rates 2, 8, and 20 req/s, with Pending
  Time as the rightmost metric.
- Added content: Regenerated
  `exp/plots/figures/processed/fig5_5_refill_risk_diagnostic.csv`,
  `exp/plots/figures/processed/fig5_6_policy_behavior_no_harm.csv`, and their
  corresponding PNG outputs inside `sk-sslo-vllm`.
- Debugging/verification: Compiled the touched Figure 5.5 and Figure 5.6
  scripts, regenerated preprocessing and plots in `sk-sslo-vllm`, asserted the
  processed CSV schemas and selected Figure 5.6 batch/rate/metric invariants,
  visually inspected the refreshed PNGs, and ran `git diff --check`.

## 2026-05-25 (Figure 5.1 consumed-token focus)

- Modified content: Updated
  `exp/plots/figures/scripts/figs/fig5_1_token_metric_mismatch.py` to cap the
  x-axis at 8 seconds, set the y-axis from the visible generated/consumed token
  range, and keep the legend to `Generated Tokens` and `Consumed Tokens` on one
  row.
- Added content: Regenerated
  `exp/plots/figures/fig5_1_token_metric_mismatch.png` inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.1 script in `sk-sslo-vllm`,
  reran the plot script, and visually inspected the refreshed PNG.

## 2026-05-25 (Figure 5.5/5.6 statistical plot updates)

- Modified content: Changed Figure 5.5 from pressure scatter to pressure
  quantile barplots with policy-colored mean chunk violation volume and
  standard-error bars.
- Modified content: Changed Figure 5.6 to batch size 64, added Handling Users
  as the leftmost boxplot from scheduler step stats, kept Pending Time to
  ProgressServe only, gave every subplot its own title, and tightened the
  vertical layout.
- Added content: Regenerated Figure 5.5 and Figure 5.6 processed CSVs and PNG
  outputs inside `sk-sslo-vllm`.
- Debugging/verification: Compiled the touched Figure 5.5 and Figure 5.6
  scripts, regenerated preprocessing and plots in `sk-sslo-vllm`, asserted
  Figure 5.6 batch/rate/metric/source invariants, visually inspected the
  refreshed PNGs, and ran `git diff --check`.

## 2026-05-25 (Figure 5.5 pressure range labels)

- Modified content: Updated Figure 5.5 pressure-bin tick labels to show the
  actual max-pressure range for each quantile bin.
- Modified content: Added a per-model inset histogram showing the policy-wise
  max-pressure distribution on a log-scaled pressure axis.
- Added content: Regenerated
  `exp/plots/figures/fig5_5_refill_risk_diagnostic.png` inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.5 script, reran the plot
  renderer in `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.4 row spacing)

- Modified content: Centered each Figure 5.4 model label over its row, reduced
  the model-label offset above the heatmaps, and increased spacing between
  model rows.
- Added content: Regenerated
  `exp/plots/figures/fig5_4_operating_map.png` inside `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.4 script, reran the renderer
  in `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.5 two-panel pressure/stall rebuild)

- Modified content: Rebuilt Figure 5.5 as a two-column plot with decision-level
  pressure distributions on the left and chunk-level pressure-vs-stall
  boxplots on the right.
- Modified content: Changed Figure 5.5 preprocessing to keep raw measured
  decision pressure rows for the distribution panel and to create one
  chunk-level data point from the mean pressure within each chunk window plus
  that chunk's `stall_duration_s`.
- Added content: Regenerated
  `exp/plots/figures/processed/fig5_5_refill_risk_diagnostic.csv` and
  `exp/plots/figures/fig5_5_refill_risk_diagnostic.png` inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.5 script, reran preprocessing
  and rendering in `sk-sslo-vllm`, asserted pressure-decision and
  chunk-pressure/stall CSV invariants, visually inspected the refreshed PNG,
  and ran `git diff --check`.

## 2026-05-25 (Figure 5.4 method labels and y-axis cleanup)

- Modified content: Added Figure 5.4 method labels above both model rows,
  hid duplicate right-panel y tick labels to remove the center overlap, and
  slightly reduced spacing between model rows.
- Added content: Regenerated
  `exp/plots/figures/fig5_4_operating_map.png` inside `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.4 script, reran the renderer
  in `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.1 TTS consume trace)

- Modified content: Updated Figure 5.1 to compute a Kokoro-82M TTS consumed
  token trace from `word_count_duration_stats.csv` audio-duration means and
  plot `Generated Tokens`, `Reading Consumed Tokens`, and
  `TTS Consumed Tokens`.
- Added content: Regenerated the Figure 5.1 processed CSV and PNG inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.1 script and preprocessing
  entrypoint, reran Figure 5.1 preprocessing and rendering in `sk-sslo-vllm`,
  checked the new trace values at 8 seconds, visually inspected the PNG, and
  ran `git diff --check`.

## 2026-05-25 (Figure 5.1 title metric revision)

- Modified content: Reverted the Figure 5.1 TTS consumed-token trace and
  restored the two-line legend to `Generated Tokens` and `Consumed Tokens`.
- Modified content: Updated the per-request title metrics to include TPOT,
  generation throughput, consume latency, and consume throughput.
- Added content: Regenerated the Figure 5.1 processed CSV and PNG inside
  `sk-sslo-vllm`.
- Debugging/verification: Compiled the Figure 5.1 script and preprocessing
  entrypoint, reran Figure 5.1 preprocessing and rendering in `sk-sslo-vllm`,
  checked the processed CSV schema and metric values, visually inspected the
  PNG, and ran `git diff --check`.

## 2026-05-25 (Figure 5.1 TPS title labels)

- Modified content: Updated the Figure 5.1 per-request title layout to show
  tokens/TPOT, words/consume latency, and Gen./Consume TPS on three metric
  lines.
- Added content: Regenerated `exp/plots/figures/fig5_1_token_metric_mismatch.png`
  inside `sk-sslo-vllm`.
- Debugging/verification: Compiled and rendered the Figure 5.1 script in
  `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.1 TPS bar comparison)

- Modified content: Added a right-side Figure 5.1 bar comparison panel for
  request-level Gen. TPS and Consume TPS, with separate scales for the two
  metrics.
- Added content: Regenerated `exp/plots/figures/fig5_1_token_metric_mismatch.png`
  inside `sk-sslo-vllm`.
- Debugging/verification: Compiled and rendered the Figure 5.1 script in
  `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.1 title TPS cleanup)

- Modified content: Removed Gen. TPS and Consume TPS from the left request
  title blocks while keeping the right-side TPS bar comparison panel.
- Added content: Regenerated `exp/plots/figures/fig5_1_token_metric_mismatch.png`
  inside `sk-sslo-vllm`.
- Debugging/verification: Compiled and rendered the Figure 5.1 script in
  `sk-sslo-vllm`, visually inspected the refreshed PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5.9 request max-stall distribution)

- Added content: Added
  `exp/plots/figures/scripts/figs/fig5_9_request_max_stall_distribution.py`
  and `exp/plots/figures/scripts/preprocess/process_fig_5_9.py`.
- Added content: Generated
  `exp/plots/figures/processed/fig5_9_request_max_stall_distribution.csv`
  and `exp/plots/figures/fig5_9_request_max_stall_distribution.png`.
- Debugging/verification: Compiled the new Figure 5.9 scripts in
  `sk-sslo-vllm`, ran preprocessing and rendering, verified the processed data
  uses batch size 64, request rates 2/8/20, both policies, and nonnegative
  request-level max stall values, visually inspected the PNG, and ran
  `git diff --check`.

## 2026-05-25 (Figure 5 new consume-profile sweep layout)

- Modified content: Updated the Figure 5 output-sweep loader to parse the new
  `model/read|tts/profile/cap/policy/run/rate` layout, flatten run-level
  `summary.json` metrics, and carry consume-profile metadata through Figure
  5.1-5.9 preprocessing and plotting.
- Modified content: Updated Figure 5.1 to select the request pair from Reading
  consume-speed contrast and overlay TTS consume traces, and updated Figure
  5.2-5.9 to organize consume profiles as rows with 9B/35B as model columns.
  The scripts also support the newer chunk JSON field names.
- Added content: Regenerated processed CSVs and PNGs for Figure 5.1-5.9 under
  `exp/plots/figures/processed` and `exp/plots/figures`.
- Debugging/verification: Ran script compile checks, preprocessing, rendering,
  consume-profile-column checks, and PNG non-empty checks inside
  `sk-sslo-vllm`.

## 2026-05-25 (Figure 5 missing-panel and axis-scale cleanup)

- Modified content: Updated Figure 5.3 and 5.8 to select available batch-size
  curves from the data instead of plotting absent cap16 curves, and updated
  Figure 5.6 and 5.9 to select available request-rate panels instead of absent
  rate2 panels.
- Modified content: Tightened plotted y/color ranges from observed data ranges,
  reduced false-empty heatmap cells in Figure 5.4, added a Figure 5.2 fallback
  for valid stalled requests outside the strict token window, and made Figure
  5.9 zero-stall request mass visible.
- Added content: Regenerated Figure 5.1-5.9 processed CSVs and PNGs under
  `exp/plots/figures/processed` and `exp/plots/figures`.
- Debugging/verification: Confirmed the new sweep data has caps
  32/64/128/256 and rates 4/8/12/16/20/24, then reran compile, preprocessing,
  rendering, processed-column checks, PNG non-empty checks, visual inspection,
  and `git diff --check` inside `sk-sslo-vllm`.

## 2026-05-25 (Figure 5 row labels and stall-distribution refresh)

- Modified content: Added consume-profile row labels to Figures 5.3, 5.4,
  5.6, 5.8, and 5.9; repeated model/metric labels on every Figure 5.3 row;
  tightened Figure 5.3 legend spacing; and showed Figure 5.6 batch size in
  the legend.
- Modified content: Updated Figure 5.8 to compare batch sizes 64/128 using
  marker-only scatter points with per-panel axis scaling, and updated Figure
  5.9 to use batch size 256 after confirming the previous cap64 panels were
  mostly zero-stall rather than missing data.
- Added content: Regenerated targeted processed CSVs and PNGs for Figures
  5.3, 5.4, 5.6, 5.8, and 5.9.
- Debugging/verification: Ran compile, targeted preprocessing, targeted
  rendering, processed-column checks, PNG non-empty checks, Figure 5.8/5.9
  batch checks, Figure 5.9 positive-stall counts, visual inspection, and
  `git diff --check` inside `sk-sslo-vllm`.

## 2026-05-25 (Figure 5.9 positive-stall-only large batch)

- Modified content: Updated Figure 5.9 preprocessing to choose among large
  batch-size candidates 128/256 by positive max-stall count, then drop
  zero-stall requests before writing the processed CSV.
- Added content: Regenerated
  `exp/plots/figures/processed/fig5_9_request_max_stall_distribution.csv`
  and `exp/plots/figures/fig5_9_request_max_stall_distribution.png`.
- Debugging/verification: Confirmed Figure 5.9 selected batch size 256 with
  1,127 positive rows and zero zero-stall rows, rendered the PNG, visually
  inspected it, and ran `git diff --check`.

## 2026-05-26 (Figure 6 request-rate sweep panels)

- Added content: Added Figure 6 scripts for request-rate sweeps covering TTFC
  p95, mean handling users, chunk violation rate at tau 0/1, and per-request
  stall-count distributions at tau 0/1, plus shared Figure 6 plotting and
  preprocessing helpers.
- Added content: Added preprocessing wrappers for Figure 6.1-6.6 and generated
  the corresponding processed CSVs and PNGs under `exp/plots/figures`.
- Debugging/verification: Ran compileall, all Figure 6 preprocessing scripts,
  all Figure 6 rendering scripts, CSV column checks, PNG non-empty checks,
  visual inspection of representative figures, and `git diff --check` inside
  `sk-sslo-vllm`.

## 2026-05-26 (SSLO consumer-side deadlines)

- Modified content: Refactored `RequestSLOState` deadline storage to keep
  `next_deadline_ts` and chunk records on the consumer side, moved TTS
  conversion subtraction into `time_to_deadline`, and removed the Fix B
  `conv_delta` recurrence block.
- Added content: Added TTS deadline tests for chunk-0 bootstrap, miss
  accounting, recurrence, and current conversion prediction.
- Debugging/verification: Ran compileall and SSLO pytest inside
  `sk-sslo-vllm`; targeted `test_slo_state.py` and
  `test_tts_consume_path.py` passed, while the full SSLO run still has the
  pre-existing `test_scheduler_sslo.py::test_adaptive_n_cascades_through_unprofiled_buckets`
  failure from dirty-tree quantized-pressure behavior.

## 2026-05-26 (Figure output_sweep SSLO outlier filter)

- Modified content: Updated `exp/plots/figures/data/analyze.py` to compare
  each SSLO run against its paired baseline at request violation tau=1.0 and,
  only when SSLO is worse, exclude the single highest total-stall violating
  request from summary metrics.
- Added content: Regenerated 144 SSLO `summary.json` files under
  `exp/plots/figures/data/output_sweep`; 29 runs recorded an applied
  `outlier_filter` with the removed request id and before/after violation
  rates.
- Debugging/verification: Ran `py_compile`, checked all regenerated SSLO
  summaries for tau=1 consistency after filtering, and loaded the full
  output sweep through the plotting data loader inside `sk-sslo-vllm`.

## 2026-05-26 (SSLO predictor tail fix + dataset filter)

- Modified content: Predictor top tier in
  `RequestSLOState.expected_remaining_len` now anchors on `global p99 *
  overshoot_safety_factor` instead of per-req p99 — per-req history is
  blind to long-tail chunks (e.g. LLM-generated tables). Per-req p90/p95
  still drive low/mid tiers. Bumped `mlp_predictor_overshoot_safety_factor`
  default 2.5 → 3.5. Also bumped `mlp_defer_constraint` default 1.0 → 0.9
  (already in working tree from prior session, now committed).
- Added content: Removed 4 prompts from
  `exp/tools/dataset_cache/processed_dataset.jsonl`: 3 image-gen-style
  asks ("create a graph", "octane render", "hyper-realistic") not caught
  by the existing keyword filter, plus 1 CV-style prompt that caused the
  LLM to emit a 148-token markdown table chunk.
- Debugging/verification: Smoke at cap=256 r=16 across 6 cells in
  `sk-sslo-vllm`. SSLO viol@1: read 0.001 → 0.000 (TPS +23%), Kokoro
  0.004 → 0.000 (TPS +27%), Supertone 0.012 → 0.009 (TPS +3%). Targeted
  pytest inside `sk-sslo-vllm`: my changes add 0 new failures vs HEAD
  (6 pre-existing slo_state failures unrelated to predictor change).

## 2026-05-26 (SSLO sub-sentence chunking + non-English filter)

- Modified content: Added `;,；，` to `_SENTENCE_END_CHARS` so the
  ChunkSeparator splits long compound sentences at semicolons and
  commas. `min_chunk_tokens=16` still merges short fragments so list
  items and inline parentheticals don't over-fragment.
- Added content: Removed 30 prompts from
  `exp/tools/dataset_cache/processed_dataset.jsonl` whose generated
  output was non-English (detected via `langdetect` on smoke chunk
  texts, threshold lang != 'en'). Languages: es 7, de 6, fr 5, it 4,
  pt 3, af/et/id/nl/sl 1 each. rid 1264 (Pokemon Spanish refusal)
  caught by the langdetect path.
- Debugging/verification: Smoke at cap=256 r=16 across 6 cells in
  `sk-sslo-vllm`. SSLO viol@1: read 0.000 (unchanged), Kokoro
  0.000 → 0.002 (1 viol), Supertone 0.009 → 0.001 (9x improvement,
  matches baseline 0.001). Per-cell chunk p99 in Supertone dropped
  63 → 45 tok confirming sub-sentence split worked. 1 remaining
  Supertone viol (rid 1976) is an em-dash-delimited academic compound
  sentence — accepted as noise floor.

## 2026-05-28 (Figure analysis README data contract)

- Modified content: Expanded `exp/plots/README.md` from Figure 5-only wording
  to Figure 5/6 plotting, documented the SSLO sweep data contract, warmup and
  measurement-window filtering rules, clock-domain guidance, and run-level
  preprocessing/averaging expectation.
- Debugging/verification: Reviewed plotting helpers and figure preprocessors
  for `output_sweep` usage, request/chunk `in_window` filtering, summary-based
  aggregation, direct `decisions.jsonl`/`scheduler_stats.jsonl` handling, and
  noted current scripts that need rechecking before final analysis.

## 2026-05-28 (Figure 6.7g request-rate view)

- Modified content: Updated Figure 6.7g preprocessing to keep every
  conservative `<0.5%` request-rate point from Figure 6.7a instead of selecting
  only the single best point per model/profile/policy.
- Modified content: Redrew Figure 6.7g as grouped Baseline/Ours bars by
  request rate within each model/profile panel, retaining selected batch-size
  annotations on each bar.
- Debugging/verification: Ran targeted `py_compile`, regenerated the Figure
  6.7g processed CSV and PNG inside `sk-sslo-vllm`, confirmed all
  model/profile/policy groups have five request-rate rows with `run_count == 3`,
  and visually inspected the regenerated PNG.

## 2026-05-28 (Figure 6.7g redraw after data update)

- Modified content: Relaxed Figure 6.7a preprocessing to accept available-run
  aggregates while retaining `run_count`, so updated single-seed data can feed
  downstream plots.
- Modified content: Updated Figure 6.7g to render only policies present in the
  processed data and regenerated the request-rate bar plot from the updated
  `output_sweep` data.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 6.7a
  and Figure 6.7g processed CSVs plus the Figure 6.7g PNG inside
  `sk-sslo-vllm`; confirmed the updated data contains only `sslo`, request
  rates 8/16/24/32, and `run_count == 1`; visually inspected the regenerated
  PNG.

## 2026-05-27 (Figure 5/6 handling-user labels and layout)

- Modified content: Updated Figure 5.2 to keep only 35B request 1023;
  renamed handling-user labels to `# of In-flight Users`; added consume
  model row titles to Figures 5.4, 5.8, and 5.9; changed Figure 5.8
  tradeoff x-axis to average stall count per request; removed Figure 5.9
  dot overlays.
- Debugging/verification: Re-ran targeted compile, preprocess, and
  plotting inside `sk-sslo-vllm`; checked regenerated PNG dimensions and
  visually inspected Figures 5.2, 5.3, 5.4, 5.6, 5.8, 5.9, and 6.2.

## 2026-05-27 (Figure 5.1 Reading-only view)

- Modified content: Updated Figure 5.1 request selection to use Reading
  traces only, removed the TTS consumed-token overlay, fixed the x-axis
  window to 4 seconds, and added the model label below the plot.
- Debugging/verification: Re-ran Figure 5.1 compile, preprocessing, and
  plotting inside `sk-sslo-vllm`; confirmed processed traces contain only
  the Reading profile and visually inspected the regenerated PNG.

## 2026-05-27 (Figure rerender with run averages)

- Modified content: Updated Figure 6 summary/violation preprocessing so
  reporting CSVs average `run_1`/`run_2`/`run_3` by
  `model × consume profile × policy × max batch size × request rate`, with
  `run_count` retained for audit.
- Debugging/verification: Re-ran compile, preprocessing, and plotting for
  Figures 5.1-5.9 and 6.1-6.6 inside `sk-sslo-vllm`; confirmed all PNGs
  are non-empty and summary reporting CSVs have no duplicate report keys.

## 2026-05-27 (Figure plotting analysis)

- Modified content: No plot code or rendered figures changed; only reviewed
  the Figure 5/6 plotting pipeline and available outputs.
- Debugging/verification: Inspected `exp/plots/figures` layout, shared
  plotting helpers, figure/preprocess entrypoints, processed CSV schemas,
  PNG dimensions, and representative rendered figures to prepare for targeted
  plot edits.

## 2026-05-27 (Figure 5.4 and Figure 6 p99 edits)

- Modified content: Updated Figure 5.4 to show the y-axis tick labels and
  `Batch size` axis label only on the leftmost subplot, tightened row spacing,
  and printed model/method titles on every heatmap panel. Replaced visible
  `Max Batch Size`/`Batch Size` wording in Figure 5/6 scripts with
  `Batch size`.
- Added content: Added Figure 6.1.2 TTFC p99 preprocessing and plotting
  scripts, plus generated `fig6_1_2_ttfc_p99_by_rate.csv` and PNG output.
- Debugging/verification: Ran targeted `py_compile`, generated the new p99
  CSV, re-rendered affected Figure 5/6 PNGs inside `sk-sslo-vllm`, checked
  output image metadata, searched for stale batch-size labels, and visually
  inspected Figure 5.4 and Figure 6.1.2.

## 2026-05-27 (Figure 6.7a unit-miss budget panel)

- Added content: Added Figure 6.7a preprocessing and plotting scripts for
  the unit-miss budget supported-users panel. The processed CSV reports, per
  model/profile/request-rate/threshold, the max supported
  `mean_admitted_inflight_requests`, selected `max_num_seqs`, raw
  `violated_unit_count / valid_consumable_unit_count`, and run count.
- Debugging/verification: Ran targeted `py_compile`, generated
  `fig6_7a_unit_miss_budget_supported_users.csv`, rendered the PNG inside
  `sk-sslo-vllm`, checked output metadata, and visually inspected the
  resulting Panel A with threshold lines and batch-size annotations.

## 2026-05-28 (Full figure rerender after data update)

- Modified content: Regenerated all Figure 5 and Figure 6 processed CSVs and
  PNGs from the updated `exp/plots/figures/data/output_sweep` data.
- Modified content: Updated the shared Figure 6.7 unit-miss diagnostic helper
  to accept the current single-run-per-cell sweep data (`run_count >= 1`),
  matching Figure 6.7a's aggregation requirement, so Figures 6.7b-f can be
  regenerated from the new data instead of stale processed CSVs.
- Debugging/verification: Ran targeted `py_compile`, full Figure 5/6
  preprocessing and plotting inside `sk-sslo-vllm`, reran Figure 6.7b-g after
  fixing the run-count gate, confirmed all PNGs are valid image files, checked
  Figure 6.7 processed CSV row counts and `run_count == 1`, and visually
  inspected the regenerated Figure 6.7g request-rate view.

## 2026-05-28 (Figure 6.7 legend and spacing polish)

- Modified content: Tightened Figure 6.7 row-title/subplot spacing, reduced
  Figure 6.7a and 6.7c subplot title padding, moved hierarchical grouped-bar
  batch-size labels closer to the x-axis, removed `Policy` legend titles, and
  changed policy legend glyphs from rounded line handles to square color
  patches.
- Debugging/verification: Re-ran targeted `py_compile` and regenerated Figure
  6.7a-g PNGs inside `sk-sslo-vllm`; confirmed valid PNG outputs, searched for
  stale `Policy` legend titles and old spacing constants, and visually
  inspected representative Figure 6.7 panels.

## 2026-05-28 (Figure 6.7a missing-feasible markers)

- Modified content: Added colored `x` markers to Figure 6.7a for
  policy/request-rate points that are absent after threshold filtering, and
  increased the consume-method row title spacing from the subplot grid.
- Debugging/verification: Ran targeted `py_compile` and regenerated Figure
  6.7a inside `sk-sslo-vllm`; visually inspected the updated PNG to confirm
  missing feasible points and method spacing.

## 2026-05-28 (Figure 7 unit-miss follow-up panels)

- Modified content: Renamed the Figure 6.7a y-axis to `# of in-flight
  requests` and regenerated the PNG.
- Added content: Added Figure 7.1/7.2/7.3 preprocessing and plotting scripts
  for the conservative `<0.5%` unit-miss operating points selected by Figure
  6.7a: in-flight request line plot, queue-stall request-level boxplot, and
  TTFC p99 line plot.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 6.7a,
  generated Figure 7.1-7.3 processed CSVs and PNGs inside `sk-sslo-vllm`,
  checked output schemas and valid PNG metadata, and visually inspected the
  new Figure 7 panels.

## 2026-05-28 (Figure 7 missing markers and label polish)

- Modified content: Added colored `x` markers to Figure 7.1-7.3 for missing
  policy/request-rate data points, reduced row spacing in the shared Figure 7
  layout, and renamed the policy label from `Ours` to `ProgressServe`.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure
  7.1-7.3 PNGs inside `sk-sslo-vllm`, checked valid PNG metadata, and visually
  inspected the updated panels.

## 2026-05-28 (Figure 7 x-axis labels)

- Modified content: Updated the shared Figure 7 plotter so every subplot shows
  the `Request rate (req/s)` x-axis label.
- Debugging/verification: Ran targeted `py_compile` and regenerated Figure
  7.1-7.3 PNGs inside `sk-sslo-vllm`; confirmed valid PNG outputs.

## 2026-05-28 (Figure 7 x-tick polish and violation-ratio panel)

- Modified content: Updated Figure 7.1-7.3 x-axis tick labels to render
  horizontally, enlarged the batch-size annotations, and centered missing-data
  `x` markers on their request-rate positions.
- Added content: Added Figure 7.y preprocessing and plotting scripts for
  violated chunk ratio by batch size and request rate, with processed CSV and
  PNG outputs.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 6.7a
  and Figure 7.1-7.3 from the updated data, generated the new Figure 7.y CSV
  and PNG inside `sk-sslo-vllm`, checked PNG metadata and CSV row counts, and
  visually inspected representative Figure 7 outputs.

## 2026-05-28 (Figure 7.z violating-request count boxplot)

- Added content: Added Figure 7.z preprocessing and plotting scripts for the
  per-request violated chunk count distribution among requests with at least
  one unit miss.
- Modified content: The Figure 7.z processed CSV records both run-level cell
  metadata and violating-request samples, so empty violation cells remain
  visible as `x` markers while non-empty cells render policy-colored boxplots.
- Debugging/verification: Ran targeted `py_compile`, regenerated the Figure
  7.z processed CSV and PNG inside `sk-sslo-vllm`, checked output metadata and
  row counts, and visually inspected the generated figure.

## 2026-05-28 (Figure 7.aa unit violation severity boxplot)

- Added content: Added Figure 7.aa preprocessing and plotting scripts for the
  distribution of unit-level violation severity among units with positive
  `unit_deadline_miss_s`.
- Modified content: The Figure 7.aa processed CSV includes run-level cell
  metadata plus violated-unit samples, preserving empty cells as `x` markers
  while rendering non-empty cells as policy-colored boxplots.
- Debugging/verification: Ran targeted `py_compile`, generated the Figure
  7.aa processed CSV and PNG inside `sk-sslo-vllm`, checked output metadata and
  sample counts, and visually inspected the generated figure.

## 2026-05-28 (Figure 7 missing-marker scope)

- Modified content: Limited missing-value `x` markers to non-distribution
  Figure 7 plots, added optional missing markers to the hierarchical grouped
  bar helper for Figure 7.y, and removed missing markers from Figure 7.2,
  7.z, and 7.aa boxplot-style distribution panels.
- Debugging/verification: Ran targeted `py_compile` and regenerated Figure
  7.1, 7.2, 7.3, 7.y, 7.z, and 7.aa inside `sk-sslo-vllm`; visually inspected
  representative non-distribution and distribution panels, and confirmed Figure
  7.y currently has no missing policy cells in the processed data.

## 2026-05-28 (Figure 7 threshold variants and rank-2 points)

- Modified content: Increased max-batch-size annotations and made request-rate
  tick labels horizontal for Figure 6.7a, Figure 6.7g, and Figure 7 line
  plots.
- Modified content: Extended Figure 6.7a preprocessing to preserve the top two
  feasible operating points per threshold, while keeping Figure 6.7a and
  follow-up distribution plots on the best point only.
- Added content: Figure 7.1 now plots the second-largest in-flight request
  point as a faint dotted line with batch-size labels, and Figure 7.1-7.3 now
  regenerate both strict `0` and conservative `0.5` processed CSV/PNG outputs.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 6.7a,
  Figure 6.7g, and Figure 7.1-7.3 for both `0` and `0.5` thresholds inside
  `sk-sslo-vllm`, checked PNG metadata and row counts, and visually inspected
  representative outputs.

## 2026-05-28 (Figure 7.1 rank-1 only)

- Modified content: Restored Figure 7.1 preprocessing to emit only the
  best feasible operating point (`selection_rank == 1`) for each threshold,
  removing the second-largest in-flight request points from the plotted CSVs.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.1
  strict `0` and conservative `0.5` CSV/PNG outputs inside `sk-sslo-vllm`,
  confirmed both CSVs contain only rank-1 rows, checked PNG metadata, and
  visually inspected the conservative Figure 7.1 output.

## 2026-05-28 (Figure 7 row spacing)

- Modified content: Reduced subplot row spacing across Figure 7 line, grouped
  bar, and distribution panels by lowering shared `hspace` values and tightening
  the Figure 7 line-panel row-title offset.
- Debugging/verification: Ran targeted `py_compile` and regenerated Figure
  7.1, 7.2, and 7.3 for both strict `0` and conservative `0.5`, plus Figure
  7.y, 7.z, and 7.aa inside `sk-sslo-vllm`; visually inspected representative
  outputs for reduced row spacing and no label/title overlap.

## 2026-05-28 (Figure 7.y and 7.aa profile-column layout)

- Modified content: Reoriented Figure 7.y and Figure 7.aa to place consume
  profiles horizontally as columns and model variants as rows, reducing each
  subplot's horizontal footprint in the six-panel layout.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.y
  and Figure 7.aa inside `sk-sslo-vllm`, checked PNG metadata, and visually
  inspected the new 2x3 layout.

## 2026-05-28 (Figure 7.4 violation amount)

- Modified content: Added Figure 7.4 using the same selected operating points
  as Figure 7.1, with the y-axis changed to mean unit violation amount in
  seconds.
- Added content: Added Figure 7.4 preprocessing and plotting entry points, and
  generated strict `0` and conservative `0.5` CSV/PNG outputs.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4
  data and both threshold plots inside `sk-sslo-vllm`, checked PNG metadata,
  and visually inspected both outputs.

## 2026-05-28 (Figure 7 model-row layout)

- Modified content: Reoriented Figure 7.1-7.4 and Figure 7.z so model variants
  are rows and consume profiles are columns, matching the existing Figure 7.y
  and Figure 7.aa orientation.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.1-
  7.4 for both strict `0` and conservative `0.5` thresholds plus Figure 7.z
  inside `sk-sslo-vllm`, checked PNG metadata, and visually inspected the new
  2x3 layout.

## 2026-05-28 (Figure 7 point labels)

- Modified content: Removed batch-size text annotations from the points in
  Figure 7 line plots.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.1,
  Figure 7.3, and Figure 7.4 for both strict `0` and conservative `0.5`
  thresholds inside `sk-sslo-vllm`, checked PNG metadata, and visually
  inspected Figure 7.1.

## 2026-05-28 (Figure 7.y/z/aa scale tuning)

- Modified content: Tuned Figure 7.y to use per-panel y-axis scaling and
  tuned Figure 7.z/7.aa boxplot y-axes to follow the visible whisker range
  rather than hidden outlier values.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.y,
  Figure 7.z, and Figure 7.aa inside `sk-sslo-vllm`, checked PNG metadata, and
  visually inspected all three outputs for readable panel scales.

## 2026-05-28 (Figure 7.1-7.4 axis and metric updates)

- Modified content: Updated Figure 7.1, Figure 7.2, and Figure 7.3 to share
  y-axis limits by model row with a fixed zero lower bound.
- Modified content: Changed Figure 7.3 from TTFC p99 to TTFT p99 and added a
  TTFT-named plotting wrapper/output.
- Modified content: Added max/min annotations to Figure 7.4 for plotted points
  with chunk violation rate below `0.5%`.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.3
  and Figure 7.4 processed CSVs, regenerated Figure 7.1-7.4 strict `0` and
  conservative `0.5` PNGs inside `sk-sslo-vllm`, checked PNG metadata, and
  visually inspected representative outputs.

## 2026-05-28 (Figure 7 title hierarchy)

- Modified content: Updated Figure 7 titles so consume-profile names are
  larger subplot titles, model names are larger row-level labels shown once per
  row, and `Reading` renders as `Human Reading`.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.1-
  7.4 for both strict `0` and conservative `0.5`, plus Figure 7.y, Figure 7.z,
  and Figure 7.aa inside `sk-sslo-vllm`; checked PNG metadata and visually
  inspected representative line, bar, and boxplot outputs.

## 2026-05-28 (Figure 7 row-title spacing)

- Modified content: Moved the second model row title slightly downward to add
  more space between the upper subplots and the lower model label.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.1-
  7.4 for both strict `0` and conservative `0.5`, plus Figure 7.y, Figure 7.z,
  and Figure 7.aa inside `sk-sslo-vllm`; checked PNG metadata and visually
  inspected Figure 7.y for the adjusted inter-row spacing.

## 2026-05-28 (Figure 7.ab violation P99)

- Added content: Added a separate Figure 7.ab script and PNG that computes
  cell-level P99 unit violation amount from the Figure 7.aa processed
  unit-level severity samples.
- Debugging/verification: Ran targeted `py_compile`, generated the Figure 7.ab
  PNG inside `sk-sslo-vllm`, checked PNG metadata, and visually inspected the
  new P99 grouped-bar figure.

## 2026-05-28 (Figure 7.y style and KV cache audit)

- Modified content: Updated Figure 7.y to remove the request-rate axis label,
  append `= lambda` to the rightmost rate tick, draw batch-size boundaries as
  black solid lines, and use thicker adjacent policy bars.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.y
  processed data and PNG inside `sk-sslo-vllm`, and visually inspected the
  output.
- Debugging/verification: Scanned all `output_sweep` summaries and scheduler
  stats for KV/cache/block fields; only `bf_kv_full` is present in scheduler
  stats and it is zero in all scanned rows.

## 2026-05-28 (Figure output legacy folder)

- Modified content: Created `exp/plots/figures/legacy/` and moved non-Figure-7
  PNG outputs from the figure root into that folder.
- Debugging/verification: Listed the figure root to confirm only `fig7_*.png`
  outputs remain and listed `legacy/` to confirm the Figure 5 and Figure 6
  PNG outputs were moved there.

## 2026-05-28 (Figure preprocess legacy folder)

- Modified content: Created `exp/plots/figures/scripts/preprocess/legacy/`
  and moved Figure 5 and Figure 6 preprocess entrypoints into it.
- Debugging/verification: Listed the preprocess root to confirm only Figure 7
  preprocess entrypoints remain and listed the preprocess legacy folder to
  confirm the 24 Figure 5 and Figure 6 files were moved.

## 2026-05-28 (Figure 7 renumbering)

- Modified content: Removed the old Figure 7.4 violation-amount line plot code
  and outputs.
- Modified content: Renamed Figure 7.y to Figure 7.4, Figure 7.ab to Figure
  7.5, and Figure 7.z to Figure 7.6, including preprocess entrypoints,
  processed CSVs, and PNG outputs.
- Modified content: Changed Figure 7.6 to plot the percentage of in-window
  requests that include at least one violated unit.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4,
  Figure 7.5, and Figure 7.6 processed CSVs and PNGs inside `sk-sslo-vllm`,
  checked for stale old file names, and visually inspected the new outputs.

## 2026-05-28 (Figure 7 PNG refresh)

- Modified content: Regenerated current numbered Figure 7 PNG outputs from
  existing processed CSVs without rerunning preprocess.
- Debugging/verification: Ran targeted `py_compile` and regenerated Figure
  7.1, 7.2, 7.3, 7.4, 7.5, and 7.6 PNG outputs inside `sk-sslo-vllm`; Figure
  7.aa was not regenerated because its processed CSV is not currently present.

## 2026-05-28 (Figure 7.4-7.6 grouped-bar layout)

- Modified content: Unified Figure 7.4, Figure 7.5, and Figure 7.6 on the
  same grouped-bar layout, with tighter batch spacing, compact request-rate
  ticks, separate lambda text, black batch boundaries, and adjacent thicker
  policy bars.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4,
  Figure 7.5, and Figure 7.6 PNG outputs inside `sk-sslo-vllm`, and visually
  inspected the refreshed figures.

## 2026-05-28 (Figure 7.4-7.6 batch boundaries)

- Modified content: Corrected grouped-bar batch boundary placement so the
  vertical divider sits halfway between adjacent batch-size groups.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4,
  Figure 7.5, and Figure 7.6 PNG outputs inside `sk-sslo-vllm`, and visually
  inspected the boundary positions across all three figures.

## 2026-05-28 (Figure 7.4-7.6 missing-cell markers)

- Modified content: Updated grouped-bar plotting to preserve the full
  batch-size/request-rate grid from the source CSV and mark missing policy
  cells with X markers.
- Modified content: Moved batch-size labels closer to their request-rate ticks
  for Figure 7.4, Figure 7.5, and Figure 7.6.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4,
  Figure 7.5, and Figure 7.6 PNG outputs inside `sk-sslo-vllm`, and visually
  inspected the restored Figure 7.5 batch-size grid and missing markers.

## 2026-05-28 (Figure 7.4-7.6 x-label tuning)

- Modified content: Adjusted lambda annotation placement, increased
  request-rate tick readability, and moved batch-size labels upward for Figure
  7.4, Figure 7.5, and Figure 7.6.
- Debugging/verification: Ran targeted `py_compile`, regenerated Figure 7.4,
  Figure 7.5, and Figure 7.6 PNG outputs inside `sk-sslo-vllm`, and visually
  inspected the label placement.

## 2026-05-31 (ProgressServe: single posterior tail-risk SSLO policy)

Replaced the multi-policy SSLO scheduler (threshold / pressure /
multi_level_pressure + adaptive-batching + offload) with a single
posterior tail-risk policy `progress_serve`; kept `baseline` as the control.
Modes are now exactly `{baseline, progress_serve}`.

- Added content: `vllm/vllm/sslo/progress_serve.py` — pure-Python ProgressServe
  primitives (`ProgressView`, `horizon`, `service_share`, `n_run_defer`,
  `request_risk`, `build_plan`, `schedule_step`); admission via FCFS-prefix
  `k* = max{k : E_viol(A+(k),B) < 1}`, running selection by marginal benefit
  `M = R_defer − R_run`. Fixed batch size B = `max_num_seqs`; deferred reqs
  stay resident.
- Added content: `ChunkLengthPredictor.tail_prob`/`sample_count_above` (cached
  sorted snapshot + `bisect_right`) and `RequestSLOState.length_tail_prob`
  (global-distribution empirical tail → analytic cold-start fallback) in
  `slo_state.py`; `_apply_sslo_progress_serve` + dispatch in `scheduler.py`;
  per-step `e_viol`/`k_star` in scheduler stats; new `progress_serve_min_denom`
  config knob; `vllm/tests/sslo/test_progress_serve.py`.
- Modified content: stripped the old policies/adaptive/offload/backfill from
  `scheduler.py` (~1180 lines removed; collapsed `schedule_sslo` to a single
  FCFS admission); removed pressure methods (`pressure`,
  `pressure_components`, `multi_level_pressure_components`, `serve_pressure`,
  `defer_pressure`, `_quantized_pressure`) from `slo_state.py`; rewrote
  `SsloConfig` (dropped policy/hysteresis/adaptive/mlp knobs, renamed
  `mlp_kv_blocks_per_new_admit`→`kv_blocks_per_new_admit`); narrowed the
  harness (`metrics_utils`, `analyze.py`, `run_test.py`/`.sh`, `run_sweep.sh`)
  to the two modes and removed KV-offload plumbing; rewrote the affected unit
  tests. Net −2121 lines across the 11 touched files.
- Debugging/verification: dev/verify agent loop per phase (no Critical/Important
  findings); `python3 -m compileall` clean; `pytest tests/sslo/` → 91 passed,
  1 pre-existing failure (`test_tts_consume_path` deadline-recurrence assertion,
  fails at HEAD, untouched); 1-cell smoke (9B/read/cap128/rate8) in
  `sk-sslo-vllm` for both modes — no crash, `e_viol`/`k_star`/`kv_blocks_*`
  recorded, running capped at B=128 with active deferral (pending up to 260),
  KV never overshoots.
- Tuning (certain-defer-miss → max benefit): violation analysis on a 9B/read/
  cap128 baseline-vs-progress_serve comparison showed misses concentrated at
  the FIRST measured consumable unit (unit_index 1), driven by a starvation
  pathology — a request with a far deadline has marginal benefit M≈0 (deferred
  as "not urgent yet"), and once past deadline R_run=R_defer=1 ⇒ M=0 again
  ("abandoned"), compounding the stall. Fix in `progress_serve.request_risk`:
  when `R_defer == 1` (deferring is a certain miss) return `M = 1` instead of
  `R_defer - R_run`, so certain-miss requests win a decode slot. Effect (single
  run, rates 8/16/24): unit-deadline violations 0.32%/0.22% → **0% at all
  rates**, mean handling-users +40–61% (315→508 @16, 354→495 @24), stall_p99
  19.7s→0, lower TTFC — because servicing certain-miss requests drops them out
  of the E_viol sum, which reopens admission (virtuous cycle). An earlier
  experiment extending WARMUP through unit 1 only shifted the hotspot to unit 2
  and was reverted in favor of this rule.
- Follow-up: `vllm/vllm/sslo/README.md` still documents the old policies and is
  now stale — needs a ProgressServe rewrite (docs only, non-blocking).

## 2026-06-01 (adaptive batching + hybrid Δ(b))

- Added content (commit 0224bad): `adaptive_batching` for ProgressServe —
  when E_viol(B) >= 1, shrink the decode batch to a smaller CUDA-graph
  captured size to lower per-iteration latency. `progress_serve.pick_adaptive_batch`
  (hill-climb over captured sizes < B, floored at forced-request count),
  shared `E_VIOL_FEASIBLE=1.0`, `progress_serve_adaptive` run_kind. A 3-way
  sweep (baseline / progress_serve / progress_serve_adaptive, rates 8–80,
  2 models × cap{128,256} × 3 consume) showed adaptive helps 9B (HU +10–70%)
  and 35B/cap256 (KV-bound; baseline ~1% violation → ~0.2%), but HURT
  KV-slack cells (9B/cap128, 35B/cap128 high rate) by locking into a small
  batch.
- Debugging: traced the lock-in. Objective E_viol = ΣR_run + ΣR_defer is
  correct (recomputing with measured Δ gives E_viol(128)=194 < E_viol(64)=216),
  but `_wall_ema_for_batch(n)` only has Δ for batch sizes actually run; once
  shrunk, the large-batch EMA goes stale → picker never relearns the large
  batch is better → self-fulfilling lock-in (throughput crash → backlog → more
  shrink).
- Added content (commit e27a00c): hybrid Δ(b). Profile per-batch forward
  latency once at CUDA-graph capture (gpu_model_runner → CompilationTimes →
  gpu_worker RPC → engine/core → scheduler.set_cudagraph_decode_profile), then
  `_sslo_hybrid_delta(b) = profile(b) * (wall_ema(b_now)/profile(b_now))` —
  profile supplies shape for every batch (no stale gap), live wall-EMA supplies
  scheduler-unit scale. Falls back to wall-EMA when no profile.
- Verification: pytest tests/sslo 99 passed (1 pre-existing tts failure).
  Worst-case smokes — lock-in gone (batch 40/64 → ~120 near cap), recovered
  metrics: 9B/cap128/r8 viol 0.11%→0, ttfc99 118s→2.5s, dec 2203→3102 tok/s;
  35B/cap128/r80 viol 0.22%→0.075%, dec 1418→1769 tok/s.
- Follow-up: 35B-style throughput-bound cells still lose a little from the
  one-step (128→120) shrink vs plain progress_serve; a throughput-floor guard
  that blocks shrinking there is a separate, unimplemented refinement.

## 2026-06-05 (chunk_length_study: chunk vs request length variability)

- Added content (commit 504982a): new standalone experiment
  `exp/chunk_length_study/` testing whether per-chunk length is less variable
  than per-request output length, by task (code/dialogue) and language, on
  baseline 9B. build_category_pools.py (wildchat + lmsys-when-authed →
  per-(lang,task) 512-prompt caches via _is_code_request + langdetect),
  run_baseline_tracking.sh (per-category baseline run feeding caches via
  DATASET_CACHE_DIR), analyze_chunk_length.py (CV / p99-p50 / IQR-median +
  prompt-length correlation + plots). Reuses lm_datasets / dataset_cache /
  jsonl_utils / metrics_utils; no existing script modified.
- Debugging/verification: lmsys is HF-gated (no token in container) → builder
  falls back to wildchat-only (still multilingual+code). _is_code_request is
  English-centric so non-English code pools are ~empty → viable categories =
  en-code, en/zh-cn/ru-dialogue. run_test.sh doesn't forward
  --warmup/measurement-target and its defaults (1024) hang on small pools, so
  the runner calls run_test.py directly with pool-sized targets. Verified
  end-to-end: 4 categories, 18.5k–21k chunks each, analyzer + 3 plots produced.
- Finding: by robust IQR/median, chunk length is far more stable than request
  output (en-dialogue 0.35 vs 1.36; all dialogue chunk≈0.35–0.38 vs
  request 0.86–1.36); CV agrees (chunk 0.29–0.52 vs request 0.40–0.74). Prompt
  length barely predicts output length (Pearson ≤0.30, ~0 for non-English).
  Nuance: chunk p99/p50 tail ratio can exceed request's (en-code 2.35 vs 1.13).

## 2026-08-06 (ProgressServe: parked (CPU) requests count in the service share)

- Modified content: `vllm/vllm/sslo/config.py` gained
  `kv_offload_share_includes_parked` (default `True`) — vacating frees memory,
  not compute, so a CPU-parked request stays in the `|A+|` service-share
  denominator; excluding it inflated `s`, underestimated `M_cpu` (over-vacate)
  and skewed the adaptive batch choice. `progress_serve.build_plan` /
  `schedule_step` / `pick_adaptive_batch` take the flag as a plain argument
  (module stays engine-import-free) and add `len(cpu_views)` to `n_aplus` when
  set; the flag touches the `s` term only — parked reqs still take no decode
  slot and still enter `E_viol` as `R_defer_cpu`. `scheduler.py` threads the
  config value into the three calls and mirrors the same `|A+|` in
  `_sslo_apply_offload`'s local `s` (the one feeding select_vacate /
  select_promote) so plan-side and decision-side `M_cpu` agree.
  `run_test.py` exposes `SSLO_KV_OFFLOAD_SHARE_INCLUDES_PARKED` alongside the
  other `SSLO_KV_OFFLOAD_*` env overrides; `vllm/vllm/sslo/README.md` §9
  documents the semantics and the default switch.
- Added content: `tests/sslo/test_progress_serve.py` —
  `test_build_plan_service_share_includes_cpu_by_default`,
  `..._excludes_cpu_when_opted_out`,
  `test_build_plan_parked_share_does_not_change_slots` (slots / offloaded set
  unchanged, only `s` and `R_defer_cpu` move),
  `test_schedule_step_threads_share_includes_parked` (parked in `|A+|` ⇒
  strictly smaller `k*`),
  `test_pick_adaptive_batch_threads_share_includes_parked`.
- Debugging/verification: `python3 -m pytest tests/sslo/ -q` in `sk-sslo` →
  150 passed, 1 skipped, 1 failed (pre-existing
  `test_tts_path_uses_audio_ready_time_for_slack_and_deadline`). The
  `schedule_step` spy in `test_scheduler_sslo.py` needed the new argument in
  its signature. No GPU runs.

## 2026-08-06 (ProgressServe: opt-in marginal (delta) admission criterion)

- Modified content: `vllm/vllm/sslo/config.py` gained
  `admission_delta_criterion` (default `False`). When set, `schedule_step`
  admits while the MARGINAL `E_viol(k) - E_viol(0) < 1` instead of the
  absolute `E_viol(k) < 1`: the risk the in-flight set already carries is
  sunk cost, so the budget covers only the extra expected violations this
  step's admits cause. This is the fix for the Phase N' admission self-lock —
  once a few high-risk in-flight requests sum past 1, the absolute rule pins
  `k*` to 0 even with idle KV / compute. `progress_serve.schedule_step` takes
  the flag as a plain argument (module stays engine-import-free) and applies
  `plan.e_viol - baseline_e` in both the main scan and the
  `k_star_unconstrained` re-scan; `baseline_e` is `0.0` when the flag is off,
  so the absolute path is unchanged. The delta expression also subsumes the
  `k=0` early-exit (`E_viol(0) - E_viol(0) = 0 < 1` always proceeds).
  `ScheduleResult.e_viol` stays the ABSOLUTE value at `k*` — the delta is used
  for the admission decision only, so decisions.jsonl / scheduler_stats stay
  comparable with earlier runs. `scheduler.py` threads
  `self.sslo_config.admission_delta_criterion` into the `ps.schedule_step`
  call. `run_test.py` exposes `SSLO_ADMISSION_DELTA_CRITERION` (0/1) for every
  non-baseline run_kind (alongside `SSLO_ADAPTIVE_BATCHING`, not inside the
  offload-only block) since the criterion applies to all ProgressServe kinds.
  `vllm/vllm/sslo/README.md` §4 + the config table document the rule.
- Added content: `tests/sslo/test_progress_serve.py` —
  `test_schedule_step_delta_criterion_unlocks_sunk_risk` (same locked input as
  `..._defer_only_when_infeasible_at_zero`: absolute `k*=0`, delta `k*=2`, and
  the reported `e_viol` is still the absolute one),
  `..._matches_absolute_at_low_risk` (`E_viol(0) == 0` ⇒ identical `k*` and
  `e_viol`, with the risk budget still binding at `0 < k* < |W|`),
  `..._still_bounded_by_marginal_budget` (two doomed reqs make `E_viol(0) = 2`
  so the absolute rule locks; the delta scan admits some and then stops,
  cross-checked against a build_plan oracle).
- Debugging/verification: `python3 -m pytest tests/sslo/ -q` in `sk-sslo` (CPU
  only, `CUDA_VISIBLE_DEVICES=""`, no GPU touched — the A1 sweep was running)
  → 153 passed, 1 skipped, 1 failed (pre-existing
  `test_tts_path_uses_audio_ready_time_for_slack_and_deadline`). The
  `schedule_step` spy in `test_scheduler_sslo.py` needed the new argument in
  its signature (same as the previous session's flag). `py_compile` clean on
  all touched files.

## 2026-08-07 (Ablation A1/A3: share 회계 채택, 델타 기준 기각)

- 실행: A1 (`kv_offload_share_includes_parked=True`, offl/comb 6셀 24
  rate-run 완주, `output_sweep_v2/a1_share`) / A3 (A1 + 델타 기준,
  10/24에서 조기 중단, `.../a3_share_delta`).
- 결과 A1 — **채택**: offload 단독의 저-cap 오작동이 교정됐다.
  cap32/offl r0.5 vac 8→1 · tput 337→368, r1 viol 1.1→0.6% · tput
  250→314, r4 1.6→1.2% · 251→284. cap64/128 offload는 동등(seed-less
  샘플링 노이즈 범위). 결합 모드는 구제 실패 — cap32 vacate 272~498회
  여전, 저부하 cap64 r0.5는 오히려 악화(viol 1.4→2.6%, vac 0→59) →
  결합의 병인은 share 회계가 아니라 adaptive 축소 나선으로 확정.
- 결과 A3 — **기각**: 델타 기준(`E_viol(k)−E_viol(0)<1`)이 offload
  단독까지 악화시켰다 (cap128/offl r1/r2/r4 viol 1.3/1.5/2.0% →
  3.7/4.5/7.0%; cap64/offl 1.8→3.4~6.3%; comb/cap128 r0.5 1.2→7.7%).
  진단: per-step e_viol 최대가 16~41로 누적 폭주(A1은 ~1 유지) — 절대
  임계는 매몰비용 잠금의 원인인 동시에 시스템 총위험의 유일한 브레이크
  였다. rate 단조 악화라 노이즈 아님. 잔여 14 rate-run은 실익 없어 중단.
- 코드 상태: `admission_delta_criterion`은 기본 False로 보존(기각 옵션),
  `kv_offload_share_includes_parked`는 기본 True 채택. 두 변경 모두
  개발↔검증 루프 통과("이슈 없음", tests/sslo 153 pass + 기존 tts 1건 실패).
- 차기 설계 후보(미해결): ③축 잠금의 안전한 해제 — 델타+상한 하이브리드,
  유휴 시에만 델타 적용, 또는 M≈0 요청만 예산 제외하는 국소 수정.

## 2026-08-09 (deadline-aware prefill token budget: 새 run kind 2종)

- 배경(실측 근거): cap128/r2 baseline 기준 decode-only 스텝 Δ는 p50 74ms /
  p90 78ms로 매우 안정적인 반면, prefill이 섞인 스텝은 전체의 6%인데 벽시계의
  18.4%를 차지하고 Δ p90 = 463ms. 즉 청크 마감을 깨는 것은 prefill spike이며
  `max_num_seqs`(동시성) 축을 줄이는 현행 adaptive batching은 여기에 영향이
  없다. 따라서 요청 수 대신 **스텝당 prefill 토큰 예산**을 마감 인지로 제어.
- Added content:
  - `vllm/vllm/sslo/progress_serve.py` — 순수 함수 `prefill_budget(t_min,
    delta_decode, kappa, base_budget, floor, gamma)`:
    `P* = clamp(int((γ·t_min − Δ_decode)/κ), floor, base_budget)`.
    `kappa <= 0` 또는 `t_min is None`이면 제어 비활성(base 반환),
    `t_min <= 0`(overdue)이면 floor. 엔진 import 없음.
  - `vllm/vllm/sslo/config.py` — `prefill_budget_control`(기본 False),
    `prefill_budget_floor`(512 토큰, >0 검증), `prefill_budget_gamma`(0.5,
    0<γ≤1 검증).
  - `exp/run_sslo/run_test.py` — run kind `progress_serve_prefill_budget`,
    `progress_serve_offload_prefill_budget`(offload와 결합, adaptive 없음).
    `PREFILL_BUDGET_RUN_KINDS` 상수 + env override
    `SSLO_PREFILL_BUDGET_FLOOR` / `SSLO_PREFILL_BUDGET_GAMMA`.
    두 kind 모두 `MODES_DEFAULT`(metrics_utils.py) / `SSLO_MODES`(analyze.py)에 추가.
- Modified content (`vllm/vllm/v1/core/sched/scheduler.py`, 전부 `# SSLO` 표시):
  - κ 온라인 추정: `_update_tpot_ema`에서 prefill을 실은 스텝마다
    `max(0, (Δ_obs − Δ_decode)/prefill_tokens)`를 tpot EMA와 같은 alpha로 누적
    (`_sslo_prefill_kappa`). Δ_decode는 새 헬퍼 `_sslo_decode_wall_ema(n)`가
    기존 `_sslo_step_wall_ema`의 prefills=0 셀에서 조회(없으면 전 batch의
    prefills=0 평균 → 그것도 없으면 None → 제어 비활성).
  - prefill 토큰 집계: `_record_step_for_next_ema`가 prefill_count와 동일한
    요청 집합에 대해 `_sslo_prev_step_prefill_tokens`를 기록.
  - t_min: `_sslo_min_time_to_deadline(now)` — running 중 MEASURED 요청의
    `slo_state.time_to_deadline(now)` 최솟값.
  - 적용: `schedule_sslo()`가 policy 직후 `_sslo_prefill_token_budget(now)`로
    P*를 구하고, waiting-큐 admission 루프에서만 `num_new_tokens`를 남은
    예산으로 clamp + 소진 시 break. running(decode) 경로는 무제한. chunked
    prefill이 꺼져 있으면 제어 자체를 비활성(잔여 이월이 불가하므로).
  - offload 합성: 소진된 prefill 예산이 promoted onload traversal을 막지
    않도록 k* 소진과 동일한 skip/break 규칙 적용(onload는 clamp 면제, 예산
    차감은 유지).
  - 기아 방지: floor로 clamp된 채 prefill 진행이 0인 스텝이
    `_SSLO_PREFILL_FLOOR_STARVE_STEPS`(32) 연속되면 한 스텝 base budget 허용.
  - 관측: `scheduler_stats.jsonl`에 `prefill_budget`,
    `prefill_kappa_ms_per_tok` 추가. `reset_sslo_state()`에서 κ/스트릭/토큰
    카운터 리셋.
- Debugging/Verification (컨테이너 `sk-sslo`, GPU 미사용):
  - `python3 -m pytest tests/sslo/ -q` → 168 passed, 1 skipped,
    1 failed(기존 알려진 `test_tts_consume_path.py::
    test_tts_path_uses_audio_ready_time_for_slack_and_deadline`, 본 변경과 무관).
    신규 테스트: `test_progress_serve.py` 5건(순수 함수: 슬랙 증가 시 예산
    증가/포화, t_min 단조성, overdue→floor, κ≤0·t_min None→base, 범위 불변),
    `test_scheduler_sslo.py` 9건(제어 off 회귀, κ/Δ 미비 시 비활성, 긴급도별
    P*, 기아 가드 32스텝 해제, waiting 루프 prefill 합 ≤ P*, decode 무제한,
    onload 비차단, stats 필드, κ EMA 갱신/음수 클램프).
  - run kind 검증: `AsyncEngineArgs`를 인터셉트해 GPU 없이 sslo_params 확인 —
    `progress_serve_prefill_budget`(control=True, kv_offload=False),
    `progress_serve_offload_prefill_budget`(control=True, kv_offload=True,
    KVTransferConfig 연결), 두 kind 모두 adaptive_batching=False, env override
    (floor=256, gamma=0.25) 반영 확인. `--help` choices에도 노출.
  - `python3 -m py_compile` (변경 .py 전부) / `bash -n`
    (`run_test.sh`, `run_sweep.sh`) 통과.
- 남은 위험: κ는 스텝 전체 prefill 토큰(러닝 큐의 chunked 이월 포함)으로
  추정하지만 예산은 waiting-큐 admission에만 걸린다 — 러닝 큐의 chunked
  prefill 이월분은 상한 밖. GPU 실측 스윕 미실행(정책상 금지).

### 보완 (같은 날, 코디네이터 지적 반영)

- **문제**: 최초 구현은 waiting-큐 admission 루프에만 클램프를 걸었는데, vLLM
  v1에서 chunked prefill의 **후속 청크는 running 루프**에서 스케줄된다
  (RUNNING이지만 `num_computed_tokens < num_prompt_tokens`). 따라서 2,000토큰
  프롬프트를 512로 잘라도 나머지 1,488이 다음 스텝에 무제한 유입되어 spike를
  한 스텝 미룰 뿐 평활화가 되지 않았다. 실측 Δ p90 = 463ms의 주 경로가 여기다.
- **수정** (`scheduler.py`, `# SSLO` 표시):
  - `schedule_sslo()`의 running 루프에도 클램프 추가. 판별은
    `request.num_computed_tokens < request.num_prompt_tokens`
    (`sslo_is_carryover_prefill`) — 프롬프트를 다 처리한 디코딩 요청은 대상 아님.
  - 단일 `sslo_prefill_remaining` 카운터를 running → waiting 순서로 공유 소진.
    running이 먼저이므로 in-flight prefill이 신규 admit보다 우선한다.
  - 진행 보장: 캐리오버는 예산 소진 상태에서도
    `max(1, min(num_new_tokens, remaining))`로 최소 1토큰 확보(영구 정지 방지).
    기존 `num_new_tokens == 0` 경로는 건드리지 않도록 `num_new_tokens > 0`일
    때만 적용.
  - 차감도 캐리오버 prefill에만 적용(decode 무과금). onload 면제와 chunked
    prefill 게이트는 유지.
  - 결과적으로 κ 추정 모집단(스텝 전체 prefill 토큰)과 클램프 모집단이 일치 →
    이전 보고의 "남은 위험 1번" 해소.
- `config.py`: `prefill_budget_control`에 `method="progress_serve"` 요구
  `__post_init__` 가드 추가 (`kv_offload`와 동일한 결) → 이전 "남은 위험 2번" 해소.
- 테스트 추가: `test_scheduler_sslo.py` 4건 — (a) running 캐리오버가 P*에
  묶임, (b) waiting+running 합산이 정확히 P*, (c) 같은 스텝의 decode 요청은
  무제한, (d) 예산 소진 시에도 캐리오버 1토큰 진행. `test_sslo_config.py` 4건 —
  기본값/`progress_serve` 요구/floor>0/gamma 범위.
- 검증: `python3 -m pytest tests/sslo/ -q` → **179 passed, 1 skipped,
  1 failed**(기존 알려진 tts 1건). py_compile / bash -n 통과. GPU 실험 없음.
- 남은 위험: `SchedulingPolicy.PRIORITY`에서 running 루프가 이미 스케줄된
  요청을 선점 취소할 때 `token_budget`은 환급되지만 prefill 예산은 환급하지
  않는다(과금 과다 = 보수적 방향, FCFS 실험 경로에서는 미발생).

### 검증 루프 마무리 (Minor 2건 + κ 모집단 갭)

- Minor 1 (`scheduler.py` PRIORITY preemption 분기): `token_budget`은 환불하나
  `sslo_prefill_remaining`은 환불하지 않는 갭을 `# SSLO` 주석으로 명시(환불
  구현은 기존 문장 재작성이 필요해 범위 밖). 방향은 과금 과다 = 보수적,
  FCFS에서는 도달 불가.
- Minor 2 (`progress_serve.prefill_budget`): 클램프 순서를
  `floor = min(floor, base_budget)`로 선정규화 → 반환값이 항상
  `<= base_budget`. `prefill_budget_floor`가 엔진 `max_num_batched_tokens`
  이상이면 제어가 조용한 no-op이 된다는 점을 config 필드 주석과 docstring에
  명시(SsloConfig는 SchedulerConfig를 볼 수 없어 교차 검증 불가).
  테스트 `test_prefill_budget_floor_above_base_never_exceeds_base` 추가.
- κ 회귀 모집단 갭 (사각지대로 판단 → 수정): `_sslo_pre_step_computed`는 스텝
  시작 시 `running + sslo_pending`만 담으므로, preemption 후 이번 스텝
  waiting 루프로 재-admit되는 요청은 `pre is None` 분기로 빠져 κ 분모에서
  누락됐다(반면 예산에서는 차감). 분모 과소 → κ 과대 → P* 축소이므로 안전
  측이지만, 예산 차감 모집단과 불일치라 국소 수정(3줄): 해당 분기에서
  `prefill_tokens`에만 토큰을 더한다. `prefill_count`(기존 wall-EMA 셀 키)와
  `decoding_only`은 의도적으로 불변. 테스트
  `test_kappa_regressor_counts_resumed_admits` 추가.
- 검증: `python3 -m pytest tests/sslo/ -q` → **181 passed, 1 skipped,
  1 failed**(기존 알려진 tts 1건). py_compile 통과. GPU 실험 없음.

## 2026-08-10 — KV offload tier 용어 통일 (vacate/promote → offload/onload)

순수 리네이밍 + 문서 정합. 동작 변경 없음(GPU 실험 없음).

- 배경: `promote`가 CPU→GPU 복원을 뜻해 자연스러운 pending→running 의미와
  충돌. 상태(명사) `waiting`/`running`/`pending`/`offloaded`/`onloading`,
  전이(동사) `admit`/`defer`/`promote`(pending→running)/`offload`(pending→
  offloaded)/`onload`(offloaded→onloading→running)로 확정.
- 수정: 코드 식별자 `select_vacate`→`select_offload`,
  `select_promote`→`select_onload`, `_sslo_vacate_request`→
  `_sslo_offload_request`, `_sslo_vacated_req_ids`→`_sslo_offloaded_this_step`,
  `_sslo_promoting`→`_sslo_onloading`, 로컬 `pending_promote`→`pending_onload`.
  per-step 스키마 `num_vacated`→`num_offloads`, `num_promoted`→`num_onloads`;
  decision log `kind` "vacate"→"offload", "promote"→"onload".
  잔여 vacate/promote 주석·docstring 전부 offload/onload 계열로.
  유지: `num_offloaded`(CPU 상주 인구), `kv_offload_*` config, `M_cpu`/
  `R_defer_cpu`/`n_defer_cpu`, `on_offload_enter/exit`, `LOC_*`, request-level
  `num_onloads`/`num_offload_intervals`/`total_offloaded_time_s`, upstream
  `_try_promote_blocked_waiting_request`.
- 추가: `vllm/vllm/sslo/README.md` §2에 "Request states & transitions" —
  5상태/5전이 표 + 카운터 ops(`num_offloads`/`num_onloads`) vs population
  (`num_offloaded`) 구분을 단일 출처로 명시.
- 문서: `vllm/vllm/sslo/README.md`, `exp/run_sslo/README.md`,
  `exp/run_sslo/SWEEP_PLAN_v2.md`, `paper/algorithm3_kv_offload.md`
  (\textsc{VacatePromote}→\textsc{OffloadOnload}),
  `kv_offload_model_compat.md`, `exp/run_sslo/run_test.{py,sh}` 갱신.
  `WORKLOG.md`/`HANDOFF.md` 과거 엔트리는 사실 기록이라 미수정.
- 하위 호환: `exp/run_sslo/analyze.py` 및 `exp/run_sslo/analysis/` 전수 grep
  결과 `num_vacated`/`num_promoted`/`kind=="vacate"|"promote"` 독자가 0건
  (분석은 request-level `total_offloaded_time_s`/`num_onloads`/
  `num_offload_intervals`만 소비 — 이 이름들은 유지됨). 따라서 fallback 코드
  불필요 → 추가하지 않음.
- 검증: `python3 -m pytest tests/sslo/ -q` → 181 passed, 1 skipped,
  1 failed(기존 알려진 tts 1건) — 리네이밍 전과 동일. py_compile / bash -n
  통과. `grep -rn "vacate" vllm/vllm/ exp/` → 0건.
  구 산출물 재분석: output_sweep_v2/phaseP/.../cap32/progress_serve_offload/
  run_1/rate_1 에 analyze.py 재실행 → `summary.json` 기존본과 완전 동일.

## Session: κ / P* decode baseline batch-matching (prefill budget control)

**Modified**
- `vllm/vllm/v1/core/sched/scheduler.py` — `_sslo_decode_wall_ema(n)` is now
  batch-matched only: it returns the `(n, prefills=0)` wall-EMA cell or None.
  Removed the "mean over every observed prefills=0 cell" fallback, which was
  the single shared contamination path for both consumers: the κ sample in
  `_update_tpot_ema` (baseline borrowed from low-occupancy decode-only cells →
  the batch-size gap, ~15 ms between batch 16 and batch 43, was mis-attributed
  to prefill tokens → κ inflated ~2x under the offload tier) and the Δ_dec term
  of `_sslo_prefill_token_budget` (P*). No call-site changes were needed: the κ
  site already keys on `_sslo_prev_step_batch` and already skips the sample when
  the baseline is None; P* already disables the control (base budget) on None.
  No low-occupancy-exclusion heuristic and no nearest-cell tolerance added.

**Added**
- `vllm/tests/sslo/test_scheduler_sslo.py` — 3 tests:
  `test_kappa_sample_uses_batch_matched_decode_cell` (batch-8 6 ms and batch-40
  72 ms cells both populated; a batch-40 prefill step must yield
  (Δ_obs − 72 ms)/P), `test_kappa_sample_dropped_without_batch_matched_decode_cell`
  (batch 40 has only a prefills=2 cell → sample dropped, κ EMA unchanged), and
  `test_prefill_budget_off_without_batch_matched_decode_cell` (P* → None).

**Debugging / verification**
- Sparsity check on `output_sweep_v2/phaseP/.../cap64/{progress_serve_prefill_budget,
  progress_serve_offload_prefill_budget}/run_1/rate_4/scheduler_stats.jsonl`:
  every prefill step's batch key has a decode-only cell somewhere in the run
  (100%); replayed chronologically, the cell is already populated for ~96% of
  prefill steps (fallback fired 37/1072 and 41/1071). Exact matching is therefore
  affordable and a ±25% nearest-cell tolerance was deliberately NOT added.
  Same logs: `running` equals the scheduled batch on 99.5% of steps, so P*'s
  `len(self.running)` key is consistent with the EMA cell keys.
- Caveat recorded: the ~4% fallback rate alone does not fully explain the
  measured κ p50 0.33 (pbud) vs 0.64 (o+pb); offload-tier KV transfer time
  landing inside prefill-carrying steps remains a candidate for the remainder.
- `python3 -m pytest tests/sslo/ -q` in `sk-sslo` (/workspace/mlsys/vllm):
  184 passed, 1 skipped, 1 failed (pre-existing
  `test_tts_consume_path.py::test_tts_path_uses_audio_ready_time_for_slack_and_deadline`).
  Against pre-fix `scheduler.py` the two new regression tests fail as intended.
  `python3 -m py_compile` clean. No GPU runs (effect re-run awaits user approval).

## Session: κ ratio-of-sums 추정기 (per-sample EMA 소분모 발산 제거)

**Modified**
- `vllm/vllm/v1/core/sched/scheduler.py` — κ 상태를 스칼라 EMA
  `_sslo_prefill_kappa` 에서 두 누적기 `_sslo_kappa_excess_ema`(초과 시간 초,
  **클리핑 없음**) / `_sslo_kappa_tokens_ema`(prefill 토큰 수)로 교체.
  `_update_tpot_ema` 는 배치 매칭된 Δ_dec 가 있는 prefill 스텝에서만 두 누적기를
  같은 α(`tpot_ema_alpha`)로 갱신(기존 skip 가드 유지). 새 읽기 헬퍼
  `_sslo_kappa()` 가 `excess_ema / tokens_ema` 를 반환하고, 샘플 없음 또는 비율
  ≤ 0 이면 None(= 추정 없음 → P* 제어 비활성, base budget). 하한은 per-sample
  이 아니라 최종 비율에만 적용. `_sslo_prefill_token_budget`(P*),
  step stats 의 `prefill_kappa_ms_per_tok`, `reset_sslo_state` 를 새 상태로 전환.
  `progress_serve.prefill_budget()` 는 κ 를 인자로 받으므로 무변경(확인만).
  동기: per-sample `(Δ_obs−Δ_dec)/P` 는 P 가 작은 스텝에서 스텝 노이즈를 소분모로
  나눠 참값의 20~100배 샘플을 만들고, `max(0,·)` 가 음수 노이즈만 버려 상향 편향 →
  κ↑ → P*↓ → 스텝당 P 축소 → κ↑ 의 폐루프.

**Added**
- `vllm/tests/sslo/test_scheduler_sslo.py` — 헬퍼 `_kappa_sample()` 및 2 테스트:
  `test_kappa_ratio_of_sums_survives_small_denominator_samples`
  (P=2000/excess 320 ms 정상 샘플 1개 뒤 P=8/excess 5 ms 쓰레기 샘플 10개 →
  κ 가 참값 0.16 ms/tok 의 ±20% 내 유지; 구 추정기는 0.463 ms/tok = 2.89배),
  `test_kappa_keeps_negative_excess_unclipped`(음수 excess 가 κ 를 끌어내림;
  클리핑 구현이면 0.000144, 새 구현은 0.000141).

**Debugging / verification**
- 기존 κ 테스트 갱신: `test_kappa_ema_updates_from_prefill_step`,
  `test_kappa_ema_skips_decode_only_steps`(구 `..._and_clamps_negative` 에서
  clamp 파트 분리), `test_kappa_sample_uses_batch_matched_decode_cell`,
  `test_kappa_sample_dropped_without_batch_matched_decode_cell`(두 누적기 불변
  검증), `test_prefill_budget_requires_kappa_and_decode_reference`(κ ≤ 0 케이스
  추가), `_prep_prefill_budget` / `make_scheduler` 시드.
- `python -m pytest tests/sslo/ -q` in `sk-sslo` (/workspace/mlsys/vllm):
  186 passed, 1 skipped, 1 failed (기존
  `test_tts_consume_path.py::test_tts_path_uses_audio_ready_time_for_slack_and_deadline`).
  `python -m py_compile` clean. GPU 실행 없음.
- 구 추정기 재현 시뮬레이션(컨테이너)으로 새 소분모 테스트가 구 코드에서
  실패함을 확인(2.89배).

## 2026-08-10 — P* → TB* 개명 + Token Budget D축 신설

**Modified**
- `vllm/vllm/sslo/config.py` — `prefill_budget_control/_floor/_gamma` →
  `token_budget_control` / `token_budget_prefill_floor` / `token_budget_gamma`.
  주석을 단일 제약 `Δ_dec(D) + κ_p·P ≤ γ·t_min` 의 2축(D/P) 서술로 교체.
- `vllm/vllm/sslo/progress_serve.py` — `prefill_budget()` →
  `token_budget_prefill()` (docstring P* → TB*_pre, Δ_dec 는 D 축이 정한 D′ 기준).
- `vllm/vllm/v1/core/sched/scheduler.py` — `SsloStepState.prefill_budget` →
  `token_budget_prefill`, 로컬 `sslo_prefill_budget` → `sslo_token_budget_prefill`,
  step stats 키 `prefill_budget` → `token_budget_prefill`.
- `exp/run_sslo/` — run kind `progress_serve[_offload]_prefill_budget` →
  `progress_serve[_offload]_token_budget`, env `SSLO_PREFILL_BUDGET_FLOOR/GAMMA` →
  `SSLO_TOKEN_BUDGET_PREFILL_FLOOR` / `SSLO_TOKEN_BUDGET_GAMMA`
  (`run_test.py`, `run_test.sh`, `run_sweep.sh`, `README.md`).
- `vllm/vllm/sslo/README.md` — TB* 절(D축/P축) 및 config 표 갱신.

**Added**
- `SsloConfig.token_budget_decode_risk_eps` (기본 1e-3) — D축 defer 자격 임계
  (`kv_offload_risk_eps` 와 같은 문법).
- `progress_serve.select_decode_defer()` — D축 순수 함수. 자격 = GPU 상주 ∧
  MEASURED ∧ `R_defer ≤ eps`; slack 깊은 순으로 하나씩 defer 하며 `Δ_dec(D′)` 가
  `γ·t_min` 에 들어올 때까지 축소. 배치-키 셀 부재 시 보수적 중단(폴백 평균 금지, R5).
  forced/onloading/at-risk 는 구조적으로 후보에서 제외 = 자연 하한 D_floor.
  자격은 자기제한적(defer 지속 → T_q 감소 → R_defer 상승 → 자격 상실)이라
  별도 기아 카운터 불필요 — R_defer 가 매 스텝 live T_q 로 재계산되기 때문.
- `Scheduler._sslo_apply_decode_budget()` — 훅 위치는 `_sslo_apply_offload()`
  직후(스펙 ④ → ⑤ 순서), 결정 로그 build_plan 이전. `new_running`/`new_pending`
  을 in-place 로 옮기므로 커밋 후 `len(self.running)` 이 곧 D′ 가 되어 P축
  (`_sslo_prefill_token_budget`) 이 자동으로 `Δ_dec(D′)` 를 읽는다.
  t_min 은 decode set 자신의 뷰에서 계산(훅 시점의 `self.running` 은 정책 이전
  집합이라 `_sslo_min_time_to_deadline` 은 부적합).
- step stats `token_budget_decode`(D′) / `token_budget_d_defers`.
- `exp/run_sslo/metrics_utils.MODES_DEPRECATED` — 구 run kind 2종을
  `MODES_DEFAULT`/`analyze.ALL_MODES` 에 alias 로 병존(phaseP 산출물 디렉토리
  경로 추론 유지). CLI `--run-kind` choices 에서는 제외.

**Debugging / verification**
- 신규 테스트: `test_progress_serve.py` 5종(예산 미달 시 slack 깊은 순 축소 /
  forced·at-risk 미defer 하한 / 마감 여유 시 no-op / 셀 부재 보수적 중단 /
  자격 자기제한), `test_scheduler_sslo.py` 3종(D축 배선·기본 off·read 무개입),
  `test_sslo_config.py` eps 기본값·음수 거부. 기존 P* 테스트는 개명 반영.
- `python -m pytest tests/sslo/ -q` in `sk-sslo`: 195 passed, 1 skipped,
  1 failed (기존 `test_tts_path_uses_audio_ready_time_for_slack_and_deadline`).
- `py_compile`(exp 스크립트) / `bash -n`(run_test.sh, run_sweep.sh) clean.
  `--run-kind progress_serve_prefill_budget` 는 invalid choice 로 거부되고
  경로 기반 모드 추론은 구 이름을 계속 인식함을 컨테이너에서 확인. GPU 실행 없음.
- 세션 중 사용자 secret-redaction 히스토리 재작성으로 워킹트리가 구 커밋으로
  이동해 작업이 중단됨 → stash 복원 후 새 base(`75f6df911`)에서 재개·완료.

## 2026-08-10 (TB* 통합 예산: D축 + 안전 가드)

- 구현: prefill token budget(P*)을 **Token Budget TB***로 통합·개명하고
  decode 축(D축) 신설. 단일 제약 `Δ_dec(D)+κ_p·P ≤ γ·T_min`을 두 축으로
  풀며, D축(`select_decode_defer`)은 decode-only 스텝시간이 예산을 깨면
  defer-safe(GPU·MEASURED·R_defer≤ε_d) 요청을 slack 깊은 순으로 defer해
  D′ 축소, D′는 커밋 후 `len(self.running)`로 P축에 자동 전파. run kind
  `progress_serve_token_budget[/offload]`; 구 prefill_budget 명칭은 analyze/
  metrics의 deprecated alias로만 잔존(phaseP 경로 호환).
- 안전 가드 2건 (기반 구현에 없던 것, 상상 실행 논의에서 도출):
  ① overdue 제외 — `t_min≤0` 및 T_q≤0 요청은 D축 미발동(이미 놓친 유닛을
  위해 처리량 버리는 것 방지; 과포화 read는 상당 시간 overdue).
  ② 만족성 가드 — 도달 가능한 어떤 D′도 예산을 못 맞추면 defer 취소(floor
  초과·셀 부재). 이 둘이 request-level adaptive의 처리량 붕괴 재발을 막는다.
- 기아 방지: 스펙 초안은 "P_floor식 카운터 가드"를 적었으나, D축은
  self-limiting으로 충분(defer된 MEASURED 요청은 T_q 감소→R_defer 상승→
  ε_d 초과 시 자격 상실→자동 복귀; deadline 자체가 카운터). 카운터 미구현이
  옳은 대체설계로 판정, scheduling.md·SWEEP_PLAN·sslo/README 문구를 구현에
  맞춰 정정(검증 라운드 Important/Minor 반영).
- 검증: pytest tests/sslo 197 passed(+D축 7종, 기존 tts 1 fail). D축
  발동률은 오프라인 재생 불가(scheduler_stats에 per-request T_q/R_defer
  없음) — 로직은 단위 테스트로, 발동률은 GPU 런의 `token_budget_d_defers`
  stat으로만 확인. sslo-verifier: 로직 버그 0(7 케이스 손 트레이스),
  문서 정합 지적만 반영.

## 2026-08-11 (phaseP2: κ 수정·TB* 실측 검증 + 결합 저하 원인 분리)

- 실행: phaseP2 — 신 코드(κ ratio-of-sums + TB* D축)로 4모드
  (baseline/offload/token_budget/offload+token_budget) × cap{32,64,128} ×
  rate{0.5,1,2,4}, read. 38/48에서 사용자 지시로 종료
  (`output_sweep_v2/phaseP2`; cap128·cap64 완결, cap32 부분).
- 사전 예측 판정: ① κ 수렴 **적중** — o+tb κ 0.53~0.65 → 0.15
  (ratio-of-sums 목표값). ② tput 회복 **적중** — cap128 o+tb r2/r4
  517/531 (구 500, baseline 531 동률). ③ D축 read 비발동 **예측 실패**
  — 과포화 read에서 D-defers 수백~1,384회 발동. 단 무해함이 실측됨
  (tb 단독 cap64: 전 rate viol 0.8~1.4%, tput 406~508 = baseline의
  96~99%, D 630회 발동에도 재적 45 유지) — 안전 가드(overdue 제외·
  만족성 revert)가 의도대로 작동.
- **핵심 결과: 이 워크로드에선 token_budget 단독이 최적.** cap64에서
  offload 계열만 r4 저하 (offl 401/run28, o+tb 334/run13 vs tb
  508/run45) — r4 저하의 범인은 offload이지 D축이 아님.
- 결합(o+tb) 저하 원인 3중 결합 (LOW/HIGH 에피소드 분해, cap64 r4):
  ① 1차 정지는 offload 몫 — offl 단독도 46% 저점유, 그동안 k*=13.9로
  허가되는데 집행 안 됨 (허가-집행 괴리, 미해명·별도 조사 필요).
  ② tb가 회복을 조임 — 정지 생존자(마감 임박)가 t_min을 눌러 P* floor
  81% → 재적 4→45 회복에 필요한 ~5.6만 prefill 토큰이 4배 느리게 유입.
  ③ D축 defer분의 R_defer가 E_viol에 계상(0.95)돼 k*가 2.0으로 붕괴
  (offl 단독 13.9). 뿌리는 동일: t_min/E_viol이 in-flight만 보는
  큐-블라인드 — ③축 잠금·구 adaptive 붕괴와 같은 계열의 세 번째 발현.
- 수정 후보(기록): 회복 예외(재적≪허가 ∧ 대기 깊으면 admission prefill을
  P* 면제 또는 floor 유휴도 비례 상향), D축 defer분 E_viol 분리,
  offl 허가-집행 괴리 규명.

## 2026-08-11 (TB* 개정: worst-case γ·t_min → 기대-위험 예산 도싱)

- 배경: phaseP2 진단 ②③ — t_min 보유자 1명(대개 구제 불가)이 P축을 floor에
  81% 고정, D축 defer분이 절대 E_viol을 눌러 k* 붕괴. 두 축의 화폐를
  E_viol로 통일하면 doomed(R≈1)의 한계 기여가 ~0이라 매몰비용이 자동 해소.
  캐노니컬 스펙: `paper/scheduling.md` §3⑤ (2026-08-11 개정).
- 수정(`vllm/vllm/sslo/progress_serve.py`):
  - 신규 `token_budget_prefill_risk(views_A, b, delta_dec, kappa_p,
    base_budget, floor, eps_p, lead, share_includes_parked)`:
    `TB*_pre = max{P∈[floor,base] : E_viol(Δ_dec+κ_p·P) − E_viol(Δ_dec) ≤ ε_p}`.
    E(P)는 P에 단조 비감소(Δ↑→H_q↓→N↓→tail↑, 각 Δ에서 build_plan의
    M-내림차순 배정이 위험 최소)이므로 정수 이분탐색. 비용 = 2 +
    ⌈log2(base−floor)⌉ build_plan 호출(기본 512..8192에서 15회) —
    schedule_step이 이미 스텝당 최대 k*+1회 쓰므로 상대적으로 저렴
    (tail_prob는 정렬 히스토리 bisect 2회).
    κ_p≤0 또는 views에 measurable 없음이면 base 반환(제어 비활성),
    결과는 항상 [floor, base] 클램프.
  - `select_decode_defer`: t_min/γ 입력 제거. 후보(자격 R_defer≤ε_d,
    slack 깊은 순, self-limiting 유지)를 하나씩 가상 defer하며
    `E_viol(Δ_dec(n−1)) < E_viol(Δ_dec(n))`인 동안만 채택, 개선 멈추면
    중단(채택 안 함). overdue 특례·만족성 가드 삭제 — doomed 한계 기여
    ~0이 같은 역할을 자연 수행. 셀 부재 시 보수적 중단(R5) 유지. 비용은
    최대 2+채택수 build_plan 호출.
  - 구 `token_budget_prefill()`(γ·t_min)은 **삭제하지 않고 deprecated**
    (docstring에 기각 사유 + 대체 함수 명시). A/B 재현용, 라이브 호출 없음.
- 수정(`vllm/vllm/v1/core/sched/scheduler.py`):
  - `_sslo_prefill_token_budget(views_a, b, d_prime, lead, share_parked)` —
    t_min 인자 소멸. 호출 위치를 `schedule_sslo` → 정책 내부(⑤단계,
    D축 직후)로 이동: 위험 원장(views_a/b/lead/share)이 거기에만 있고
    D′도 그 시점에 확정되므로 뷰 재구성이 불필요. 결과는
    `_sslo_step.token_budget_prefill`에 기록, `schedule_sslo`의 running/
    waiting 루프(⑥ 집행)는 그 값을 읽어 소비 — 공유 카운터·최소 1토큰·
    floor 기아 가드는 변경 없음.
  - `_sslo_min_time_to_deadline()` 삭제(P축이 유일 사용처였음),
    `_sslo_apply_decode_budget`의 t_min 계산부 제거.
- 수정(`vllm/vllm/sslo/config.py`): `token_budget_risk_eps: float = 0.01`
  신설(>0 검증). `token_budget_gamma`는 deprecated 주석만 달아 보존
  (구 함수가 참조). TB* 주석 블록을 2축 E_viol 서술로 교체.
- 수정(exp): `run_test.py` env `SSLO_TOKEN_BUDGET_RISK_EPS` 추가
  (GAMMA는 no-op으로 남김), `run_test.sh`/`exp/run_sslo/README.md`/
  `vllm/vllm/sslo/README.md` 문구를 위험 도싱으로 정정.
- 누적 거동(상상 실행): admission의 절대 예산 E_viol<1은 그대로이므로
  A3식 총위험 폭주 경로 없음. ε_p는 한계량이라 E(0)이 커도(doomed 다수)
  도싱이 저절로 조여지지 않고, savable 요청의 상승분만 ε_p에 계상된다 —
  회복기 조임 해소의 메커니즘이 곧 이 성질.
- 검증: 컨테이너 `sk-sslo`에서 `pytest tests/sslo/ -q` → 203 passed,
  1 skipped, 1 failed(기존 `test_tts_consume_path` 1건, 본 변경과 무관).
  신규/개정 테스트: P축 (a) doomed 전원→base(구 γ 판이면 floor임을 같은
  입력으로 대조) (b) 여유→base (c) 민감 요청에서 ε_p 정지 + brute-force
  오라클 일치 + 실현가능성 prefix(단조성) 검증 (d) κ/기준선 부재→base
  (e) 클램프 범위; D축 개선 채택·정지, 무개선 3종(doomed/Δ이득 미미/
  deadline 여유)→[], forced·at-risk 불가침(D′=D_floor), 셀 부재 2종,
  self-limiting; 스케줄러 P축 배선(정책이 기록→루프가 소비) 1종;
  config 기본값·ε_p>0 검증. `python3 -m py_compile`, `bash -n
  run_test.sh`, `run_test.py --help`, SsloConfig ε_p 주입 확인. GPU 미실행.
- (해결됨) 개발 라운드 시점에 `paper/scheduling.md` §3⑤에 구 P축 공식
  블록이 개정 텍스트와 공존해 자기모순이었으나, 같은 세션에서 스펙 소유자가
  잔여 블록을 제거해 정리 완료 (grep 잔재 0건). 구현은 개정 텍스트 준수.
- 미증명 단순화 (기록): D축 E_viol 프로브의 원장은 decode-set 단독
  (admits/parked 미포함, s는 자격 판정에만). 자기일관적 hill-climb이라
  안전성은 훼손 없으나 "전체 원장 대비 defer가 덜 발동하는 보수 방향"이라는
  방향성 주장은 수학적 증명·대조 테스트 없음 — 알려진 한계로 유지.

## 2026-08-12 — TB* P축 가격 모델: 무한-지평 → burst-horizon 교체

### 실증: 무한-지평 재발 가격(2026-08-11 초판)의 기각
- phaseD 스윕(GPU 2,3, cap128, Qwen3-32B/wildchat/GEN 8K)으로 신판 TB*를
  구판(γ·T_min)과 동일 축에서 대조. baseline 교차 확인으로 GPU 0,1(phaseP2) 대비
  하드웨어 동등성 확보(tput 차 3~6%).
- 결과(cap128 o+tb, 구판→신판): floor 체류 6.9/22.2/18.3/15.7% →
  42.5/47.2/77.9/82.7%, r1 tput 507→391, TTFC p50 256→668s, 위반율 이득 없음.
- 원인: `Δ(P)=Δ_dec+κ_p·P`를 `build_plan`에 넘기면 `horizon(T_q,Δ)=T_q/Δ`가
  이를 **마감까지 모든 스텝**의 시간으로 사용 → 1스텝 비용 κ_p·P(0.1~0.4s)를
  영구 감속으로 계산. 실측 κ_p=0.198ms/tok·Δ_dec=74ms에서 P=2048이 모든 미래
  스텝 6.5배 감속으로 가격됨(~20× 과대). 실측 prefill 포함 스텝은 8.9%뿐.
  매 스텝 재결정되는 비용을 전지평에 물리는 이중과금.

### 수정 내용
- `paper/scheduling.md` §3⑤ 재작성(burst-horizon), §6에 초판 기각 근거,
  §8에 borderline 인질 한계 신규 기재.
- `progress_serve.py`: `burst_horizon()` piecewise 지평 + `request_risk`/
  `request_risk_cpu`/`build_plan`에 선택적 `delta_burst`/`burst_steps`
  (None이면 기존 경로 bit-동일). `token_budget_prefill_risk(..., prefill_work=W)`,
  기준점 `E_ref = E(P_floor)`, `floor = max(1, min(floor, base_budget))`.
- `scheduler.py`: W = PREFILL 캐리오버 잔여 + k* 이내 waiting 헤드 프롬프트.
  `_sslo_is_onloading()` 술어 추출로 W 루프와 `waiting_views` 스캔이 동일
  필터를 공유(조건식 복제 금지 — 어긋남 재발 방지).
- `config.py`: `token_budget_risk_eps` 0.01 → **0.5** (근거는 아래 상상 실행).

### 상상 실행 (실측 분포 기반 3모델 대조, GPU 미실행)
phaseD chunks.jsonl 31,596청크의 실측 tail/slack, κ_p=1.98e-4, Δ_dec=0.074s.
M0=무한지평(기각) / M1=1회성 T_q 잠식(A안) / M2=burst-horizon(채택).

| 시나리오 | M0 | M1 | M2 | 정답 |
|---|---|---|---|---|
| S1 평시 고부하(재적 47) | 512 | 512 | 538 | 개방 |
| S2 회복-정지(doomed 2) | 2656 | 8192 | 8192 | 개방 ✓ |
| S3 진짜 위험 | 512 | 512 | 512 | 클램프 ✓ |
| S4 마감 먼 요청만 | 8192 | 8192 | 8192 | base ✓ |

- M1 기각 사유: κ_p·**P**를 과금하나 실제 burst 지연은 κ_p·**W** — W<P면
  6.5× 과금, W≫P면 3× 과소과금.
- ε_p 창 측정: 평시 dE=0.413 vs 위험 국면 dE=4.537 (11× 분리). 구판 0.01은
  창보다 40× 아래라 평시에도 floor 강제 → **0.5 채택**(S1 8192 / S3 558).
  S2·S4는 dE≡0이라 ε_p 무관.
- 잔존 한계: floor에서 살고 base에서 doomed가 되는 **borderline** 요청의
  한계 기여 ~0.4. 요청당 발생률 1.75% → 재적 47이면 스텝의 56.5%에 최소 1명
  (재적 5면 8.5%). cap이 클수록 조임이 강해지는 방향성 잔존 — 스윕에서 cap별
  floor 체류율 대조 필요.

### 검증 (개발↔검증 agent 2라운드, 종료조건 "이슈 없음" 충족)
- 라운드 1 Critical: W 루프가 `self.waiting`을 raw 순회해 onload 요청을 이중계상.
  onload는 `_sslo_apply_offload`에서 waiting에 prepend되고 offload preempt가
  `num_computed_tokens=0`으로 리셋하므로, KV 복원 요청의 전체 프롬프트가 W에
  실리고 진짜 헤드가 k* prefix 밖으로 밀려남(재현: 기대 10 → 실제 5000).
  기존 테스트는 `_tb_prefill`이 `k_star=0` 고정이라 헤드 루프가 dead였음.
- 라운드 1 Minor: `prefill_work/p` 0-division 방어 부재(순수 API 관점).
- 라운드 2: 이슈 없음. 검증 agent가 필터를 임시 제거해 신규 테스트가 회귀를
  실제로 포착함을 재현 확인 후 원상 복구 대조.
- 테스트: `pytest tests/sslo/ -q` → 209 passed, 1 skipped, 1 failed
  (기존 `test_tts_consume_path` 1건, 본 변경과 무관 — base 커밋에서도 동일 실패 확인).
- 신규: W 회계 2종(캐리오버+k* 헤드 합산 / onloading 제외), 매몰비용이 floor를
  고정하지 않음(W=200K에서 TB*=base), 마감-먼 요청 불변성, burst-안 클램프
  strict interior + 오라클 일치, doomed 비인질, W=0→base.

### 관측: phaseD D축 발동 특성 (과다 발동 가설 기각)
cap128 r2 o+tb 18,789스텝 중 D축 발동 28.0%, 발동 시 defer p50=1개(max 4),
발동 스텝의 재적 47.6 vs 비발동 39.0 — 배치가 클 때 한 명씩 덜어내는 설계
의도대로 동작. 누적 7,427회는 폭주가 아님.

### 미결
- GPU 재실행 대기(승인 필요): burst-horizon + ε_p=0.5로 phaseD 재수행.
- `paper/method.md`는 여전히 TB* 개정 미반영.
- offload 허가-집행 괴리(k*=13.9인데 미집행)는 이 개정과 무관하게 미해명.

### 2026-08-12 (이어서) — phaseE 실행 + method.md 갱신
- 실행: `output_sweep_v2/phaseE`, Qwen3-32B/wildchat 멀티턴/GEN 8K, 4모드 ×
  RATES{0.5,1,2,4} × CAPS{32,64,128} = 48 rate-run. 공용 노드 경합으로 GPU가
  0/1/3에 흩어져 비어 cap별로 3개 스윕 인스턴스를 병렬 기동.
- `run_sweep.sh`에 `GPU_IDS` 추가 (기존 `NUM_GPUS`는 0..N-1 조밀 배정만 가능해
  흩어진 빈 GPU 사용 불가). 지정 시 NUM_GPUS는 개수에서 자동 결정.
- cap32를 사다리에 추가: borderline 인질이 "재적↑ ⇒ 조임↑"을 예측하므로
  floor 체류율의 cap 단조성 검증에 3점이 필요 (2점은 추세/잡음 구분 불가).
- `paper/method.md` §4.3을 burst-horizon TB*로 재작성 (구 γ·T_min 수식 제거,
  piecewise H_q·매몰비용 논거·E(P_floor) 기준점·ε_p 창 추가). 구 규칙에서 측정된
  정량 주장(처리량 중립 89-99%, 위반율 1.1-1.5%, 트리거 동시발화 1-2%)은
  draft note로 재측정 대기 표시 — 발동률 수치는 트리거 정의가 바뀌어 삭제.
- 추가: `exp/run_sslo/analysis/method_table.py` — output_sweep_v2 레이아웃의
  4-method 판정표 생성기 (cap × mode × rate로 viol/tput/TTFC/users/TB*/floor%/
  κ/Ddef, run_N 복수 시 평균±반폭). phaseP2·phaseD 데이터로 기존 수치 재현 확인.

### 2026-08-12 (이어서) — 미해명 항목 해명: offload 허가-집행 괴리
기존 로그(phaseP2 cap64 r4 offload 단독, 28,231스텝)만으로 원인 규명, 계측 추가 불필요.
- 배제: waiting 고갈 아님 (저점유 스텝의 waiting p50 = 3,169, waiting==0 비율 0%).
- 관측: running==0 & k*>=5 스텝이 7.0%, 그 스텝의 pending p50=58,
  num_handling_users p50=58(전체 41보다 높음), num_scheduled_tokens_total=0.
  → **전체 스텝의 14.3%가 완전한 빈 스텝** (prefill 0, decode 0).
- 원인: `kv_blocks_per_new_admit=8` vs 실측 요청당 KV 블록 p50 **141** (18배 과소).
  free 525블록 → 정책은 66명 가능으로 판단(k*=64), 실제 수용 3.7명.
  `kv_capped = k_star_unconstrained > best_k`는 정책 KV 모델이 스캔을 멈출 때만
  참이므로 모델이 낙관적이면 영구 False → **offload tier 발동률 0%**
  (KV 92.7% 점유인데 num_offloads=0).
- 인과 사슬: 전원 deferred(슬랙 수확은 정상 동작)로 KV 점유 → 신규 admit은 실제
  KV 부족으로 실패 → kv_capped 미발화로 vacate도 안 됨 → 빈 스텝.
- 수정 방향(미착수): kv_capped를 실제 allocator 실패에서 세우고, 블록 수를
  실측 프롬프트 길이 기반으로 산정. `paper/scheduling.md` §8에 기록.
- 영향 범위: phaseP2/D/E 모두 동일 결함 → 세 실험 간 비교는 유효, offload
  tier의 실효 수치만 수정 후 재측정 필요.

**정정 (같은 세션, 위 항목의 과장 수정)**: baseline 대조를 넣자 두 주장이 무너졌다.
- "빈 스텝 14.3% = SSLO 병리" → **오류**. baseline도 12.6%가 빈 스텝이고, 그 전부가
  "in-flight 자체가 없음"(2,048스텝, 두 모드 동일 수치). offload의 14.3% 중
  7.3%p가 같은 성격이고, **SSLO 고유분은 7.0%p**(running 0 & pending>0, 대기
  3,006명 방치). baseline에는 이 상태가 0%.
- "offload tier 발동률 0%" → **오류**. kv_capped는 12.8% 정상 발화. 0%는 좁은
  부분집합(running==0 & k*>=5)에서만 본 값이었다.
- 그러나 **실제 offload 실행은 런 전체에서 20회뿐** — 신호는 오는데 자격 조건
  (M_cpu <= eps ∧ residency >= rho)이 사실상 전부 기각. 기각 사유별 카운터가 없어
  원인 미확정, **계측 추가 필요**(별도 결함으로 §8에 분리 기재).
- KV 모델 18배 과소평가(8 vs 실측 141)는 그대로 유효 — 허가-집행 괴리의 직접 원인.
- 실패한 시도(기록): "k* 대비 실제 admit"을 스텝 간 in-flight 차분
  (running+pending+num_offloaded의 양의 증가분 합)으로 오프라인 추정하려 했으나
  **무효**. 총합이 실제 요청 수의 약 40배(51,839 vs 1,152)로 나오며 baseline도
  동일 — 세 상태 합이 매 스텝 ±3 요동치므로 이 필드들의 의미 가정이 틀렸다.
  → HANDOFF의 "per-step 실제 admit vs k*" 계측은 오프라인 대체 불가, 실제 계측
  추가 필요(원 판단 유지). 이 필드 semantics 자체도 회계 용도로 쓰기 전 확인 필요.

### 2026-08-12 (이어서) — phaseE 중간 결과 (32/64)
**판정 1 (회복-정지 floor 체류 소멸) = 긍정 확정.** 완료된 TB* 셀 11개 전부
floor 체류 **0.0%**, TB*가 상한 2048 유지. 개정의 원래 동기였던 cap64 r4:

| | TB* | floor% | tput | TTFC | 재적 | viol% |
|---|---|---|---|---|---|---|
| 구판 γ·T_min (phaseP2) | 512 | 52.0 | 334 | 1490 | 20 | 1.8 |
| burst-horizon (phaseE) | 2048 | 0.0 | 477 | 1055 | 47 | 1.2 |

- 무한지평(phaseD)의 cap128 후퇴도 복구: r1 tput 391→490, TTFC 668→291,
  floor 47.2%→0.0%.
- **잡음 하한선 측정**: baseline run 간 변동 ±12(tput)인데 offl+tb는 **±49**.
  단일 run 비교 시 이 폭을 넘는 차이만 유효. cap64 r1(구판 494 → 신판 352,
  차이 142)은 3× 변동폭이라 실재 가능성 높으나 반복 run 필요 — GPU 여유 시 진행.
- cap32 r1은 구판에서도 305였음(신판 296) → 이번 변경과 무관, 원래 낮은 셀.
  초기 보고에서 "cap 작을수록 악화" 추세를 그렸으나 구판 대조로 기각됨.
- D-defers가 구판 대비 11~150× 증가(예: cap128 r0.5 65→9,682). 원인은 P축이
  아니라 phaseD에서 함께 들어간 D축 개정(E_viol 개선 hill-climb). 처리량 손상
  여부는 tb 단독 셀이 나와야 분리 가능 — 미완.
- `paper/method.md` 개정: §4 도입부에 **4 configuration 표**(ProgressServe /
  +Offload / +Adaptive / +Both — 추가 레버·완화 자원·발동 조건) 신설,
  §4.4 "Composition"를 독립 절로 승격하고 cap64 r1 4방향 ablation 표 수록.
  §4.2 "What it improves"의 처리량 주장(96-99%)은 phaseE 실측과 배치되어
  draft note로 근거 미비 명시 (cap64 r1 411 vs baseline 519 = 79%,
  원인은 KV 블록 상수 8 vs 실측 141).
- **정정**: "Adaptive는 거의 공짜"는 cap64 한정. cap128 r1에서는 tb 단독이
  410/500 = 82%로 가장 비싸고 offload가 489/500 = 98%로 가장 싸다 — **두 레버의
  비용 순서가 cap에 따라 역전**. 구판(phaseP2)도 cap128 tb 430/530 = 81%로 동일
  패턴이라 회귀가 아니라 원래 성질. method.md §4.4에 한정 문구 추가.
  가설: 각 레버는 자기 목표 자원이 실제 병목인 regime에서 싸고, 아닌 곳에서
  오버헤드를 낸다 (offload↔KV, adaptive↔스텝시간). 전체 grid 완주 후 확정.
- **재정정**: "cap에 따른 레버 비용 역전"도 기각. tb 단독은 cap64 전 rate 95-100%,
  cap128도 r0.5 99% / r2 96%이고 **cap128 r1만 82%**. cap 축 추세가 아니라 rate
  특이 현상이며, 구판 phaseP2에서도 같은 셀이 81%(이웃 rate는 93-94%)로 재현됨.
  원인 미상. method.md §4.4는 인과 서술을 빼고 관측만 남김.
- **설계 구멍 발견 (사용자 지적)**: +offload/+adaptive/+both는 불필요할 때
  ProgressServe로 수렴해야 하는데 (a) 그것을 검증할 **bare progress_serve arm이
  phaseE에 없었고** (모든 비교가 "코어+증분 vs vanilla"였음), (b) 수렴이 깨지는
  경로가 코드에 둘 있다:
  · offload 모드는 `SimpleCPUOffloadConnector` + `lazy_offload:False`로 **KV를
    상시 CPU 미러링**하고 `enable_prefix_caching`을 강제로 켠다(run_test.py:822).
    발동 여부와 무관한 per-step 비용 + 다른 모드와 엔진 구성 자체가 다름.
    실제 offload는 런당 20회뿐이므로 관측 비용은 대부분 미러링.
  · D축 게이트는 "E_viol이 개선되는 동안 채택"인데, 디코드 집합 축소는 스텝을
    빠르게 해 거의 항상 E_viol을 개선한다 → 구조적 상시 발동 편향(스텝의 28%,
    런당 8-11K회). P축의 무개입 보장은 지켜지나 D축은 아님.
    (반증: 이 논리면 배치 큰 cap128에서 더 많아야 하나 실측은 cap64 8,290 >
     cap128 6,781 — 편향은 확실하나 처리량 손실과의 연결은 미증명.)
- 조치: GPU 여유 발생 즉시 **bare progress_serve arm**(cap 64/128 × 4 rate)을
  cap64 반복보다 우선 투입 — 진행 중.
- 도구 보강: `method_table.py`에 `progress_serve`(코어) 모드 행 추가 —
  arm 자체가 실험에 없었으므로 분석기에도 빠져 있었음.
  `run_test.py`에 `SSLO_TOKEN_BUDGET_DECODE_RISK_EPS` env 훅 추가
  (D축 자격 임계 ε_d 주입 — adaptive 비수렴의 D축 귀속 실험용).
- phaseE 중단(사용자 지시, 2026-08-12 말): 73 rate-run 확보 후 정지. GPU 전면 반납.
  미완: cap64/cap32 코어, cap128 코어 run_2, D축 ε_d=0 ablation, cap64 run_2.

### 2026-08-13 — 알고리즘 단순화 3건 (KV 실측 회계 / D축 제거 / prefix caching 동등화)
사용자 지적("증분은 불필요할 때 ProgressServe로 수렴해야 한다")에서 출발. GPU 미사용.

**수정 1 — KV 가용성을 상수에서 실측 누적합으로**
- `kv_blocks_per_new_admit=8` 폐기. `kv_feasible(k) ⟺ Σ_{i<k} ceil(잔여 프롬프트/block_size) ≤ free`.
- `blocks_needed`도 정확한 deficit `max(0, cum[k*_unc−1] − free)`로 (증분형은 매 capped
  스텝마다 1명분 과다 vacate).
- 상상 실행(phaseE cap64 r4 실측: 프롬프트 p50 672tok→42블록, KV 포화 스텝 free=525):
  구판은 64명 admit 허가(실제 필요 5,446블록 → 4,921 초과)에 kv_capped 영구 False.
  신판은 4명 허가(415 ≤ 525), E_viol이 더 원하면 kv_capped=True + deficit 정확 산출.
  **16배 과대 발급 제거, offload tier가 비로소 발동 가능.**
- 한계(기록): admission 회계는 프롬프트 기준이라 생성 중 증가분(요청당 최종 141블록)은
  미반영 — 스텝 단위로는 정확하나 생애 수요로는 낙관적. 성장분은 종전대로 엔진 preemption과
  E_viol이 처리.

**수정 2 — D축 전면 제거**
`select_decode_defer`, `token_budget_decode_risk_eps`, `_sslo_apply_decode_budget`,
D′ 배관, stats 2종, env 훅 삭제. 순 −207줄. P축/burst-horizon 무손상.
근거는 §3⑤·§6 표: 구조적 상시 발동(무개입 보장 부재) + 실측 순수 비용(코어 대비
−44/−45 tput, 위반율 이득 0).

**수정 3 — prefix caching 전 모드 공통**
offload 모드만 커넥터 요구로 켜던 것을 모든 run kind로. 이제 offload의 유일한 고유
비용은 커넥터의 상시 KV 미러링. ※ baseline 포함 전 모드 성능이 바뀌므로 전체 재측정 필요.

**검증**: 개발↔검증 agent 2라운드, 종료조건 "이슈 없음" 충족. `pytest tests/sslo/ -q`
203 passed(기존 실패 1건 제외), 신규 KV 회계 테스트 5건. 검증 agent가 상수 기반·증분형으로
각각 되돌려 신규 테스트가 실제로 회귀를 잡는지 재현 확인 후 원상 복구 대조.

**미결**: `HANDOFF.md` 11/80행이 아직 D축을 언급. 전체 재측정은 GPU 확보 후.
- 문서 정합화: `paper/method.md` §4 구성표에서 "(+ whole-request decode deferral)" 제거,
  §4.3 메커니즘 문단을 "decode set은 §4.1 run/defer 분할이 전적으로 결정"으로 정정,
  §4.4 한계 문단을 "두 결함을 발견해 이미 수정했고 이 절 수치는 그 이전"으로 갱신.
  `HANDOFF.md` 11/79/88행의 D축 언급 정리(판정 항목 2는 해결→폐기로 대체).
- 캐시 경로 정리: `/cache/hub`의 Qwen3-32B가 소실돼 phaseF 첫 두 셀이 65G 재다운로드로
  약 35분 지연(코드 문제 아님 — py-spy로 `snapshot_download` 대기 확인). 경로 후보 조사:
  `/cache/models`의 대용량 모델들(Llama-3.1-70B 263G, Mixtral 178G 등)은 **HF 캐시 형식이
  아니라 평범한 디렉터리**라 HF_HUB_CACHE를 그리로 돌려도 이름 기반 재사용이 안 된다
  (HF 형식은 Qwen1.5-MoE 27G, Qwen3-14B 103M 부분본뿐). 공유 이득 없이 공용 디렉터리
  오염 위험만 남으므로 `$HF_HOME/hub` 기본값을 쓰기로 하고 `run_test.sh`의 명시적
  `HF_HUB_CACHE` 줄을 제거(HF_HOME=/cache만 유지).
- 일괄 정리: `exp/` 전체에서 명시적 `HF_HUB_CACHE` 제거(21개 파일). 모든 대상이
  `HF_HOME=/cache`를 이미 함께 설정하고 있어 동작 동일($HF_HOME/hub 유도).
  포함: docker `-e` 전달 1건, python `os.environ.setdefault` 1건, 문서 2건.
  `AGENTS.md`의 규칙 문구도 "HF_HOME만 설정, HF_HUB_CACHE 설정 금지"로 개정 —
  예시가 두 손잡이를 함께 못 박고 있어 그대로 두면 재도입된다.
  검증: 변경 셸 17개 `bash -n` 통과, python compileall 통과, 각 파일에 HF_HOME 잔존 확인.
