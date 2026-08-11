# HANDOFF — 새 실험 머신 인수인계 (2026-08-11, TI1 → 후속 머신)

> TI1(RTX PRO 6000 Blackwell ×2) 세션의 아웃바운드 핸드오프. **새 머신은 GPU가
> 다르므로 실험은 전부 재수립·재실행한다** — 이 문서는 ① 환경 구축, ② 코드/정책
> 현황, ③ 하드웨어 재보정 방법론(이번 세션에서 확립), ④ 판정 항목을 담는다.
> 상세 이력은 WORKLOG.md 2026-08-04~11, 설계는 `paper/scheduling.md`(캐노니컬).

## 0. 코드 상태 (main @ `018482032`)

- **스케줄러**: ProgressServe + offload/onload tier + **TB\* 기대-위험 도싱**
  (P축 `token_budget_prefill_risk` 이분탐색 + D축 `select_decode_defer`
  E_viol-개선 채택). 용어 통일: `pending/promote/offload/onload`,
  `progress_serve_token_budget[/offload]` run kinds.
- **채택**: `kv_offload_share_includes_parked=True`, κ ratio-of-sums(배치-매칭
  기준선, 폴백 금지), `token_budget_risk_eps=0.01`.
- **기각(코드 보존, 기본 off)**: `admission_delta_criterion`(A3),
  request-level adaptive(산출물 폐기), γ·T_min worst-case TB\*(deprecated 함수).
- 테스트: `pytest tests/sslo/` → 203 passed, 1 skipped(TTS CSV 부재),
  1 failed(`test_tts_consume_path.py::test_tts_path_uses_audio_ready_time_...`
  — **사전 존재 실패**, 이 세션 범위 밖).

## 1. 환경 구축 (새 머신)

1. **컨테이너**: `run_docker.sh`는 `--privileged`+`--gpus all`인데
   **privileged가 GPU 제한을 무력화**한다. GPU를 제한하려면 privileged를 빼고
   `--gpus '"device=<ids>"'`로 직접 실행 (TI1 사례: WORKLOG 2026-08-04).
   이미지 `nvcr.io/nvidia/vllm:26.03-py3`, 리포 → `/workspace/mlsys`,
   `/data` → `/cache`.
2. **vLLM 설치 (핵심 함정)**: `vllm/`은 upstream 히스토리 없는 squash 벤더링이라
   자동 wheel 탐지가 잘못된 nightly로 폴백 → `vllm._C` ABI 크래시. 반드시:
   ```bash
   cd /workspace/mlsys/vllm
   VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.0.0+sslo \
   VLLM_PRECOMPILED_WHEEL_COMMIT=5371d6fb4023a1a08021135e46e9354ba0923e50 \
   pip install -e . --no-build-isolation
   ```
   (벤더링 트리 = upstream 2026-04-29, v0.20.1rc1.dev57. wheel variant는
   CUDA 버전 자동 탐지 — 새 GPU에서도 동일 커밋 사용.)
3. **추가 pip**: `pip install pytest tblib datasets` (dialogue 워크로드에
   `datasets` 필수).
4. **NGC 잔존물 확인**: `pip list | grep flashinfer-jit-cache` — 구버전(0.6.7)이
   남아 있으면 MoE 모델에서 `Mismatched number of arguments` 크래시.
   `pip uninstall flashinfer-jit-cache`로 제거 (dense 모델은 무관).
   ※ TI1의 SM120 특이 이슈 — 새 GPU 아키텍처에서는 재확인.
5. **모델**: offload tier는 **dense full-attention만** 지원 (hybrid/mamba 계열
   비호환 — `kv_offload_model_compat.md` 필독). TI1은 Qwen3-32B 사용
   (`hf download Qwen/Qwen3-32B`, HF_HOME=/cache HF_HUB_CACHE=/cache/hub).
   새 GPU 메모리에 맞춰 모델 크기 재선정 가능 — 단 "가중치가 커서 KV 풀이
   자연히 좁아지는" 구성이 KV-bound 실험에 필요.
6. **dialogue 캐시**: `exp/tools/dataset_cache/dialogues_wildchat_conv-en-nocode.jsonl`
   (4,000 대화)는 **리포에 포함**되어 있음 — 재구축 불필요. 다른 필터 조합이
   필요하면 캐시 파일 삭제 후 재실행 (cold-start 시 `--num-prompts`가 캐시
   상한이 되는 점 주의).

## 2. 실험 재수립 방법론 (하드웨어가 바뀌면 전부 재보정)

TI1의 CAPS{32,64,128}·RATES{0.5,1,2,4}는 **그 GPU의 산물**이다. 순서:

1. **KV 풀 산정**: (GPU 메모리×0.95 − 가중치 − 활성화) ÷ KV bytes/token.
   목표: cap 사다리가 KV-slack/전이/KV-bound 세 regime을 가르도록.
2. **Probe 런** (~30분×수 회): 중간 cap 1셀, `progress_serve_offload`,
   중간 rate — `scheduler_stats.jsonl`의 `kv_capped` 비율·`running`
   p50/p90·출력 길이 분포로 CAPS 확정. (TI1 사례: running p90이 cap에 안 닿고
   KV가 ~45-68 재적에서 binding → CAPS 하향.)
3. **용량 실측 → RATES**: probe의 완료 속도로 서비스 용량(req/s)을 재고,
   sub-saturation 1점 이상 포함해 사다리 구성 (전 rate 과포화였던 실수 반복
   금지 — TTFC가 전부 큐 대기가 됨). 최저 rate의 셀당 소요시간
   ≈ (2·cap+1024)/rate 를 계산해 시간 예산 확인.
4. **본 sweep**: `run_sweep.sh` — 4모드
   `baseline,progress_serve_offload,progress_serve_token_budget,progress_serve_offload_token_budget`,
   `DIALOGUE_PROMPTS=1 DATASET_NAME=wildchat GENERATION_MAX_TOKENS=8192
   CONSUME_CELLS="read" NUM_GPUS=<n>`. `docker exec -d`로 띄울 것
   (장수 호스트 프로세스가 끊기는 환경 대비).

## 3. 새 머신에서 판정할 항목 (TI1에서 미완/부분)

1. **TB\* 도싱 회복 거동**: 정지 회복기에 P\* floor 체류가 사라지는지
   (TI1 구판: floor 81% — 도싱 개정은 상상 실행으로만 검증됨, **실측 미완**).
   per-step `token_budget_prefill` 분포로 판정.
2. **D축 발동률·무해성**: `token_budget_d_defers` — 발동해도 tput이 baseline의
   ~95%+ 유지되는지 (TI1: tb 단독이 96-99%로 최적이었음).
3. **κ 수렴**: `prefill_kappa_ms_per_tok`이 안정값으로 수렴하는지 (TI1: 0.15).
4. **offload 허가-집행 괴리 (미해명 최우선 조사)**: offl 모드 저점유 구간에서
   k\*는 큰데(TI1: 13.9) 재적이 안 차는 현상 — E_viol도 KV도 안 막는데 admission
   집행이 안 됨. 스텝 로그로 waiting 루프의 실제 skip 사유 추적 필요
   (계측 추가 권장: per-step 실제 admit 수 vs k\*).
5. **결합(o+tb) vs 단독**: TI1에선 tb 단독 우세 — 새 HW에서 재확인.
6. **TTS 축**: 미실행. read보다 마감이 촘촘해 D축·offload의 실효가 다를 것.

## 4. 문서 맵

| 문서 | 역할 |
|---|---|
| `paper/scheduling.md` | **캐노니컬 스펙** (§3⑤ TB\* 도싱, §5 회계 원칙 R1-R5, §6 폐기 근거, §8 알려진 한계) |
| `paper/method.md` | 논문 Method 초안 (영문; TB\* 도싱 개정 반영 필요 — 미반영 상태) |
| `paper/algorithm3_kv_offload.md` | offload/onload 알고리즘 박스 (LaTeX) |
| `exp/run_sslo/SWEEP_PLAN_v2.md` | TI1 실험 전체 이력·결정·TODO (오프라인 프로파일, ③축 해제 등) |
| `kv_offload_model_compat.md` | offload 지원 모델 아키텍처 판정 |
| `vllm/vllm/sslo/README.md` | 구현 상세 (상태/전이 표, config 표) |
| `WORKLOG.md` 2026-08-04~11 | 전체 세션 이력 (병리 진단·ablation·수치) |

## 5. TI1 실측 참고치 (새 HW 대비 기준)

Qwen3-32B, GMU 0.95, wildchat multi-turn(프롬프트 평균 1.3K tok), GEN 8K:
KV 풀 ~21GB(~85K tok) / decode-only Δ p50 74ms(점유 40대) / prefill 스텝은
전체 6%인데 벽시계 18%, Δ p90 463ms / κ_p ≈ 0.16 ms/tok (청크≤512는
0.23-0.30 — per-chunk 오버헤드) / 서비스 용량 ~0.64 req/s / 문장 청크 ~40 tok /
baseline 위반율(τ=1s) 5-6% vs SSLO 모드 0.7-2.0%.
