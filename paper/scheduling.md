# ProgressServe 스케줄링 정책 — 현행 스펙 (2026-08-10 정리)

> 캐노니컬 현행 설계. 구현 상태 기준이며, 미구현 확장은 마지막 §7에 명시 구분.
> 이전 버전(슬랙-히스테리시스 pending)은 폐기됨 — 이력은 WORKLOG 참조.

## 1. 상태 기계와 용어

| 상태 | 뜻 |
|---|---|
| `waiting` | 미입장 (엔진 큐) |
| `running` | 이번 스텝 디코딩 |
| `pending` | 입장·defer됨, KV는 GPU 상주 |
| `offloaded` | defer됨, KV는 CPU 상주 |
| `onloading` | CPU→GPU 복원 중 (전이 상태) |

전이: **admit** (waiting→in-flight) / **defer** (running→pending) / **promote** (pending→running) / **offload** (pending→offloaded) / **onload** (offloaded→onloading→running).

라이프사이클: `PREFILL → WARMUP → MEASURED`. MEASURED만 청크 deadline을 보유하고 위험 계산에 참여한다. PREFILL/WARMUP은 *forced* — 무조건 슬롯을 받고 E_viol에서 제외 (첫 유닛을 내야 측정 가능해지므로).

## 2. 위험 프리미티브

단일 통계 기반: 청크 길이 예측기의 **tail posterior** $\Pr[L > c+N \mid L > c]$.

- service share $s = \min(1,\ B/|A^{+}|)$. $|A^{+}|$ = running + onloading + 신규 admit + **offloaded(parked) 포함** — offload는 메모리를 비우지 연산 부하를 없애지 않는다 (§5-R1).
- deadline horizon $H_q = \lfloor T_q/\Delta \rfloor$ → run/defer 시 마감 전 생산 가능 토큰 $N_{run} > N_{defer}$ → 위험:
  $$R_{run},\ R_{defer} = \Pr[L > c_q + N_{\cdot} \mid L > c_q], \qquad M = R_{defer} - R_{run}$$
  $M$ = 디코드 슬롯 1개의 한계 효용. $R_{defer} \ge 1$이면 $M=1$ 강제 (가망 없다고 버리면 스톨이 복리로 커지므로 — doomed guard).
- CPU 체류 위험: $N_{defer\_cpu} = N_{defer} - \ell \cdot s$ (onload lead $\ell$ 반영) → $M_{cpu} = R_{defer\_cpu} - R_{defer}$ = **CPU 체류의 위험 프리미엄**.

## 3. 스텝 파이프라인

```
① prologue: onload 완료 요청을 running에 합류, offloaded 체류 계상
② run/defer 분할 (build_plan):
     forced + onloading → 무조건 슬롯 / MEASURED는 M 내림차순으로 decode_cap까지 run,
     나머지 defer(pending)
③ admission 탐색 (schedule_step):
     k* = max{ k : E_viol(k) < 1  ∧  KV가 k명 admit 허용 }   (waiting FCFS 프리픽스)
     E_viol = Σ_run R_run + Σ_pend R_defer + Σ_offl R_defer_cpu   (절대 임계; §5-R3)
     KV가 스캔을 멈췄으면 kv_capped ← true
④ offload/onload (kv_capped일 때만):
     onload 먼저 — d−t ≤ ℓΔ+f̂ 인 offloaded를 복원 시작 (deadline 안전 우선)
     offload — pending 중 M_cpu ≤ ε ∧ residency ≥ ρ 를 slack 깊은 순으로,
               admission에 필요한 블록만큼만
⑤ Token Budget TB* 계산 — **burst-horizon 기대-위험 예산** (2026-08-12 개정.
     이력: γ·T_min worst-case → 기각(t_min 인질, 회복기 floor 81%);
     무한-지평 재발 가격 Δ(P) 전지평 적용 → 기각(§6, phaseD 실증) → 현행):
     W = 이번 burst의 잔여 prefill 작업량 — PREFILL 캐리오버의 잔여 프롬프트
         토큰 + k* 안의 큐 헤드 프롬프트. 사실값(추정 아님 — R5 무관).
     P를 고르면 burst는 W/P 스텝 × Δ_b(P)=Δ_dec(D')+κ_p·P,
     벽시계 wall(P) = (W/P)·Δ_dec + κ_p·W. burst 종료 후 Δ_dec 복귀.
     요청별 마감까지 얻는 토큰 수 (piecewise horizon):
       T_q ≤ wall(P):  H_q = T_q / Δ_b(P)            — burst 안 마감, P에 민감
       T_q > wall(P):  H_q = (T_q − κ_p·W) / Δ_dec   — P와 무관
       TB*_pre = max{ P ∈ [P_floor, P_base] : E_viol(P) − E_viol(P_floor) ≤ ε_p }
     E_viol(P)는 build_plan을 piecewise horizon으로 재평가한 값. 기준점은
     **E_viol(P_floor)** — floor는 무조건 부여되므로 항상 실현 가능한 대안이고,
     매몰비용 κ_p·W 중 어떤 P로도 피할 수 없는 흡수분이 P 선택에 과금되지
     않는다 (순수 Δ_dec 기준을 쓰면 P-무관 오프셋이 ε_p를 잠식해 floor 체류가
     다른 경로로 재발 — 구현 검증에서 발견). P↑ ⇒ burst 안 마감 요청의
     토큰율만 ↓ 이므로 E_viol(P) 단조 비감소 → bisection 유효, 기준점에서
     좌변=0이라 floor는 항상 통과.
     · 총 지연 κ_p·W는 P와 무관한 매몰비용 — 가격되는 것은 **집중도**뿐
       (burst 창 안에 마감이 걸린 요청의 한계 위험). 마감 먼 요청 과금 0.
     · doomed(R≈1)는 한계 기여 ~0 → 인질 없음 (γ·T_min 결함 비재발).
     · burst는 커밋된 작업(PREFILL은 forced 슬롯)이므로 열린-루프 선반영은
       이중과금이 아님. 정상성 가정: 선택 P가 burst 종료까지 유지된다고
       보고 가격 (매 스텝 재결정으로 자기수정).
     W에서 onloading 요청은 제외 — waiting에 prepend되지만 KV 복원이지
     prefill 작업이 아니고, waiting_views 루프의 제외와 대칭 (이중계산 방지).
     ε_p 보정: 상상 실행(2026-08-12, 실측 tail/slack 분포) 기준 평시 고부하의
     한계 비용이 0.41, 진짜 위험 국면이 4.54 — 11배 분리 창이 존재한다.
     ε_p=0.5로 설정 (평시 base 개방, 위험 국면 floor 클램프 유지).
     구판의 0.01은 창보다 40배 아래라 상시 floor를 강제했다.
     [D축] 같은 장부로 재가격: slack 깊은 순 후보(자격 R_defer≤ε_d,
           self-limiting은 종전과 동일)를 하나씩 defer해 보며
           E_viol(Δ_dec(D−1)) < E_viol(Δ_dec(D))인 동안만 채택 (개선이
           멈추면 중단·revert — 만족성 가드의 기대값 판). t_min 입력 소멸.
     Δ_dec(·)은 배치-키 셀 직접 조회, 셀 부재 시 보수적 중단 (R5).
     주의(A3 교훈): admission의 절대 예산 E_viol<1은 **그대로 유지** —
     ε_p는 prefill '도싱'에만 걸리는 한계 예산이고 prefill 총량은 도착량으로
     보존되므로 A3식 무한 누적 경로가 아님. 상상 실행으로 누적 거동 검증 필수.
     κ_p = ratio-of-sums: EMA[초과시간(무클리핑)]/EMA[P], 기준선은
     배치-매칭 셀만 (부재 시 샘플 폐기 / 제어 비활성 → base).
⑥ 집행:
     running 루프 — D'에 든 요청의 디코드 토큰은 무제한; 청킹된 프롬프트
                    캐리오버는 TB*_pre 공유 카운터에서 우선 차감 (최소 1토큰)
     waiting 루프 — k*·KV·TB*_pre 잔여 안에서 admit; onload는 클램프 면제(과금은 됨)
     P_floor 연속 클램프 + prefill 무진행 N스텝이면 1스텝 base 허용 (기아 방지)
     ※ P_floor/base 클램프와 공유 카운터·기아 가드는 도싱 개정과 무관하게 유지
```

## 4. 지렛대 × regime 매핑

| 실패 국면 | 지렛대 | 발동 조건 | 무개입 보장 |
|---|---|---|---|
| slack 낭비 (기본) | ②③ run/defer + E_viol admission | 항상 | E_viol(0)≈0이면 baseline과 동일 admit |
| KV-bound admission | ④ offload/onload | `kv_capped`일 때만 | KV 여유면 no-op (I4) |
| prefill spike | ⑤⑥ TB* P축 | prefill 작업 ∧ ΔE_viol이 ε_p에 닿음 | 위험 여유면 TB*_pre=P_base (baseline 동일) |
| decode-Δ-bound | ⑤ TB* D축 | defer가 E_viol을 실제로 낮출 때 | 개선 없으면 D'=D_policy (no-op) |

두 확장의 트리거는 실측상 **동시 발화 1~2%** (서로 다른 국면: "가득 참" vs "흡수 중") — 결합은 상호작용이 아니라 커버리지 합집합으로 동작한다.

## 5. 회계 원칙 (불변 규칙)

- **R1 — parked도 share 분모에**: offload는 메모리만 비운다. 제외하면 s가 부풀어 M_cpu 과소평가 → 과잉 offload (실측: cap32 offload 8→1회 교정).
- **R2 — parked도 E_viol에**: CPU에 내린 위험은 $R_{defer\_cpu}$로 계속 계상 (회계 연속성).
- **R3 — 절대 위반 예산 유지**: `E_viol < 1`. 한계 기준(ΔE_viol)은 매몰비용 잠금을 풀지만 총위험 브레이크를 제거해 기각됨 (e_viol 16~41 폭주 실증; 코드에 기본 off로 보존).
- **R4 — decode는 토큰 단위로 잘리지 않는다**: TB*의 D축은 요청 단위 defer로만 작용한다 (요청당 1토큰이라 부분 실행이 없음). run으로 확정된 요청의 디코드 토큰은 어떤 예산에도 잘리지 않는다. 원시 P+D 토큰 합산 캡은 단가 비대칭(KV-읽기 상각) 때문에 기각 — 예산의 화폐는 토큰 수가 아니라 **예측 스텝 시간**이다.
- **R5 — 추정기는 자기 결정에 오염되면 안 됨**: κ의 기준선은 배치-매칭 셀만(폴백 금지), 샘플은 토큰 가중(ratio-of-sums, per-sample 나눗셈·클리핑 금지). 위반 시 폐루프 실증됨 (κ↑→P*↓→소P 샘플↑→κ↑).

## 6. 폐기 결정 (근거 요약)

| 폐기안 | 사유 |
|---|---|
| request-level adaptive (`max_num_seqs` 축소) | 축이 틀림 — Δ 변동의 주범은 prefill 토큰(스텝 6%가 벽시계 18%, p90 463ms)이지 요청 수가 아님. 큐-블라인드 목적함수와 결합해 배치 1까지 붕괴 (대기 2,691명 방치). TB*(P축)로 대체 |
| admission 한계 기준 (ΔE_viol) | R3 참조 — 총위험 무한 누적 |
| 원시 P+D 토큰 캡 | 두 토큰의 단가가 다를 수 있고(KV-읽기 상각 비대칭) 교환비가 regime·컨텍스트 의존 — 시간 단위 예산(§7)으로 일반화해야 함 |
| TB* 무한-지평 재발 가격 (2026-08-11 초판) | Δ(P)=Δ_dec+κ_p·P를 마감까지 **모든** 미래 스텝에 적용 — 1스텝 κ_p·P(~0.1–0.4s) 비용을 영구 감속으로 ~20× 과대가격 (실측 prefill 포함 스텝은 8.9%뿐). phaseD 실증: cap128 floor 체류 7~22%→43~83%, r1 tput 507→391, TTFC 2.6×, 위반율 이득 없음. 매 스텝 재결정되는 비용을 전지평에 물리는 이중과금 — burst-horizon(§3⑤)으로 대체 |

## 7. Staged 확장 (미구현 — TODO, SWEEP_PLAN_v2.md와 동기)

1. **오프라인 프로파일**: prefill 비용곡선 cost(P) (κ 스칼라 제거 — 토큰당 비용이 청크 크기 의존: ≤512에서 0.23~0.30 vs 2048에서 0.16), decode 비용은 (배치 × 컨텍스트) 그리드 필수 (관측 데이터는 공선성으로 분리 불가 실증). 기존 CUDA-graph 프로파일의 하이브리드(오프라인 shape × 라이브 anchor) 패턴 재사용.
2. (승격됨 → §3⑤ TB*: 통합 예산은 2026-08-10 현행 정책으로 편입. κ_d 기울기 분해 없이 Δ_dec(D) 직접 조회로 구현 — 공선성 장애물은 선형 분해에만 해당했음. TTS 축 실측은 D축 실효 검증 항목으로 유지.)
3. **③축 admission 잠금의 안전한 해제**: 델타+상한 하이브리드 / 유휴 조건부 델타 / M≈0 국소 제외 — 미결.

## 8. 알려진 한계 (기록)

- ③축 self-lock: 소수 고위험 in-flight의 위험 합이 절대 예산을 소진하면 KV·연산이 남아도 k*≈0 (과포화 + 만기 경과 큐에서 발생; R3 트레이드오프의 대가).
- o+pb cap64 저하: κ_p 오염(소분모 발산)으로 진단·수정 완료 — GPU 재실행으로 최종 확인 대기 (예측: κ→0.15 수렴, tput 392→~470).
- `request_risk_cpu`에는 doomed guard($R_{defer}\ge1 \Rightarrow$ offload 부적격)가 없음 — §2의 run-side 가드와 비대칭. 현행 kv_capped 게이트 하에서 실해가 관측되지 않아 미구현 (후보 유지).
- **[해명됨 2026-08-12] offload 허가-집행 괴리 = KV 모델 18배 과소평가**:
  `kv_blocks_per_new_admit=8`은 실측(멀티턴 wildchat, 요청당 KV 블록 p50 **141**)
  대비 18배 과소평가다. 그 결과 free 525블록에서 정책은 66명 admit 가능으로 보고
  k*=64를 허가하지만 실제로는 3.7명만 들어간다. 더 나쁜 것은 신호 경로다 —
  `kv_capped = (k_star_unconstrained > best_k)`는 **정책의 KV 모델이 스캔을
  멈췄을 때만** 참이므로, 모델이 낙관적이면 스캔이 KV에서 안 멈춰 kv_capped가
  영구 False가 된다. offload tier는 kv_capped 게이트라 **한 번도 발동하지 않는다**
  (실측 num_offloads=0, KV 92.7% 점유 상태에서).
  귀결: in-flight 전원이 소비자보다 앞서 deferred(pending 58, running 0)로 KV만
  붙들고, 신규 admit은 실제 KV 부족으로 막히며, **전체 스텝의 14.3%가 토큰을
  하나도 스케줄하지 않는 빈 스텝**이 된다 (대기 3,169명 방치).
  수정 방향: (a) kv_capped를 정책 추정이 아니라 **실제 allocator 실패**에서
  세우고, (b) 요청당 블록 수를 상수가 아니라 실측 프롬프트 길이에서 산정.
  ※ 이 결함은 phaseP2/phaseD/phaseE 전부에 동일하게 존재하므로 세 실험 간
  비교는 유효하나, offload tier의 실효는 수정 후 재측정해야 한다.
- **borderline 인질** (burst-horizon의 잔존 한계): floor에서는 살릴 수 있으나
  base에서는 doomed가 되는 요청(R 0.6→1.0)은 한계 기여가 ~0.4로 크다. 완전
  doomed(기여 0)와 마감 먼 요청(기여 0)은 해소됐지만 이 경계층은 남는다.
  상상 실행 실측: 요청당 발생률 1.75% → 재적 47명이면 스텝의 56.5%에 최소
  1명 존재(재적 5명이면 8.5%). ε_p=0.5 보정으로 실효는 억제되나 원리적으로는
  잔존 — regime별 재보정 없이 ε_p 하나로 두는 한 cap이 클수록 조임이 강해진다.
- TB* D축 프로브의 원장은 decode-set 단독(admits/parked 미포함) — 자기일관적
  hill-climb이라 안전성 문제는 없으나, 전체 원장 대비 방향성(defer 과소 발동
  = 보수)은 미증명 (검증 라운드 2026-08-11 지적, 알려진 한계로 유지).
- offload tier는 full-attention(MLA 포함) 모델 한정 — hybrid(mamba/linear-attn)는 upstream 비호환 (`kv_offload_model_compat.md`).
