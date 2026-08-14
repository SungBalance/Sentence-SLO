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
     **KV 가용성은 요청별 실측 블록 수의 누적합으로 판정** (2026-08-13 개정):
       blocks(q) = ceil(잔여 프롬프트 토큰 / block_size),
       kv_feasible(k) ⟺ Σ_{i<k} blocks(q_i) ≤ free_blocks
     상수 `kv_blocks_per_new_admit`(=8)은 **폐기** — 실측 요청당 141블록 대비
     18배 과소평가라 (a) k*가 실제 수용량의 ~17배로 과대 발급되고
     (b) 그 결과 스캔이 KV에서 멈추지 않아 kv_capped가 영구 False가 되며
     (c) kv_capped 게이트인 offload tier가 KV 99% 포화에서도 깨어나지 못했다.
     실측 근거: cap64/128에서 KV 점유 98.8~99.0%, 재적이 cap과 무관하게 48로
     고정(=KV가 상한을 정함)인데 kv_capped 발생률 0.0%.
     KV가 스캔을 멈췄으면 kv_capped ← true (이제 실제 용량 기준)
④ offload/onload (kv_capped일 때만):
     onload 먼저 — d−t ≤ ℓΔ+f̂ 인 offloaded를 복원 시작 (deadline 안전 우선)
     offload — pending 중 M_cpu ≤ ε ∧ residency ≥ ρ 를 slack 깊은 순으로,
               admission에 필요한 블록만큼만
     **두 판정은 서로 다른 share를 쓴다 (2026-08-14 개정)**:
       onload  : s_now  = B / (|A| + onloading + k* + parked)   — 지금의 실제 상태
       offload : s_post = B / (|A| + onloading + **k\*_unconstrained** + parked)
                 — 비운 자리를 실제로 admit에 쓴 뒤의 상태
     이유: offload 자격 "M_cpu ≤ ε"는 *CPU에 둬도 위험이 거의 안 는다*는 판정인데,
     이를 offload 이전 share로 하면 **자기 결정의 전제를 스스로 무효화한다** —
     비운 블록으로 admit이 일어나면 |A+|가 커져 s가 떨어지고, 그 순간 방금 내보낸
     요청의 M_cpu가 ε를 넘어 즉시 onload 대상이 된다.
     실측(phaseF cap64 offload r4, offload 이벤트 21건): offload 스텝에 k*=26이
     한꺼번에 허가되어 재적 32→48, e_viol 0.12 → +1스텝 0.96 → +3스텝 1.36,
     k*→0으로 잠김. offload→onload 간격 **중앙값 2스텝(최소 1)**. 왕복 106/105회에
     `num_offloaded` 중앙값 0 — 내보낸 것이 상주하지 못한다.
     붕괴 실증: 재적 7~13, KV 점유 17~32%(메모리가 남는데 아무도 못 받음),
     tput은 baseline의 58~65%. 코어·tb 단독에는 없고 offload 계열에만 발생.
     **자격은 두 share 모두에서 성립해야 하고, doomed는 제외한다**:
       offload 자격 ⟺ max( M_cpu(s_now), M_cpu(s_post) ) ≤ ε
                      ∧ R_defer_cpu(s_now) < 1  ∧  R_defer_cpu(s_post) < 1
     둘째 조건이 없으면 doomed 요청(R_defer≈1 ⇒ M_cpu≈0)이 자격을 통과하는데,
     onload의 안전망 `R_defer_cpu ≥ 1`이 곧바로 되불러 왕복이 재발한다 — max
     보장은 onload의 M_cpu leg만 덮기 때문이다. §2의 run-side doomed guard와
     대칭을 맞추는 것이고, §8에 후보로 남아 있던 비대칭을 해소한다.
     의미: **가망 없는 요청은 CPU로 내리지 않는다.** 내려도 곧 되불러야 하므로
     자리를 못 벌고 전송만 낭비한다.
     그러면 자격 통과 ⇒ M_cpu(s_now) ≤ ε ⇒ 같은 시점 onload 조건
     (M_cpu(s_now) > ε)을 **정의상** 만족하지 않는다. 즉시 왕복이 원천 차단된다.
     ※ 정정(2026-08-14, 구현 검증에서 발견): 초안은 "s_post ≤ s_now이므로
     M_cpu(s_post) ≥ M_cpu(s_now)"라는 단조성을 근거로 s_post 단독 판정을
     지시했으나 **이 단조성은 거짓이다.** 국소적으로 M_cpu ≈ f(c+N_defer)·ℓ·s
     (f = tail 밀도)라 s가 커지면 평가점 간격이 벌어져 M_cpu가 오히려 커지는
     구간이 있고, 특히 ℓ·s_post < 1이면 floor 이산화로 N_defer와 N_defer_cpu가
     같은 토큰에 붙어 M_cpu(s_post)=0이 된다. 실측 그리드(s∈[0.35,1], ε=1e-3,
     ℓ=2)에서 **11%(229/2100)** 조합이 "s_post 통과 & s_now onload 대상"으로
     남았다. parked가 |A+|를 2B 이상으로 밀 때 열리므로 **offload가 활발할수록
     커지는 구멍**이었다. max 형태만이 구조적 보장을 준다.
⑤ Token Budget TB* 계산 — **burst-horizon 기대-위험 예산** (2026-08-12 개정.
     이력: γ·T_min worst-case → 기각(t_min 인질, 회복기 floor 81%);
     무한-지평 재발 가격 Δ(P) 전지평 적용 → 기각(§6, phaseD 실증) → 현행):
     W = 이번 burst의 잔여 prefill 작업량 — PREFILL 캐리오버의 잔여 프롬프트
         토큰 + k* 안의 큐 헤드 프롬프트. 사실값(추정 아님 — R5 무관).
     P를 고르면 burst는 W/P 스텝 × Δ_b(P)=Δ_dec+κ_p·P,
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
     Δ_dec은 현재 디코드 집합 크기의 배치-키 셀 직접 조회, 셀 부재 시
     제어 비활성 → base (R5).
     **[D축 제거됨 — 2026-08-13]** decode 요청을 추가로 defer해 Δ_dec을 줄이던
     축은 폐기했다. 근거: (a) 게이트가 "defer해서 E_viol이 개선되는 동안"인데
     디코드 집합 축소는 스텝을 빠르게 해 모든 horizon을 늘리므로 **거의 항상**
     개선이 성립 — 구조적으로 상시 발동(스텝의 28%, 런당 8~11천 회)하며
     무개입 보장이 없다. (b) 실측상 순수 비용: cap128 r1/r2에서 코어 대비
     처리량 −44/−45인데 위반율 이득 없음(1.2% vs 0.9%). (c) 이 워크로드에서
     P축은 무개입(TB*=base 상시, floor 체류 0%)이므로 adaptive 모드의 측정된
     효과는 **전부 D축**이었고, 그것이 곧 순손실이었다.
     제거 대상: `select_decode_defer`, `token_budget_decode_risk_eps`,
     Δ_dec(D′) 탐색과 D′ 배관, 관련 stats(token_budget_decode/_d_defers).
     R4(“decode는 토큰 단위로 잘리지 않는다”)는 더 강해진다 — 이제 decode는
     예산의 대상이 전혀 아니며 run/defer 분할(②)만이 디코드 집합을 정한다.
     주의(A3 교훈): admission의 절대 예산 E_viol<1은 **그대로 유지** —
     ε_p는 prefill '도싱'에만 걸리는 한계 예산이고 prefill 총량은 도착량으로
     보존되므로 A3식 무한 누적 경로가 아님. 상상 실행으로 누적 거동 검증 필수.
     κ_p = ratio-of-sums: EMA[초과시간(무클리핑)]/EMA[P], 기준선은
     배치-매칭 셀만 (부재 시 샘플 폐기 / 제어 비활성 → base).
⑥ 집행:
     running 루프 — ②가 run으로 정한 요청의 디코드 토큰은 무제한; 청킹된
                    프롬프트 캐리오버는 TB*_pre 공유 카운터에서 우선 차감 (최소 1토큰)
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

**엔진 구성 동등성 (2026-08-13)**: offload 모드만 커넥터 요구로
`enable_prefix_caching`을 강제로 켜던 것을 **전 모드 공통으로** 바꾼다. 켜고 끄는
차이가 모드 간 교란이 되면 안 되기 때문이다(멀티턴에서 prefix 캐시는 큰 이득).
남는 offload 고유 비용은 커넥터의 **상시 KV 미러링**(`lazy_offload=False`)이며,
이는 tier의 내재 비용으로 보고 대상이다 — lazy로 돌리면 `is_fully_mirrored`가
항상 False가 되어 tier 자체가 무력화되므로 선택지가 아니다.
실측 방증: KV가 병목이 아닌 cap32(점유 63~66%)에서 offload는 완전한 no-op이어야
하는데 처리량이 baseline의 58~68%다 — 결정이 0번인 곳의 손실이므로 원인은
결정이 아니라 미러링이다.
※ 이 변경은 baseline 포함 전 모드의 성능을 바꾸므로 **전체 재측정 필요**.

두 확장의 트리거는 실측상 **동시 발화 1~2%** (서로 다른 국면: "가득 참" vs "흡수 중") — 결합은 상호작용이 아니라 커버리지 합집합으로 동작한다.

## 5. 회계 원칙 (불변 규칙)

- **R1 — parked도 share 분모에**: offload는 메모리만 비운다. 제외하면 s가 부풀어 M_cpu 과소평가 → 과잉 offload (실측: cap32 offload 8→1회 교정).
- **R2 — parked도 E_viol에**: CPU에 내린 위험은 $R_{defer\_cpu}$로 계속 계상 (회계 연속성).
- **R3 — 절대 위반 예산 유지**: `E_viol < 1`. 한계 기준(ΔE_viol)은 매몰비용 잠금을 풀지만 총위험 브레이크를 제거해 기각됨 (e_viol 16~41 폭주 실증; 코드에 기본 off로 보존).
- **R4 — decode는 어떤 예산의 대상도 아니다**: 디코드 집합은 오직 ②의 run/defer 분할이 정하며, TB*는 prefill 토큰에만 걸린다 (D축 폐기, 2026-08-13). run으로 확정된 요청의 디코드 토큰은 잘리지 않는다. 원시 P+D 토큰 합산 캡은 단가 비대칭(KV-읽기 상각) 때문에 기각 — 예산의 화폐는 토큰 수가 아니라 **예측 스텝 시간**이다.
- **R5 — 추정기는 자기 결정에 오염되면 안 됨**: κ의 기준선은 배치-매칭 셀만(폴백 금지), 샘플은 토큰 가중(ratio-of-sums, per-sample 나눗셈·클리핑 금지). 위반 시 폐루프 실증됨 (κ↑→P*↓→소P 샘플↑→κ↑).

## 6. 폐기 결정 (근거 요약)

| 폐기안 | 사유 |
|---|---|
| request-level adaptive (`max_num_seqs` 축소) | 축이 틀림 — Δ 변동의 주범은 prefill 토큰(스텝 6%가 벽시계 18%, p90 463ms)이지 요청 수가 아님. 큐-블라인드 목적함수와 결합해 배치 1까지 붕괴 (대기 2,691명 방치). TB*(P축)로 대체 |
| admission 한계 기준 (ΔE_viol) | R3 참조 — 총위험 무한 누적 |
| 원시 P+D 토큰 캡 | 두 토큰의 단가가 다를 수 있고(KV-읽기 상각 비대칭) 교환비가 regime·컨텍스트 의존 — 시간 단위 예산(§7)으로 일반화해야 함 |
| TB* D축 (decode defer로 Δ_dec 축소, 2026-08-10~08-13) | 게이트가 "E_viol 개선 시 채택"인데 디코드 집합 축소는 거의 항상 개선을 만들어 **구조적 상시 발동**(스텝 28%, 런당 8~11천 회) — 무개입 보장 부재. 실측 순수 비용: cap128 r1/r2에서 코어 대비 −44/−45 tput, 위반율 이득 없음. 상세 §3⑤ |
| TB* 무한-지평 재발 가격 (2026-08-11 초판) | Δ(P)=Δ_dec+κ_p·P를 마감까지 **모든** 미래 스텝에 적용 — 1스텝 κ_p·P(~0.1–0.4s) 비용을 영구 감속으로 ~20× 과대가격 (실측 prefill 포함 스텝은 8.9%뿐). phaseD 실증: cap128 floor 체류 7~22%→43~83%, r1 tput 507→391, TTFC 2.6×, 위반율 이득 없음. 매 스텝 재결정되는 비용을 전지평에 물리는 이중과금 — burst-horizon(§3⑤)으로 대체 |

## 7. Staged 확장 (미구현 — TODO, SWEEP_PLAN_v2.md와 동기)

1. **오프라인 프로파일**: prefill 비용곡선 cost(P) (κ 스칼라 제거 — 토큰당 비용이 청크 크기 의존: ≤512에서 0.23~0.30 vs 2048에서 0.16), decode 비용은 (배치 × 컨텍스트) 그리드 필수 (관측 데이터는 공선성으로 분리 불가 실증). 기존 CUDA-graph 프로파일의 하이브리드(오프라인 shape × 라이브 anchor) 패턴 재사용.
2. (승격됨 → §3⑤ TB*: 통합 예산은 2026-08-10 현행 정책으로 편입. κ_d 기울기 분해 없이 Δ_dec(D) 직접 조회로 구현 — 공선성 장애물은 선형 분해에만 해당했음. TTS 축 실측은 P축 실효 검증 항목으로 유지 — D축은 폐기됨.)
3. **③축 admission 잠금의 안전한 해제**: 델타+상한 하이브리드 / 유휴 조건부 델타 / M≈0 국소 제외 — 미결.

## 8. 알려진 한계 (기록)

- ③축 self-lock: 소수 고위험 in-flight의 위험 합이 절대 예산을 소진하면 KV·연산이 남아도 k*≈0 (과포화 + 만기 경과 큐에서 발생; R3 트레이드오프의 대가).
- o+pb cap64 저하: κ_p 오염(소분모 발산)으로 진단·수정 완료 — GPU 재실행으로 최종 확인 대기 (예측: κ→0.15 수렴, tput 392→~470).
- (해결됨 2026-08-14) `request_risk_cpu`의 doomed guard 비대칭 — §3④에 offload 자격 조건 `R_defer_cpu < 1`로 반영. 미구현으로 두었던 근거("실해 미관측")는 offload가 발동조차 못 하던 상태의 산물이었고, KV 회계 수정으로 발동이 시작되자 왕복 경로로 드러났다.
- **[부분 해명 2026-08-12] offload 허가-집행 괴리 — 두 개의 독립 결함**
  (실측: phaseP2 cap64 r4, offload 단독 28,231스텝 vs baseline 16,282스텝):
  1. **KV 모델 18배 과소평가**: `kv_blocks_per_new_admit=8` vs 실측 요청당 KV
     블록 p50 **141**. free 525블록에서 정책은 66명 가능으로 보고 k*=64를
     허가하나 실제 수용은 3.7명 — 관측된 "허가-집행 괴리"의 직접 원인.
     수정 방향: 요청당 블록 수를 상수가 아니라 실측 프롬프트 길이에서 산정.
  2. **offload tier가 신호는 받되 거의 집행되지 않음**: `kv_capped`는 정상
     발화(12.8%)하는데 전체 런에서 실제 offload는 **20회**뿐. 후보 게이트는
     순서대로 ① `min_residency`(기본 0 — no-op이므로 배제) ②
     `is_fully_mirrored` ③ `is_measurable` ④ `M_cpu ≤ ε`. 여기에 더해
     `blocks_needed = (k*_unconstrained − k*) × per_admit`이 결함 1의
     per_admit=8을 그대로 쓰므로 **요구량 자체가 18배 과소** — 결함 1과 2는
     독립이 아니라 같은 상수를 공유한다.
     **선행 가설(미확정)**: `is_fully_mirrored`는 in-flight store가 있으면
     False인데(`manager.py:676`), 디코딩 중에는 매 토큰이 새 블록을 만들어
     미러링이 계속 밀린다. deferred로 내려가야 블록이 안정되어 미러링이
     완료되는 구조라, 미러링 완료 전에 run/defer 재평가가 요청을 다시
     promote하면 영구히 자격 미달이 된다. 확인하려면 게이트별 기각 카운터가
     필요 — **계측 추가 필요**.
  귀결(SSLO 고유): in-flight 전원이 소비자보다 앞서 deferred되어 KV만 붙들고
  (running 0 / pending 58), 신규 admit은 실제 KV 부족으로 막히며, **전체 스텝의
  7.0%가 대기 3,006명을 둔 채 유휴**가 된다. baseline에는 이 상태가 0%다.
  ※ 정정: 초판에서 "빈 스텝 14.3% = SSLO 병리", "offload 발동률 0%"라 적었으나
  둘 다 과장이었다. 빈 스텝 14.3% 중 7.3%p는 in-flight 자체가 없는 구간으로
  baseline(12.6%)에도 동일하게 존재한다. SSLO 고유분은 7.0%p. kv_capped 0%는
  좁은 부분집합(running==0 & k*≥5)에서만 관측된 값이었다.
  ※ 이 결함은 phaseP2/phaseD/phaseE 전부에 동일하게 존재하므로 세 실험 간
  비교는 유효하나, offload tier의 실효는 수정 후 재측정해야 한다.
- **admission이 신규 admit의 미래 위험에 눈감음** (2026-08-14 관측, 미해결):
  신규 admit은 PREFILL/WARMUP 동안 *forced*라 E_viol에서 제외되므로, k* 탐색은
  "이들이 MEASURED가 된 뒤 지게 될 위험"을 계산에 넣지 않는다. 실측상 k*=26을
  한 스텝에 허가한 직후 e_viol이 0.12→1.36으로 뛰고 k*=0으로 잠기는 진동이
  나타난다. §3④의 share 개정은 offload 왕복을 막지만 이 과다-발급 자체는
  남는다. 후보: admit 램프(비운 만큼을 한 번에 쓰지 않음) / forced 요청의
  예상 위험을 할인 계상. 현행은 관측만 기록.
- **borderline 인질** (burst-horizon의 잔존 한계): floor에서는 살릴 수 있으나
  base에서는 doomed가 되는 요청(R 0.6→1.0)은 한계 기여가 ~0.4로 크다. 완전
  doomed(기여 0)와 마감 먼 요청(기여 0)은 해소됐지만 이 경계층은 남는다.
  상상 실행 실측: 요청당 발생률 1.75% → 재적 47명이면 스텝의 56.5%에 최소
  1명 존재(재적 5명이면 8.5%). ε_p=0.5 보정으로 실효는 억제되나 원리적으로는
  잔존 — regime별 재보정 없이 ε_p 하나로 두는 한 cap이 클수록 조임이 강해진다.
- offload tier는 full-attention(MLA 포함) 모델 한정 — hybrid(mamba/linear-attn)는 upstream 비호환 (`kv_offload_model_compat.md`).
- **KV 수요 추정이 prefix-cache 히트를 반영하지 않음 (2026-08-13, 신규 회계의
  잔존 한계)**: `blocks(q) = ceil(잔여 프롬프트 토큰 / block_size)`는 프롬프트
  전체를 새로 할당한다고 본다. 실제로는 prefix caching이 켜져 있으면 공유
  프리픽스만큼 신규 할당이 줄어드므로 요구량이 상방 편향된다 → k*를 과소
  발급하고 `kv_capped`를 과다 발화시킬 수 있다. 방향이 **보수적**(과다 admit
  이 아니라 과소 admit)이라 상수 8의 18배 과소평가와 달리 안전 측 오차이므로
  이번에는 두었다. 다만 §4의 엔진 구성 동등성 개정으로 prefix caching이 이제
  **전 모드 공통**이고 dialogue(멀티턴) 워크로드는 프리픽스 공유율이 높아
  편향이 커지는 방향이다. 여기서 추정기(히트율 보정)를 넣지 않는 이유는 R5 —
  자기 결정에 오염될 수 있는 양을 실측 없이 모델링하지 않는다.
  **확인 방법**: prefix caching을 켠 재측정에서 (a) 재적(num_handling_users)이
  KV 점유 대비 낮게 눌리는지, (b) `kv_capped` 발생률이 KV 점유율과
  괴리되는지(예: 점유 70%대인데 kv_capped가 상시 발화)를 dialogue 대 단일턴
  셀로 대조한다. 괴리가 확인되면 그때 실측 히트율 기반 보정을 도입한다.
