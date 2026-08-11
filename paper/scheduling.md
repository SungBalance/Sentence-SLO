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
⑤ Token Budget TB* 계산 — **기대-위험 예산 도싱** (2026-08-11 개정:
     worst-case 제약 γ·T_min은 t_min 보유자 1명이 전체를 인질 잡는 근시안
     — 회복기 P* floor 81% 실증 — 이라 기각. E_viol 화폐로 통일):
       TB*_pre = max{ P : E_viol(Δ(P, D')) − E_viol(Δ(0, D')) ≤ ε_p }
     E_viol(Δ)은 기존 build_plan을 Δ(P)=Δ_dec(D')+κ_p·P로 재평가한 값 —
     in-flight(running+pending+offloaded) **전원**의 위험 합이므로:
     · 한 명의 임박 마감이 아니라 총 기대 피해로 도싱을 정함
     · doomed(R≈1) 요청은 한계 기여 ~0 → 회복기의 가망 없는 생존자가
       prefill을 막지 못함 (매몰비용 자동 해소; overdue 특례 불필요)
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

## 7. Staged 확장 (미구현 — TODO, SWEEP_PLAN_v2.md와 동기)

1. **오프라인 프로파일**: prefill 비용곡선 cost(P) (κ 스칼라 제거 — 토큰당 비용이 청크 크기 의존: ≤512에서 0.23~0.30 vs 2048에서 0.16), decode 비용은 (배치 × 컨텍스트) 그리드 필수 (관측 데이터는 공선성으로 분리 불가 실증). 기존 CUDA-graph 프로파일의 하이브리드(오프라인 shape × 라이브 anchor) 패턴 재사용.
2. (승격됨 → §3⑤ TB*: 통합 예산은 2026-08-10 현행 정책으로 편입. κ_d 기울기 분해 없이 Δ_dec(D) 직접 조회로 구현 — 공선성 장애물은 선형 분해에만 해당했음. TTS 축 실측은 D축 실효 검증 항목으로 유지.)
3. **③축 admission 잠금의 안전한 해제**: 델타+상한 하이브리드 / 유휴 조건부 델타 / M≈0 국소 제외 — 미결.

## 8. 알려진 한계 (기록)

- ③축 self-lock: 소수 고위험 in-flight의 위험 합이 절대 예산을 소진하면 KV·연산이 남아도 k*≈0 (과포화 + 만기 경과 큐에서 발생; R3 트레이드오프의 대가).
- o+pb cap64 저하: κ_p 오염(소분모 발산)으로 진단·수정 완료 — GPU 재실행으로 최종 확인 대기 (예측: κ→0.15 수렴, tput 392→~470).
- `request_risk_cpu`에는 doomed guard($R_{defer}\ge1 \Rightarrow$ offload 부적격)가 없음 — §2의 run-side 가드와 비대칭. 현행 kv_capped 게이트 하에서 실해가 관측되지 않아 미구현 (후보 유지).
- TB* D축 프로브의 원장은 decode-set 단독(admits/parked 미포함) — 자기일관적
  hill-climb이라 안전성 문제는 없으나, 전체 원장 대비 방향성(defer 과소 발동
  = 보수)은 미증명 (검증 라운드 2026-08-11 지적, 알려진 한계로 유지).
- offload tier는 full-attention(MLA 포함) 모델 한정 — hybrid(mamba/linear-attn)는 upstream 비호환 (`kv_offload_model_compat.md`).
