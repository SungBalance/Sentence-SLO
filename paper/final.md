# ProgressServe: 연속 배칭 LLM 서빙을 위한 소모 가능 진행도 백프레셔

> 수정 스타일: LLM 추론 시스템 분야의 교수 PI 관점에서 작성. 목표는 폭넓은 서술이 아니라 정밀성이다. 모든 주장은 런타임의 불변식으로 방어 가능해야 하며, 그렇지 않다면 실험 계획에 포함되어야 한다.

---

## 0. 섹션 개요

- **초록** — 도전 과제 → 통찰 → 기여 (`abstract.md`의 버전 2).
- **§1 서론** — 도전 과제로 시작(Part A 버전 4) + 하나의 기여가 여러 장점을 만드는 파이프라인(Part C 버전 1). 이 도전 과제는 현재 서빙 런타임의 *튜닝 격차*가 아니라 *역량 격차*로 재구성된다.
- **§2 관련 연구** — 세 주제, 각 주제는 우리가 추가하는 정확한 메커니즘으로 끝난다.
- **§3 방법** — 표기법, CP-SLO 정의, 콜드 스타트 처리를 포함한 요청별 상태, 불변식과 복잡도를 갖춘 백프레셔 스케줄링, 같은 신호를 공유하는 세 정책.
- **§4 실험 계획** — 가설 → 워크로드 → 베이스라인 → 지표 → 어블레이션 행렬 → 민감도 → 적대 사례. 수치는 넣지 않으며 자리표시는 `[TBM]`으로 표시한다.
- **§5 논의** — 예상되는 리뷰어 우려와 타당도 위협.
- **§6 결론** — 승리 선언이 아니라 범위 진술.

문단 역할은 역방향 아웃라이닝 점검(skill `does-my-writing-flow-source.md`)을 위해 `<!-- role: ... -->`로 주석 처리되어 있다.

---

## 초록
<!-- abstract: Version 2 (Challenge → Insight → Contribution); scope kept honest -->

<!-- role: task -->
연속 배칭 엔진은 대화형 LLM 서빙을 대규모로 실현 가능하게 했다. 그러나 이들이 노출하는 SLO(TTFT, 토큰 간 지연, 요청 완료)는 토큰 또는 요청 중심이므로, 인간 독자나 다운스트림 소비자(예: 스트리밍 TTS 엔진)가 실제로 진행을 인지하는 단위에는 눈이 멀어 있다.
<!-- role: challenge -->
디코드 경합이 있을 때, 토큰 수준 지표가 같은 두 요청도 가시 동작은 매우 다를 수 있다. 한쪽은 사용자가 아직 읽고 있는 긴 문장 전체를 이미 버퍼링해 두었고, 다른 쪽은 방금 짧은 문장을 끝냈으며 다음 문장은 길다. 기존 스케줄러는 이러한 비대칭성을 표현하지 못한다. 그 상태 안에는 소비자가 보유한 *소모 가능 버퍼*가 담겨 있지 않기 때문이다.
<!-- role: insight -->
문제를 가르는 변수는 토큰을 얼마나 빨리 내보내는지가 아니라, 다음 진행 단위를 만들 것으로 예상되는 시간에 비해 요청이 이미 소비자에게 얼마나 많은 소모 가능 텍스트를 예치해 두었는가이다.
<!-- role: contribution -->
우리는 이를 **소모 가능 진행도 SLO(Consumable-Progress SLO, CP-SLO)** 로 정식화하고 **ProgressServe**를 제안한다. ProgressServe는 연속 배칭 런타임으로서 (i) 애플리케이션이 정의한 진행 단위 경계를 온라인으로 감지하고 (ii) 오프라인 캘리브레이션 없이 진행 단위별 생성 시간의 EMA / 윈도우 백분위 추정기를 유지하며 (iii) 그 결과로 나온 압력을 요청별 단일 slack 점수로 축약하여, 하나의 warmup, hysteresis, pressure 상수와 hysteresis 제어 규칙 아래 디코드 우선순위, 적응형 배치 크기 조정, 대기 큐 입장, KV 상주성을 움직인다.
<!-- role: benefit -->
백프레셔는 양의 slack을 가진 요청에만 가해진다. 마감에 가깝거나 이미 지난 요청은 예외 없이 실행되며 연속 demotion에 대한 하드 캡이 기아 방지를 보장한다.
<!-- role: experiment-summary (planned, not claimed) -->
우리는 문장 및 문단 단위 워크로드에서 ProgressServe가 총 처리량을 떨어뜨리지 않으면서 FCFS, 요청 우선순위, TTFT/TBT 인지 베이스라인 대비 CP-SLO miss rate와 tail consumable-stall time을 줄이는지 검증할 실험 계획을 함께 정리한다. **이 초안에서는 어떤 경험적 주장도 하지 않는다. 수치는 계획된 스윕 이후 채워진다.**

---

## §1 서론
<!-- introduction: Part A Version 4 (challenge in opening) + Part C Version 1 (one core contribution, multiple advantages) -->

<!-- role: opening + immediate exposure of the challenge -->
스트리밍 LLM 애플리케이션, 즉 채팅 어시스턴트, 라이브 캡셔닝, 실시간 TTS는 토큰 단위로 텍스트를 생성하지만 소비자는 더 큰 단위로 이를 읽거나 말한다. 연속 배칭(vLLM~\cite{kwon2023vllm}, Orca 스타일 스케줄러~\cite{yu2022orca}, Hugging Face TGI~\cite{tgi}) 위에 구축된 프로덕션 서빙 스택은 TTFT, 토큰 간 지연, 요청 완료 지연을 최적화한다. 이 지표들은 평균값으로는 유용하지만 대화형 사용에서 사용자 경험을 결정하는 경우에는 실패한다. 토큰 수준 지연이 건강해 보이더라도, 소비자가 이전 문장을 다 읽기 전에 다음 문장을 완료하지 못하는 요청을 포착하지 못한다.

<!-- role: prior-work-1 — name what current systems can represent -->
토큰 중심 SLO는 모든 출력 토큰을 똑같이 긴급한 것으로 다룬다. 경합 상황에서 이는 실질적으로 다른 두 상태를 하나로 붕괴시킨다. 첫 번째는 소비자가 아직 긴 버퍼 문장을 읽고 있는 요청이고, 두 번째는 소비자가 방금 가시 버퍼를 소진한 요청이다. 스케줄러에는 이 둘을 구분하는 상태가 없으므로 두 번째 요청을 첫 번째보다 우선할 수 없다.

<!-- role: prior-work-2 — show why even tighter token SLOs do not fix this -->
더 엄격한 토큰 수준 SLO인 time-between-tokens(TBT)와 그 tail percentile은 증상을 덜 자주 만들 수는 있지만 원인은 해결하지 않는다. 이들 역시 소비자의 남은 버퍼가 아니라 다음 토큰의 타이밍의 함수이기 때문이다. 요청 수준 지연 SLO는 반대로 전체 응답을 하나의 마감으로 붕괴시켜 스트리밍 전달의 구조를 무시한다. 스트리밍 인지 시스템은 출력 포매팅이나 다운스트림 합성을 위해 문장이나 구 경계를 감지하지만, 경계를 출력 측 관심사로 다룬다. 경계 인지 마감이 아직 런타임으로 되돌아가 스케줄링 입력이 되지 않은 셈이다.

<!-- role: technical challenge — be precise about what a fix must satisfy -->
해결책은 세 제약을 동시에 만족해야 한다. 위 접근들이 충분하지 않은 까닭이 여기에 있다. 우선 마감은 요청의 벽시계 시간이나 토큰 속도가 아니라 소비자의 실제 버퍼의 함수여야 한다. 다음으로 요청별 압력 신호는 모델 속도에 대한 요청별 오프라인 캘리브레이션 없이 관측 가능한 스트리밍 상태로부터 온라인으로 계산되어야 한다. 마지막으로 이 신호는 연속 배칭의 하드 불변식, 즉 고정 형태 디코드 스텝과 동시에 디코드되는 시퀀스 수에 대한 엄격한 상한 $\mathit{maxSeqs}$(vLLM의 `max_num_seqs`)와 깔끔하게 합성되어야 한다. 그래야 진동이나 기아를 도입하지 않고 모든 경합 관련 정책(우선순위, 배치 크기, 입장, KV 상주성)을 움직일 수 있다.

<!-- role: pipeline / our method (Pipeline Version 1) -->
우리는 소모 가능 진행도 slack을 런타임 백프레셔로 바꾸는 연속 배칭 LLM 서빙 런타임 **ProgressServe**를 제안한다. 각 요청은 작은 온라인 상태 객체(경계 감지기, consume-time 추정기, EMA / percentile generation-time 추정기)를 지니며 이 객체는 하나의 scalar slack을 스케줄러로 넘긴다. 하나의 hysteresis 제어 규칙이 이 scalar를 요청별 결정으로 바꾼다. 같은 scalar가 **디코드 우선순위, 적응형 배치 크기 조정, 대기 큐 입장, KV 상주성**을 움직인다. 토큰 또는 요청 중심 스케줄러와 달리 ProgressServe는 위험한 요청만, 그것도 시스템이 실제로 경합 상태일 때만 보호한다. 임시방편 우선순위 기법과 달리 이 규칙은 명시적인 anti-starvation cap으로 경계가 정해져 있으며 엔진의 $\mathit{maxSeqs}$ 불변식을 구성상 준수한다.

<!-- role: contributions (kept tight; the four-policy phrasing matches the abstract verbatim) -->
**기여.**
1. **CP-SLO**: 토큰이나 전체 응답이 아니라 애플리케이션 정의 진행 단위에 고정된 마감 정의(§3.2).
2. **백프레셔 스케줄링**: CP-SLO 압력을 하나의 이진 결정으로 바꾸는 step별 규칙. 구성상 (a) $|R| \le \mathit{maxSeqs}$, (b) 요청당 최대 $\mathit{maxConsec}$번의 연속 demotion, (c) 대기 큐가 비어 있을 때 demotion 없음이 보장된다(§3.4).
3. **단일 신호 통합**: 같은 slack 점수를 **디코드 우선순위, 적응형 배치 크기 조정, 대기 큐 입장, KV 상주성** 전반에 쓰며, 하나의 warmup, hysteresis, pressure 상수를 공유한다(§3.5).
4. **공개 평가 계획**(§4): 위 기여들을 반증할 수 있을 만큼 충분히 명시적인 가설, 베이스라인, 지표, 어블레이션, 민감도 연구와 함께 ProgressServe가 *도움이 되지 않을 것으로 예상되는* 사례 목록(§5).

---

## §2 관련 연구

### 2.1 연속 배칭 LLM 서빙
<!-- role: paradigm + capability gap -->
연속 배칭 엔진(PagedAttention을 갖춘 vLLM~\cite{kwon2023vllm}, Orca~\cite{yu2022orca}, TGI~\cite{tgi}, LightLLM~\cite{lightllm})은 매 step마다 요청 간 prefill과 decode token을 interleave하며 주요 preemption 신호로 KV-cache pressure에 의존한다. 이들의 기본 정책은 FCFS 또는 static priority이다. 어느 것도 요청별 출력 측 상태(무엇이 스트리밍되었는지, 어떤 단위가 진행 중인지)를 유지하지 않으므로 소비자 버퍼에 연결된 마감을 표현하지 못한다. ProgressServe는 vLLM 위에 구축되며 바로 이 누락된 상태를 더한다.

### 2.2 지연 및 SLO 인지 LLM 스케줄링
<!-- role: paradigm + capability gap -->
Sarathi-Serve~\cite{agrawal2024sarathi}, FastServe~\cite{wu2024fastserve} 및 관련 연구는 chunked prefill 또는 preemptive scheduling으로 토큰 수준 지연을 노린다. 이 시스템들은 특정 토큰 수준 분포(TTFT 또는 TBT)를 개선하지만, 한 요청의 모든 토큰이 하나의 공유 마감을 갖는 것으로 다룬다. 한 요청 내부에서 소모 가능 버퍼가 고갈되면서 발생하는 비대칭한 긴급성은 모델링하지 않는다. ProgressServe는 직교적이다. slack 점수는 같은 엔진 위에서 chunked prefill(Sarathi-Serve 스타일)과 결합될 수 있고, 우리는 그런 결합을 가장 강한 베이스라인으로 쓴다(§4.3).

### 2.3 스트리밍 인지 출력 구조화
<!-- role: paradigm + capability gap -->
스트리밍 TTS와 캡셔닝 파이프라인은 다운스트림 소비를 위해 LLM 출력을 문장, 구, 문단으로 세분화한다. 최근 평가 연구는 문장 수준 지연을 측정한다~\cite{streamtts, captionlatency}. 이 시스템들은 합성, 캐싱, 사후 지연 보고를 위해 출력 측에서 경계를 쓰지만 경계를 엔진 스케줄러로 다시 전파하지 않는다. ProgressServe는 경계, consume-time estimate, generation-time estimate를 통합된 스케줄러 입력으로 받아 이 루프를 닫는다.

### 2.4 실시간 및 마감 인지 스케줄러
<!-- role: connection + difference -->
§3.4의 pending/running 재분할 규칙은 고전적인 실시간 시스템의 deadline-aware scheduling, 즉 earliest-deadline-first~\cite{liu1973scheduling} 및 slack-based scheduling~\cite{stankovic1995deadline}과 개념적으로 관련이 있다. 차이는 실용적인 데 있다. 우리의 마감은 주어진 것이 아니라 추정된다. 마감 사이의 작업 자체도 온라인으로 추정된다. 또한 스케줄러는 고전적 RT 이론이 다루지 않는 연속 배칭 불변식을 준수해야 한다.

---

## §3 방법

### 3.1 표기법과 설정
<!-- role: notation table to keep terminology stable; columns: math symbol / code-or-prose name / meaning -->

| 기호 | 코드 / prose 이름 | 의미 |
|---|---|---|
| $r$ | `req` | 스트리밍 요청. |
| $c_i$ | `chunk_i` | $r$의 $i$번째 진행 단위(문장 / 문단 / 구). |
| $\mathrm{consume}(c_i)$ | `consume(c_i)` | 소비자가 $c_i$를 소비하는 데 필요한 애플리케이션 정의 초 단위 시간. |
| $t_0(r)$ | `decoding_start(r)` | $r$의 첫 디코드 토큰이 나온 벽시계 시간. |
| $t_{\text{end}}(c_i)$ | `end_time(c_i)` | $c_i$가 사용 가능해지는 벽시계 시간. |
| $\widehat{g}(r)$ | `gen_time(r)` | $r$의 한 진행 단위를 생성하는 데 필요한 초 단위 시간의 추정기. |
| $\widehat{\tau}(r)$ | `per_token_time(r)` | $r$의 단어/토큰당 초 단위 시간 추정기. |
| $\widehat{w}(r)$ | `chunk_word_count(r)` | $r$의 진행 단위당 단어 수 추정기. |
| $w_{\text{cur}}(r)$ | `current_chunk_word_count(r)` | $r$의 진행 중인 단위에서 이미 방출된 단어 수. |
| $\widehat{f}(r)$ | (derived) | 진행 중인 단위의 예측 완료 시간: $(\widehat{w}(r) - w_{\text{cur}}(r))\,\widehat{\tau}(r)$. |
| $\mathrm{slack}_i$ | `cumulative_slack` | 단위 $i$의 누적 slack(§3.2에서 정의). |
| $\mathrm{slack}(t)$ | `realtime_slack(t)` | 벽시계 $t$에서 다음 마감까지의 slack(§3.3에서 정의). |
| $R, P, W$ | `running, pending, waiting` | 한 step의 런타임 집합(§3.4). |
| $N$ | — | 한 step에서 $\lvert R \cup P\rvert$. |
| $n_p$ | `pending_count` | $\lvert P\rvert$(pressure 항에서 사용). |
| $\mathit{maxSeqs}$ | `max_num_seqs` | 엔진 `InputBatch` 용량(E1). |

런타임은 연속 배칭 엔진으로서 각 step마다 admitted request의 부분집합에 대해 고정 형태 디코드 배치를 commit한다. 두 엔진 불변식은 주어진 것으로 본다. (E1) $|R| \le \mathit{maxSeqs}$; (E2) preemption은 허용되지만 비용이 크다(recomputation 또는 KV offload).

### 3.2 소모 가능 진행도 SLO
<!-- role: motivation -->
토큰 수준 마감은 스케줄러에게 토큰이 *언제* 생성되어야 하는지를 말해 준다. 다만 지연이 *얼마나 큰 손해인지*는 알려 주지 않는다. 단위 $c_i$에서 소비자가 지연을 견디는 정도는 $c_0, \ldots, c_{i-1}$를 통해 누적된 버퍼에 따라 결정되며, 애플리케이션은 이를 $\mathrm{consume}(c_i)$로 정량화할 수 있다.

<!-- role: design -->
우리는 단위 $i$의 누적 slack을 다음과 같이 정의한다.

$$
\mathrm{slack}_i \;=\; t_0(r) \;+\; \sum_{j<i} \mathrm{consume}(c_j) \;-\; t_{\text{end}}(c_i).
$$

첫 항은 소비자가 단위 $c_0, \ldots, c_{i-1}$를 끝냈을 벽시계 시점이다(연속 소비를 가정). 둘째 항은 $c_i$가 실제로 사용 가능해진 시점이다. $\mathrm{slack}_i \ge 0$이면 소비자가 단위 $i$에서 한 번도 멈추지 않았다는 의미이다. $\mathrm{slack}_i < 0$은 *가시* miss이다. **CP-SLO**는 제공된 모든 요청의 모든 단위에 대해 $\mathrm{slack}_i \ge 0$을 요구한다.

<!-- role: advantage / scope -->
이 정의는 런타임이 관측할 수 있는 시간($t_0$, $t_{\text{end}}$)과 애플리케이션이 제공할 수 있는 양($\mathrm{consume}$)에만 기댄다. 토큰 속도, 시퀀스 길이, 큐 깊이에 기대지 않으므로 어떤 스케줄링 정책 아래에서도 잘 정의되며, 모든 베이스라인을 사후 평가할 수 있다.

### 3.3 요청별 상태와 콜드 스타트
<!-- role: motivation -->
$\mathrm{slack}_i$를 사후 계산하기는 간단하다. 이를 *다음 step*을 결정하는 데 쓰려면 다음 마감과 그 마감을 맞추는 데 남은 작업을 추정해야 한다.

<!-- role: design -->
각 요청은 모두 온라인으로 동작하는 세 구성요소를 가진 `RequestSLOState`를 지닌다.

1. **경계 감지기.** 플러그형 `ChunkBoundaryDetector`는 스트리밍 텍스트를 스캔하고, 공백이 뒤따르는 문장 종결 구두점, 문단 구분(`\n\n`), 사용자 제공 predicate에서 단위를 flush한다. `min_chunk_tokens` guard는 약어처럼 micro-chunk가 생기는 경우 이를 병합해 가짜 flush를 피한다.
2. **Consume 추정기.** 플러그형 `ConsumeEstimator`는 단위 텍스트를 초 단위 시간으로 매핑한다. 기본 `WordRateEstimator`는 `seconds_per_word`를 쓴다. TTS의 경우 이는 per-model calibration pass(1회, 오프라인)에서 얻은 audio-duration estimator로 대체된다.
3. **생성 시간 추정기.** smoothing $\alpha$를 가진 EMA 추정기 또는 windowed-percentile 추정기(구성 가능한 window에서 기본 p99). 둘 다 $\widehat{g}, \widehat{\tau}, \widehat{w}$를 산출한다(§3.1). 관측 sample 수가 $\mathit{warmup}$에 도달할 때까지 추정기는 *cold*로 본다. §3.4의 cold-start gate는 cold request가 pending으로 demote되는 것을 막는다.

이제 벽시계 시간 $t$에서 realtime slack을 다음과 같이 정의한다.

$$
\mathrm{slack}(t) \;=\; t_0(r) \;+\; \sum_{j \le k} \mathrm{consume}(c_j) \;-\; t,
$$

여기서 $c_k$는 가장 최근에 flush된 단위이다. 스케줄러는 $\mathrm{slack}_i$(chunk-aligned cumulative score)를 우선순위 key로 쓰고 $\mathrm{slack}(t)$를 §3.4의 $\textsc{ShouldEnter}$ / $\textsc{ShouldExit}$ predicate 입력으로 쓴다.

<!-- role: advantage -->
세 구성요소는 모두 요청별이며 온라인이다. 스케줄링 경로에는 오프라인 per-model speed calibration이 필요하지 않다. cold-start guard는 요청 자신의 generation-time distribution이 관측되기 전에 해당 요청이 demote되지 않도록 보장한다.

### 3.4 백프레셔 스케줄링
<!-- role: motivation -->
연속 배칭 step은 하나의 고정 형태 배치를 commit해야 하므로, 런타임은 요청별 압력을 이진 `run` / `pause` 결정으로 축약해야 한다. 세 실패 모드는 구성상 배제되어야 한다. (F1) 진동(같은 요청이 매 step 상태를 뒤집음), (F2) 기아(낮은 우선순위 요청이 영원히 실행되지 않음), (F3) 용량 위반($|R| > \mathit{maxSeqs}$).

<!-- role: design — algorithm box, ready for direct paste into LaTeX with `algorithm2e` package -->

```latex
\begin{algorithm}[t]
\DontPrintSemicolon
\caption{ProgressServe per-step scheduler (\textsc{ScheduleStep}).}
\label{alg:schedule-step}
\KwIn{running set $R$, pending set $P$, waiting queue $W$, time $t$;\\
\quad knobs $\Theta = \{\kappa_{\text{enter}}, \Delta, \lambda_p,\, \mathit{maxConsec},\, \mathit{warmup},\, \mathit{adaptive}\}$;\\
\quad engine cap $\mathit{maxSeqs}$.}
\KwOut{$R',\,P'$ with $|R'| \le \mathit{maxSeqs}$.}
\BlankLine
$C \leftarrow R \cup P$\;
sort $C$ by ascending $\mathrm{slack}_i$ \tcp*{가장 긴급한 요청부터}
$P_{\mathrm{prev}} \leftarrow P$;\quad $R' \leftarrow \emptyset$;\quad $P' \leftarrow \emptyset$\;
\ForEach{$r \in C$}{
  $c \leftarrow \mathit{consec}[r]$\;
  \uIf(\tcp*[h]{F2: anti-starvation}){$c \ge \mathit{maxConsec}$}{$e \leftarrow \textbf{false}$}
  \uElseIf(\tcp*[h]{cold start / no contention}){$r$ is cold $\lor$ $W = \emptyset$}{$e \leftarrow \textbf{false}$}
  \uElseIf{$r \in P_{\mathrm{prev}}$}{$e \leftarrow \neg\,\textsc{ShouldExit}(r,t,|P|)$}
  \Else{$e \leftarrow \textsc{ShouldEnter}(r,t,|P|)$}
  \uIf{$e$}{$P' \leftarrow P' \cup \{r\}$;\quad $\mathit{consec}[r] \mathrel{+}= 1$}
  \Else{$R' \leftarrow R' \cup \{r\}$;\quad $\mathit{consec}[r] \leftarrow 0$}
}
\BlankLine
$\mathit{cap} \leftarrow \mathit{maxSeqs}$\;
\If(\tcp*[h]{adaptive batch}){$\mathit{adaptive} \land \exists\, r \in R': \mathrm{slack}_r < 0$}{$\mathit{cap} \leftarrow \lfloor \mathit{cap}/2 \rfloor$}
\If(\tcp*[h]{F3: engine capacity}){$|R'| > \mathit{cap}$}{
  move the $|R'| - \mathit{cap}$ highest-slack requests from $R'$ to $P'$\;
}
\Return $(R', P')$\;
\end{algorithm}
```

요청별 predicate는 다음과 같다.

```latex
\begin{align*}
\textsc{ShouldEnter}(r,t,n_p) &\;:=\; \mathrm{slack}(t) \;>\; (\kappa_{\text{enter}} + \lambda_p\, n_p)\,\widehat{g}(r),\\
\textsc{ShouldExit}(r,t,n_p)  &\;:=\; \mathrm{slack}(t) \;\le\; (\kappa_{\text{enter}} + \lambda_p\, n_p - \Delta)\,\widehat{g}(r) \\
                              &\quad\;\lor\; \widehat{f}(r) \;\ge\; \mathrm{slack}(t) + \widehat{\tau}(r),
\end{align*}
```

여기서 $\widehat{g}(r)$는 진행 단위별 생성 시간 추정값, $\widehat{\tau}(r)$는 토큰당 시간, $\widehat{f}(r) = (\widehat{w}(r) - w_{\text{cur}}(r)) \cdot \widehat{\tau}(r)$는 진행 중인 단위의 예측 완료 시간이다($\widehat{w}$: 추정 단위 단어 수; $w_{\text{cur}}$: 진행 중인 단위에서 이미 방출된 단어 수).

<!-- role: invariants and complexity -->
**불변식.** F1(진동)은 $\textsc{ShouldEnter}$와 $\textsc{ShouldExit}$ 사이의 hysteresis gap $\Delta > 0$ 및 cold-start gate(pending eligible이 되기 전에 $\widehat{g}$가 $\geq \mathit{warmup}$ sample을 요구함)가 억제한다. F2(기아)는 $\mathit{maxConsec}$ guard가 강제한다. 이 guard는 요청별 loop의 *첫 번째* check이므로 slack 기반 결정보다 우선한다. F3(용량)는 post-loop overflow step이 무조건 강제한다. 곧 F3가 F2를 override할 수 있다는 의미이다(엔진 cap이 starvation prevention보다 우선함). 이는 의도적인 선택이다. $\mathit{maxSeqs}$를 위반하면 decode batch shape가 손상되기 때문이다.

**복잡도.** Step당 복잡도는 sort에 $O(N \log N)$, loop에 $O(N)$이며 $N = |R \cup P|$이다. 요청별 상태는 EMA 추정기의 경우 $O(1)$, windowed-percentile 추정기의 경우 $O(W_{p99})$이다(window $W_{p99} = $ `chunk_gen_p99_window`, 기본값 100). 메모리 오버헤드는 boundary detector의 pending-text buffer가 지배하며, 이 버퍼는 진행 중인 가장 긴 progress unit으로 상한이 정해진다.

### 3.5 단일 신호 통합
<!-- role: motivation -->
압력 신호는 엔진의 모든 경합 관련 정책이 이를 받아 쓸 때 가장 유용하다. 추가 정책 셋은 §3.4의 slack score를 추가 상태 없이 공유한다.

<!-- role: design — three policies, one signal; line refs are descriptive, not numeric -->
- **적응형 배치 크기 조정.** Post-warmup running request 중 하나라도 $\mathrm{slack}_i < 0$이면 해당 step의 유효 $\mathit{maxSeqs}$가 절반으로 줄어든다(Algorithm 1에서 F3 overflow check 직전의 cap-halve step). 근거는 다음과 같다. 최소 한 소비자가 이미 starving 상태라면 decode batch를 줄여 처리량을 희생하는 대신 토큰 수준 interference를 줄인다. 시스템은 적어도 하나의 CP-SLO가 위반된 동안에만 이 비용을 지불한다.
- **대기 큐 입장 gate.** $W$로부터의 admission은 (i) $|R| \ge \mathit{cap}$ (절반으로 줄어들 수 있는 cap)이거나 (ii) 어떤 $r \in P$가 이제 $\textsc{ShouldExit}(r, t, |P|)$를 만족하는 경우, 곧 이전에 demote된 요청이 곧 자신의 slot을 돌려받아야 하는 경우 중지된다. 이는 “해를 끼치지 말라”는 no-harm 원칙이다. 기존 요청이 곧 capacity를 필요로 할 때 새 요청에 capacity를 약속하지 않는다.
- **KV-offload victim selection.** `kv_cache_manager.allocate_slots(...)`가 실패하면 런타임은 $\arg\max_{r \in R \cup P} \mathrm{slack}_r$를 offload victim으로 고른다. KV connector는 cache를 CPU에 저장하고 resume 시 복원한다. 직관은 §3.4와 대칭이다. slack-rich request는 다음 소모 가능 마감까지 시간이 가장 많으므로 offload latency를 가장 잘 견딘다.

<!-- role: advantage and honest caveat -->
**장점.** 세 경합 관련 결정이 하나의 신호, 하나의 warmup, 하나의 hysteresis, 하나의 pressure constant를 받아 쓴다. 정책 사이에 조정해야 하는 별도 priority knob가 없으며 잘못 설정된 구성요소가 다른 구성요소와 조용히 불일치할 수 없다.

**주의점.** 단일 신호 설계는 slack score가 잘 calibrate되어 있을 때만 강점이다. 잘못 조정된 `seconds_per_word`나 trace 중간에 `gen_time` 분포가 이동하는 모델은 세 정책을 동시에 나빠지게 만들 수 있다. 우리는 이러한 입력에 대한 민감도를 일급 평가 목표로 다룬다(§4.6).

### 3.6 구현과 하이퍼파라미터
<!-- role: reproducibility table; one default value + range we will sweep + role -->

ProgressServe는 vLLM(V1 scheduler) 내부에 구현된다. 추가된 요청별 상태는 `vllm.sslo` 아래에 자리한다. §3.4–3.5의 네 정책은 하나의 `SsloConfig` flag(`enabled=True`)로 gate되므로 이를 비활성화하면 vanilla scheduler가 정확히 복구된다. 추정기 셋(boundary detector, consume estimator, generation-time estimator)은 모두 플러그형이다.

| 그룹 | 기호 / flag | 기본값 | 테스트 범위(§4.6) | 역할 |
|---|---|---|---|---|
| 추정기 | `chunk_unit` | `sentence` | $\{$ sentence, paragraph $\}$ | 경계 granularity. |
|  | `chunk_gen_estimator` | `ema` | $\{$ ema, p99 $\}$ | 생성 시간 추정기. |
|  | $\alpha$ (`ema_alpha`) | 0.2 | fixed | EMA smoothing factor. |
|  | `chunk_gen_p99_window` | 100 | fixed | p99 variant의 sliding-window size. |
|  | $\mathit{secPerWord}$ (`seconds_per_word`) | 0.28 | $\{0.20, 0.28, 0.36\}$ | 기본 reading rate(Phase 1, axis 6). |
|  | $\mathit{minChunk}$ (`min_chunk_tokens`) | 16 | $\{0, 16, 32\}$ | Micro-chunk merge guard(Phase 1, axis 5). |
| 스케줄러 | $\kappa_{\text{enter}}$ (`pending_enter_factor`) | 2.5 | $\{1.5, 2.5, 3.5\}$ | Pending-enter threshold multiplier(Phase 1, axis 1). |
|  | $\Delta$ (`pending_hysteresis_gap`) | 0.5 | $\{0.25, 0.5, 1.0\}$ | Hysteresis gap, F1 방지(Phase 1, axis 2). |
|  | $\lambda_p$ (`pending_pressure_lambda`) | 0.05 | $\{0, 0.05, 0.10\}$ | $\lvert P\rvert$에 대한 pressure pre-multiplier. Subscript $p$는 arrival rate $\lambda$와 구분한다(Phase 1, axis 3). |
|  | $\mathit{warmup}$ (`pending_warmup_chunks`) | 5 | $\{3, 5, 10\}$ | Pending-eligible 전 cold-start guard(Phase 1, axis 4). |
|  | $\mathit{maxConsec}$ (`max_consecutive_pending`) | 5 | fixed | Anti-starvation cap(F2). |
|  | $\mathit{adaptive}$ (`adaptive_batch_size`) | `True` | $\{$on, off$\}$ ablation A5 | Overdue 시 cap 절반(§3.5). |
|  | $\mathit{offload}$ (`offloading`) | `True` | $\{$on, off$\}$ ablation A7 | Slack 기반 KV offload(§3.5). |
| 엔진 | `max_num_seqs` | model-specific | inherited | Hard `InputBatch` capacity(E1). |
|  | `max_num_batched_tokens` | model-specific | inherited | Per-step token budget. |

기본값은 `vllm.sslo.config.SsloConfig`에 commit된 값과 일치한다. Phase-1 민감도(§4.6)는 기본 operating point 주변에서 한 번에 한 row씩 변형한다. Phase 2는 pre-registered adversarial corner를 측정한다. Phase 3는 살아남은 config를 두 번째 cell로 transfer한다. “fixed”로 표시된 것은 전체 실험 계획에서 고정해 sweep을 다루기 쉽게 한다. paper-level claim이 그것에 의존하는 경우에만 fixed knob를 다시 검토한다.

---

## §4 실험 계획

이 섹션은 *계획*이지 결과 보고가 아니다. 모든 정량 주장은 `[TBM]`(to be measured)으로 표시되며 반증 가능한 가설과 짝지어진다. 계획된 sweep 이후 수치가 `[TBM]`을 대체한다.

### 4.1 가설

우리는 **contention knee**를 정책 $\pi$의 CP-SLO miss rate가 reference workload에서 처음으로 고정된 작은 threshold(1% 사용)를 초과하는 arrival rate $\lambda_{\text{knee}}^\pi$로 정의한다. 이렇게 하면 각 정책은 자신의 knee를 갖는다. 가설은 이 knee를 기준으로 진술된다.

**H1 (효과성).** 지속적인 decode contention($\lambda > \lambda_{\text{knee}}^{\text{FCFS}}$) 아래에서 ProgressServe는 같거나 더 높은 goodput에서 FCFS, request-priority, TBT-aware scheduler 대비 CP-SLO miss rate를 줄인다.

**H2 (저부하 no-harm).** $\lambda < \lambda_{\text{knee}}^{\text{FCFS}}$일 때, ProgressServe는 모든 primary 및 secondary metric에서 FCFS와 통계적으로 구별되지 않는다.

**H3 (신호 충분성).** 단일 신호 소비자 넷(priority, adaptive batch, admission gate, KV-offload-by-slack) 중 하나를 빼면, 나머지를 유지하더라도 CP-SLO miss rate 또는 tail stall이 측정 가능하게 나빠진다.

**H4 (no starvation).** 어떤 sweep에서도 요청이 $\mathit{maxConsec}$개의 연속 scheduling step보다 더 오래 pending 상태에 머물지 않는다.

**H5 (추정기 noise에 대한 강건성).** $\mathit{secPerWord}$의 $\pm 30\%$ perturbation이나 EMA에서 p99 estimator로의 전환은 CP-SLO miss rate를 strongest baseline과의 gap보다 덜 변화시킨다.

각 가설에는 아래에 계획된 test가 붙어 있다. H1과 H3는 논문의 중심 주장을 담고 있으며 반드시 필요한 실험을 결정한다.

### 4.2 워크로드
**데이터셋.** ShareGPT-derived prompts(single-turn) 및 paragraph-granularity stress를 위한 long-form completion mix(예: WritingPrompts).
**Granularities.** Sentence, paragraph, TTS phrase. 각 granularity는 그에 맞는 `ChunkBoundaryDetector`와 `ConsumeEstimator`를 쓴다.
**Arrival processes.** 세 regime을 둔다.
- *Steady.* Underloaded → knee → overloaded 범위를 포괄하려고 여러 $\lambda$에서 Poisson(λ).
- *Bursty.* Transient 상황에서 admission gate와 adaptive batch를 시험하려는 on/off arrivals(예: 90초마다 30초 동안 5×λ).
- *Mixed.* 공존 요청 사이의 slack-asymmetry를 시험하려는 길이가 다른 요청(short Q&A + long-form).

각 (dataset, granularity, arrival) tuple에 대해 FCFS knee의 60–140%를 포괄하는 다섯 지점에서 $\lambda$를 sweep한다. 각 cell은 ≥ 3 seeds로 실행한다.

### 4.3 베이스라인
1. **FCFS-CB.** Vanilla vLLM continuous batching.
2. **Priority-CB.** Request length를 기준으로 static request priority를 적용한 vLLM(naive QoS의 proxy).
3. **TBT-aware (Sarathi-Serve-style).** 아래에 상세 reproduction.
4. **Oracle-slack.** 별도 run에서 offline-computed ground truth로 `gen_time`과 `consume`을 대체한 ProgressServe. Estimator noise로 인한 gap의 상한을 정한다. 배포 가능한 baseline은 아니지만 ProgressServe의 calibration ceiling이다.
5. **ProgressServe.**

모든 baseline은 같은 engine, 같은 model, 같은 hardware, 같은 `max_num_seqs`에서 실행된다. SSLO state는 모든 정책에 대해 *계산*된다(따라서 FCFS, Priority, TBT-aware에도 CP-SLO miss rate가 잘 정의된다). 다만 ProgressServe와 Oracle-slack만 그 값을 *행동에 사용*한다.

**TBT-aware baseline reproduction.** Sarathi-Serve는 (i) prefill을 $C$ token의 fixed-size chunk로 나누어 decode token과 interleave하고 (ii) worst-case TBT가 target $T^\star$ 아래에 있도록 고른 token budget $B$로 per-step compute를 cap해 per-step token-level latency를 노린다. vLLM upstream은 이미 chunked prefill을 지원하므로 이를 재구현하지 않고 같은 엔진 위에서 reproduce한다. 우리는 `enable_chunked_prefill=True`를 설정하고, low-load probe trace에서 prefill chunk size $C$를 고정하며, single-request-at-a-time probe에서 TBT p99가 목표 $T^\star$와 같아지도록 `max_num_batched_tokens` $= B$로 설정한다(35 B-class model에서는 $T^\star = 50$ ms 사용; (model, hardware) pair당 한 번 calibration run). Base scheduler는 FCFS로 둔다. 이후 다른 baseline과 같은 workload 및 arrival trace에 TBT-aware engine을 적용한다. 이는 *토큰 수준* latency를 노리는 가장 강한 prior art이다. TBT p99를 구성상 개선하므로 우리는 정확히 그 축에서 앞서고 싶지 않다(H2 no-harm). 그러므로 ProgressServe가 이 baseline 대비 CP-SLO miss rate에서 보이는 이득은 더 엄격한 token timing만으로는 설명될 수 없다.

### 4.4 지표

$\mathcal{R}$을 완료된 요청의 집합, $\mathcal{C}_r = \{c_0, c_1, \ldots\}$를 $r \in \mathcal{R}$의 progress unit으로 둔다. 이 subsection의 표기법은 §3.1을 따른다.

**Primary.**

- **CP-SLO miss rate** $\downarrow$: CP-SLO를 miss한 progress unit의 비율.

$$
\mathrm{Miss} \;=\; \frac{\bigl|\{(r, c_i) : r \in \mathcal{R},\, c_i \in \mathcal{C}_r,\, \mathrm{slack}_i < 0\}\bigr|}{\sum_{r \in \mathcal{R}} |\mathcal{C}_r|}.
$$

- **Tail consumable-stall** $\downarrow$: 요청별 최악 stall duration의 tail.

$$
\mathrm{Stall}_q \;=\; q\text{-th percentile of}\; \bigl\{\, \max\bigl(0,\; -\min_{c_i \in \mathcal{C}_r} \mathrm{slack}_i\bigr) \;:\; r \in \mathcal{R} \,\bigr\},\qquad q \in \{95, 99\}.
$$

- **Goodput** $\uparrow$: CP-SLO miss가 0인 요청으로 제한한 completion rate.

$$
\mathrm{Goodput} \;=\; \frac{\bigl|\{r \in \mathcal{R} : \min_{c_i} \mathrm{slack}_i \ge 0\}\bigr|}{\text{total wall-clock duration of the trace}}.
$$

**Secondary.**

- Aggregate throughput(tokens / s, requests / s).
- Token-level latency: TTFT, TBT at p50 / p95 / p99.
- *Pending health*: request당 pending-interval count, max consecutive pending steps(H4를 위해 $\mathit{maxConsec}$와 비교), total pending time.
- *Offload health*($\mathit{offload} = \text{on}$일 때): offload count, restore latency, CP-SLO를 miss한 offloaded request의 비율.

Secondary token-level metrics는 target이 아니라 no-harm check(H2)로 보고한다. ProgressServe는 TBT-aware scheduler 대비 TBT를 개선하도록 설계된 것이 아니다.

### 4.5 어블레이션(행렬)
Contention knee에서의 (sentence, ShareGPT, steady) cell에 대해:

| Variant | §3.6 기본값 대비 knob 변경 | 테스트 |
|---|---|---|
| A0 — Full | — | reference |
| A1 — No priority | slack score를 sort key로 쓰지 않음 | H3의 priority component |
| A2 — No backpressure | pending state 비활성화(force $P' = \emptyset$) | H3의 scheduling component |
| A3 — No hysteresis | $\Delta = 0$ | F1(oscillation) |
| A4 — No pressure | $\lambda_p = 0$ | bursty에서 cascading-pending |
| A5 — Fixed batch | $\mathit{adaptive} = \text{off}$ | H3의 adaptive-batch component |
| A6 — Greedy admission | admission gate 비활성화 | H3의 admission component |
| A7 — Default offload | $\mathit{offload} = \text{off}$, FCFS-tail로 fallback | H3의 KV-residency component |
| A8 — EMA → p99 | `chunk_gen_estimator: ema → p99` | estimator robustness(H5의 subset) |

각 variant는 같은 $\lambda$에서 ≥ 3 seeds로 실행한다. A1–A2, A5–A7 중 어떤 것도 CP-SLO metrics를 나빠지게 만들지 못하면, 해당 component에 대해 H3는 *반증*된다.

### 4.6 민감도
하이퍼파라미터 여섯의 full factorial sweep은 비현실적이다($3^6 = 729$ configs $\times$ 3 seeds = 2187 runs). 우리는 비용을 제한하면서도 H5를 반증할 수 있는 세 단계 계획을 쓴다.

**기본 operating point.** $\kappa_{\text{enter}} = 2.5$, $\Delta = 0.5$, $\lambda_p = 0.05$, $\mathit{warmup} = 5$, $\mathit{minChunk} = 16$, $\mathit{secPerWord} = 0.28$.

**Phase 1 — 기본값 주변 one-at-a-time** (cell: sentence × ShareGPT × steady × $\lambda_{\text{knee}}^{\text{FCFS}}$).
다른 값은 기본값으로 고정하고 한 번에 한 축씩 세 수준으로 변화시킨다. 여섯 축 $\times$ 2 non-default levels $= 12$ configs(기본값은 공유). 각 config $\times$ 3 seeds $= 36$ runs.

| Axis | Levels |
|---|---|
| $\kappa_{\text{enter}}$ | 1.5, **2.5**, 3.5 |
| $\Delta$ | 0.25, **0.5**, 1.0 |
| $\lambda_p$ | 0, **0.05**, 0.10 |
| $\mathit{warmup}$ | 3, **5**, 10 |
| $\mathit{minChunk}$ | 0, **16**, 32 |
| $\mathit{secPerWord}$ | 0.20, **0.28**, 0.36 |

Phase 1은 H5를 시험한다. 각 축은 CP-SLO miss rate가 strongest baseline과의 gap 이내임을 보여야 한다. 실패한 축은 Phase 2로 escalate된다.

**Phase 2 — Adversarial corners** (같은 cell). 상호작용하는 축(estimator $\times$ pressure $\times$ hysteresis)에 대해 불리하도록 설계된 pre-registered corner config 넷을 측정한다. $(\Delta = 0.25, \lambda_p = 0)$, $(\Delta = 0.25, \kappa_{\text{enter}} = 1.5)$, $(\mathit{warmup} = 3, \mathit{secPerWord} = 0.20)$, $(\mathit{warmup} = 3, \mathit{secPerWord} = 0.36)$. $4 \times 3$ seeds $= 12$ runs.

**Phase 3 — Cross-cell transfer.** Phases 1–2에서 config 셋(**default**, Phase-1 **best** config, Phase-1(또는 Phase-2) **worst** config)을 가져와 두 번째 cell에서 실행한다. paragraph $\times$ WritingPrompts $\times$ bursty $\times$ $\lambda_{\text{knee}}^{\text{FCFS}}$. $3 \times 3$ seeds $= 9$ runs. Transfer test는 CP-SLO miss rate에 따른 config의 *순서*가 보존될 때만 통과한다(no inversion). 그렇지 않으면 “single config” 주장은 workload class에 조건부이며, 이를 그대로 보고한다.

**총 예산.** $36 + 12 + 9 = 57$ sensitivity runs. “no per-case retuning” 주장은 (a) Phase 1 worst가 primary cell에서 strongest baseline을 이기고, *그리고* (b) Phase 3에서 ordering inversion이 없을 때만 성립한다.

### 4.7 적대 / 실패 모드 사례
- *Drift.* Token rate에 step-change(예: t = 60s에서 model switch)를 주입하고 EMA vs. p99 estimator의 time-to-recover를 보고한다.
- *Wrong consume estimate.* `seconds_per_word`를 true value의 절반 / 두 배로 설정해 실행한다. ProgressServe는 *over-* 또는 *under-*demote할 것으로 예상된다. starvation cap(H4)이 여전히 유지되는지, goodput이 collapse되는지를 보고한다.
- *Heavy KV pressure.* KV preemption이 지배하도록 고른 workload. KV-offload-by-slack이 FCFS-tail보다 나은지, 아니면 해를 끼치는지(offloaded request가 여전히 포화된 GPU로 돌아오는 경우) 시험한다.
- *No-slack workload.* 모든 요청이 매우 짧은 출력(한 문장)을 가지는 경우. ProgressServe는 no-harm tolerance(H2) 안에서 FCFS-CB로 수렴해야 한다.

### 4.8 보고

**표.** `booktabs` 스타일: caption은 위에, header에는 `↑/↓` 화살표, best/second-best 강조, vertical rule 없음. 한 표 = 한 메시지.

**Figure 1 — Teaser (single-trace comparison).** 하나의 contended trace에 대해 같은 wall-clock $x$-axis를 공유하는 2행 2열 figure.

- **Row layout.** 각 row는 하나의 요청 $r_A$와 $r_B$에 해당한다. 두 요청은 모두 $t = 0$에서 시작한다. $r_A$는 긴 opening sentence(큰 initial consumable buffer)를 가지고, $r_B$는 짧은 opening sentence(작은 initial buffer)를 가진다.
- **Per-request tracks.** 각 row 안에는 stacked track 셋이 있다.
  - Track 1 — *token emissions*(token당 vertical tick 하나).
  - Track 2 — *progress-unit boundaries*(문장이 flush되는 지점의 downward triangle). 두 triangle 사이의 horizontal segment는 소비자가 *그 시점에 보유한* consumable buffer를 나타내도록 shading한다. Segment는 소비자가 $1/\mathit{secPerWord}$ rate로 이를 소진함에 따라 오른쪽으로 갈수록 dim해진다.
  - Track 3 — *consumer playback*(소비자가 해당 consume rate로 단위를 끝냈을 upward triangle). Consumable-stall은 unit $i+1$에 대한 Track 3의 triangle이 unit $i+1$에 대한 Track 2의 triangle보다 먼저 도착하는 region이다. 이 region은 빨간색으로 shading한다.
- **Column (a): FCFS-CB.** 두 row가 같게 경쟁한다. $r_B$의 두 번째 문장은 consumable deadline을 miss한다(row B에서 첫 번째와 두 번째 progress-unit triangle 사이에 빨간 region).
- **Column (b): ProgressServe.** 같은 arrival이지만, $r_A$가 positive slack을 축적하는 순간 scheduler는 $r_A$를 *pending*으로 demote한다(row A의 token track에 “P” marker로 annotation하여 그 interval에는 token이 emit되지 않음을 나타냄). 이제 $r_B$의 두 번째 문장은 소비자가 drain하기 전에 도착한다. 빨간 region은 사라진다. Row A의 이후 token emission은 (a)에 견주어 약간 지연되지만, $r_A$의 buffer가 여전히 앞서 있었기 때문에 *가시 stall은 없다*.
- **Annotations.** (a)의 빨간 region을 가리키는 “consumable stall” label의 arrow 하나. (b)의 두 row 사이에 “slack reallocated” label의 arrow 하나.
- **Why this teaser works.** 한 trace에서 (i) $r_A$의 token-level rate가 가시적으로 나빠지지 않고, (ii) (a)와 (b) 사이에서 바뀌는 metric은 오직 consumable-stall이며, (iii) 메커니즘은 throughput improvement가 아니라 reallocation임이 드러난다.

**Figure 2 — Miss-rate vs. arrival rate.** (granularity, dataset) pair마다 panel 하나. $x$-axis: arrival rate $\lambda$, FCFS knee로 정규화($\lambda / \lambda_{\text{knee}}^{\text{FCFS}}$, 무차원). $y$-axis: CP-SLO miss rate(linear, 0–1). Policy별 곡선 하나(FCFS, Priority, TBT-aware, ProgressServe, Oracle-slack). Shaded bands: seed 사이 $\pm$ one standard error. $x = 1$의 dashed vertical line은 knee를 표시한다. H1의 경험 주장은 다음으로 축약된다. ProgressServe의 곡선은 knee 오른쪽 전 구간에서 FCFS, Priority, TBT-aware보다 낮고 왼쪽에서는 no-harm tolerance 안에 있다.

**Figure 3 — ProgressServe pipeline diagram.** 관측 가능한 스트리밍 상태가 스케줄링 결정이 되는 방식을 드러내는 horizontal three-stage data-flow diagram.

- **Stage A (left) — Per-request observables.** `RequestSLOState` label의 box 안에 sub-block 셋을 세로로 쌓는다.
  - `ChunkBoundaryDetector` — input arrow label “streamed text $\Delta$”; output arrows label “unit text $c_i$” 및 “boundary timestamp $t_{\text{end}}(c_i)$”.
  - `ConsumeEstimator`(예: `WordRateEstimator`) — input arrow label “unit text $c_i$”; output arrow label “$\mathrm{consume}(c_i)$”.
  - `ChunkGenerationEstimator`(`EMA` 또는 `p99`) — input arrow label “$(t_{\text{end}}(c_i) - t_{\text{end}}(c_{i-1})$, word_count$(c_i))$”; output arrows label “$\widehat{g}(r), \widehat{\tau}(r), \widehat{w}(r)$”.
- **Stage B (center) — Slack fusion.** 작은 box `slack(r, t)` 하나가 $\mathrm{consume}(\cdot)$, $t_0(r)$, latest $t_{\text{end}}$, $\widehat{g}, \widehat{\tau}, \widehat{w}$를 입력으로 받아 두 output을 낸다. $\mathrm{slack}_i$(priority key)와 $\mathrm{slack}(t)$(predicate input). Box 위에는 formula $\mathrm{slack}_i = t_0 + \sum_{j<i} \mathrm{consume}(c_j) - t_{\text{end}}(c_i)$를 둔다.
- **Stage C (right) — Policies.** Slack output에서 fan out되는 policy box 다섯.
  - `Priority` — “ascending-slack sort key (Algorithm 1, sort step)”로 annotation.
  - `Backpressure (run/pause)` — “$\textsc{ShouldEnter}/\textsc{ShouldExit}$ predicates, with F2 / cold-start / contention guards”로 annotation.
  - `Adaptive batch size` — “halve $\mathit{cap}$ when any $\mathrm{slack}_r < 0$ in $R'$”로 annotation.
  - `Admission gate` — “stop admitting from $W$ if any $r \in P$ wants exit”로 annotation.
  - `KV offload victim` — “$\arg\max_{r \in R \cup P} \mathrm{slack}_r$ on `allocate_slots` failure”로 annotation.

  Layout note: box 다섯이 가로로 맞지 않으면 `Priority`를 `Backpressure` 안으로 접어 넣는다(둘은 같은 input ordering을 공유한다).
- **Annotations.** Engine invariants(E1, E2)는 오른쪽 edge에 작은 badge로 둔다. “engine step output”에서 Stage A로 돌아가는 dashed feedback loop를 두고, label은 “next streamed $\Delta$”로 한다. 색상: data-flow arrow와 control/decision arrow를 서로 다른 색으로 한다.
- **Why this diagram works.** 한 figure에서 (i) 추정기 셋이 모두 *요청별이며 온라인*이고 offline calibration이 없으며, (ii) Stage B에서 나가는 scalar가 하나뿐이므로 Stage C가 priority에 대해 “불일치”할 수 없고, (iii) engine invariants가 policy 내부가 아니라 외부 제약임이 드러난다.

**Figure 4 — Phase-1 sensitivity bars.** §4.6의 축별 panel 6개. 각 축의 sweep level 셋에서 CP-SLO miss rate를 bar chart로 표시하며 나머지 knob는 기본값으로 고정한다. 같은 cell에서 *strongest baseline*의 miss rate를 horizontal reference line으로 표시한다. H5 claim은 모든 panel의 모든 bar가 이 line 아래에 있을 때 시각적으로 드러난다.

**Figure 5 — Phase-3 transfer scatter.** $x$-axis: cell 1(sentence $\times$ ShareGPT $\times$ steady)의 CP-SLO miss rate. $y$-axis: cell 2(paragraph $\times$ WritingPrompts $\times$ bursty)의 같은 metric. 점 셋(default / best / worst Phase-1 configs)과 각 축의 strongest-baseline point를 포함한다. Reference로 diagonal $y=x$를 둔다. Ordering preservation은 점 셋이 diagonal을 기준으로 서로 같은 순서를 유지할 때만 보인다.

**재현성.** Sweep을 재현할 모든 scripts와 configurations는 `exp/run_sslo/`에 있다. 각 cell은 result file header에 (config, seed, commit) tuple을 기록한다.

---

## §5 논의

### 5.1 예상되는 리뷰어 우려
**“단일 신호 설계는 brittle하다.”** 인정한다. §4.6–4.7을 바로 이 brittleness를 bound하기 위해 계획했다. H5 falsification criterion도 명시되어 있다. H5가 실패하면, 논문의 주장은 “slack estimate가 true의 X% 이내일 때 single-signal design이 동작한다”로 약화된다.

**“Slack-richest를 KV-offload victim으로 삼는 것은 직관에 반한다. 그 요청이 종종 context를 가장 많이 갖고 있기 때문이다.”** 이 trade-off는 offload latency와 visible stall 사이에서 우리가 visible stall을 고른 결과이다. §4.7의 stress test(heavy KV pressure)는 이 선택이 지는 경우를 드러내도록 정확히 설계되었다. 우리는 그런 사례를 정직하게 보고할 것이다.

**“이것은 chunked prefill과 어떻게 합성되는가?”** Sarathi-Serve-style chunked prefill은 가장 강한 baseline으로 채택된다(§4.3). 또한 ProgressServe와 *같은 엔진*에서 실행된다. 두 메커니즘은 서로 다른 layer에서 작동한다(token-budget shaping vs. request-level demotion). 우리는 둘이 합성될 것으로 예상하지만 더 깊은 integration study는 이 논문의 범위를 벗어난다.

**“왜 RL-based scheduling이 아닌가?”** 학습된 policy는 원칙적으로 이 규칙을 능가할 수 있다. 다만 그것은 training data가 필요하고 deployment dependency를 더한다. 이 논문의 요지는 명시적 불변식을 갖는 작고 해석 가능한 규칙만으로도 관심 metric을 이동시킬 수 있음을 보이는 것이다. 학습된 policy는 후속 질문이다.

### 5.2 타당도 위협
- *Construct validity.* `consume(c_i)`는 애플리케이션이 제공한다. 애플리케이션의 estimate가 실제 consumer behavior와 벌어지면 CP-SLO는 더 이상 user experience를 반영하지 않는다. 이 논문에서는 실제 사용자를 측정하지 않는다.
- *Internal validity.* 모든 baseline은 policy를 isolate하기 위해 같은 engine 안에서 실행된다. runtime-overhead confound는 배제하지만, 우리 policy에 유리한 implementation bias까지 배제하지는 않는다.
- *External validity.* Sweep은 단일 GPU configuration과 작은 model set(§4.2)에서 수행된다. Cross-hardware behavior는 future work이다.

---

## §6 결론
<!-- conclusion: scope statement, no oversell -->

<!-- role: restate -->
우리는 대화형 LLM 서빙 SLO가 소비자가 실제로 진행을 인지하는 단위에 고정되어야 한다고 주장하며, 그 구체 실현으로 소모 가능 진행도 SLO와 ProgressServe를 내놓았다. ProgressServe는 CP-SLO pressure를 하나의 요청별 slack signal로 바꾸고, 이를 하나의 hysteresis-controlled rule을 거쳐 decode priority, adaptive batch sizing, waiting-queue admission, KV residency에 쓴다.
<!-- role: planned evidence -->
함께 내놓은 실험 계획(§4)은 중심 가설 넷을 확인하거나 반증하도록 설계되어 있다. 우리는 negative result를 포함한 결과를 released runtime과 함께 보고할 것이다.
<!-- role: scope limitation -->
현재 formulation은 애플리케이션이 progress-unit boundaries와 consume-time function을 정의할 것을 요구한다. 구조화되지 않은 streaming output을 가진 workload나, text의 함수로 표현할 수 없는 consumer behavior를 가진 workload는 범위 밖이다.
<!-- role: future work -->
확장 둘이 자연스럽다. 한쪽은 chunked-prefill scheduling과 더 깊이 통합하는 것(prefill token budget 자체를 running-set slack의 함수로 다룸)이고, 다른 한쪽은 static `consume` function을 gaze, comprehension, downstream synthesis backpressure에 적응하는 consumer model로 대체하는 것이다.

---

## Appendix A — 자기 검토 체크리스트
<!-- skill paper-review.md, five dimensions; honest about gaps -->

| # | 차원 | 질문 | 상태 |
|---|---|---|---|
| 1 | 기여 | 기술 아이디어가 잘 탐구된 practice를 넘어 non-obvious한가? | **pass** — CP-SLO + single-signal integration은 우리가 아는 한 새롭다. |
| 1 | 기여 | 명시적 novelty type이 최소 하나 있는가? | **pass** — new SLO, new scheduling rule, new system integration. |
| 2 | 글쓰기 | Method가 재현 가능한가? | **pass** — Algorithm 1 + notation table + invariants + complexity. |
| 2 | 글쓰기 | 용어가 안정적인가? | **pass** — `progress unit`, `consumable buffer`, `slack`, `gen_time`이 한 번 정의된다. |
| 3 | 경험적 | 강한 baseline이 포함되어 있는가? | **needs new experiment** — TBT-aware reproduction(Sarathi-Serve-style)과 Oracle-slack ceiling은 *계획*되어 있으며 아직 실행되지 않았다. |
| 3 | 경험적 | Multi-dataset / multi-granularity인가? | **needs new experiment** — ≥ 2 datasets × 3 granularities × 3 arrivals planned. |
| 4 | 평가 완전성 | 설계 선택마다 ablation이 있는가? | **planned** — §4.5 matrix A0–A8 참조. |
| 4 | 평가 완전성 | Sensitivity sweep이 있는가? | **planned** — §4.6. |
| 5 | 건전성 | 숨은 기술 결함이 있는가? | **pass** — F1(oscillation), F2(starvation), F3(capacity)는 구성상 처리되며, trade-off도 문서화됨. |
| 5 | 건전성 | Per-case retuning이 필요한가? | **needs new experiment** — claim은 §4.6 worst case가 strongest baseline을 이기는지에 조건부이다. |

---

## Appendix B — 주장–증거 맵

| # | 주장 | 증거 | 상태 |
|---|---|---|---|
| C1 | Token-/request-centric metrics는 consumable-buffer state에 blind하다. | §1 prior-work analysis; §3.2 definition shows independence; §4.1 H1 quantifies. | needs evidence(Fig. 2, Table 1) |
| C2 | 하나의 요청별 slack score가 경합 관련 정책 넷을 움직일 수 있다. | §3.4–3.5 construction; §4.5 ablations A1, A2, A5, A6, A7. | needs evidence |
| C3 | 이 규칙은 $|R| \le \mathit{maxSeqs}$, anti-starvation, no demotion under no-contention을 보장한다. | §3.4 invariants by construction; §4.1 H4 falsifiable test. | partial — invariants are by construction; H4는 implementation이 spec과 일치하는지에 대한 *empirical* check이다. |
| C4 | ProgressServe는 FCFS / Priority / TBT-aware 대비 CP-SLO miss rate를 줄인다. | §4.1 H1 → §4.4 primary metrics. | needs evidence |
| C5 | Low load에서 throughput / TBT가 regression되지 않는다. | §4.1 H2 → §4.4 secondary metrics. | needs evidence |
| C6 | 이 규칙은 $\mathit{secPerWord}$의 $\pm 30\%$ misestimation과 estimator choice에 강건하다. | §4.1 H5 → §4.6 sensitivity, §4.7 wrong-consume case. | needs evidence |
| C7 | KV-offload-by-slack은 heavy KV pressure에서 개선한다. | §4.7 stress test. | needs evidence; honest negative result acceptable. |

---

## Appendix C — 용어 노트
- `consumable`은 이전의 `readable` 대신 전반적으로 쓴다. 약어는 **CP-SLO**이다(`RP-SLO`가 아님).
- `progress unit`: 애플리케이션 정의 단위(sentence / paragraph / TTS phrase).
- qualifier 없는 `slack`은 $\mathrm{slack}_i$(cumulative, per progress unit)를 뜻한다. $\mathrm{slack}(t)$는 항상 명시한다.
- `pending`(engine state: KV를 보유하지만 decode되지 않음)은 `waiting`(아직 admitted되지 않음)과 구분된다.
- `[TBM]`은 §4 plan 실행에 의존하는 모든 정량 주장을 표시한다.

---

## Appendix D — Citation Keys (자리표시자)

이 초안에서 쓴 `\cite{...}` marker와 그 의미. LaTeX로 옮길 때 정확한 BibTeX entry로 대체할 것. Key는 venue-canonical하기보다 안정적이고 사람이 읽기 쉽게 골랐다.

| Key | 가리키는 대상 | 사용 위치 |
|---|---|---|
| `kwon2023vllm` | vLLM / PagedAttention (Kwon et al., SOSP 2023). | §1, §2.1 |
| `yu2022orca` | Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022). | §1, §2.1 |
| `tgi` | Hugging Face Text Generation Inference (TGI) — official GitHub. | §1, §2.1 |
| `lightllm` | LightLLM — official GitHub (ModelTC). | §2.1 |
| `agrawal2024sarathi` | Sarathi-Serve / chunked prefill (Agrawal et al., OSDI 2024). | §2.2, §4.3 |
| `wu2024fastserve` | FastServe — preemptive scheduling for LLM inference (Wu et al.). | §2.2 |
| `streamtts` | Streaming TTS evaluation work — 작성 시점의 canonical reference를 위한 placeholder. | §2.3 |
| `captionlatency` | Streaming-caption / live-captioning latency study — placeholder. | §2.3 |
| `liu1973scheduling` | Liu & Layland, *Scheduling algorithms for multiprogramming in a hard-real-time environment* (JACM 1973). | §2.4 |
| `stankovic1995deadline` | Stankovic et al., *Deadline scheduling for real-time systems: EDF and related algorithms*. | §2.4 |

placeholder 둘(`streamtts`, `captionlatency`)은 submission 전에 canonical citation을 확인해야 하는 영역을 가리킨다. 이 초안은 어느 특정 논문에도 아직 commit하지 않는다.
