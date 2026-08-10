# Algorithm 3 — KV-Offload Tier (offload / onload)

Algorithm 1(`ScheduleStep`)이 만든 run/pause 분할 위에, KV가 admission의 binding constraint가 될 때만 개입하는 3-상태 확장이다: `run` / `pause`(KV GPU 상주) / **`parked`(KV CPU 상주)**. 붙여넣기용 `algorithm2e` 블록과 술어 정의, invariant, 복잡도 순.

<!-- role: design — algorithm box, ready for direct paste into LaTeX with `algorithm2e` package -->

```latex
\begin{algorithm}[t]
\DontPrintSemicolon
\caption{KV-offload tier (\textsc{OffloadOnload}), run once per step after \textsc{ScheduleStep}.}
\label{alg:kv-offload}
\KwIn{run set $R'$, pause set $P'$ (KV on GPU), parked set $O$ (KV on CPU), time $t$;\\
\quad step latency $\Delta$, onload lead $\ell$ (iterations), risk budget $\varepsilon$,\\
\quad min-residency $\rho$ (steps), KV admission signal $\mathit{kvCapped}$ from the\\
\quad admission gate (Alg.~1, waiting-gate step).}
\KwOut{updated $(R', P', O)$; connector ops \textsc{Save}/\textsc{Load} issued.}
\BlankLine
\tcp{--- Onload path first: deadline safety dominates reclamation ---}
\ForEach{$r \in O$}{
  \uIf(\tcp*[h]{restore before the deadline horizon closes}){$d_r - t \;\le\; \ell\Delta + \widehat{f}(r)$}{
    issue \textsc{Load}$(r)$;\quad $O \leftarrow O \setminus \{r\}$;\quad $R' \leftarrow R' \cup \{r\}$
    \tcp*{ONLOADING: holds a slot, excluded from $E_{\mathrm{viol}}$ until resident}
  }
}
\BlankLine
\tcp{--- Offload path: only when KV, not risk, blocks admission ---}
\If{$\mathit{kvCapped}$}{
  $V \leftarrow \{\, r \in P' \;:\; M_{\mathrm{cpu}}(r) \le \varepsilon \;\land\; \mathrm{res}(r) \ge \rho \,\}$
  \tcp*{cheap CPU stay + anti-thrash}
  sort $V$ by descending $\mathrm{slack}_r$ \tcp*{deepest slack parked first}
  \ForEach{$r \in V$ \textbf{while} $\mathit{kvCapped}$}{
    issue \textsc{Save}$(r)$;\quad $P' \leftarrow P' \setminus \{r\}$;\quad $O \leftarrow O \cup \{r\}$\;
    recompute $\mathit{kvCapped}$ with the freed blocks\;
  }
}
\BlankLine
\Return $(R', P', O)$\;
\end{algorithm}
```

술어·기호 정의 (Algorithm 1의 표기 위에 추가):

```latex
\begin{align*}
R_{\mathrm{defer}}(r)      &\;=\; \Pr\!\big[\,L_r > c_r + N_{\mathrm{defer}}(r) \;\big|\; L_r > c_r\,\big],\\
R_{\mathrm{defer\_cpu}}(r) &\;=\; \Pr\!\big[\,L_r > c_r + N_{\mathrm{defer\_cpu}}(r) \;\big|\; L_r > c_r\,\big],
\qquad N_{\mathrm{defer\_cpu}} = N_{\mathrm{defer}} - \ell\,/\,\widehat{\tau}(r),\\
M_{\mathrm{cpu}}(r)        &\;=\; R_{\mathrm{defer\_cpu}}(r) - R_{\mathrm{defer}}(r) \;\ge\; 0,
\end{align*}
```

- $L_r$: 진행 중 청크의 최종 길이(토큰), $c_r$: 현재까지 생성량 — tail posterior는 청크 길이 예측기(§predictor)에서 나온다.
- $N_{\mathrm{defer}}(r)$: GPU-defer 상태로 다음 deadline $d_r$까지 생산 가능한 토큰 수; CPU-defer는 onload lead $\ell\Delta$만큼 예산이 준다.
- $M_{\mathrm{cpu}}$: **CPU 체류의 위험 프리미엄**. $\varepsilon$(기본 $10^{-3}$) 이하 = "CPU에 내려도 deadline 위험이 사실상 늘지 않는" 요청만 offload 후보.
- $\mathrm{res}(r)$: 현 위치(GPU/CPU) 연속 체류 스텝 수 — $\rho$ 미만이면 이동 금지(진동 방지).
- parked 요청도 위반 예산에는 $R_{\mathrm{defer\_cpu}}$로 **계속 계상**된다(회계 연속성): $E_{\mathrm{viol}} = \sum_{run} R_{\mathrm{run}} + \sum_{pause} R_{\mathrm{defer}} + \sum_{parked} R_{\mathrm{defer\_cpu}}$.

<!-- role: invariants and complexity -->
**Invariants.**
(I1) *Deadline safety*: onload 조건 $d_r - t \le \ell\Delta + \widehat{f}(r)$이 offload 조건보다 먼저 평가되고, offload 자격이 $M_{\mathrm{cpu}} \le \varepsilon$이므로 parked 요청의 추가 위반 확률은 $\le \varepsilon$로 상한된다.
(I2) *No-loss*: offload는 \textsc{Save} 완료(커넥터의 블록 스냅샷) 후에만 블록을 반환하고, ONLOADING 요청은 상주 완료까지 슬롯을 점유한 채 $E_{\mathrm{viol}}$에서 제외되므로 상태 유실·이중 계상이 없다.
(I3) *Anti-thrash*: $\rho$-residency가 offload↔onload 진동을 구조적으로 차단한다(히스테리시스의 KV판).
(I4) *Non-interference*: $\mathit{kvCapped}=\textbf{false}$이면 이 알고리즘은 no-op이다 — KV-slack 구간에서 Algorithm 1의 결정은 변하지 않는다.

**Complexity.** 스텝당 $O(|O| + |V|\log|V|)$; $V \subseteq P'$이므로 전체는 Algorithm 1의 $O(N\log N)$에 흡수된다. 요청당 추가 상태는 위치·residency 카운터 $O(1)$과 커넥터의 CPU 블록 미러(용량 상수 `CPU_OFFLOAD_GB`로 상한).

**Scope.** 커넥터가 paged KV 블록을 저장·복원하므로 full-attention(MLA 포함) 모델에 적용된다; recurrent state(mamba/linear-attn hybrid)는 블록 의미론 밖이라 제외된다(`kv_offload_model_compat.md`).
