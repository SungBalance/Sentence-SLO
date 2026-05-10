# ProgressServe: Consumable-Progress Backpressure for Continuous-Batching LLM Serving

> Revision style: written from the perspective of a faculty PI in LLM inference systems. Goal is precision over breadth — every claim should either be defensible from the runtime's invariants or be scheduled into the experimental plan.

---

## 0. Section Outline

- **Abstract** — Challenge → Insight → Contribution (Version 2 of `abstract.md`).
- **§1 Introduction** — Open with challenge (Part A Version 4) + one-contribution-multi-advantage pipeline (Part C Version 1). The challenge is reframed as a *capability gap* in current serving runtimes, not a tuning gap.
- **§2 Related Work** — Three topics, each ending with the exact mechanism we add.
- **§3 Method** — Notation, CP-SLO definition, per-request state with cold-start handling, backpressure scheduling with invariants and complexity, three policies sharing the same signal.
- **§4 Experimental Plan** — Hypotheses → workloads → baselines → metrics → ablation matrix → sensitivity → adversarial cases. No numbers; placeholders marked `[TBM]`.
- **§5 Discussion** — Anticipated reviewer concerns, threats to validity.
- **§6 Conclusion** — Scope statement, not a victory lap.

Paragraph roles are annotated as `<!-- role: ... -->` for reverse-outlining checks (skill `does-my-writing-flow-source.md`).

---

## Abstract
<!-- abstract: Version 2 (Challenge → Insight → Contribution); scope kept honest -->

<!-- role: task -->
Continuous-batching engines have made interactive LLM serving viable at scale, yet the SLOs they expose — TTFT, inter-token latency, request completion — are token- or request-centric and therefore blind to the unit at which a human reader or a downstream consumer (e.g., a streaming TTS engine) actually perceives progress.
<!-- role: challenge -->
Under decode contention, two requests with identical token-level metrics can have very different visible behavior: one has buffered an entire long sentence the user is still reading, the other has just finished a short sentence and faces a long next one. Existing schedulers cannot represent this asymmetry, because nothing in their state carries the *consumable buffer* the consumer holds.
<!-- role: insight -->
The decisive variable is not how fast we emit tokens, but how much consumable text a request has already deposited with the consumer relative to the expected time to produce the next progress unit.
<!-- role: contribution -->
We formalize this as the **Consumable-Progress SLO (CP-SLO)** and present **ProgressServe**, a continuous-batching runtime that (i) detects application-defined progress-unit boundaries online, (ii) maintains an EMA / windowed-percentile estimator of per-unit generation time without offline calibration, and (iii) collapses the resulting pressure into a single per-request slack score that drives decode priority, adaptive batch sizing, waiting-queue admission, and KV residency under one hysteresis-controlled rule with a single warmup, hysteresis, and pressure constant.
<!-- role: benefit -->
Backpressure is exerted only on requests with positive slack; requests near or past their deadline always run, and a hard cap on consecutive demotions provides a starvation guarantee.
<!-- role: experiment-summary (planned, not claimed) -->
We outline an experimental plan to test, on sentence- and paragraph-granularity workloads, whether ProgressServe reduces CP-SLO miss rate and tail consumable-stall time relative to FCFS, request-priority, and TTFT/TBT-aware baselines without regressing aggregate throughput. **No empirical claims are made in this draft; numbers will be filled from the planned sweep.**

---

## §1 Introduction
<!-- introduction: Part A Version 4 (challenge in opening) + Part C Version 1 (one core contribution, multiple advantages) -->

<!-- role: opening + immediate exposure of the challenge -->
Streaming LLM applications — chat assistants, live captioning, real-time TTS — produce text token-by-token while the consumer reads or speaks it in larger units. Production serving stacks built on continuous batching (vLLM~\cite{kwon2023vllm}, Orca-style schedulers~\cite{yu2022orca}, Hugging Face TGI~\cite{tgi}) optimize TTFT, inter-token latency, and request completion latency. These metrics are useful averages, but they fail at the case that determines user experience in interactive use: a request whose next sentence cannot be completed before the consumer finishes the previous one, even when token-level latency looks healthy.

<!-- role: prior-work-1 — name what current systems can represent -->
Token-centric SLOs treat every output token as equally urgent. Under contention this collapses two materially different states into one: a request whose consumer is still reading a long buffered sentence, and a request whose consumer has just exhausted its visible buffer. The scheduler has no state distinguishing them, so it cannot prioritize the second over the first.

<!-- role: prior-work-2 — show why even tighter token SLOs do not fix this -->
Tighter token-level SLOs — time-between-tokens (TBT) and its tail percentiles — make the symptom less frequent but do not address the source: they are still functions of the next token's timing, not of the consumer's remaining buffer. Request-level latency SLOs go to the opposite extreme by collapsing the entire response into one deadline, ignoring the structure of the streamed delivery. Streaming-aware systems detect sentence or phrase boundaries for output formatting and downstream synthesis, but treat the boundary as an output-side concern; the boundary-aware deadline has not been fed back into the runtime as a scheduling input.

<!-- role: technical challenge — be precise about what a fix must satisfy -->
A fix must satisfy three constraints simultaneously, which is why none of the above suffices. First, the deadline must be a function of the consumer's actual buffer, not of the request's wall time or token rate. Second, the per-request pressure signal must be computable online from observable streaming state, without per-request offline calibration of model speed. Third, the signal must compose cleanly with continuous batching's hard invariants — fixed-shape decode steps and a strict cap $\mathit{maxSeqs}$ on the number of concurrently decoded sequences (vLLM's `max_num_seqs`) — so it can drive *all* contention-relevant policies (priority, batch size, admission, KV residency) without introducing oscillation or starvation.

<!-- role: pipeline / our method (Pipeline Version 1) -->
We present **ProgressServe**, a continuous-batching LLM serving runtime that turns consumable-progress slack into runtime backpressure. Each request maintains a small online state object (boundary detector, consume-time estimator, EMA / percentile generation-time estimator) that exposes one scalar slack to the scheduler. A single hysteresis-controlled rule converts this scalar into a per-request decision; the same scalar drives **decode priority, adaptive batch sizing, waiting-queue admission, and KV residency**. In contrast to token- or request-centric schedulers, ProgressServe protects only the at-risk request and only when the system is actually contended; in contrast to ad-hoc priority schemes, the rule is bounded by an explicit anti-starvation cap and respects the engine's $\mathit{maxSeqs}$ invariant by construction.

<!-- role: contributions (kept tight; the four-policy phrasing matches the abstract verbatim) -->
**Contributions.**
1. **CP-SLO**, a deadline definition anchored to application-defined progress units rather than to tokens or whole responses (§3.2).
2. **Backpressure Scheduling**, a per-step rule that converts CP-SLO pressure into one binary decision while guaranteeing, by construction, (a) $|R| \le \mathit{maxSeqs}$, (b) at most $\mathit{maxConsec}$ consecutive demotions per request, and (c) no demotion when the waiting queue is empty (§3.4).
3. **Single-signal integration** of the same slack score across **decode priority, adaptive batch sizing, waiting-queue admission, and KV residency**, with a single warmup, hysteresis, and pressure constant (§3.5).
4. **An open evaluation plan** (§4), with explicit hypotheses, baselines, metrics, ablations, and sensitivity studies sufficient to falsify the above contributions, and an explicit list of cases (§5) where ProgressServe is expected to *fail* to help.

---

## §2 Related Work

### 2.1 Continuous-batching LLM serving
<!-- role: paradigm + capability gap -->
Continuous-batching engines — vLLM with PagedAttention~\cite{kwon2023vllm}, Orca~\cite{yu2022orca}, TGI~\cite{tgi}, LightLLM~\cite{lightllm} — interleave prefill and decode tokens across requests at every step and rely on KV-cache pressure as the principal preemption signal. Their default policies are FCFS or static priority. None maintain per-request output-side state — what has been streamed, what unit is in flight — so they cannot express a deadline tied to a consumer's buffer. ProgressServe is built on top of vLLM and adds exactly this missing state.

### 2.2 Latency- and SLO-aware LLM scheduling
<!-- role: paradigm + capability gap -->
Sarathi-Serve~\cite{agrawal2024sarathi}, FastServe~\cite{wu2024fastserve}, and related work target token-level latency through chunked prefill or preemptive scheduling. These systems improve specific token-level distributions (TTFT or TBT) but treat all tokens of a request as having a single shared deadline. They do not model the asymmetric urgency that arises within a single request as its consumable buffer drains. ProgressServe is orthogonal: the slack score can be combined with chunked prefill (Sarathi-Serve-style) on the same engine, and we use such a combination as our strongest baseline (§4.3).

### 2.3 Streaming-aware output structuring
<!-- role: paradigm + capability gap -->
Streaming TTS and captioning pipelines segment LLM output into sentences, phrases, or paragraphs for downstream consumption; recent evaluation work measures sentence-level latency~\cite{streamtts, captionlatency}. These systems use boundaries on the output side — for synthesis, caching, or post-hoc latency reporting — and never propagate the boundary back into the engine's scheduler. ProgressServe closes this loop by treating the boundary, the consume-time estimate, and the generation-time estimate as a unified scheduler input.

### 2.4 Real-time and deadline-aware schedulers
<!-- role: connection + difference -->
The pending/running re-partition rule in §3.4 is conceptually related to deadline-aware scheduling in classic real-time systems — earliest-deadline-first~\cite{liu1973scheduling} and slack-based scheduling~\cite{stankovic1995deadline}. The differences are practical: our deadlines are estimated, not given; the work between deadlines is itself estimated online; and the scheduler must respect continuous-batching invariants that classical RT theory does not address.

---

## §3 Method

### 3.1 Notation and Setting
<!-- role: notation table to keep terminology stable; columns: math symbol / code-or-prose name / meaning -->

| Symbol | Code / prose name | Meaning |
|---|---|---|
| $r$ | `req` | A streaming request. |
| $c_i$ | `chunk_i` | The $i$-th progress unit of $r$ (sentence / paragraph / phrase). |
| $\mathrm{consume}(c_i)$ | `consume(c_i)` | Application-defined seconds the consumer needs for $c_i$. |
| $t_0(r)$ | `decoding_start(r)` | Wall-clock time of the first decode token of $r$. |
| $t_{\text{end}}(c_i)$ | `end_time(c_i)` | Wall-clock time $c_i$ becomes available. |
| $\widehat{g}(r)$ | `gen_time(r)` | Estimator of seconds to produce one progress unit of $r$. |
| $\widehat{\tau}(r)$ | `per_token_time(r)` | Estimator of seconds per word/token of $r$. |
| $\widehat{w}(r)$ | `chunk_word_count(r)` | Estimator of words per progress unit of $r$. |
| $w_{\text{cur}}(r)$ | `current_chunk_word_count(r)` | Words already emitted in $r$'s in-progress unit. |
| $\widehat{f}(r)$ | (derived) | Predicted finish time of in-progress unit: $(\widehat{w}(r) - w_{\text{cur}}(r))\,\widehat{\tau}(r)$. |
| $\mathrm{slack}_i$ | `cumulative_slack` | Cumulative slack of unit $i$ (defined §3.2). |
| $\mathrm{slack}(t)$ | `realtime_slack(t)` | Slack to next deadline at wall-clock $t$ (defined §3.3). |
| $R, P, W$ | `running, pending, waiting` | Runtime sets at a step (§3.4). |
| $N$ | — | $\lvert R \cup P\rvert$ at a step. |
| $n_p$ | `pending_count` | $\lvert P\rvert$ (used in pressure term). |
| $\mathit{maxSeqs}$ | `max_num_seqs` | Engine `InputBatch` capacity (E1). |

The runtime is a continuous-batching engine that, at each step, commits a fixed-shape decode batch over a subset of admitted requests. Two engine invariants are taken as given: (E1) $|R| \le \mathit{maxSeqs}$; (E2) preemption is allowed but expensive (recomputation or KV offload).

### 3.2 The Consumable-Progress SLO
<!-- role: motivation -->
Token-level deadlines tell the scheduler when a token *should* be produced; they do not tell it how *bad* it is to be late. The consumer's tolerance for lateness on unit $c_i$ depends on the buffer accumulated through units $c_0, \ldots, c_{i-1}$, which the application can quantify through $\mathrm{consume}(c_i)$.

<!-- role: design -->
We define the cumulative slack of unit $i$:

$$
\mathrm{slack}_i \;=\; t_0(r) \;+\; \sum_{j<i} \mathrm{consume}(c_j) \;-\; t_{\text{end}}(c_i).
$$

The first term is the wall-clock instant by which the consumer will have finished units $c_0, \ldots, c_{i-1}$ (assuming continuous consumption). The second term is when $c_i$ actually became available. $\mathrm{slack}_i \ge 0$ means the consumer never stalled at unit $i$; $\mathrm{slack}_i < 0$ is a *visible* miss. **CP-SLO** is the requirement $\mathrm{slack}_i \ge 0$ for every unit of every served request.

<!-- role: advantage / scope -->
The definition is a function only of times the runtime can observe ($t_0$, $t_{\text{end}}$) and a quantity the application can supply ($\mathrm{consume}$). It does not depend on token rate, sequence length, or queue depth, so it is well-defined under any scheduling policy and can be evaluated post-hoc for any baseline.

### 3.3 Per-Request State and Cold Start
<!-- role: motivation -->
Computing $\mathrm{slack}_i$ after the fact is trivial; using it to decide *the next step* requires an estimate of the next deadline and of the work remaining to meet it.

<!-- role: design -->
Each request carries a `RequestSLOState` with three components, all online:

1. **Boundary detector.** A pluggable `ChunkBoundaryDetector` scans the streamed text and flushes a unit at sentence-ending punctuation followed by whitespace, at paragraph breaks (`\n\n`), or at a user-supplied predicate. A `min_chunk_tokens` guard merges micro-chunks (e.g., abbreviations) to avoid spurious flushes.
2. **Consume estimator.** A pluggable `ConsumeEstimator` maps unit text to seconds. The default `WordRateEstimator` uses `seconds_per_word`. For TTS, this is replaced by an audio-duration estimator obtained from a per-model calibration pass (one-time, offline).
3. **Generation-time estimator.** Either an EMA estimator with smoothing $\alpha$, or a windowed-percentile estimator (default p99 over a configurable window). Both expose $\widehat{g}, \widehat{\tau}, \widehat{w}$ (§3.1). Until the number of observed samples reaches $\mathit{warmup}$, the estimator is considered *cold*; the cold-start gate in §3.4 prevents any cold request from being demoted to pending.

We then define the realtime slack at wall-clock time $t$:

$$
\mathrm{slack}(t) \;=\; t_0(r) \;+\; \sum_{j \le k} \mathrm{consume}(c_j) \;-\; t,
$$

where $c_k$ is the most recently flushed unit. The scheduler uses $\mathrm{slack}_i$ (the chunk-aligned cumulative score) as its priority key and $\mathrm{slack}(t)$ as the input to the $\textsc{ShouldEnter}$ / $\textsc{ShouldExit}$ predicates in §3.4.

<!-- role: advantage -->
All three components are per-request and online; no offline per-model speed calibration is required for the scheduling path. The cold-start guard guarantees that no request is demoted before its own generation-time distribution is observed.

### 3.4 Backpressure Scheduling
<!-- role: motivation -->
Continuous-batching steps must commit a single fixed-shape batch, so the runtime must reduce per-request pressure to a binary `run` / `pause` decision. Three failure modes must be ruled out by construction: (F1) oscillation (the same request flips state every step), (F2) starvation (a low-priority request is never run), (F3) capacity violation ($|R| > \mathit{maxSeqs}$).

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
sort $C$ by ascending $\mathrm{slack}_i$ \tcp*{most urgent first}
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

with the per-request predicates

```latex
\begin{align*}
\textsc{ShouldEnter}(r,t,n_p) &\;:=\; \mathrm{slack}(t) \;>\; (\kappa_{\text{enter}} + \lambda_p\, n_p)\,\widehat{g}(r),\\
\textsc{ShouldExit}(r,t,n_p)  &\;:=\; \mathrm{slack}(t) \;\le\; (\kappa_{\text{enter}} + \lambda_p\, n_p - \Delta)\,\widehat{g}(r) \\
                              &\quad\;\lor\; \widehat{f}(r) \;\ge\; \mathrm{slack}(t) + \widehat{\tau}(r),
\end{align*}
```

where $\widehat{g}(r)$ is the estimated per-unit generation time, $\widehat{\tau}(r)$ the per-token time, and $\widehat{f}(r) = (\widehat{w}(r) - w_{\text{cur}}(r)) \cdot \widehat{\tau}(r)$ the predicted finish time of the in-progress unit ($\widehat{w}$: estimated unit word count; $w_{\text{cur}}$: words already emitted in the in-progress unit).

<!-- role: invariants and complexity -->
**Invariants.** F1 (oscillation) is suppressed by the hysteresis gap $\Delta > 0$ between $\textsc{ShouldEnter}$ and $\textsc{ShouldExit}$ and by the cold-start gate ($\widehat{g}$ requires $\geq \mathit{warmup}$ samples before pending is eligible). F2 (starvation) is enforced by the $\mathit{maxConsec}$ guard, which is the *first* check in the per-request loop and therefore dominates the slack-based decision. F3 (capacity) is enforced unconditionally by the post-loop overflow step; this means F3 can override F2 (the engine cap takes priority over starvation prevention), a deliberate choice because violating $\mathit{maxSeqs}$ corrupts the decode batch shape.

**Complexity.** Per step, $O(N \log N)$ for the sort plus $O(N)$ for the loop, where $N = |R \cup P|$. State per request is $O(1)$ for the EMA estimator and $O(W_{p99})$ for the windowed-percentile estimator (with window $W_{p99} = $ `chunk_gen_p99_window`, default 100). Memory overhead is dominated by the boundary detector's pending-text buffer, which is bounded by the longest in-progress progress unit.

### 3.5 Single-Signal Integration
<!-- role: motivation -->
A pressure signal is most useful when every contention-relevant policy in the engine consumes it. Three further policies share the slack score from §3.4 with no additional state.

<!-- role: design — three policies, one signal; line refs are descriptive, not numeric -->
- **Adaptive batch sizing.** When any post-warmup running request has $\mathrm{slack}_i < 0$, the step's effective $\mathit{maxSeqs}$ is halved (the cap-halve step in Algorithm 1, just before the F3 overflow check). Rationale: at least one consumer is already starving; shrinking the decode batch reduces token-level interference at the cost of throughput, and the system pays this cost only while at least one CP-SLO is violated.
- **Waiting-queue admission gate.** Admission from $W$ stops if (i) $|R| \ge \mathit{cap}$ (the possibly-halved cap), or (ii) any $r \in P$ now satisfies $\textsc{ShouldExit}(r, t, |P|)$ — i.e., a previously demoted request is about to need its slot back. This is a "do no harm" rule: do not promise a new request if an existing one is about to require capacity.
- **KV-offload victim selection.** When `kv_cache_manager.allocate_slots(...)` fails, the runtime selects $\arg\max_{r \in R \cup P} \mathrm{slack}_r$ as the offload victim. The KV connector saves its cache to CPU and restores it on resume. The intuition is symmetric to §3.4: slack-rich requests have the most time before their next consumable deadline and tolerate offload latency best.

<!-- role: advantage and honest caveat -->
**Advantage.** Three contention-relevant decisions consume one signal, one warmup, one hysteresis, and one pressure constant. There is no separate priority knob to tune across policies, and a misconfigured component cannot silently disagree with the others.

**Caveat.** The single-signal design is a strength only when the slack score is well-calibrated. A poorly tuned `seconds_per_word` or a model whose `gen_time` distribution shifts mid-trace can hurt all three policies simultaneously. We treat sensitivity to these inputs as a first-class evaluation target (§4.6).

### 3.6 Implementation and Hyperparameters
<!-- role: reproducibility table; one default value + range we will sweep + role -->

ProgressServe is implemented inside vLLM (V1 scheduler). The added per-request state lives under `vllm.sslo`; the four policies in §3.4–3.5 are gated by a single `SsloConfig` flag (`enabled=True`) so that disabling them recovers the vanilla scheduler exactly. All three estimators (boundary detector, consume estimator, generation-time estimator) are pluggable.

| Group | Symbol / flag | Default | Range tested (§4.6) | Role |
|---|---|---|---|---|
| Estimators | `chunk_unit` | `sentence` | $\{$ sentence, paragraph $\}$ | Boundary granularity. |
|  | `chunk_gen_estimator` | `ema` | $\{$ ema, p99 $\}$ | Generation-time estimator. |
|  | $\alpha$ (`ema_alpha`) | 0.2 | fixed | EMA smoothing factor. |
|  | `chunk_gen_p99_window` | 100 | fixed | Sliding-window size for the p99 variant. |
|  | $\mathit{secPerWord}$ (`seconds_per_word`) | 0.28 | $\{0.20, 0.28, 0.36\}$ | Default reading rate (Phase 1, axis 6). |
|  | $\mathit{minChunk}$ (`min_chunk_tokens`) | 16 | $\{0, 16, 32\}$ | Micro-chunk merge guard (Phase 1, axis 5). |
| Scheduler | $\kappa_{\text{enter}}$ (`pending_enter_factor`) | 2.5 | $\{1.5, 2.5, 3.5\}$ | Pending-enter threshold multiplier (Phase 1, axis 1). |
|  | $\Delta$ (`pending_hysteresis_gap`) | 0.5 | $\{0.25, 0.5, 1.0\}$ | Hysteresis gap, prevents F1 (Phase 1, axis 2). |
|  | $\lambda_p$ (`pending_pressure_lambda`) | 0.05 | $\{0, 0.05, 0.10\}$ | Pressure pre-multiplier on $\lvert P\rvert$. Subscript $p$ distinguishes from arrival rate $\lambda$ (Phase 1, axis 3). |
|  | $\mathit{warmup}$ (`pending_warmup_chunks`) | 5 | $\{3, 5, 10\}$ | Cold-start guard before pending-eligible (Phase 1, axis 4). |
|  | $\mathit{maxConsec}$ (`max_consecutive_pending`) | 5 | fixed | Anti-starvation cap (F2). |
|  | $\mathit{adaptive}$ (`adaptive_batch_size`) | `True` | $\{$on, off$\}$ ablation A5 | Halve cap on overdue (§3.5). |
|  | $\mathit{offload}$ (`offloading`) | `True` | $\{$on, off$\}$ ablation A7 | Slack-based KV offload (§3.5). |
| Engine | `max_num_seqs` | model-specific | inherited | Hard `InputBatch` capacity (E1). |
|  | `max_num_batched_tokens` | model-specific | inherited | Per-step token budget. |

Defaults match the values committed to `vllm.sslo.config.SsloConfig`. Phase-1 sensitivity (§4.6) varies one row at a time around the default operating point; Phase 2 evaluates pre-registered adversarial corners; Phase 3 transfers the surviving configs to a second cell. Anything marked "fixed" is held constant across the entire experimental plan to keep the sweep tractable; we revisit any fixed knob only if a paper-level claim depends on it.

---

## §4 Experimental Plan

This section is a *plan*, not a results report. Every quantitative claim is marked `[TBM]` (to be measured) and is paired with a falsifiable hypothesis. Numbers will replace `[TBM]` after the planned sweep.

### 4.1 Hypotheses

We define the **contention knee** as the arrival rate $\lambda_{\text{knee}}^\pi$ above which CP-SLO miss rate of policy $\pi$ first exceeds a fixed small threshold (we use 1%) on a reference workload; this gives every policy its own knee. Hypotheses are stated relative to this knee.

**H1 (effectiveness).** Under sustained decode contention ($\lambda > \lambda_{\text{knee}}^{\text{FCFS}}$), ProgressServe reduces CP-SLO miss rate vs. FCFS, request-priority, and TBT-aware schedulers, at equal or higher goodput.

**H2 (no-harm under low load).** For $\lambda < \lambda_{\text{knee}}^{\text{FCFS}}$, ProgressServe is statistically indistinguishable from FCFS on every primary and secondary metric.

**H3 (signal sufficiency).** Removing any one of the four single-signal consumers (priority, adaptive batch, admission gate, KV-offload-by-slack) measurably degrades CP-SLO miss rate or tail stall, while keeping the others.

**H4 (no starvation).** No request is held pending for more than $\mathit{maxConsec}$ consecutive scheduling steps in any sweep.

**H5 (robustness to estimator noise).** A $\pm 30\%$ perturbation of $\mathit{secPerWord}$ or a switch from EMA to p99 estimator changes CP-SLO miss rate by less than the gap to the strongest baseline.

Each hypothesis has a planned test below; H1 and H3 carry the paper's central claims and dictate the must-have experiments.

### 4.2 Workloads
**Datasets.** ShareGPT-derived prompts (single-turn) and a long-form completion mix (e.g., WritingPrompts) for paragraph-granularity stress.
**Granularities.** Sentence, paragraph, TTS phrase. Each granularity uses the matching `ChunkBoundaryDetector` and `ConsumeEstimator`.
**Arrival processes.** Three regimes:
- *Steady.* Poisson(λ) at multiple λ to span underloaded → knee → overloaded.
- *Bursty.* On/off arrivals (e.g., 5×λ for 30s every 90s) to test admission gate and adaptive batch under transients.
- *Mixed.* Heterogeneous request lengths (short Q&A + long-form) to test slack-asymmetry across coexisting requests.

For each (dataset, granularity, arrival) tuple we sweep λ over five points covering 60–140% of the FCFS knee. Each cell runs ≥ 3 seeds.

### 4.3 Baselines
1. **FCFS-CB.** Vanilla vLLM continuous batching.
2. **Priority-CB.** vLLM with static request priority by request length (proxy for naive QoS).
3. **TBT-aware (Sarathi-Serve-style).** Detailed reproduction below.
4. **Oracle-slack.** ProgressServe with `gen_time` and `consume` replaced by an offline-computed ground truth from a separate run. Bounds the gap attributable to estimator noise; not a deployable baseline but a calibration ceiling for ProgressServe.
5. **ProgressServe.**

All baselines run on the same engine, same model, same hardware, and the same `max_num_seqs`. The SSLO state is *computed* for every policy (so CP-SLO miss rate is well-defined for FCFS, Priority, and TBT-aware too) but only *acted on* by ProgressServe and Oracle-slack.

**TBT-aware baseline reproduction.** Sarathi-Serve targets per-step token-level latency by (i) splitting prefill into fixed-size chunks of $C$ tokens that are interleaved with decode tokens, and (ii) capping the per-step compute by a token budget $B$ chosen so that the worst-case TBT stays under a target $T^\star$. We reproduce this on the same engine without re-implementing it, since vLLM upstream already supports chunked prefill: we set `enable_chunked_prefill=True`, fix the prefill chunk size $C$ from a low-load probe trace, and set `max_num_batched_tokens` $= B$ so that the TBT p99 on a single-request-at-a-time probe equals our target $T^\star$ (we use $T^\star = 50$ ms for the 35 B-class model; one calibration run per (model, hardware) pair). The base scheduler stays FCFS. We then apply the TBT-aware engine to the same workloads and arrival traces as the other baselines. This is the strongest prior art that targets *token-level* latency: it improves TBT p99 by construction, which is exactly what we want to *not* be ahead of along that axis (H2 no-harm). Any gain ProgressServe shows on CP-SLO miss rate against this baseline therefore cannot be explained by tighter token timing alone.

### 4.4 Metrics

Let $\mathcal{R}$ be the set of completed requests and $\mathcal{C}_r = \{c_0, c_1, \ldots\}$ the progress units of $r \in \mathcal{R}$. Notation in this subsection follows §3.1.

**Primary.**

- **CP-SLO miss rate** $\downarrow$: fraction of progress units that miss the CP-SLO,

$$
\mathrm{Miss} \;=\; \frac{\bigl|\{(r, c_i) : r \in \mathcal{R},\, c_i \in \mathcal{C}_r,\, \mathrm{slack}_i < 0\}\bigr|}{\sum_{r \in \mathcal{R}} |\mathcal{C}_r|}.
$$

- **Tail consumable-stall** $\downarrow$: tail of the worst per-request stall duration,

$$
\mathrm{Stall}_q \;=\; q\text{-th percentile of}\; \bigl\{\, \max\bigl(0,\; -\min_{c_i \in \mathcal{C}_r} \mathrm{slack}_i\bigr) \;:\; r \in \mathcal{R} \,\bigr\},\qquad q \in \{95, 99\}.
$$

- **Goodput** $\uparrow$: completion rate restricted to requests with zero CP-SLO misses,

$$
\mathrm{Goodput} \;=\; \frac{\bigl|\{r \in \mathcal{R} : \min_{c_i} \mathrm{slack}_i \ge 0\}\bigr|}{\text{total wall-clock duration of the trace}}.
$$

**Secondary.**

- Aggregate throughput (tokens / s, requests / s).
- Token-level latency: TTFT, TBT at p50 / p95 / p99.
- *Pending health*: pending-interval count per request, max consecutive pending steps (compared against $\mathit{maxConsec}$ for H4), total pending time.
- *Offload health* (when $\mathit{offload} = \text{on}$): offload count, restore latency, fraction of offloaded requests that miss CP-SLO.

The secondary token-level metrics are reported as a no-harm check (H2) rather than as targets; ProgressServe is not designed to improve TBT against TBT-aware schedulers.

### 4.5 Ablations (matrix)
For the (sentence, ShareGPT, steady) cell at the contention knee:

| Variant | Knob change vs. §3.6 default | Tests |
|---|---|---|
| A0 — Full | — | reference |
| A1 — No priority | slack score not used as sort key | priority component of H3 |
| A2 — No backpressure | pending state disabled (force $P' = \emptyset$) | scheduling component of H3 |
| A3 — No hysteresis | $\Delta = 0$ | F1 (oscillation) |
| A4 — No pressure | $\lambda_p = 0$ | cascading-pending under bursty |
| A5 — Fixed batch | $\mathit{adaptive} = \text{off}$ | adaptive-batch component of H3 |
| A6 — Greedy admission | admission gate disabled | admission component of H3 |
| A7 — Default offload | $\mathit{offload} = \text{off}$, fallback to FCFS-tail | KV-residency component of H3 |
| A8 — EMA → p99 | `chunk_gen_estimator: ema → p99` | estimator robustness (subset of H5) |

Each variant is run for ≥ 3 seeds at the same λ. A failure of any of A1–A2, A5–A7 to degrade CP-SLO metrics would *falsify* H3 for that component.

### 4.6 Sensitivity
A full factorial sweep of six hyperparameters is infeasible ($3^6 = 729$ configs $\times$ 3 seeds = 2187 runs). We use a three-phase plan that bounds cost while still falsifying H5.

**Default operating point.** $\kappa_{\text{enter}} = 2.5$, $\Delta = 0.5$, $\lambda_p = 0.05$, $\mathit{warmup} = 5$, $\mathit{minChunk} = 16$, $\mathit{secPerWord} = 0.28$.

**Phase 1 — One-at-a-time around the default** (cell: sentence × ShareGPT × steady × $\lambda_{\text{knee}}^{\text{FCFS}}$).
Vary one axis at a time, three levels per axis, holding others at the default. Six axes $\times$ 2 non-default levels $= 12$ configs (the default is shared). Each $\times$ 3 seeds $= 36$ runs.

| Axis | Levels |
|---|---|
| $\kappa_{\text{enter}}$ | 1.5, **2.5**, 3.5 |
| $\Delta$ | 0.25, **0.5**, 1.0 |
| $\lambda_p$ | 0, **0.05**, 0.10 |
| $\mathit{warmup}$ | 3, **5**, 10 |
| $\mathit{minChunk}$ | 0, **16**, 32 |
| $\mathit{secPerWord}$ | 0.20, **0.28**, 0.36 |

Phase 1 tests H5: each axis must show CP-SLO miss rate within the gap to the strongest baseline. A failing axis is escalated to Phase 2.

**Phase 2 — Adversarial corners** (same cell). For axes that *interact* (estimator $\times$ pressure $\times$ hysteresis), evaluate four pre-registered corner configs designed to be unfavorable: $(\Delta = 0.25, \lambda_p = 0)$, $(\Delta = 0.25, \kappa_{\text{enter}} = 1.5)$, $(\mathit{warmup} = 3, \mathit{secPerWord} = 0.20)$, $(\mathit{warmup} = 3, \mathit{secPerWord} = 0.36)$. $4 \times 3$ seeds $= 12$ runs.

**Phase 3 — Cross-cell transfer.** Take three configs from Phases 1–2 — the **default**, the **best** Phase-1 config, the **worst** Phase-1 (or Phase-2) config — and run on a second cell: paragraph $\times$ WritingPrompts $\times$ bursty $\times$ $\lambda_{\text{knee}}^{\text{FCFS}}$. $3 \times 3$ seeds $= 9$ runs. The transfer test passes iff the *ordering* of configs by CP-SLO miss rate is preserved (no inversion); otherwise the "single config" claim is conditional on workload class and we report it as such.

**Total budget.** $36 + 12 + 9 = 57$ sensitivity runs. The "no per-case retuning" claim stands only if (a) Phase 1 worst beats the strongest baseline on the primary cell, *and* (b) Phase 3 shows no ordering inversion.

### 4.7 Adversarial / Failure-Mode Cases
- *Drift.* Inject a step-change in token rate (e.g., model switch at t = 60s) and report time-to-recover for the EMA vs. p99 estimator.
- *Wrong consume estimate.* Run with `seconds_per_word` set to half / double the true value; ProgressServe is expected to *over-* or *under-*demote. Report whether the starvation cap (H4) still holds and whether goodput collapses.
- *Heavy KV pressure.* Workload chosen so KV preemption dominates; tests whether KV-offload-by-slack outperforms FCFS-tail or whether it hurts (offloaded request returns to a still-saturated GPU).
- *No-slack workload.* All requests with very short outputs (one sentence). ProgressServe should reduce to FCFS-CB to within the no-harm tolerance (H2).

### 4.8 Reporting

**Tables.** `booktabs` style: caption above, `↑/↓` arrows in headers, best/second-best highlighted, no vertical rules. One table = one message.

**Figure 1 — Teaser (single-trace comparison).** A two-row, two-column figure, both rows sharing the same wall-clock $x$-axis over a single contended trace.

- **Row layout.** Each row corresponds to one request, $r_A$ and $r_B$. Both requests start at $t = 0$; $r_A$ has a long opening sentence (large initial consumable buffer), $r_B$ has a short opening sentence (small initial buffer).
- **Per-request tracks.** Within each row, three stacked tracks:
  - Track 1 — *token emissions* (vertical ticks, one per token).
  - Track 2 — *progress-unit boundaries* (downward triangles where a sentence flushes). The horizontal segment between two triangles is shaded to represent the consumable buffer the consumer *holds at that time*; segments dim toward the right as the consumer drains them at rate $1/\mathit{secPerWord}$.
  - Track 3 — *consumer playback* (upward triangles where the consumer would finish the unit at its consume rate). A consumable-stall is a region where Track 3's triangle for unit $i+1$ would arrive before Track 2's triangle for unit $i+1$; we shade those regions red.
- **Column (a): FCFS-CB.** Both rows compete equally; $r_B$'s second sentence misses its consumable deadline (a red region on row B between its first and second progress-unit triangles).
- **Column (b): ProgressServe.** Same arrival, but at the moment $r_A$ accumulates positive slack, the scheduler demotes $r_A$ to *pending* (annotated with a "P" marker on row A's token track, indicating tokens are not emitted in that interval). $r_B$'s second sentence now lands before its consumer drains; no red region remains. Row A's later token emissions show a small delay relative to (a) but *no* visible stall, because $r_A$'s buffer was still ahead.
- **Annotations.** A single arrow in (a) labeled "consumable stall" pointing at the red region; a single arrow in (b) labeled "slack reallocated" between the two rows.
- **Why this teaser works.** It shows in one trace that (i) token-level rate is *not regressed* for $r_A$ in any visible way, (ii) the metric that flips between (a) and (b) is exclusively consumable-stall, and (iii) the mechanism is reallocation, not throughput improvement.

**Figure 2 — Miss-rate vs. arrival rate.** One panel per (granularity, dataset) pair. $x$-axis: arrival rate $\lambda$, normalized to the FCFS knee ($\lambda / \lambda_{\text{knee}}^{\text{FCFS}}$, dimensionless). $y$-axis: CP-SLO miss rate (linear, 0–1). One curve per policy (FCFS, Priority, TBT-aware, ProgressServe, Oracle-slack). Shaded bands: $\pm$ one standard error across seeds. Dashed vertical at $x = 1$ marks the knee. The empirical claim H1 reduces to: ProgressServe's curve is below FCFS, Priority, and TBT-aware everywhere right of the knee, and within the no-harm tolerance left of it.

**Figure 3 — ProgressServe pipeline diagram.** A horizontal three-stage data-flow diagram showing how observable streaming state becomes a scheduling decision.

- **Stage A (left) — Per-request observables.** A box labeled `RequestSLOState` containing three sub-blocks stacked vertically:
  - `ChunkBoundaryDetector` — input arrow labeled "streamed text $\Delta$"; output arrows labeled "unit text $c_i$" and "boundary timestamp $t_{\text{end}}(c_i)$".
  - `ConsumeEstimator` (e.g., `WordRateEstimator`) — input arrow labeled "unit text $c_i$"; output arrow labeled "$\mathrm{consume}(c_i)$".
  - `ChunkGenerationEstimator` (`EMA` or `p99`) — input arrow labeled "$(t_{\text{end}}(c_i) - t_{\text{end}}(c_{i-1})$, word_count$(c_i))$"; output arrows labeled "$\widehat{g}(r), \widehat{\tau}(r), \widehat{w}(r)$".
- **Stage B (center) — Slack fusion.** A single small box `slack(r, t)` taking $\mathrm{consume}(\cdot)$, $t_0(r)$, the latest $t_{\text{end}}$, and $\widehat{g}, \widehat{\tau}, \widehat{w}$ as inputs and producing two outputs: $\mathrm{slack}_i$ (priority key) and $\mathrm{slack}(t)$ (predicate input). Above the box: the formula $\mathrm{slack}_i = t_0 + \sum_{j<i} \mathrm{consume}(c_j) - t_{\text{end}}(c_i)$.
- **Stage C (right) — Policies.** Five policy boxes, fanned out from the slack outputs:
  - `Priority` — annotated as "ascending-slack sort key (Algorithm 1, sort step)".
  - `Backpressure (run/pause)` — annotated as "$\textsc{ShouldEnter}/\textsc{ShouldExit}$ predicates, with F2 / cold-start / contention guards".
  - `Adaptive batch size` — annotated as "halve $\mathit{cap}$ when any $\mathrm{slack}_r < 0$ in $R'$".
  - `Admission gate` — annotated as "stop admitting from $W$ if any $r \in P$ wants exit".
  - `KV offload victim` — annotated as "$\arg\max_{r \in R \cup P} \mathrm{slack}_r$ on `allocate_slots` failure".

  Layout note: if five boxes do not fit horizontally, fold `Priority` into `Backpressure` (they share the same input ordering).
- **Annotations.** Engine invariants (E1, E2) as small badges on the right edge. A dashed feedback loop: from "engine step output" back into Stage A, labeled "next streamed $\Delta$". Color: data-flow arrows in one color, control/decision arrows in another.
- **Why this diagram works.** It shows in one figure that (i) all three estimators are *per-request and online*, no offline calibration; (ii) only one scalar leaves Stage B, so Stage C cannot "disagree" on priority; (iii) the engine invariants are external constraints, not internals of the policy.

**Figure 4 — Phase-1 sensitivity bars.** Six panels (one per axis from §4.6), bar chart of CP-SLO miss rate at the three sweep levels for that axis, with all other knobs fixed at the default. A horizontal reference line: the *strongest baseline*'s miss rate on the same cell. The H5 claim is visible as: every bar in every panel sits below this line.

**Figure 5 — Phase-3 transfer scatter.** $x$-axis: CP-SLO miss rate on cell 1 (sentence $\times$ ShareGPT $\times$ steady). $y$-axis: same metric on cell 2 (paragraph $\times$ WritingPrompts $\times$ bursty). Three points (default / best / worst Phase-1 configs) plus the strongest-baseline point on each axis. Diagonal $y=x$ as reference. Ordering preservation is visible iff the three points stay on the same side of each other across the diagonal.

**Reproducibility.** All scripts and configurations to reproduce the sweep live in `exp/run_sslo/`. Each cell logs its (config, seed, commit) tuple in the result file header.

---

## §5 Discussion

### 5.1 Anticipated Reviewer Concerns
**"The single-signal design is brittle."** Acknowledged. We schedule §4.6–4.7 specifically to bound the brittleness; the H5 falsification criterion is explicit. If H5 fails, the paper's claim weakens to "single-signal design works when the slack estimate is within X% of true."

**"Slack-richest as KV-offload victim is counter-intuitive: that request often has the most context."** The trade-off is offload latency vs. visible stall, and we are choosing visible stall. The stress test in §4.7 (heavy KV pressure) is designed exactly to expose cases where this choice loses; we will report those cases honestly.

**"How does this compose with chunked prefill?"** Sarathi-Serve-style chunked prefill is taken as our strongest baseline (§4.3) and is run *on the same engine* as ProgressServe; the two mechanisms operate at different layers (token-budget shaping vs. request-level demotion) and we expect them to compose, but a deeper integration study is out of scope for this paper.

**"Why not RL-based scheduling?"** A learned policy could in principle dominate this rule, but it would also require training data and add a deployment dependency. The point of this paper is to show that a small, interpretable rule with explicit invariants already moves the metric of interest; a learned policy is a follow-up question.

### 5.2 Threats to Validity
- *Construct validity.* `consume(c_i)` is application-supplied; if the application's estimate diverges from real consumer behavior, CP-SLO no longer reflects user experience. We do not measure real users in this paper.
- *Internal validity.* All baselines run inside the same engine to isolate the policy; this rules out runtime-overhead confounds but does not rule out implementation bias toward our policy.
- *External validity.* Sweeps are on a single GPU configuration and a small set of models (§4.2); cross-hardware behavior is future work.

---

## §6 Conclusion
<!-- conclusion: scope statement, no oversell -->

<!-- role: restate -->
We argue that interactive LLM serving SLOs should be anchored to the unit at which the consumer actually perceives progress, and we present a concrete realization: the Consumable-Progress SLO and ProgressServe, a continuous-batching runtime that converts CP-SLO pressure into a single per-request slack signal driving decode priority, adaptive batch sizing, waiting-queue admission, and KV residency through one hysteresis-controlled rule.
<!-- role: planned evidence -->
The accompanying experimental plan (§4) is designed to either confirm or falsify the four central hypotheses; we will report results, including negative ones, alongside the released runtime.
<!-- role: scope limitation -->
The current formulation requires the application to define progress-unit boundaries and a consume-time function. Workloads with unstructured streaming output, or with consumer behavior not expressible as a function of text, are out of scope.
<!-- role: future work -->
Two extensions are natural: a deeper integration with chunked-prefill scheduling (treating the prefill token budget itself as a function of running-set slack), and replacing the static `consume` function with a consumer model that adapts to gaze, comprehension, or downstream synthesis backpressure.

---

## Appendix A — Self-Review Checklist
<!-- skill paper-review.md, five dimensions; honest about gaps -->

| # | Dim. | Question | Status |
|---|---|---|---|
| 1 | Contribution | Is the technical idea non-obvious beyond well-explored practice? | **pass** — CP-SLO + single-signal integration is, to our knowledge, new. |
| 1 | Contribution | At least one explicit novelty type? | **pass** — new SLO, new scheduling rule, new system integration. |
| 2 | Writing | Method reproducible? | **pass** — Algorithm 1 + notation table + invariants + complexity. |
| 2 | Writing | Terms stable? | **pass** — `progress unit`, `consumable buffer`, `slack`, `gen_time` defined once. |
| 3 | Empirical | Strong baselines included? | **needs new experiment** — TBT-aware reproduction (Sarathi-Serve-style) and Oracle-slack ceiling are *planned*, not yet run. |
| 3 | Empirical | Multi-dataset / multi-granularity? | **needs new experiment** — ≥ 2 datasets × 3 granularities × 3 arrivals planned. |
| 4 | Eval. completeness | Ablation per design choice? | **planned** — see §4.5 matrix A0–A8. |
| 4 | Eval. completeness | Sensitivity sweep? | **planned** — §4.6. |
| 5 | Soundness | Hidden technical defects? | **pass** — F1 (oscillation), F2 (starvation), F3 (capacity) handled by construction; trade-offs documented. |
| 5 | Soundness | Per-case retuning required? | **needs new experiment** — claim is conditional on §4.6 worst case beating strongest baseline. |

---

## Appendix B — Claim–Evidence Map

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Token-/request-centric metrics are blind to consumable-buffer state. | §1 prior-work analysis; §3.2 definition shows independence; §4.1 H1 quantifies. | needs evidence (Fig. 2, Table 1) |
| C2 | A single per-request slack score can drive four contention-relevant policies. | §3.4–3.5 construction; §4.5 ablations A1, A2, A5, A6, A7. | needs evidence |
| C3 | The rule guarantees $|R| \le \mathit{maxSeqs}$, anti-starvation, and no demotion under no-contention. | §3.4 invariants by construction; §4.1 H4 falsifiable test. | partial — invariants are by construction; H4 is the *empirical* check that the implementation matches the spec. |
| C4 | ProgressServe reduces CP-SLO miss rate vs. FCFS / Priority / TBT-aware. | §4.1 H1 → §4.4 primary metrics. | needs evidence |
| C5 | Throughput / TBT are not regressed under low load. | §4.1 H2 → §4.4 secondary metrics. | needs evidence |
| C6 | The rule is robust to $\pm 30\%$ misestimation of $\mathit{secPerWord}$ and to estimator choice. | §4.1 H5 → §4.6 sensitivity, §4.7 wrong-consume case. | needs evidence |
| C7 | KV-offload-by-slack improves under heavy KV pressure. | §4.7 stress test. | needs evidence; honest negative result acceptable. |

---

## Appendix C — Notes on Terminology
- `consumable` is used throughout in place of the earlier `readable`. The acronym is **CP-SLO** (not `RP-SLO`).
- `progress unit`: application-defined unit (sentence / paragraph / TTS phrase).
- `slack` without qualifier denotes $\mathrm{slack}_i$ (cumulative, per progress unit). $\mathrm{slack}(t)$ is always written explicitly.
- `pending` (engine state: holds KV, not decoded) is distinct from `waiting` (not yet admitted).
- `[TBM]` marks every quantitative claim that depends on running the §4 plan.

---

## Appendix D — Citation Keys (Placeholders)

`\cite{...}` markers used in this draft and what they refer to. Replace with the correct BibTeX entries when porting to LaTeX; keys are chosen to be stable and human-readable rather than venue-canonical.

| Key | Refers to | Used in |
|---|---|---|
| `kwon2023vllm` | vLLM / PagedAttention (Kwon et al., SOSP 2023). | §1, §2.1 |
| `yu2022orca` | Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022). | §1, §2.1 |
| `tgi` | Hugging Face Text Generation Inference (TGI) — official GitHub. | §1, §2.1 |
| `lightllm` | LightLLM — official GitHub (ModelTC). | §2.1 |
| `agrawal2024sarathi` | Sarathi-Serve / chunked prefill (Agrawal et al., OSDI 2024). | §2.2, §4.3 |
| `wu2024fastserve` | FastServe — preemptive scheduling for LLM inference (Wu et al.). | §2.2 |
| `streamtts` | Streaming TTS evaluation work — placeholder for the canonical reference at the time of writing. | §2.3 |
| `captionlatency` | Streaming-caption / live-captioning latency study — placeholder. | §2.3 |
| `liu1973scheduling` | Liu & Layland, *Scheduling algorithms for multiprogramming in a hard-real-time environment* (JACM 1973). | §2.4 |
| `stankovic1995deadline` | Stankovic et al., *Deadline scheduling for real-time systems: EDF and related algorithms*. | §2.4 |

Two placeholders (`streamtts`, `captionlatency`) point to areas where the canonical citation should be confirmed before submission; this draft does not commit to a specific paper for either.
