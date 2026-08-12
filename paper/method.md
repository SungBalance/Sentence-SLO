# Method Sections (§2–§4) — draft

> Scope: paper Method sections only (no experimental history). Terminology follows the
> unified state/transition scheme: `waiting / running / pending / offloaded / onloading`
> (states), `admit / defer / promote / offload / onload` (transitions).

---

## Section 2. Motivation

**LLM output is consumed, not delivered.** In interactive serving, the model's output
stream is not the end product: a human reads it, or a TTS engine converts it into audio
that plays in real time. Consumption is *incremental* and *paced* — a reader advances
sentence by sentence at reading speed; a TTS pipeline synthesizes and plays one utterance
while the next is being generated. The user experience is therefore governed not by when
individual tokens arrive, but by whether each *consumable unit* of output is ready by the
time the consumer reaches it. When it is not, the consumer stalls mid-stream — a pause in
the middle of reading, or an audible gap in speech — which is qualitatively worse than an
equivalently longer initial wait.

**Existing SLO formulations do not capture this.** Token-level metrics (TTFT, TPOT/TBT)
constrain the token stream uniformly, but consumers do not consume tokens: a 40-token
sentence delivered as a burst after a 2-second gap reads identically to one delivered
token-by-token, while a uniform token cadence that happens to straddle a sentence boundary
can still starve the consumer. Request-level latency is coarser still — it says nothing
about pacing within the stream. Serving under token- or request-level SLOs therefore
either over-provisions (enforcing a uniform token cadence far stricter than consumption
requires) or under-protects (meeting average latency while stalling the consumer
mid-stream).

**The gap is an opportunity, not just a mismatch.** Generation is typically much faster
than consumption: a modern GPU decodes a sentence in a fraction of the time a human takes
to read it. Every in-flight request that is ahead of its consumer holds *slack* — time
during which its next unit is not yet needed. From the scheduler's perspective this slack
is a resource: tokens the GPU does not owe anyone yet. A scheduler aware of per-unit
deadlines can *spend* that slack — deferring requests that are ahead, admitting more
concurrent requests into the freed capacity — and thereby serve more users per GPU while
every consumer still receives each unit on time. Two obstacles stand between this idea
and a working system:

1. **Memory.** Admitting more requests consumes KV-cache memory; on long-context
   workloads the KV pool, not compute, becomes the binding constraint on admission.
   A deferred request that is merely *waiting ahead of its consumer* still occupies
   GPU memory that could admit a new user (§4.2).
2. **Compute interference.** Admission is not free at the step level: prefilling a new
   request's prompt inflates the iteration time for everyone. In our measurements,
   steps containing prefill work are only ~6% of steps but account for ~18% of
   wall-clock time, with p90 step latency 6× the decode-only baseline (463 ms vs 78 ms).
   These spikes are precisely what breaks unit pacing for already-running requests (§4.3).

The remainder of this paper formalizes per-unit deadlines (§3) and presents ProgressServe,
a scheduler that spends slack safely against an explicit violation budget (§4.1), together
with two resource-specific mechanisms — memory-side offloading (§4.2) and compute-side
adaptive token batching (§4.3) — that remove the two obstacles above.

---

## Section 3. Consumable-Unit SLO

**Consumable units.** We segment a response stream into *consumable units* (CUs): the
smallest spans a consumer ingests atomically. For reading we use sentences (paragraphs
are an alternative granularity); for TTS we use the utterance units the synthesizer
accepts. Segmentation runs online over the token stream with a streaming boundary
detector; sub-minimal fragments are merged so that pathological outputs (e.g., long
comma-separated lists) do not produce degenerate units.

**Consumption model and deadline recurrence.** Each consumer class defines a per-unit
consumption time. For reading, unit $i$ with $w_i$ words takes $c_i = w_i \cdot t_{word}$
(seconds-per-word). For TTS, $c_i$ is the *audio playback duration* of unit $i$, and unit
readiness additionally requires the synthesis (conversion) time, both taken from a
measured per-word-count profile of the target TTS engine. Consumption is sequential and
begins when the first unit is delivered ($t_{start}$). The deadline of unit $i$ is the
moment the consumer finishes everything before it:

$$d_i \;=\; t_{start} \;+\; \sum_{j < i} c_j .$$

A unit that completes generation at $g_i$ has slack $d_i - g_i$; a negative value means
the consumer stalled for $g_i - d_i$ seconds. Note the recurrence is *self-paced*: fast
generation builds slack, and one late unit shifts all subsequent deadlines (the consumer
resumes where it left off), so the SLO measures the stall actually experienced rather
than an accumulating fiction.

**CU-SLO.** A request satisfies its CU-SLO at tolerance $\tau$ if no unit stalls its
consumer by more than $\tau$: $\max_i (g_i - d_i) \le \tau$. We report the fraction of
requests violating this (per-request), plus stall-time distributions (per-unit). Requests
that have not yet delivered their first unit have *no* unit deadlines — the time to first
consumable unit (TTFC) is reported separately, as queueing policy, not pacing, governs it.

**What the scheduler can know online.** Enforcing $d_i$ ahead of time requires knowing
how many tokens remain in the unit under generation — which is unknown until the boundary
appears. We maintain an online *unit-length predictor*: a per-request estimator blended
with a shared global estimator (warm-started across requests), exposing an empirical
*tail posterior*

$$\Pr\big[\,L > c + N \,\big|\, L > c\,\big]$$

— the probability that a unit already $c$ tokens long needs more than $N$ further tokens.
Requests progress through a lifecycle (`PREFILL → WARMUP → MEASURED`): only MEASURED
requests (first units delivered, estimator warmed) carry deadlines and participate in the
risk calculations of §4. This tail posterior is the single statistical primitive on which
all scheduling decisions below are built.

---

## Section 4. ProgressServe: Scheduling for CU-SLO

ProgressServe replaces the engine's per-step scheduling decision. Its design premise:
*every* placement question — who decodes this step, who waits, who enters, whose memory
stays resident, how much prefill to take on — is the same question, "what does this do to
the probability that some unit misses its deadline?", and should be answered by the same
model. §4.1 develops that model and the core run/defer/admit decision. §4.2 and §4.3 then
extend it along the two resource axes identified in §2: KV memory and step-time compute.
Each extension is gated so that it acts only in the regime it targets and provably (or
measurably) leaves other regimes untouched.

**The four configurations we evaluate.** The two extensions are independent and
separately switchable, giving four schedulers. We evaluate all four against the vanilla
engine, because the interesting question is not "does the full system win" but *which
lever pays for itself, in which regime, and do they compose*.

| Configuration | Lever added | Binding resource it relieves | Acts when |
|---|---|---|---|
| **ProgressServe** (§4.1) | run/defer + $E_{viol}$-budgeted admission | none — it *harvests* slack and spends it on concurrency | always |
| **+ Offload** (§4.2) | park deferred requests' KV on CPU, restore before the deadline | **KV memory** — deferred requests keep their blocks | `kv-capped` steps only |
| **+ Adaptive** (§4.3) | per-step prefill token budget $TB^{*}$ (+ whole-request decode deferral) | **step-time compute** — prefill spikes freeze every consumer | prefill work coexists with at-risk in-flight requests |
| **+ Both** | both of the above | either | union of the two triggers |

Throughout, *ProgressServe* names the §4.1 core that all four share; *Offload* and
*Adaptive* name the increments. The four differ only in which increments are enabled —
the risk model, the deadline recurrence, and the admission budget are identical across
them, which is what makes the comparison an ablation rather than four separate systems.
The composed configuration is our full system; §4.4 reports when composing helps and
when it does not.

### 4.1 CU-SLO-aware Scheduling

**Why the baseline is not enough.** A vanilla continuous-batching scheduler admits
whenever capacity allows and decodes everyone every step. Under CU-SLO this throws
away exactly the resource that matters: it gives tokens to requests that are seconds
ahead of their consumer at the same priority as requests about to stall, and its only
admission throttle — total memory — is blind to pacing. The operator is left choosing a
static concurrency cap: small caps meet deadlines but idle the GPU and cap the number of
concurrent users; large caps raise occupancy but let stalls grow unchecked. ProgressServe
replaces this static trade-off with a per-step, per-request decision made against an
explicit violation budget.

**Risk primitives.** Consider a step with batch capacity $B$ and $|A^{+}|$ requests
sharing service (running, onloading, and new admits — and offloaded requests, see §4.2).
Each request's expected per-step token allocation is the *service share*
$s = \min(1, B/|A^{+}|)$. For a MEASURED request $q$ with time $T_q$ to its next unit
deadline and step time $\Delta$, the deadline horizon is $H_q = \lfloor T_q/\Delta \rfloor$
steps, from which we derive the tokens producible before the deadline if the request runs
this step ($N_{run}$) or is deferred one step ($N_{defer} < N_{run}$). The tail posterior
of §3 converts token budgets into risks:

$$R_{run} = \Pr[L > c_q + N_{run} \mid L > c_q], \qquad
  R_{defer} = \Pr[L > c_q + N_{defer} \mid L > c_q],$$

and the *marginal benefit* of a decode slot is $M = R_{defer} - R_{run} \ge 0$: how much
violation probability this one slot removes. Requests without deadlines (PREFILL/WARMUP)
are *forced* — they always receive service (they must produce their first units to become
measurable at all) and are excluded from the risk sums.

**Run/defer partition.** Each step, decode slots go to measurable requests in descending
$M$; the remainder are *deferred* to the `pending` state — admitted, KV resident, but not
decoding this step. Deferral is the mechanism by which slack is harvested: a request far
ahead of its consumer has $R_{defer} \approx R_{run} \approx 0$, so $M \approx 0$, and
yielding its slot costs (in expectation) nothing. A `pending` request is re-evaluated
every step and *promotes* back to `running` the moment its marginal benefit justifies a
slot — typically well before its deadline horizon closes.

**Admission under a violation budget.** The harvested capacity is spent on admission.
Let $E_{viol}(k)$ be the expected number of unit-deadline violations this step if the
$k$ oldest waiting requests are admitted:

$$E_{viol}(k) \;=\; \sum_{\text{scheduled}} R_{run} \;+\; \sum_{\text{deferred}} R_{defer}
\qquad\text{(sums over measurable requests, at share } s(k)\text{)} .$$

New admits carry no deadlines, so they enter $E_{viol}$ only *indirectly* — each admission
shrinks the service share and the decode budget, raising every incumbent's risk. Since
$E_{viol}(k)$ is monotone in $k$, ProgressServe admits the largest FCFS prefix
$k^{*} = \max\{k : E_{viol}(k) < 1\}$, i.e., it packs users up to the point where the
system expects (at most) one unit violation — an interpretable, workload-independent
budget. Admission is additionally capped by free KV blocks; when the KV cap, not the risk
budget, is what stops the scan, the step is flagged **kv-capped**. This flag is the
hand-off point to §4.2: it identifies, per step, that memory rather than pacing risk is
the binding constraint.

**What it improves.** In KV-slack regimes the mechanism converts slack into concurrency:
occupancy rises (more handling users per GPU) while stalls stay within the budget. Its
limits are equally instructive: it can only *reorder and admit* — when the constraint is
memory (deferred requests still hold KV) or single-step compute (prefill spikes), the
risk model correctly diagnoses the pressure but has no lever to relieve it. Those levers
are §4.2 and §4.3.

### 4.2 Offloading

**Why it is needed.** Deferral separates a request's *service* from its *memory*: a
pending request consumes no compute but still occupies its full KV footprint. On
long-context workloads this is the dominant admission constraint — the KV pool saturates
while the risk budget $E_{viol}$ still has room ($k^{*}$ wants to admit; free blocks do
not allow it). The slack harvested by §4.1 is then stranded: it cannot be spent on new
users because departed-but-parked state is holding the room. Offloading un-strands it by
moving the KV of the *safest* pending requests to host memory, on the observation that a
request that is many units ahead of its consumer does not need its KV on the GPU *right
now* — it needs it back *before its deadline*.

**Mechanism.** Offloading extends the state machine with `offloaded` (deferred, KV on
CPU) and the transient `onloading` (KV streaming back). Three rules govern it, all
expressed in the §4.1 risk model:

- *Eligibility (offload).* For a pending request, define $N_{defer\_cpu}$ as $N_{defer}$
  minus the tokens forgone during the *onload lead* $\ell$ (the iterations needed to
  stream KV back before it can decode), and
  $$M_{cpu} \;=\; R_{defer\_cpu} - R_{defer} \;\ge\; 0$$
  — the *risk premium of a CPU stay*. Only requests with $M_{cpu} \le \varepsilon$
  (default $10^{-3}$) may be offloaded: parking them is, to within $\varepsilon$,
  free in violation probability. A minimum-residency guard suppresses thrashing.
- *Trigger.* Offloading fires only on **kv-capped** steps (§4.1), and frees only as many
  blocks as admission actually needs. On every other step the mechanism is a no-op —
  in KV-slack regimes the system is bit-for-bit the scheduler of §4.1.
- *Return (onload).* An offloaded request begins onloading when its deadline horizon
  approaches the lead time ($d - t \le \ell\Delta + \hat{f}$), holds a slot while
  streaming (so its return is never crowded out by new admissions), and rejoins
  `running` with KV intact — the deadline-safety check precedes any new offload.

Two accounting rules keep the extension consistent with §4.1's model. First, offloaded
requests remain in the service-share denominator $|A^{+}|$: offloading frees *memory*,
not *compute* — the parked request will return and reclaim its share of tokens, and
pretending otherwise would systematically understate every risk in the system (and, in
particular, understate $M_{cpu}$ itself, causing spurious offloads). Second, offloaded
requests keep contributing $R_{defer\_cpu}$ to $E_{viol}$: risk parked on the CPU is
still risk.

**What it improves.** Offloading targets exactly the KV-bound regime: large concurrency
caps, long prompts, long generations. There it converts stranded slack into admissions.
In our evaluation setting (32B model, multi-turn prompts averaging ~1.3K tokens, 8K
generation cap) it cuts CU-SLO violation rates from ~6% (baseline) to 1.4–2.0%. Where KV
is not binding, the trigger never fires; the correct behavior there is to do nothing, and
the gating guarantees it.

> **Draft note (2026-08-12) — the throughput claim is currently unsupported.** Measured
> throughput for this configuration is regime-dependent and, at the smaller concurrency
> cap, clearly below the §4.1 core: at cap 64 / rate 1 it reaches 411 tok/s against a
> 519 baseline (79%), while at cap 128 it matches baseline (503 vs 500). We traced the
> deficit to an implementation defect rather than the mechanism: the admission KV model
> assumes a fixed 8 blocks per new admit, whereas the measured footprint on this workload
> is ~141 blocks (an 18$\times$ underestimate). The same constant sizes
> `blocks_needed` for the offload search, so the tier is also asked to free ~1/18 of what
> admission actually requires — over one full run it executed only 20 offloads. Until
> that constant is derived from the real prompt length, the numbers here characterise a
> largely inert tier, and the violation reductions above should be attributed mostly to
> the §4.1 core it sits on. Re-measurement is required before this section makes a
> throughput claim.

### 4.3 Adaptive Token Batching

**Why it is needed.** §4.1–4.2 manage *which requests* get service; neither controls
*what a step costs*. Iteration time is, however, not uniform: decode-only steps are
tightly concentrated (p50 74 ms, p90 78 ms in our measurements), while steps that carry
prompt-prefill work exhibit heavy tails (p90 463 ms) — a single large prefill chunk
freezes every in-flight request for the duration. For CU-SLO this is the worst possible
disturbance: it hits all consumers simultaneously, and it is invisible to per-request
scheduling — no run/defer/offload decision changes the cost of the step itself. Note
that shrinking the *request-level* batch (the classical adaptive-batching knob) does not
address this: the spike is caused by prefill *tokens*, not by the number of concurrent
requests, and reducing concurrency taxes throughput without flattening the tail. The
correct control variable is the *token composition of the step*.

**Mechanism.** ProgressServe bounds the prefill tokens per step with a *token budget*
$TB^{*}$ priced in the same currency as admission: expected violations. Model the step
time as $\Delta(P) \approx \Delta_{dec} + \kappa_p P$, where $P$ is the step's prefill
token count, $\Delta_{dec}$ is the decode-only baseline (read from a batch-matched
step-time table) and $\kappa_p$ (s/token) is estimated online from prefill-carrying steps
as a ratio of sums (EMA of excess step time over EMA of prefill tokens), so that the
estimator cannot be biased by its own clamping decisions.

The key modelling question is *what a prefill choice actually costs*. Charging
$\Delta(P)$ to every step until every deadline — the natural reading of a per-step step
time — is badly wrong: the budget is recomputed every step, prefill-carrying steps are a
minority (8.9% in our traces), and the one-step cost $\kappa_p P$ would be priced as a
permanent slowdown, over-charging by roughly $20\times$ at measured $\kappa_p$. We
instead price the *burst*. Let $W$ be the committed prefill work — the residual prompt
tokens of in-flight prefills plus those of the prefix of the queue this step admits.
Choosing $P$ makes the burst $W/P$ steps long at $\Delta_b(P) = \Delta_{dec} + \kappa_p P$
each, i.e. a wall-clock window
$\mathrm{wall}(P) = (W/P)\,\Delta_{dec} + \kappa_p W$ after which the step time returns to
$\Delta_{dec}$. A request's deadline horizon is then piecewise:

$$H_q \;=\;
\begin{cases}
\lfloor T_q / \Delta_b(P) \rfloor, & T_q \le \mathrm{wall}(P) \\[2pt]
\lfloor (T_q - \kappa_p W) / \Delta_{dec} \rfloor, & T_q > \mathrm{wall}(P)
\end{cases}$$

Two properties follow, and they are what make the budget well-posed. First, the total
delay $\kappa_p W$ is *sunk*: it does not depend on $P$, so $P$ prices only the
**concentration** of that delay, and requests whose deadlines fall outside the burst
window pay exactly nothing. Second, a request that will miss regardless
($R \approx 1$ at every $P$) contributes no marginal risk, so it cannot hold the budget
hostage. Re-evaluating $E_{viol}$ of §4.1 under this horizon gives

$$TB^{*} \;=\; \max\{\, P \in [P_{floor}, P_{base}] \;:\;
E_{viol}(P) - E_{viol}(P_{floor}) \le \varepsilon_p \,\},$$

found by bisection since $E_{viol}(P)$ is non-decreasing in $P$. The reference point is
$E_{viol}(P_{floor})$ — the always-available alternative — rather than a burst-free
baseline, so the unavoidable part of the sunk cost is not charged against the margin
$\varepsilon_p$. In words: *buy prefill tokens until the in-flight set's total expected
violation count has risen by $\varepsilon_p$ over what the floor already costs.* The
budget is enforced over the step's total prefill — chunked carry-over of already-admitted
prompts and newly admitted prompts drain a single shared counter (in-flight prefills
first) — while decode tokens are never limited; the decode side is controlled only by
whole-request deferral, which the same $E_{viol}$ ledger accepts while it strictly
improves. Two guards complete the design: a positive floor $P_{floor}$ with a starvation
counter ensures prefill always progresses (bounding TTFC inflation), and when the
estimator is cold or no measured request exists, the budget rests at $P_{base}$ —
i.e., the control degrades to the baseline scheduler, never below it.

**What it improves — and when it does nothing.** The control is *smoothing, not
reduction*: total prefill work is conserved and spread across steps that were going to
run anyway, so it is throughput-neutral by construction (89–99% of baseline tokens/s in
our measurements, versus 20–40% losses for request-level batch shrinking, which we
report as a rejected alternative). It engages only when prefill work and a genuinely
at-risk in-flight set coexist; on steps with no pending prefill, or when every deadline
lies outside the burst window, the scheduler is unmodified by construction. In the same
evaluation setting it matches offloading's violation reductions (1.1–1.5% vs baseline
5.4–6.2%) through an entirely disjoint mechanism.

> **Draft note (2026-08-12).** The quantitative claims in this paragraph were measured
> under the earlier worst-case budget $P^{*} = (\gamma T_{min} - \Delta_{dec})/\kappa_p$,
> which the burst-horizon formulation above replaces. Throughput neutrality and the
> violation-reduction range are being re-measured on the shipped rule (sweep `phaseE`,
> caps 32/64/128); the engagement-rate figure is deliberately omitted until then, since
> the trigger condition itself changed. The tolerance $\varepsilon_p$ is calibrated
> against the separation between the marginal cost under normal load and under genuine
> deadline pressure — measured at 0.41 vs 4.54 expected violations on our workload, an
> $11\times$ window.

### 4.4 Composition: when the two levers add up

**Why they should compose.** The extensions bind on different resources in different
phases: offloading acts when the system is *full* (KV blocks admission; prefill work is
scarce because little is being admitted), adaptive token batching acts when the system is
*absorbing* (prompts queued or in flight; KV still has room). Empirically the triggers
co-fire on only 1–2% of steps under the earlier budget rule — far below the product of
their individual rates, and pending re-measurement with the rest of §4.3. On that basis
the combination should behave as a coverage *union* rather than an interaction: each
mechanism handles the failure phase the other cannot see, on a shared risk model and
without contending for the same control variable.

**What the four-way ablation actually shows.** Disjoint triggers make composition safe,
not automatically profitable — a lever that costs throughput in its own regime still
costs it when combined. One representative operating point (cap 64, rate 1; throughput in
tokens/s, violation rate at $\tau=1$s):

| Configuration | Throughput | vs. vanilla | Violation rate |
|---|---|---|---|
| vanilla engine | 519 | 100% | 5.6% |
| ProgressServe + Adaptive | 499 | 96% | **1.3%** |
| ProgressServe + Offload | 411 | 79% | 2.0% |
| ProgressServe + Both | 352 | 68% | 1.8% |

Every configuration achieves the headline result — violations fall from ~6% to under 2%
— so the differentiator is what each pays for it. At this operating point Adaptive is
nearly free (96% of baseline throughput) while Offload is not, and composing inherits
Offload's cost rather than cancelling it.

**One operating point does not settle the ordering.** Across the wider grid Adaptive
holds 95–100% of baseline throughput at every cell measured *except* cap 128 / rate 1,
where it drops to 82%. That exception reproduces: an earlier sweep of the identical cell
gives 81% (430 against a 530 baseline) while its neighbouring rates give 93–94%, so it is
a rate-specific effect rather than run-to-run noise or a monotone trend in the cap. We do
not yet have an explanation for it.

Two caveats bound how far this table can be read. First, the composed configuration
carries markedly wider run-to-run spread than either single lever ($\pm$49 vs $\pm$13
tokens/s at cap 128), which suggests the two controls interact more than their 1–2%
trigger overlap implies. Second, and more fundamentally, **every row above is measured
against the vanilla engine, not against the §4.1 core**, so it reports the cost of
"core + increment" rather than of the increment. By design each increment should collapse
to the core outside its target regime; two implementation details are known to break that
(§4.2's connector mirrors KV eagerly and forces prefix caching on, both paid per step
regardless of whether an offload fires; §4.3's decode-deferral gate accepts any strictly
positive $E_{viol}$ improvement, which shrinking the decode set almost always produces).
The core-only arm needed to separate these effects is in flight; the attribution in this
section is provisional until it lands.
