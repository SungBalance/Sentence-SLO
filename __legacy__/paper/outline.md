- Title (working)
- Sentence-SLO: Slack-Aware Sentence-Level Scheduling for Interactive LLM-to-TTS Serving

- Abstract
- Problem: token/sequence latency metrics miss user-perceived sentence jitter.
- Method: Sentence-SLO objective + quantile next-sentence length predictor + selective KV offloading.
- Result preview: better SLO-goodput frontier than vLLM default scheduling.

- 1. Introduction
- Interactive TTS/reading workloads and why sentence gaps matter.
- Failure modes of token-level and sequence-level scheduling for UX.
- Contributions:
- Formal sentence-jitter SLO formulation.
- Slack-aware Sentence-EDF scheduler with quantile prediction.
- Selective KV offloading strategy under memory pressure.
- Reproducible evaluation protocol on Qwen3.5-14B.

- 2. Background and Motivation
- LLM serving pipeline basics (prefill/decode/KV cache).
- Why cognitive slack is schedulable opportunity.
- Resource waste patterns during sentence consumption windows.

- 3. Problem Formulation
- Definitions: sentence ready time, consume end time, jitter, deadline.
- SLO constraint: violation probability target (`eps`) under `delta`.
- Optimization objective: maximize goodput under SLO + resource constraints.

- 4. Method
- 4.1 Scheduler Core: Sentence-EDF priority with slack margin.
- 4.2 Next-Sentence Length Predictor:
- Quantile classification head design.
- Calibration and quantile selection (`q`).
- 4.3 Selective KV Offloading:
- Offload trigger conditions under memory pressure.
- Reload penalty estimation and safety margin.
- 4.4 End-to-end scheduling algorithm and complexity.

- 5. System and Implementation
- Integration points in vLLM serving loop.
- Data paths and logging schema.
- Platform constraints: 4x RTX PRO 6000, no NVLink.
- Practical overheads and operational knobs.

- 6. Experimental Setup
- Model: Qwen3.5-14B-base (single-model first phase).
- Workloads and traffic patterns (burst + long-context mix).
- TTS profiling pipeline using Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice.
- Deterministic sentence segmentation (fixed abbreviation exceptions) and zero-fallback headline policy.
- TTS duration model: piecewise word-count bucket means per language (`en`, `ko`).
- `word_count` definition: whitespace-delimited tokens on normalized sentence text.
- Predictor classes: remaining-to-sentence-end tokens in {<=8, 9-16, 17-32, 33-64, 65-128, 129-256, 257-512, >512}.
- Baseline and ablations:
- Baseline: vLLM default scheduler.
- Ablations: Sentence-EDF only, +predictor only, +offload only, full method.
- Metrics and statistics:
- Primary: goodput_tps, slo_violation_rate.
- Secondary: jitter tail, TTFT, E2E latency, memory/offload bandwidth.
- Seeds, paired replay, bootstrap confidence intervals.

- 7. Results
- 7.1 SLO-goodput frontier across `delta in {0, 50, 100}`.
- 7.2 Ablation study and component contribution.
- 7.3 Tail behavior and failure case analysis.
- 7.4 Sensitivity to quantile `q` and load levels.

- 8. Discussion
- When offloading helps or hurts in PCIe-only settings.
- Practical deployment guidance for interactive services.
- Fairness and multi-tenant considerations.

- 9. Related Work
- LLM scheduling and KV cache management.
- Real-time serving and deadline-aware systems.
- TTS/interactive latency QoE literature.

- 10. Limitations and Future Work
- Dependence on accurate consume-duration profiling.
- Domain shift and predictor calibration drift.
- Extension to multi-modal and multi-tenant production traces.

- Appendix
- Additional algorithm details and pseudocode.
- Full hyperparameter/config tables.
- Extra plots and reproducibility checklist.
