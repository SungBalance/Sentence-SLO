# SSLO Analysis Notes

`RequestOutput.slo_chunk_records` is populated only on the final output for a
request. Each record carries per-chunk `chunk_idx`, `cumulative_slack`,
`gen_time`, `pending_time`, and `word_count`; the experiment runner flattens
these into `baseline_chunks.jsonl` and `sslo_chunks.jsonl`.

`RequestOutput.sslo_metrics` is also populated only on the final output. It is
an `SsloRequestStats` value from `vllm.sslo.slo_state.RequestSLOState` with:

- `final_cumulative_slack`: last chunk's cumulative slack, or `0.0` when no
  chunk was recorded.
- `min_cumulative_slack`: minimum cumulative slack across recorded chunks.
- `neg_slack_chunk_count`: count of chunks with `cumulative_slack < 0`.
- `total_pending_time_s`: sum of chunk pending time.
- `num_pending_intervals`: number of pending intervals entered by the request.
- `max_consecutive_pending`: longest consecutive pending-enter streak.
- `final_ema_gen_time_s`: final chunk generation-time estimator value.
- `final_ema_per_word_time_s`: final estimator per-token field as currently
  exposed by `RequestSLOState.compute_stats()`.

The audited flow sets `final_cumulative_slack` and `neg_slack_chunk_count` from
the completed chunk records in `compute_stats()`, and
`output_processor.py` attaches that value to `RequestOutput.sslo_metrics` when
generation finishes. Analysis scripts can consume either flattened chunk rows
for per-chunk distributions or `sslo_metrics` from final `RequestOutput`
objects for per-request aggregate counters.
