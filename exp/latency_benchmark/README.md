# latency_benchmark

Consumer-side latency benchmark measuring TTFT, TPOT, E2E latency, and chunk-level cumulative slack.

All timing is recorded at the async consumer (client side), not inside the vLLM engine. No SSLO vLLM extensions are required — the only project dependency is `exp/tools/lm_datasets.py`.

## Metrics

**Per-request** (`requests.jsonl`):

| Field | Description |
|-------|-------------|
| `ttft_s` | Time from request submission to first token received at consumer |
| `tpot_ms` | (e2e − ttft) / (output_tokens − 1) × 1000 |
| `e2e_latency_s` | Time from submission to last token |
| `num_output_tokens` | Total generated tokens |
| `num_chunks` | Number of sentence/paragraph chunks emitted |

**Per-chunk** (`chunks.jsonl`):

| Field | Description |
|-------|-------------|
| `chunk_idx` | 0-based chunk index within the request |
| `word_count` | Words in this chunk |
| `chunk_end_ts` | Monotonic timestamp when chunk arrived at consumer |
| `decoding_start_ts` | Monotonic timestamp of first token for this request |
| `cumulative_consume` | Σ word_count[0..k−1] × spw — the reading deadline |
| `cumulative_slack` | cumulative_consume − (chunk_end_ts − decoding_start_ts) |

Positive slack means the model is ahead of the reader. Negative means the reader has to wait.

## Running

```bash
bash exp/latency_benchmark/run_experiment.sh
```

Both GPUs must be available in container `sk-sslo-vllm`. The two models run in parallel on separate GPUs; sentence and paragraph chunk units run sequentially within each GPU.

## Output layout

```
exp/latency_benchmark/outputs/
  {model_slug}/{dataset_slug}/batch_{N}/{chunk_unit}/
    requests.jsonl
    chunks.jsonl
    summary.json
```
