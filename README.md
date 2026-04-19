# Sentence-SLO

## vLLM submodule

`vLLM` is vendored as a git submodule at `third_party/vllm` and pinned to
the stable tag `v0.17.0`.

## Docker (NGC PyTorch 25.12)

Run:

```bash
./scripts/run_ngc_pytorch_25_12.sh
```

Install vLLM (editable) and dependencies inside the container:

```bash
./scripts/install_vllm_editable.sh
```

If docker permission is restricted, run with `sudo`:

```bash
sudo ./scripts/run_ngc_pytorch_25_12.sh
```

## Async Benchmarks

Inside the container (`/workspace/Sentence-SLO`), run:

```bash
# 128 concurrent request latency benchmark (optional per-request CSV)
./scripts/run_async_128_latency.sh \
  --num-requests 128 \
  --max-tokens 64 \
  --model Qwen/Qwen3.5-0.8B \
  --output-csv results/latency_128.csv

# Decode batch size sweep with iteration-level TPOT CSV output
./scripts/run_async_decode_batch_tpot.sh \
  --decode-batch-sizes 1,2,4,8,16,32,64,128 \
  --num-requests 128 \
  --max-tokens 64 \
  --model Qwen/Qwen3.5-0.8B \
  --output-csv results/tpot_by_decode_batch.csv
```
