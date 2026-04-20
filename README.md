# Sentence-SLO

## Current Workspace Layout

- This repository is mounted into the `sk-sslo` Docker container at
  `/workspace/mlsys`.
- The editable vLLM checkout is under `vllm/`.
- SSLO-specific vLLM code should live under `vllm/vllm/sslo/`.
- Experiment code and launch scripts should live under `exp/`.

## Docker

Start the local experiment container:

```bash
./run_docker.sh
```

Inside the container, use `/workspace/mlsys` as the repository root. The host
`/data` directory is mounted as `/cache`, and experiment scripts should use it
for Hugging Face model cache:

```bash
export HF_HOME=/cache
export HF_HUB_CACHE=/cache/hub
```

## Install vLLM In The Container

Run the editable install inside `sk-sslo`:

```bash
docker exec sk-sslo bash -lc '
  git config --global --add safe.directory /workspace/mlsys
  cd /workspace/mlsys/vllm
  VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.0.0+sslo \
    pip install -e . --no-build-isolation
'
```

If build metadata generation complains about missing local build helpers, install
them first and rerun the command above:

```bash
docker exec sk-sslo bash -lc '
  pip install "setuptools_scm>=8.0" "setuptools>=77,<81" \
    "packaging>=24.2" wheel ninja "cmake>=3.26.1" jinja2
'
```

`--no-build-isolation` avoids a slow isolated build dependency install that can
try to reinstall large packages such as PyTorch inside the container.

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
