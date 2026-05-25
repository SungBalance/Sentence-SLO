"""Seed-agnostic JSONL cache for the SSLO sampling pool.

Single shared file: `exp/tools/dataset_cache/processed_dataset.jsonl`
holds the curated prompt pool (e.g. wildchat + lmsys after our
filter_prompt pipeline). Seed only re-orders the returned list — pool
content is identical across seeds.

When any run_sslo/ script requests the "combined" dataset (via
`load_or_build_combine_pool`), this module returns the contents of
processed_dataset.jsonl. The `build_fn` argument is invoked ONLY when
the file is absent — i.e. fresh cold-start with no prior cache.

Override the cache location with the `DATASET_CACHE_DIR` env var.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable

DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / "dataset_cache")
PROCESSED_FILENAME = "processed_dataset.jsonl"


def cache_dir() -> Path:
    return Path(os.environ.get("DATASET_CACHE_DIR", DEFAULT_CACHE_DIR))


def combine_cache_path(num_datasets: int = 2) -> Path:
    """Path to the shared processed pool. `num_datasets` kept for API
    compatibility with older callers but does not affect the path."""
    del num_datasets  # one canonical pool regardless of source count
    return cache_dir() / PROCESSED_FILENAME


def load_or_build_combine_pool(
    *,
    num_datasets: int,
    seed: int,
    build_fn: Callable[[], list[str]],
) -> list[str]:
    """Return the curated prompt pool shuffled by `seed`.

    Reads `processed_dataset.jsonl` if present; otherwise calls
    `build_fn()` to construct the pool from scratch and persists it.
    Seed only re-orders the returned list.
    """
    path = combine_cache_path(num_datasets)
    if path.exists():
        with path.open() as f:
            prompts = [json.loads(line)["prompt"]
                       for line in f if line.strip()]
    else:
        prompts = list(build_fn())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
        tmp.replace(path)

    # Deterministic per-seed shuffle of the shared pool.
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    return shuffled
