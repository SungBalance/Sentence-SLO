"""Seed-agnostic JSONL cache for the SSLO sampling pool.

Single shared file: `exp/tools/dataset_cache/processed_dataset.jsonl`
holds the curated prompt pool (e.g. wildchat + lmsys after our
filter_prompt pipeline). Seed only re-orders the returned list — pool
content is identical across seeds.

When any run_sslo/ script requests the "combined" dataset (via
`load_or_build_combine_pool`), this module returns the contents of
processed_dataset.jsonl. The `build_fn` argument is invoked ONLY when
the file is absent — i.e. fresh cold-start with no prior cache.

Multi-turn workloads use a parallel cache
(`load_or_build_dialogue_pool`) keyed by source dataset and filter
combination, storing raw `{"messages": [...]}` rows before any chat
template is applied so the same file is reusable across models.

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
DIALOGUE_FILENAME_TMPL = "dialogues_{dataset}_{filters}.jsonl"


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


def dialogue_cache_path(
    dataset_name: str,
    *,
    conversation_only: bool,
    english_only: bool,
    exclude_code: bool,
) -> Path:
    """Path to the raw-dialogue cache for one source/filter combination."""
    flags = [
        ("conv", conversation_only),
        ("en", english_only),
        ("nocode", exclude_code),
    ]
    filters = "-".join(tag for tag, on in flags if on) or "all"
    return cache_dir() / DIALOGUE_FILENAME_TMPL.format(
        dataset=dataset_name, filters=filters)


def load_or_build_dialogue_pool(
    *,
    dataset_name: str,
    conversation_only: bool,
    english_only: bool,
    exclude_code: bool,
    max_dialogues: int,
    seed: int,
    build_fn: Callable[[], list[list[dict]]],
) -> list[list[dict]]:
    """Return up to `max_dialogues` raw dialogues shuffled by `seed`.

    Reads the per-source dialogue cache if present; otherwise calls
    `build_fn()` (which does the dataset streaming/filtering) and
    persists the result. The cache holds raw `{role, content}` message
    lists, so it is model-agnostic. A cache smaller than requested is
    used as-is — delete the file to rebuild it larger.
    """
    path = dialogue_cache_path(
        dataset_name,
        conversation_only=conversation_only,
        english_only=english_only,
        exclude_code=exclude_code)
    if path.exists():
        with path.open() as f:
            dialogues = [json.loads(line)["messages"]
                         for line in f if line.strip()]
    else:
        dialogues = list(build_fn())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for messages in dialogues:
                f.write(json.dumps({"messages": messages},
                                   ensure_ascii=False) + "\n")
        tmp.replace(path)

    shuffled = list(dialogues)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < max_dialogues:
        print(f"dialogue cache {path} holds {len(shuffled)} dialogues "
              f"(< requested {max_dialogues}); using all of them. "
              "Delete the file to rebuild it larger.")
    return shuffled[:max_dialogues]
