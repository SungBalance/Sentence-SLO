#!/usr/bin/env python3
"""Shared dataset loading for LM experiments.

Usage:
    from exp.tools.lm_datasets import load_prompts

    prompts = load_prompts("koala", num_prompts=50)
    prompts = load_prompts("wildchat", num_prompts=100)
    prompts = load_prompts("lmsys", num_prompts=200)
    prompts = load_prompts("allenai/WildChat-4.8M", split="train", num_prompts=200)
    prompts = load_prompts("lmsys/lmsys-chat-1m", num_prompts=200)
"""

from __future__ import annotations

import re
from typing import Any

KOALA_DATASET_ID = "HuggingFaceH4/Koala-test-set"
WILDCHAT_DATASET_ID = "allenai/WildChat-4.8M"
LMSYS_DATASET_ID = "lmsys/lmsys-chat-1m"

SUPPORTED_DATASETS: dict[str, str] = {
    "koala": KOALA_DATASET_ID,
    KOALA_DATASET_ID: KOALA_DATASET_ID,
    "wildchat": WILDCHAT_DATASET_ID,
    WILDCHAT_DATASET_ID: WILDCHAT_DATASET_ID,
    "lmsys": LMSYS_DATASET_ID,
    LMSYS_DATASET_ID: LMSYS_DATASET_ID,
}

# Default eval splits per dataset.
_DEFAULT_SPLITS: dict[str, str] = {
    KOALA_DATASET_ID: "test",
    WILDCHAT_DATASET_ID: "train",
    LMSYS_DATASET_ID: "train",
}


def load_prompts(
    dataset_name: str,
    *,
    split: str | None = None,
    num_prompts: int | None = None,
) -> list[str]:
    """Return a list of clean prompt strings from the named dataset.

    Args:
        dataset_name: Canonical HF dataset ID or short alias
                      ('koala', 'wildchat').
        split: Dataset split. Defaults to each dataset's natural eval split
               ('test' for Koala, 'train' for WildChat).
        num_prompts: Maximum prompts to return. None returns all available.

    Returns:
        List of cleaned, non-empty prompt strings.
    """
    dataset_id = _normalize(dataset_name)
    resolved_split = split if split is not None else _DEFAULT_SPLITS[dataset_id]

    if dataset_id == KOALA_DATASET_ID:
        return _load_koala(split=resolved_split, num_prompts=num_prompts)
    if dataset_id == WILDCHAT_DATASET_ID:
        return _load_wildchat(split=resolved_split, num_prompts=num_prompts)
    if dataset_id == LMSYS_DATASET_ID:
        return _load_lmsys(split=resolved_split, num_prompts=num_prompts)
    raise ValueError(f"No loader implemented for dataset id: {dataset_id}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(dataset_name: str) -> str:
    try:
        return SUPPORTED_DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            f"Supported names: {supported}."
        ) from exc


def _clean(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        raise ValueError("empty prompt")
    return text


def _select(items: list[str], num_prompts: int | None) -> list[str]:
    if num_prompts is None:
        return items
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive when set.")
    return items[: min(num_prompts, len(items))]


def _load_koala(*, split: str, num_prompts: int | None) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(KOALA_DATASET_ID, split=split)
    prompts: list[str] = []
    for row in dataset:
        try:
            prompts.append(_clean(row["prompt"]))
        except (KeyError, ValueError):
            continue
    if not prompts:
        raise ValueError(f"{KOALA_DATASET_ID} split={split!r} returned no prompts.")
    return _select(prompts, num_prompts)


def _load_wildchat(*, split: str, num_prompts: int | None) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 4.8 M rows into memory.
    dataset = load_dataset(WILDCHAT_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        conversation = row.get("conversation") or []
        first_user = next(
            (m for m in conversation if m.get("role") == "user"),
            None,
        )
        if first_user is None:
            continue
        try:
            prompts.append(_clean(first_user["content"]))
        except ValueError:
            continue
    if not prompts:
        raise ValueError(f"{WILDCHAT_DATASET_ID} split={split!r} returned no prompts.")
    return prompts


def _load_lmsys(*, split: str, num_prompts: int | None) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 1 M rows into memory.
    dataset = load_dataset(LMSYS_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        conversation = row.get("conversation") or []
        first_user = next(
            (m for m in conversation if m.get("role") == "user"),
            None,
        )
        if first_user is None:
            continue
        try:
            prompts.append(_clean(first_user["content"]))
        except ValueError:
            continue
    if not prompts:
        raise ValueError(f"{LMSYS_DATASET_ID} split={split!r} returned no prompts.")
    return prompts
