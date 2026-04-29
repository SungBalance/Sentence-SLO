#!/usr/bin/env python3
"""Load and clean supported evaluation datasets for SSLO experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


KOALA_DATASET_ID = "HuggingFaceH4/Koala-test-set"
SUPPORTED_DATASETS = {
    "koala": KOALA_DATASET_ID,
    KOALA_DATASET_ID: KOALA_DATASET_ID,
}


@dataclass(frozen=True)
class EvalDatasetItem:
    item_id: str
    prompt: str


def clean_prompt(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        raise ValueError("Dataset row has an empty prompt.")
    return text


def normalize_dataset_name(dataset_name: str) -> str:
    try:
        return SUPPORTED_DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported eval dataset {dataset_name!r}. Supported: {supported}."
        ) from exc


def load_eval_dataset(
    *,
    dataset_name: str,
    split: str,
    num_prompts: int | None,
) -> list[EvalDatasetItem]:
    dataset_id = normalize_dataset_name(dataset_name)
    if dataset_id != KOALA_DATASET_ID:
        raise ValueError(f"Unsupported eval dataset id: {dataset_id}")
    return _select_items(
        _load_koala(split=split),
        num_prompts=num_prompts,
    )


def _load_koala(*, split: str) -> list[EvalDatasetItem]:
    from datasets import load_dataset

    dataset = load_dataset(KOALA_DATASET_ID, split=split)
    items: list[EvalDatasetItem] = []
    for row_idx, row in enumerate(dataset):
        items.append(
            EvalDatasetItem(
                item_id=str(row.get("id") or f"koala_{row_idx}"),
                prompt=clean_prompt(row["prompt"]),
            )
        )
    if not items:
        raise ValueError(f"{KOALA_DATASET_ID} split {split!r} has no prompts.")
    return items


def _select_items(
    items: list[EvalDatasetItem],
    *,
    num_prompts: int | None,
) -> list[EvalDatasetItem]:
    if num_prompts is None:
        return items
    if num_prompts <= 0:
        raise ValueError("--num-prompts must be positive when set.")

    # Clamp oversized requests to the available dataset instead of repeating rows.
    return items[: min(num_prompts, len(items))]
