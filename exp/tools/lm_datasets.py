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
    # SSLO: synthetic mix of koala + wildchat + lmsys, seed-shuffled.
    "combine": "__combine__",
    "COMBINE": "__combine__",
}

# Default eval splits per dataset.
_DEFAULT_SPLITS: dict[str, str] = {
    KOALA_DATASET_ID: "test",
    WILDCHAT_DATASET_ID: "train",
    LMSYS_DATASET_ID: "train",
    "__combine__": "train",  # passed to streaming sources; koala ignores it
}


# Code-generation detection: a prompt is treated as code-related if either
# the user prompt or (when available) the first assistant response shows
# strong code-request signals. Keep the regex conservative — false
# negatives (some code prompts slip through) are preferable to false
# positives (chat prompts wrongly excluded).
_CODE_PROMPT_PATTERN = re.compile(
    r"```"
    r"|write\s+(a\s+)?(function|code|program|script|class|method)"
    r"|implement\s+(a\s+)?(function|class|method|algorithm|program)"
    # action verb + (later in same prompt) a programming language name
    r"|(write|implement|build|create|generate|code)\s+[^.?!]{0,80}\b"
    r"(python|javascript|typescript|rust|c\+\+|java|go|sql|html|css)\b"
    r"|def\s+\w+\s*\("
    r"|function\s+\w*\s*\("
    r"|(python|javascript|typescript|rust|c\+\+|java|go|sql|html|css)"
    r"\s+(code|function|implementation|script|class|method|snippet)",
    re.IGNORECASE,
)


def _is_code_request(prompt: str, response: str | None = None) -> bool:
    """Return True if the prompt (or its first response) signals code-gen.

    - Prompt-side: regex on common code-request phrasing / fenced blocks.
    - Response-side (when available): triple-backtick fenced code block
      is a near-certain signal the model interpreted it as code-gen.
    """
    if _CODE_PROMPT_PATTERN.search(prompt):
        return True
    if response and "```" in response:
        return True
    return False


def load_prompts(
    dataset_name: str,
    *,
    split: str | None = None,
    num_prompts: int | None = None,
    exclude_code: bool = False,
    seed: int = 42,
) -> list[str]:
    """Return a list of clean prompt strings from the named dataset.

    Args:
        dataset_name: Canonical HF dataset ID or short alias
                      ('koala', 'wildchat', 'lmsys', 'combine').
        split: Dataset split. Defaults to each dataset's natural eval split
               ('test' for Koala, 'train' for WildChat / LMSYS). Ignored
               by `combine`.
        num_prompts: Maximum prompts to return. None returns all available.
        exclude_code: If True, filter out prompts that look like code-gen
            requests (prompt regex + first assistant response check when
            available). Streaming datasets keep iterating until enough
            non-code prompts are collected.
        seed: Random seed used by the `combine` dataset shuffle.
            Ignored for single-source datasets.

    Returns:
        List of cleaned, non-empty prompt strings.
    """
    dataset_id = _normalize(dataset_name)
    resolved_split = split if split is not None else _DEFAULT_SPLITS[dataset_id]

    if dataset_id == KOALA_DATASET_ID:
        return _load_koala(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code)
    if dataset_id == WILDCHAT_DATASET_ID:
        return _load_wildchat(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code)
    if dataset_id == LMSYS_DATASET_ID:
        return _load_lmsys(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code)
    if dataset_id == "__combine__":
        return _load_combine(
            num_prompts=num_prompts, exclude_code=exclude_code, seed=seed)
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


def _load_koala(
    *, split: str, num_prompts: int | None, exclude_code: bool = False,
) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(KOALA_DATASET_ID, split=split)
    prompts: list[str] = []
    for row in dataset:
        try:
            text = _clean(row["prompt"])
        except (KeyError, ValueError):
            continue
        # Koala has no assistant response in this split → prompt-only filter.
        if exclude_code and _is_code_request(text):
            continue
        prompts.append(text)
    if not prompts:
        raise ValueError(f"{KOALA_DATASET_ID} split={split!r} returned no prompts.")
    return _select(prompts, num_prompts)


def _first_user_and_assistant(conversation: list[dict]) -> tuple[str | None, str | None]:
    """Extract the first user message and first assistant response (if any)."""
    first_user = None
    first_asst = None
    for msg in conversation:
        role = msg.get("role")
        if role == "user" and first_user is None:
            first_user = msg.get("content")
        elif role == "assistant" and first_asst is None:
            first_asst = msg.get("content")
        if first_user is not None and first_asst is not None:
            break
    return first_user, first_asst


def _load_wildchat(
    *, split: str, num_prompts: int | None, exclude_code: bool = False,
) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 4.8 M rows into memory.
    dataset = load_dataset(WILDCHAT_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        conversation = row.get("conversation") or []
        first_user, first_asst = _first_user_and_assistant(conversation)
        if first_user is None:
            continue
        try:
            text = _clean(first_user)
        except ValueError:
            continue
        if exclude_code and _is_code_request(text, first_asst):
            continue
        prompts.append(text)
    if not prompts:
        raise ValueError(f"{WILDCHAT_DATASET_ID} split={split!r} returned no prompts.")
    return prompts


def _load_lmsys(
    *, split: str, num_prompts: int | None, exclude_code: bool = False,
) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 1 M rows into memory.
    dataset = load_dataset(LMSYS_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        conversation = row.get("conversation") or []
        first_user, first_asst = _first_user_and_assistant(conversation)
        if first_user is None:
            continue
        try:
            text = _clean(first_user)
        except ValueError:
            continue
        if exclude_code and _is_code_request(text, first_asst):
            continue
        prompts.append(text)
    if not prompts:
        raise ValueError(f"{LMSYS_DATASET_ID} split={split!r} returned no prompts.")
    return prompts


def _load_combine(
    *,
    num_prompts: int | None,
    exclude_code: bool = False,
    seed: int = 42,
) -> list[str]:
    """Mix prompts from koala + wildchat + lmsys, shuffled by `seed`.

    Each source contributes roughly num_prompts/3 prompts (with a 50%
    over-sample to absorb shuffle/filter losses). The combined pool is
    shuffled with a seeded RNG so the same `seed` always returns the
    same ordering — useful for reproducible sweeps.
    """
    import random

    if num_prompts is None or num_prompts <= 0:
        per_source = 200  # arbitrary default when caller asks for "all"
    else:
        per_source = max(1, (num_prompts + 2) // 3)
    over = max(1, int(per_source * 1.5))

    # Catch broad Exception so one source failing (HF auth, gated dataset,
    # network) doesn't kill the others. LMSYS is gated and may raise
    # `DatasetNotFoundError` if HF_TOKEN isn't set.
    import sys
    pool: list[str] = []
    for name, loader, split in (
        ("koala", _load_koala, "test"),
        ("wildchat", _load_wildchat, "train"),
        ("lmsys", _load_lmsys, "train"),
    ):
        try:
            pool += loader(split=split, num_prompts=over,
                           exclude_code=exclude_code)
        except Exception as e:  # noqa: BLE001 — intentional broad catch
            print(f"[combine] {name} skipped: {type(e).__name__}: {e}",
                  file=sys.stderr)

    if not pool:
        raise ValueError(
            "combine dataset: all three sources returned no prompts.")

    rng = random.Random(seed)
    rng.shuffle(pool)
    if num_prompts is not None and num_prompts > 0:
        return pool[:num_prompts]
    return pool
