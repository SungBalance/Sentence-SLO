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


def _max_response_chunk_chars(response: str) -> int:
    """Run SSLO's ChunkSeparator on the response and return the longest
    chunk length (chars). Used to identify dump-style responses (long
    comma lists, ASCII outputs, repeated tokens) that lack sentence-end
    punctuation and would produce mega-chunks at SSLO runtime."""
    # Local import: vllm package is only available inside the SSLO
    # container. Keeping the import lazy lets non-SSLO callers use the
    # rest of this module without pulling vllm in.
    from vllm.sslo.slo_state import ChunkSeparator

    sep = ChunkSeparator(chunk_unit="sentence", min_chunk_tokens=0)
    mx = 0
    for chunk in sep.feed(response, num_tokens=max(1, len(response))):
        if len(chunk) > mx:
            mx = len(chunk)
    tail = sep.flush()
    if tail is not None and len(tail) > mx:
        mx = len(tail)
    return mx


def load_prompts(
    dataset_name: str,
    *,
    split: str | None = None,
    num_prompts: int | None = None,
    exclude_code: bool = False,
    seed: int = 42,
    conversation_only: bool = False,
    english_only: bool = False,
    max_response_chunk_chars: int | None = None,
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
        english_only: If True, keep only rows whose row-level `language`
            field equals exactly "English". WildChat-4.8M and
            LMSYS-Chat-1M both expose this string field. Non-English,
            missing, or "Nolang" rows are dropped.
        max_response_chunk_chars: If set, drop rows whose first assistant
            response produces a chunk longer than this many chars under
            SSLO's ChunkSeparator (sentence mode). Filters out
            dump-style responses (long comma lists, ASCII output,
            repeated tokens) that lack sentence-end punctuation and
            would yield mega-chunks at runtime.

    Returns:
        List of cleaned, non-empty prompt strings.
    """
    dataset_id = _normalize(dataset_name)
    resolved_split = split if split is not None else _DEFAULT_SPLITS[dataset_id]

    if dataset_id == KOALA_DATASET_ID:
        if conversation_only:
            raise ValueError(
                "Koala is a single-turn instruction set — no conversation "
                "rows. Use wildchat / lmsys / combine with "
                "conversation_only=True.")
        # Koala is English-only by construction; english_only is a no-op.
        return _load_koala(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code)
    if dataset_id == WILDCHAT_DATASET_ID:
        return _load_wildchat(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code, conversation_only=conversation_only,
            english_only=english_only,
            max_response_chunk_chars=max_response_chunk_chars)
    if dataset_id == LMSYS_DATASET_ID:
        return _load_lmsys(
            split=resolved_split, num_prompts=num_prompts,
            exclude_code=exclude_code, conversation_only=conversation_only,
            english_only=english_only,
            max_response_chunk_chars=max_response_chunk_chars)
    if dataset_id == "__combine__":
        return _load_combine(
            num_prompts=num_prompts, exclude_code=exclude_code, seed=seed,
            conversation_only=conversation_only,
            english_only=english_only,
            max_response_chunk_chars=max_response_chunk_chars)
    raise ValueError(f"No loader implemented for dataset id: {dataset_id}")


# SSLO
def load_dialogues(
    dataset_name: str,
    *,
    split: str | None = None,
    max_dialogues: int | None = None,
    conversation_only: bool = False,
    english_only: bool = False,
    exclude_code: bool = False,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Return filtered dialogues as lists of {role, content} dicts."""
    if max_dialogues is not None and max_dialogues <= 0:
        raise ValueError("max_dialogues must be positive when set.")

    dataset_id = _normalize(dataset_name)
    resolved_split = split if split is not None else _DEFAULT_SPLITS[dataset_id]

    if dataset_id == KOALA_DATASET_ID:
        raise ValueError(
            "Koala is a single-turn instruction set; use wildchat, lmsys, "
            "or combine for dialogue rows."
        )
    if dataset_id == WILDCHAT_DATASET_ID:
        return _load_wildchat_dialogues(
            split=resolved_split,
            max_dialogues=max_dialogues,
            conversation_only=conversation_only,
            english_only=english_only,
            exclude_code=exclude_code,
        )
    if dataset_id == LMSYS_DATASET_ID:
        return _load_lmsys_dialogues(
            split=resolved_split,
            max_dialogues=max_dialogues,
            conversation_only=conversation_only,
            english_only=english_only,
            exclude_code=exclude_code,
        )
    if dataset_id == "__combine__":
        return _load_combine_dialogues(
            max_dialogues=max_dialogues,
            conversation_only=conversation_only,
            english_only=english_only,
            exclude_code=exclude_code,
            seed=seed,
        )
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


# SSLO: strict-English text check — rejects any character from the
# major non-Latin script blocks (CJK, Hangul, Kana, Cyrillic, Arabic,
# Hebrew, Devanagari, Thai). Greek (U+0370..U+03FF) is intentionally NOT
# included so math symbols Δ, λ, θ in technical English pass through.
# Latin Extended (é, ü, ñ, ...) also passes through.
# Motivation: row-level `language=="English"` lets bilingual / mis-tagged
# rows leak, and SSLO's tail-outlier behavior is very sensitive to those
# leaks (Chinese / Korean text inflates audio_duration and skews the
# tail of the chunk-length distribution).
_NON_LATIN_RE = re.compile(
    r"[一-鿿"   # CJK Unified Ideographs
    r"가-힯"    # Hangul Syllables
    r"぀-ゟ"    # Hiragana
    r"゠-ヿ"    # Katakana
    r"Ѐ-ӿ"    # Cyrillic
    r"؀-ۿ"    # Arabic
    r"֐-׿"    # Hebrew
    r"ऀ-ॿ"    # Devanagari
    r"฀-๿"    # Thai
    r"＀-￯"    # Halfwidth / Fullwidth forms (CJK punctuation)
    r"]"
)


# SSLO
def _is_strict_english_text(text: str) -> bool:
    """Return True iff `text` contains zero non-Latin-script characters.

    Strict companion to the row-level `language=="English"` check. Use
    when downstream consumers (e.g. SSLO TTS profiling) are sensitive to
    tail outliers from bilingual / mis-tagged rows.
    """
    if not text:
        return True
    return _NON_LATIN_RE.search(text) is None


def _is_conversation(conversation: list[dict]) -> bool:
    """Multi-turn conversation = at least one full
    user-assistant-user-assistant cycle (i.e., the user came back with a
    follow-up). Rows with just the opening prompt and one assistant reply
    are single-shot instructions, not conversations."""
    n_user = sum(1 for m in conversation if m.get("role") == "user")
    n_asst = sum(1 for m in conversation if m.get("role") == "assistant")
    return n_user >= 2 and n_asst >= 2


# SSLO
def _apply_dialogue_filters(
    rows_iter: Any,
    *,
    content_key: str,
    max_dialogues: int | None,
    conversation_only: bool,
    english_only: bool,
    exclude_code: bool,
):
    yielded = 0
    for row in rows_iter:
        if max_dialogues is not None and yielded >= max_dialogues:
            break
        if english_only and row.get("language") != "English":
            continue
        conversation = row.get(content_key) or []
        if conversation_only and not _is_conversation(conversation):
            continue

        dialogue: list[dict[str, Any]] = []
        has_code_turn = False
        # SSLO
        has_non_english_turn = False
        for message in conversation:
            try:
                content = _clean(message.get("content", ""))
            except (AttributeError, ValueError):
                continue
            if exclude_code and _is_code_request(content):
                has_code_turn = True
                break
            # SSLO: strict-English check on every turn — any non-Latin
            # script content (CJK, Hangul, Cyrillic, Arabic, ...) drops
            # the whole dialogue. row-level language=='English' alone
            # leaks bilingual / mistagged rows.
            if english_only and not _is_strict_english_text(content):
                has_non_english_turn = True
                break
            dialogue.append(
                {
                    "role": str(message.get("role", "")).strip().lower(),
                    "content": content,
                }
            )
        if has_code_turn or has_non_english_turn or not dialogue:
            continue
        yielded += 1
        yield dialogue


# SSLO
def _load_wildchat_dialogues(
    *,
    split: str,
    max_dialogues: int | None,
    conversation_only: bool = False,
    english_only: bool = False,
    exclude_code: bool = False,
) -> list[list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(WILDCHAT_DATASET_ID, split=split, streaming=True)
    return list(
        _apply_dialogue_filters(
            dataset,
            content_key="conversation",
            max_dialogues=max_dialogues,
            conversation_only=conversation_only,
            english_only=english_only,
            exclude_code=exclude_code,
        )
    )


# SSLO
def _load_lmsys_dialogues(
    *,
    split: str,
    max_dialogues: int | None,
    conversation_only: bool = False,
    english_only: bool = False,
    exclude_code: bool = False,
) -> list[list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(LMSYS_DATASET_ID, split=split, streaming=True)
    return list(
        _apply_dialogue_filters(
            dataset,
            content_key="conversation",
            max_dialogues=max_dialogues,
            conversation_only=conversation_only,
            english_only=english_only,
            exclude_code=exclude_code,
        )
    )


# SSLO
def _load_combine_dialogues(
    *,
    max_dialogues: int | None,
    conversation_only: bool = False,
    english_only: bool = False,
    exclude_code: bool = False,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    import random
    import sys

    if max_dialogues is None:
        over = 300
    else:
        over = max(1, int(max_dialogues * 1.5))

    pool: list[list[dict[str, Any]]] = []
    for name, loader, split in (
        ("wildchat", _load_wildchat_dialogues, "train"),
        ("lmsys", _load_lmsys_dialogues, "train"),
    ):
        try:
            pool += loader(
                split=split,
                max_dialogues=over,
                conversation_only=conversation_only,
                english_only=english_only,
                exclude_code=exclude_code,
            )
        except Exception as e:  # noqa: BLE001 - intentional broad catch
            print(
                f"[combine] {name} skipped: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    if not pool:
        raise ValueError("combine dataset: both sources returned no dialogues.")

    rng = random.Random(seed)
    rng.shuffle(pool)
    if max_dialogues is not None:
        return pool[:max_dialogues]
    return pool


def _load_wildchat(
    *, split: str, num_prompts: int | None, exclude_code: bool = False,
    conversation_only: bool = False, english_only: bool = False,
    max_response_chunk_chars: int | None = None,
) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 4.8 M rows into memory.
    dataset = load_dataset(WILDCHAT_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        if english_only and row.get("language") != "English":
            continue
        conversation = row.get("conversation") or []
        if conversation_only and not _is_conversation(conversation):
            continue
        first_user, first_asst = _first_user_and_assistant(conversation)
        if first_user is None:
            continue
        try:
            text = _clean(first_user)
        except ValueError:
            continue
        # SSLO: strict-English text check (see _is_strict_english_text).
        if english_only and (
                not _is_strict_english_text(text)
                or (first_asst is not None
                    and not _is_strict_english_text(first_asst))):
            continue
        if exclude_code and _is_code_request(text, first_asst):
            continue
        if (max_response_chunk_chars is not None and first_asst
                and _max_response_chunk_chars(first_asst)
                > max_response_chunk_chars):
            continue
        prompts.append(text)
    if not prompts:
        raise ValueError(f"{WILDCHAT_DATASET_ID} split={split!r} returned no prompts.")
    return prompts


def _load_lmsys(
    *, split: str, num_prompts: int | None, exclude_code: bool = False,
    conversation_only: bool = False, english_only: bool = False,
    max_response_chunk_chars: int | None = None,
) -> list[str]:
    from datasets import load_dataset

    # Streaming avoids pulling all 1 M rows into memory.
    dataset = load_dataset(LMSYS_DATASET_ID, split=split, streaming=True)
    prompts: list[str] = []
    for row in dataset:
        if num_prompts is not None and len(prompts) >= num_prompts:
            break
        if english_only and row.get("language") != "English":
            continue
        conversation = row.get("conversation") or []
        if conversation_only and not _is_conversation(conversation):
            continue
        first_user, first_asst = _first_user_and_assistant(conversation)
        if first_user is None:
            continue
        try:
            text = _clean(first_user)
        except ValueError:
            continue
        # SSLO: strict-English text check (see _is_strict_english_text).
        if english_only and (
                not _is_strict_english_text(text)
                or (first_asst is not None
                    and not _is_strict_english_text(first_asst))):
            continue
        if exclude_code and _is_code_request(text, first_asst):
            continue
        if (max_response_chunk_chars is not None and first_asst
                and _max_response_chunk_chars(first_asst)
                > max_response_chunk_chars):
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
    conversation_only: bool = False,
    english_only: bool = False,
    max_response_chunk_chars: int | None = None,
) -> list[str]:
    """Mix prompts from wildchat + lmsys, shuffled by `seed`.

    Each source contributes roughly num_prompts/2 prompts (with a 50%
    over-sample to absorb shuffle/filter losses). The combined pool is
    shuffled with a seeded RNG so the same `seed` always returns the
    same ordering — useful for reproducible sweeps. Koala is excluded
    (single-turn instruction set; the sentence-SLO benchmark targets
    chat-style prompts).
    """
    import random

    if num_prompts is None or num_prompts <= 0:
        per_source = 200  # arbitrary default when caller asks for "all"
    else:
        per_source = max(1, (num_prompts + 1) // 2)
    over = max(1, int(per_source * 1.5))

    # Catch broad Exception so one source failing (HF auth, gated dataset,
    # network) doesn't kill the others. LMSYS is gated and may raise
    # `DatasetNotFoundError` if HF_TOKEN isn't set.
    import sys
    pool: list[str] = []
    for name, loader, split in (
        ("wildchat", _load_wildchat, "train"),
        ("lmsys", _load_lmsys, "train"),
    ):
        try:
            pool += loader(
                split=split, num_prompts=over,
                exclude_code=exclude_code,
                conversation_only=conversation_only,
                english_only=english_only,
                max_response_chunk_chars=max_response_chunk_chars)
        except Exception as e:  # noqa: BLE001 — intentional broad catch
            print(f"[combine] {name} skipped: {type(e).__name__}: {e}",
                  file=sys.stderr)

    if not pool:
        raise ValueError(
            "combine dataset: both sources returned no prompts.")

    rng = random.Random(seed)
    rng.shuffle(pool)
    if num_prompts is not None and num_prompts > 0:
        return pool[:num_prompts]
    return pool
