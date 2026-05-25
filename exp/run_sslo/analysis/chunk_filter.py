"""Chunk-level noise filter for analysis.

Used by analyze.py / progress_metrics.py to drop chunks whose text is:
  - foreign language (CJK Han/Hiragana/Katakana/Hangul)
  - unicode garbage (>5% non-ASCII/Latin-1 chars)
  - meaningless repetition (low char diversity or repeating n-gram)

These chunks pollute downstream metrics because the consume_duration
estimator (`word_count × seconds_per_word` or TTS profile by wc) under-
estimates the actual reading/synthesis cost when the chunk text is not
real human-readable text. See R4 violation analysis (smoke run).
"""
from __future__ import annotations

import re
from collections import Counter

CJK_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿"
    r"぀-ゟ゠-ヿ"
    r"가-힯ᄀ-ᇿ㄰-㆏]"
)

_ALLOWED = re.compile(
    r"[\x00-\x7F"
    r" -ÿ"
    r" -⁯"
    r"‐-‧‰-⁞]"
)

# Tuned to catch GARBAGE/non-readable text without flagging normal English
# prose, source code, markdown tables, or emoji-containing output.
CJK_DOMINANT_THRESHOLD = 0.30   # only drop if ≥30% chars are CJK
NON_ASCII_THRESHOLD = 0.30      # only drop if ≥30% chars outside Latin-1
CHAR_DIVERSITY_THRESHOLD = 0.05 # normal English ≈0.12, garbage repeats ≪0.05
MIN_LEN_FOR_DIVERSITY_CHECK = 60
REPEAT_NGRAM = 3
REPEAT_RATIO_THRESHOLD = 0.50   # garbage repeats (》:( pattern) hit ≥0.95
MIN_LEN_FOR_NGRAM_CHECK = 60


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if CJK_RE.match(ch)) / len(text)


def unicode_noise_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if not _ALLOWED.match(ch))
    return bad / len(text)


def char_diversity(text: str) -> float:
    if not text:
        return 1.0
    return len(set(text)) / len(text)


def top_ngram_ratio(text: str, n: int = REPEAT_NGRAM) -> float:
    """Fraction of positions covered by the most common n-gram."""
    if len(text) < n * 2:
        return 0.0
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    most = Counter(ngrams).most_common(1)[0][1]
    return most * n / len(text)


def classify(text: str) -> str | None:
    """Return reason if the chunk should be dropped, else None."""
    if not text:
        return None
    if cjk_ratio(text) >= CJK_DOMINANT_THRESHOLD:
        return "cjk"
    if unicode_noise_ratio(text) > NON_ASCII_THRESHOLD:
        return "unicode_noise"
    if (len(text) >= MIN_LEN_FOR_DIVERSITY_CHECK
            and char_diversity(text) < CHAR_DIVERSITY_THRESHOLD):
        return "low_diversity"
    if (len(text) >= MIN_LEN_FOR_NGRAM_CHECK
            and top_ngram_ratio(text) > REPEAT_RATIO_THRESHOLD):
        return "repeating_ngram"
    return None


def should_keep(chunk: dict) -> bool:
    return classify(chunk.get("text") or "") is None


def split_filtered(chunks: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Return (kept_chunks, drop_count_by_reason)."""
    kept = []
    drops: dict[str, int] = {}
    for c in chunks:
        reason = classify(c.get("text") or "")
        if reason is None:
            kept.append(c)
        else:
            drops[reason] = drops.get(reason, 0) + 1
    return kept, drops
