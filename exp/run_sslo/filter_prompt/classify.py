"""Classify per-request outputs and emit list of prompts to remove.

Reads chunks.jsonl + requests.jsonl from a single rate_<r> dir, joins
chunks per request, classifies the joined output, and writes:
  classification.jsonl — one row per request: {request_id, prompt,
                          reasons: [...], num_chunks, output_chars}

A prompt is marked "bad" if its output triggers ANY of:
  - garbage_unicode: high non-ASCII/Latin-1 ratio
  - non_english:     CJK or Arabic/Persian/Thai etc. dominant
  - repetition:      low char diversity or repeating n-gram
  - markdown_table:  rows of pipes `| ... | ... |`
  - code_block:      triple-backtick fences or fenced inline patterns

The 4-line bar (very long English responses) is NOT a trigger — long
content from real prompts is desired workload.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Foreign-language ranges (CJK + other non-Latin scripts seen in our pool)
FOREIGN_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿"          # CJK Unified + Compat
    r"぀-ゟ゠-ヿ"                # Hiragana, Katakana
    r"가-힯ᄀ-ᇿ㄰-㆏"   # Hangul + Jamo
    r"؀-ۿݐ-ݿ"                # Arabic + Arabic Suppl
    r"ก-๿ກ-໿"                  # Thai, Lao
    r"א-ת"                          # Hebrew
    r"ऀ-ॿਁ-ੴఁ-౿"  # Devanagari, Gurmukhi, Telugu
    r"]"
)

# Latin-1 + general punctuation allowance (matches chunk_filter.py)
_ALLOWED = re.compile(
    r"[\x00-\x7F"
    r" -ÿ"
    r" -⁯"
    r"‐-‧‰-⁞]"
)

# Markdown table detector: 2+ lines that look like `| ... | ... |`
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|.*$", re.MULTILINE)
TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|", re.MULTILINE)

# Code block: triple backtick fence
CODE_FENCE_RE = re.compile(r"```")
# Also code-looking 4-space-indented blocks (>=3 consecutive lines)
INDENTED_CODE_RE = re.compile(
    r"(?:^( {4,}|\t)[^\n]+\n){3,}", re.MULTILINE)

FOREIGN_DOMINANT = 0.20      # >20% of output is foreign script
NOISE_THRESHOLD = 0.20       # >20% non-ASCII/Latin-1
# Use ABSOLUTE unique-char count rather than a length-normalized ratio.
# Natural English easily has 30+ unique chars; "aaaa..." has 1.
MIN_UNIQUE_CHARS = 15
MIN_LEN_FOR_UNIQUE_CHECK = 60
REPEAT_NGRAM = 3
REPEAT_RATIO = 0.30          # garbage glyph runs hit 0.5+; normal text ≤0.1
MIN_LEN_FOR_NGRAM_CHECK = 60
WHITESPACE_MIN_RATIO = 0.04  # natural text has >=5% whitespace
MIN_LEN_FOR_WS_CHECK = 80


def whitespace_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ch.isspace()) / len(text)


def foreign_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if FOREIGN_RE.match(ch)) / len(text)


def noise_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if not _ALLOWED.match(ch)) / len(text)


def char_diversity(text: str) -> float:
    if not text:
        return 1.0
    return len(set(text)) / len(text)


def top_ngram_ratio(text: str, n: int = REPEAT_NGRAM) -> float:
    if len(text) < n * 2:
        return 0.0
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return Counter(ngrams).most_common(1)[0][1] * n / len(text)


def has_markdown_table(text: str) -> bool:
    # Need at least 2 table-row lines AND a separator line for confidence.
    rows = len(TABLE_LINE_RE.findall(text))
    has_sep = bool(TABLE_SEP_RE.search(text))
    return rows >= 2 and has_sep


def has_code_block(text: str) -> bool:
    if CODE_FENCE_RE.search(text):
        return True
    if INDENTED_CODE_RE.search(text):
        return True
    return False


def classify(text: str) -> list[str]:
    reasons: list[str] = []
    if not text:
        return reasons
    if foreign_ratio(text) >= FOREIGN_DOMINANT:
        reasons.append("non_english")
    if noise_ratio(text) > NOISE_THRESHOLD:
        reasons.append("garbage_unicode")
    # Garbage glyph runs often lack whitespace — a strong tell.
    if (len(text) >= MIN_LEN_FOR_WS_CHECK
            and whitespace_ratio(text) < WHITESPACE_MIN_RATIO):
        reasons.append("garbage_unicode")
    # Very low unique-char count catches "aaaa..." / "ababab..."; doesn't
    # false-positive on long English (which has 30+ unique chars).
    if (len(text) >= MIN_LEN_FOR_UNIQUE_CHECK
            and len(set(text)) < MIN_UNIQUE_CHARS):
        reasons.append("repetition")
    if (len(text) >= MIN_LEN_FOR_NGRAM_CHECK
            and top_ngram_ratio(text) > REPEAT_RATIO):
        reasons.append("repetition")
    if has_markdown_table(text):
        reasons.append("markdown_table")
    if has_code_block(text):
        reasons.append("code_block")
    # dedupe
    return sorted(set(reasons))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="rate_<r> dir with chunks.jsonl + requests.jsonl")
    args = ap.parse_args()

    in_dir = Path(args.input)
    chunks_path = in_dir / "chunks.jsonl"
    requests_path = in_dir / "requests.jsonl"
    if not chunks_path.exists() or not requests_path.exists():
        raise SystemExit(f"missing chunks.jsonl or requests.jsonl in {in_dir}")

    # Join chunks per request
    chunks_by_req: dict[str, list[dict]] = defaultdict(list)
    with chunks_path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("request_id")
            if rid is None:
                continue
            chunks_by_req[str(rid)].append(row)
    for rid in chunks_by_req:
        chunks_by_req[rid].sort(key=lambda c: c.get("unit_index", 0))

    out_path = in_dir / "classification.jsonl"
    n_total = 0
    n_bad = 0
    cat_counts: Counter = Counter()
    with out_path.open("w") as fout, requests_path.open() as fin:
        for line in fin:
            req = json.loads(line)
            rid = str(req.get("request_id"))
            prompt = req.get("prompt", "")
            chunks = chunks_by_req.get(rid, [])
            full_output = "".join(c.get("text", "") for c in chunks)
            reasons = classify(full_output)
            n_total += 1
            if reasons:
                n_bad += 1
                for r in reasons:
                    cat_counts[r] += 1
            fout.write(json.dumps({
                "request_id": rid,
                "prompt": prompt,
                "reasons": reasons,
                "num_chunks": len(chunks),
                "output_chars": len(full_output),
            }, ensure_ascii=False) + "\n")

    print(f"classified {n_total} requests, {n_bad} marked bad")
    print(f"  category counts: {dict(cat_counts)}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
