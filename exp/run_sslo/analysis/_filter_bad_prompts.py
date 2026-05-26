"""Scan recent output chunks.jsonl, identify bad-pattern requests
(numeric strings, repetition), map to dataset prompts via seed=42
shuffle, dry-run removal from processed_dataset.jsonl.

Usage:
  python3 _filter_bad_prompts.py            # dry-run, print samples
  python3 _filter_bad_prompts.py --apply    # actually remove
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspace/mlsys")
CACHE = ROOT / "exp/tools/dataset_cache/processed_dataset.jsonl"
OUTPUT_DIRS = [
    ROOT / "exp/run_sslo/filter_prompt/output/sentence",
    ROOT / "exp/run_sslo/output_smoke/sentence",
]
SEED = 42                # dataset_cache shuffle seed (DATASET_SEED env)
REQUEST_RATE_SEED = 44   # base seed for run_test.py per-rate shuffle
# Per-rate seed: REQUEST_RATE_SEED + int(rate * 1000)

# Thresholds
NUMERIC_DIGIT_RATIO = 0.50       # tighter — only >50% digits
NUMERIC_MIN_TOK_PER_WORD = 8.0   # extreme density (π digits = 100+ tok / 1 word)
NUMERIC_MIN_CHUNK_TOK = 80       # only LONG numeric chunks cause deadline misses
REPEAT_NGRAM_MIN = 2             # skip 1-grams (too noisy)
REPEAT_NGRAM_MAX = 5
REPEAT_MIN_OCCURRENCES = 8       # n-gram repeated 8+ times = degenerate
REPEAT_MIN_CHUNK_TOK = 50        # only long chunks
REPEAT_UNIQUE_WORD_RATIO = 0.15  # very low diversity
# Repetition n-gram must be alphabetic (not math symbols)
REPEAT_NGRAM_ALPHA_RATIO = 0.7   # >=70% letters in ngram = natural language

# Markdown table detection
MDTABLE_PIPE_LINES_MIN = 3       # ≥3 lines with multiple pipes = table
MDTABLE_PIPES_PER_LINE_MIN = 2   # ≥2 '|' per qualifying line

# Heavy markdown bold (e.g., alphabet-A-Z chain "**A**ll **b**right ...")
# These chunks inflate token count vs word count → tight wc-based deadline
# vs slow tok-based generation → frequent deadline miss.
MDBOLD_MIN_WORD_COUNT = 8        # only consider chunks with enough words
MDBOLD_RATIO = 0.5               # bold-wraps per word ≥ 0.5 → bad
MDBOLD_MIN_WRAPS = 6             # OR ≥6 bold-wraps in one chunk

# Single-char / short quoted token spam (e.g., '"W", "M", "N", ...')
# Each `"X"` is 3 tokens (quote, char, quote) → token density blows up.
QUOTED_SHORT_MIN_COUNT = 5       # ≥5 such patterns in a chunk = spam

# Dense comma-separated list (e.g., language dump, photo specs)
# Chunk has many short comma-separated items → high tok/word ratio.
COMMA_LIST_MIN_TOK = 100         # only long chunks
COMMA_LIST_MIN_ITEMS = 10        # ≥10 comma-separated short items
COMMA_LIST_ITEM_MAX_WORDS = 3    # each item ≤ 3 words (short)

# Non-English (non-Latin script) detection
NONLATIN_LETTER_RATIO = 0.30     # ≥30% of letter chars are non-Latin = foreign
NONLATIN_MIN_CHUNK_CHARS = 30    # ignore tiny chunks
# Unicode ranges considered "non-Latin letter"
_NONLATIN_RANGES = (
    (0x0370, 0x03FF),  # Greek + Coptic
    (0x0400, 0x04FF),  # Cyrillic
    (0x0500, 0x052F),  # Cyrillic Supplement
    (0x0530, 0x058F),  # Armenian
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK Unified
    (0xAC00, 0xD7AF),  # Hangul
)


def is_numeric_chunk(text: str, num_token: int, word_count: int) -> bool:
    """Detect chunks dominated by digit strings."""
    if num_token < NUMERIC_MIN_CHUNK_TOK:
        return False
    non_space = re.sub(r"\s", "", text)
    if not non_space:
        return False
    digit_count = sum(1 for c in non_space if c.isdigit())
    digit_ratio = digit_count / len(non_space)
    if digit_ratio >= NUMERIC_DIGIT_RATIO:
        return True
    # Dense token but few words = numeric/string blob
    if (word_count > 0 and num_token / word_count >= NUMERIC_MIN_TOK_PER_WORD
            and digit_count > 20):
        return True
    return False


def _is_nonlatin(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _NONLATIN_RANGES:
        if lo <= cp <= hi:
            return True
    return False


_QUOTED_SHORT_RE = re.compile(r'"[A-Za-z0-9]{1,2}"')


def is_quoted_short_spam(text: str) -> bool:
    """Detect chunks with many `"X"` short-quoted tokens (e.g., letter
    list `"W", "M", "N"...`). Each pattern adds 3 tokens for 1 character."""
    matches = _QUOTED_SHORT_RE.findall(text)
    return len(matches) >= QUOTED_SHORT_MIN_COUNT


def is_dense_comma_list(text: str, num_token: int) -> bool:
    """Detect chunks dominated by short comma-separated items
    (e.g., language list, photo specs)."""
    if num_token < COMMA_LIST_MIN_TOK:
        return False
    items = [s.strip() for s in text.split(',')]
    short_items = [s for s in items if s and len(s.split()) <= COMMA_LIST_ITEM_MAX_WORDS]
    return len(short_items) >= COMMA_LIST_MIN_ITEMS


def is_markdown_bold_dense_chunk(text: str, word_count: int) -> bool:
    """Detect chunks with heavy `**X**` density (e.g. alphabet-A-Z with
    each word forced into markdown bold). Inflates token/word ratio."""
    if word_count < MDBOLD_MIN_WORD_COUNT:
        return False
    wraps = text.count("**") // 2
    if wraps >= MDBOLD_MIN_WRAPS and wraps >= word_count * MDBOLD_RATIO:
        return True
    return False


def is_markdown_table_chunk(text: str) -> bool:
    """Detect markdown table syntax: ≥3 lines with ≥2 pipes each."""
    qualifying = 0
    for line in text.splitlines():
        if line.count("|") >= MDTABLE_PIPES_PER_LINE_MIN:
            qualifying += 1
            if qualifying >= MDTABLE_PIPE_LINES_MIN:
                return True
    return False


def is_non_english_chunk(text: str) -> bool:
    """Detect chunks dominated by non-Latin scripts."""
    if len(text) < NONLATIN_MIN_CHUNK_CHARS:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    nonlatin = sum(1 for c in letters if _is_nonlatin(c))
    return (nonlatin / len(letters)) >= NONLATIN_LETTER_RATIO


def is_repetition_chunk(text: str, num_token: int) -> bool:
    """Detect degenerate-repetition chunks (model stuck in loop).

    Requires BOTH:
      (a) low unique-word ratio (< REPEAT_UNIQUE_WORD_RATIO), AND
      (b) some n-gram (n=2..5) repeats >= REPEAT_MIN_OCCURRENCES.

    Math/LaTeX has many repeated structural symbols but typically high
    unique-word ratio in the natural-language portion, so (a) gates them
    out.
    """
    if num_token < REPEAT_MIN_CHUNK_TOK:
        return False
    words = re.findall(r"\w+", text.lower())
    if len(words) < REPEAT_MIN_OCCURRENCES * 2:
        return False
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio >= REPEAT_UNIQUE_WORD_RATIO:
        return False
    for n in range(REPEAT_NGRAM_MIN, min(REPEAT_NGRAM_MAX,
                                          len(words) // REPEAT_MIN_OCCURRENCES) + 1):
        ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        counter = Counter(ngrams)
        if not counter:
            continue
        most_ngram, most_count = counter.most_common(1)[0]
        if most_count < REPEAT_MIN_OCCURRENCES:
            continue
        # Math/LaTeX guard: require the repeating n-gram to be mostly
        # alphabetic letters, not single-char math variables.
        alpha_chars = sum(1 for c in most_ngram if c.isalpha())
        if alpha_chars / max(1, len(most_ngram.replace(" ", ""))) < REPEAT_NGRAM_ALPHA_RATIO:
            continue
        # Each word in the ngram must be >= 3 chars (filters out math vars)
        ngram_words = most_ngram.split()
        if any(len(w) < 3 for w in ngram_words):
            continue
        return True
    return False


def scan_outputs(pool):
    """Per-rate-dir scan; map rid → prompt via requests.jsonl directly.
    Returns: bad_prompts set, samples by reason, totals."""
    del pool  # no shuffle reconstruction needed when requests.jsonl has prompt
    bad_prompts = set()
    samples = defaultdict(list)
    total_chunks = 0
    total_requests = 0

    for outroot in OUTPUT_DIRS:
        if not outroot.exists():
            continue
        for chunks_path in outroot.rglob("chunks.jsonl"):
            rate_dir = chunks_path.parent
            reqs_path = rate_dir / "requests.jsonl"
            if not reqs_path.exists():
                continue
            # Build rid → prompt map from requests.jsonl
            rid_to_prompt = {}
            with reqs_path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        rid = int(r["request_id"])
                        p = r.get("prompt") or ""
                        if p:  # only if prompt was logged
                            rid_to_prompt[rid] = p
                    except Exception:
                        continue
            if not rid_to_prompt:
                continue  # skip cells without prompt logging
            total_requests += len(rid_to_prompt)

            with chunks_path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    total_chunks += 1
                    try:
                        rid = int(r["request_id"])
                    except (KeyError, ValueError, TypeError):
                        continue
                    prompt = rid_to_prompt.get(rid)
                    if not prompt:
                        continue
                    text = r.get("text", "") or ""
                    num_tok = int(r.get("num_token") or 0)
                    wc = int(r.get("word_count") or 0)
                    is_num = is_numeric_chunk(text, num_tok, wc)
                    is_rep = is_repetition_chunk(text, num_tok)
                    is_mdt = is_markdown_table_chunk(text)
                    is_mdb = is_markdown_bold_dense_chunk(text, wc)
                    is_nel = is_non_english_chunk(text)
                    is_qsp = is_quoted_short_spam(text)
                    is_dcl = is_dense_comma_list(text, num_tok)
                    if not (is_num or is_rep or is_mdt or is_mdb or is_nel
                            or is_qsp or is_dcl):
                        continue
                    bad_prompts.add(prompt)
                    if is_num and len(samples["numeric"]) < 6:
                        samples["numeric"].append((rid, text[:140], prompt[:180]))
                    if is_rep and len(samples["repetition"]) < 6:
                        samples["repetition"].append((rid, text[:140], prompt[:180]))
                    if is_mdt and len(samples["markdown_table"]) < 6:
                        samples["markdown_table"].append((rid, text[:140], prompt[:180]))
                    if is_mdb and len(samples["markdown_bold_dense"]) < 6:
                        samples["markdown_bold_dense"].append((rid, text[:140], prompt[:180]))
                    if is_nel and len(samples["non_english"]) < 6:
                        samples["non_english"].append((rid, text[:140], prompt[:180]))
                    if is_qsp and len(samples["quoted_short_spam"]) < 6:
                        samples["quoted_short_spam"].append((rid, text[:140], prompt[:180]))
                    if is_dcl and len(samples["dense_comma_list"]) < 6:
                        samples["dense_comma_list"].append((rid, text[:140], prompt[:180]))
    return bad_prompts, samples, total_chunks, total_requests


def load_cache_prompts():
    """Read processed_dataset.jsonl in file order."""
    prompts = []
    with CACHE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    return prompts


def shuffled_by_seed(prompts, seed):
    """Reproduce load_or_build_combine_pool's shuffle."""
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite processed_dataset.jsonl")
    args = ap.parse_args()

    cache_prompts = load_cache_prompts()
    pool = shuffled_by_seed(cache_prompts, SEED)  # 'pool' as in run_test.py
    print(f"cache size: {len(cache_prompts)} prompts")

    bad_prompts, samples, total_chunks, total_reqs = scan_outputs(pool)
    print(f"scanned: {total_chunks} chunks across {total_reqs} requests")
    print(f"unique bad prompts (raw, possibly chat-templated): {len(bad_prompts)}")

    # Strip chat template wrapper so we can match against the raw cache.
    # Qwen format: <|im_start|>user\n(.*?)<|im_end|>
    chat_user_pat = re.compile(
        r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL)

    def strip_chat_template(p):
        m = chat_user_pat.search(p)
        if m:
            return m.group(1).strip()
        return p.strip()

    bad_prompts_raw = {strip_chat_template(p) for p in bad_prompts}
    print(f"unique bad prompts (chat-stripped): {len(bad_prompts_raw)}")

    for reason, items in samples.items():
        print(f"\n--- {reason} sample (max 8) — output → prompt ---")
        for rid, text, prompt in items:
            text = text.replace("\n", " ")
            prompt = prompt.replace("\n", " ")
            print(f"  rid={rid:>5d}  OUT: {text[:140]}")
            print(f"         PROMPT: {prompt[:200]}")

    bad_prompt_indices = {i for i, p in enumerate(cache_prompts)
                           if p.strip() in bad_prompts_raw}
    print(f"\ncache indices to remove: {len(bad_prompt_indices)}")

    if not args.apply:
        print(f"\n(dry-run; pass --apply to remove {len(bad_prompt_indices)} prompts)")
        return

    # Apply: rewrite processed_dataset.jsonl, keep order minus bad
    backup = CACHE.with_suffix(CACHE.suffix + ".bak_iter_filter")
    if not backup.exists():
        import shutil
        shutil.copy2(CACHE, backup)
        print(f"backup: {backup}")
    kept = [p for i, p in enumerate(cache_prompts) if i not in bad_prompt_indices]
    tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
    with tmp.open("w") as f:
        for p in kept:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    tmp.replace(CACHE)
    print(f"REWROTE: {len(kept)} kept ({len(cache_prompts) - len(kept)} removed)")


if __name__ == "__main__":
    main()
