"""Apply classification result to the cache.

Reads classification.jsonl (one row per request with `prompt` and `reasons`).
The classification prompts are chat-templated; cache prompts are raw.
We strip the chat-template wrapper (Qwen3-family <|im_start|>user\\n...
<|im_end|>) before matching so raw cache prompts compare correctly.

Backs up the original cache to combine_2.jsonl.prefilter.bak before
overwriting.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CACHE = Path("/workspace/mlsys/exp/tools/dataset_cache/combine_2.jsonl")

# Qwen3 chat template wraps user text between these markers.
USER_RE = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def strip_chat_template(text: str) -> str:
    """Extract raw user prompt from a chat-templated string. Returns the
    text unchanged if the template markers aren't found."""
    matches = USER_RE.findall(text)
    if matches:
        # Multi-turn: take the last user turn (single-turn datasets won't
        # have more, so this matches when there's exactly one).
        return matches[-1]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", required=True,
                    help="classification.jsonl produced by classify.py")
    ap.add_argument("--cache", default=str(CACHE),
                    help=f"cache file to filter (default {CACHE})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Collect bad prompts (strip chat template to match raw cache form)
    bad_prompts: set[str] = set()
    reason_counts: Counter = Counter()
    with open(args.classification) as f:
        for line in f:
            row = json.loads(line)
            if row.get("reasons"):
                bad_prompts.add(strip_chat_template(row["prompt"]))
                for r in row["reasons"]:
                    reason_counts[r] += 1

    print(f"bad prompts to remove: {len(bad_prompts)}")
    print(f"  reason breakdown: {dict(reason_counts)}")

    # Load cache, partition
    cache_path = Path(args.cache)
    rows = [json.loads(line) for line in cache_path.open() if line.strip()]
    kept = [r for r in rows if r.get("prompt") not in bad_prompts]
    removed = len(rows) - len(kept)
    print(f"cache: total={len(rows)}, kept={len(kept)}, removed={removed}")

    if args.dry_run:
        print("--dry-run: no files written")
        return

    backup = cache_path.with_suffix(".jsonl.prefilter.bak")
    if not backup.exists():
        cache_path.rename(backup)
        print(f"backed up to {backup.name}")
    with cache_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} kept prompts to {cache_path.name}")


if __name__ == "__main__":
    main()
