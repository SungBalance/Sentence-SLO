#!/usr/bin/env python3
"""Calibrate the dialogue pool's --min-response-chars threshold.

The filter keeps dialogues whose *reference* assistant answer (the message
build_dialogue_prompts drops) is long, on the assumption that those prompts
also make the model generate long answers -- which is what puts KV under
pressure. That assumption is what this script measures.

Labels come from past sweeps: every requests.jsonl row carries the rendered
prompt and the tokens actually generated, and a prompt served many times
gives a stable per-prompt median. Joining those labels back to the cache
needs a key that survives chat-template rendering, so we key on the full
sequence of user turns -- keying on the last user turn alone collides on
generic follow-ups ("continue", "ok") and inflates the estimate.

Usage:
  python3 exp/run_sslo/analysis/calibrate_response_filter.py \
      exp/tools/dataset_cache/dialogues_wildchat_conv-en-nocode.jsonl
"""
from __future__ import annotations

import collections
import glob
import json
import statistics
import sys
from pathlib import Path

SWEEP_GLOB = ("exp/run_sslo/output_sweep_v2/phase*/sentence/*/*/*/cap*/*/"
              "run_*/rate_*/requests.jsonl")
THRESHOLDS = (0, 1200, 2000, 2400, 2800, 3000, 4000)


def user_key(messages: list[dict]) -> str:
    """Join every user turn -- stable across chat-template rendering."""
    return "␟".join(m["content"] for m in messages
                         if m.get("role") == "user")


def measured_labels() -> dict[str, float]:
    """Rendered prompt -> median generated tokens, over all past sweeps."""
    seen: dict[str, list[int]] = collections.defaultdict(list)
    for path in glob.glob(SWEEP_GLOB):
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("prompt") and row.get("num_output_tokens"):
                    seen[row["prompt"]].append(row["num_output_tokens"])
    return {p: statistics.median(v) for p, v in seen.items()}


def main() -> None:
    cache = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "exp/tools/dataset_cache/"
                 "dialogues_wildchat_conv-en-nocode.jsonl")
    labels = measured_labels()
    print(f"labels: {len(labels):,} distinct prompts")

    # The rendered prompt is the join key on the label side, but the cache
    # holds raw messages, so map user-turn key -> generated length by
    # re-rendering the cache and matching. Rendering needs the tokenizer the
    # sweep used, so fall back to a substring match when it is unavailable.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B",
                                        trust_remote_code=True)

    rows = []
    with cache.open() as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            trimmed = list(messages)
            reference = None
            while trimmed and trimmed[-1].get("role") != "user":
                dropped = trimmed.pop()
                if dropped.get("role") == "assistant" and reference is None:
                    reference = dropped.get("content")
            if not trimmed or reference is None:
                continue
            rendered = tok.apply_chat_template(trimmed, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            if rendered in labels:
                rows.append((len(reference), labels[rendered],
                             user_key(trimmed)))

    print(f"joined: {len(rows):,} dialogues have both a reference answer "
          f"and a measured generation length")
    if not rows:
        return
    # Collisions on the join key would mean one label standing in for several
    # dialogues; report it rather than silently averaging them.
    keys = collections.Counter(r[2] for r in rows)
    dupes = sum(c - 1 for c in keys.values() if c > 1)
    print(f"join-key collisions: {dupes} "
          f"(non-zero means the key is too coarse)")

    print(f"\n{'min ref chars':>14s} {'kept':>7s} {'gen p50':>9s} "
          f"{'gen p90':>9s} {'gen mean':>9s}")
    for th in THRESHOLDS:
        kept = sorted(gen for ref, gen, _ in rows if ref >= th)
        if len(kept) < 10:
            continue
        print(f"{th:14d} {len(kept):7d} {kept[len(kept) // 2]:9.0f} "
              f"{kept[9 * len(kept) // 10]:9.0f} "
              f"{statistics.mean(kept):9.0f}")


if __name__ == "__main__":
    main()
