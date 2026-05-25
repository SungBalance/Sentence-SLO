#!/usr/bin/env python3
"""Adapt run_sslo chunks.jsonl rows to the schema this folder consumes.

The user copies one cell's chunks.jsonl into this folder (default name
`input_chunks.jsonl`). Each input row carries `request_id`, `chunk_idx`,
`text`, `num_words`, `num_token` from the SSLO runtime; we reshape them
into the row dict that measure_audio_duration.py expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from common import read_jsonl


def transform_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the per-chunk record for the synthesis step, or None to skip."""
    text = (row.get("text") or "").strip()
    if not text:
        return None
    request_id = str(row.get("request_id") or "")
    chunk_idx = int(row.get("chunk_idx") or 0)
    if not request_id:
        return None
    return {
        "chunk_id": f"{request_id}-{chunk_idx}",
        "request_id": request_id,
        "chunk_idx": chunk_idx,
        "text": text,
        "word_count": int(row.get("num_words") or 0),
        "num_token": int(row.get("num_token") or 0),
    }


def load(
    path: str | Path = "input_chunks.jsonl",
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = transform_row(r)
        if rec is None:
            continue
        out.append(rec)
        if max_rows is not None and len(out) >= max_rows:
            break
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input_chunks.jsonl")
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()
    rows = load(args.input)
    print(f"loaded {len(rows)} chunks from {args.input}")
    for r in rows[: args.head]:
        snippet = r["text"][:60].replace("\n", " ")
        print(f"  {r['chunk_id']:>10}  words={r['word_count']:>3}  "
              f"tokens={r['num_token']:>3}  text={snippet!r}")
