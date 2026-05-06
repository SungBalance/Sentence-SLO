#!/usr/bin/env python3
"""Merge per-mode output files into shared per-cell files.

run_test.py writes mode-suffixed JSONL files (e.g. requests_sslo.jsonl).
After each mode completes, this script appends those rows — tagged with
{"mode": <mode>, ...} — into the cell's shared files (requests.jsonl,
chunks.jsonl, scheduler_stats.jsonl, offload_log.jsonl) and removes the
per-mode source files.

Usage:
  python3 _consolidate_mode_outputs.py <out_dir> <mode>
"""
from __future__ import annotations

import json
import os
import sys

FILE_MAP = (
    ("requests",         "requests.jsonl"),
    ("chunks",           "chunks.jsonl"),
    ("_stats",           "scheduler_stats.jsonl"),
    ("_offload_log",     "offload_log.jsonl"),
)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <out_dir> <mode>", file=sys.stderr)
        sys.exit(2)
    out_dir, mode = sys.argv[1], sys.argv[2]
    for src_prefix, dst_name in FILE_MAP:
        src = os.path.join(out_dir, f"{src_prefix}_{mode}.jsonl")
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, dst_name)
        with open(src) as f_in, open(dst, "a") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                f_out.write(json.dumps({"mode": mode, **row}) + "\n")
        os.remove(src)


if __name__ == "__main__":
    main()
