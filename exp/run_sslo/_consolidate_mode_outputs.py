#!/usr/bin/env python3
"""Merge per-mode output files into shared per-cell files.

run_test.py writes mode-suffixed JSONL files (e.g. requests_sslo.jsonl).
After each mode completes, this script appends those rows — tagged with
{"mode": <mode>, ...} — into the cell's shared files (requests.jsonl,
chunks.jsonl, scheduler_stats.jsonl, offload_log.jsonl) and removes the
per-mode source files.

The per-mode sslo_config_<mode>.json sidecar (a single dict, not jsonl) is
folded into a cell-shared sslo_config.json keyed by mode.

Usage:
  python3 _consolidate_mode_outputs.py <out_dir> <mode>
  python3 _consolidate_mode_outputs.py --validity-csv <sweep_root>
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

FILE_MAP = (
    ("requests",         "requests.jsonl"),
    ("chunks",           "chunks.jsonl"),
    ("_stats",           "scheduler_stats.jsonl"),
    ("_offload_log",     "offload_log.jsonl"),
)


def consolidate_sslo_config(out_dir: str, mode: str) -> None:
    src = os.path.join(out_dir, f"sslo_config_{mode}.json")
    if not os.path.exists(src):
        return
    dst = os.path.join(out_dir, "sslo_config.json")
    merged: dict = {}
    if os.path.exists(dst):
        try:
            merged = json.loads(open(dst).read()) or {}
        except json.JSONDecodeError:
            merged = {}
    with open(src) as f:
        merged[mode] = json.loads(f.read() or "{}")
    with open(dst, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")
    os.remove(src)


VALIDITY_CSV_FIELDS = (
    "run_id", "mode", "validity_pass", "no_harm_pass", "invalid_reason"
)


def consolidate_validity_csv(sweep_root: str | os.PathLike) -> Path:
    """Walk every summary.json under sweep_root and write validity_checks.csv.

    Each per-mode summary.json is one row; invalid_reason is `;`-joined.
    Returns the output path.
    """
    root = Path(sweep_root)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        val = summary.get("validity") or {}
        meta = summary.get("run_meta") or {}
        cfg = summary.get("config") or {}
        run_id = meta.get("run_id") or str(
            summary_path.parent.relative_to(root))
        mode = (
            meta.get("policy")
            or cfg.get("run_kind")
            or summary_path.parent.parent.name
        )
        rows.append({
            "run_id": run_id,
            "mode": mode,
            "validity_pass": val.get("validity_pass"),
            "no_harm_pass": val.get("no_harm_pass"),
            "invalid_reason": ";".join(val.get("invalid_reason", []) or []),
        })

    out_path = root / "validity_checks.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(VALIDITY_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--validity-csv":
        out = consolidate_validity_csv(sys.argv[2])
        print(f"wrote {out}")
        return
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <out_dir> <mode>", file=sys.stderr)
        print(f"   or: {sys.argv[0]} --validity-csv <sweep_root>",
              file=sys.stderr)
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
    consolidate_sslo_config(out_dir, mode)


if __name__ == "__main__":
    main()
