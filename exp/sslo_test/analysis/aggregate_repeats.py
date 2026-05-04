#!/usr/bin/env python3
"""Aggregate metrics across N repeated runs.

Usage:
  python3 exp/sslo_test/analysis/aggregate_repeats.py \
      --output-root exp/sslo_test/output_sweep/sentence/seqs_64/rate_4 \
      --max-num-seqs 64 --num-runs 3
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics_utils import DISPLAY_GROUPS, MODES_DEFAULT, fmt_pair, lookup, parse_modes_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="exp/sslo_test/output")
    p.add_argument("--max-num-seqs", type=int, required=True)
    p.add_argument("--num-runs", type=int, required=True)
    p.add_argument("--modes", default=",".join(MODES_DEFAULT))
    return p.parse_args()


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args()
    modes = parse_modes_arg(args.modes)
    root = Path(args.output_root)

    per_mode_values: dict[tuple, dict[str, list[float]]] = {}
    for _group_name, specs in DISPLAY_GROUPS:
        for spec in specs:
            key = (spec.path, spec.field)
            if key not in per_mode_values:
                per_mode_values[key] = {m: [] for m in modes}

    for i in range(1, args.num_runs + 1):
        summary_path = root / f"run_{i}" / "summary.json"
        if not summary_path.exists():
            print(f"missing: {summary_path}")
            continue
        s = json.loads(summary_path.read_text())
        for _group_name, specs in DISPLAY_GROUPS:
            for spec in specs:
                key = (spec.path, spec.field)
                for mode in modes:
                    val = lookup(s, spec.path, spec.field, mode)
                    if val is not None:
                        per_mode_values[key][mode].append(float(val))

    print(f"\n=== {args.num_runs}-run aggregate (max_num_seqs={args.max_num_seqs}) ===\n")

    for group_name, specs in DISPLAY_GROUPS:
        print(f"\n=== {group_name} ===")
        header = f"{'metric':<30s}" + "".join(f"{m:>26s}" for m in modes)
        print(header)
        print("-" * len(header))
        for spec in specs:
            key = (spec.path, spec.field)
            cells = [
                fmt_pair(per_mode_values[key][m], scale=spec.scale, fmt=spec.fmt)
                for m in modes
            ]
            print(f"{spec.label:<30s}" + "".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
