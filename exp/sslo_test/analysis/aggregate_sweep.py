#!/usr/bin/env python3
"""Aggregate full sweep: seqs x rate x N trials x modes.

Reads ${base_output}/seqs_${seqs}/rate_${rate}/run_${i}/summary.json
for every cell x run, then prints per-metric 2D tables (seqs x rate).
"""
from __future__ import annotations

import argparse
import json
import signal
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics_utils import DISPLAY_GROUPS, MODES_DEFAULT, fmt_pair, lookup, parse_modes_arg


RATES = (0, 4, 8, 16)
SEQS = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-output", default="exp/sslo_test/output_sweep")
    p.add_argument("--num-runs", type=int, required=True)
    p.add_argument("--modes", default=",".join(MODES_DEFAULT))
    return p.parse_args()


def load_cell(base: Path, seqs: int, rate: int, num_runs: int, modes: tuple[str, ...]) -> dict:
    cell: dict[tuple, dict[str, list[float]]] = {}
    for _group_name, specs in DISPLAY_GROUPS:
        for spec in specs:
            key = (spec.path, spec.field)
            if key not in cell:
                cell[key] = {m: [] for m in modes}
    for i in range(1, num_runs + 1):
        path = base / f"seqs_{seqs}" / f"rate_{rate}" / f"run_{i}" / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for _group_name, specs in DISPLAY_GROUPS:
            for spec in specs:
                key = (spec.path, spec.field)
                for mode in modes:
                    v = lookup(s, spec.path, spec.field, mode)
                    if v is not None:
                        cell[key][mode].append(float(v))
    return cell


def fmt(values: list[float], scale: float, fmt_s: str) -> str:
    if not values:
        return "    n/a    "
    if len(values) == 1:
        return fmt_s.format(values[0] * scale) + "  (n=1)"
    m = st.mean(values) * scale
    sd = st.stdev(values) * scale
    return f"{fmt_s.format(m)}+/-{fmt_s.format(sd).strip()}"


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args()
    modes = parse_modes_arg(args.modes)
    base = Path(args.base_output)

    cells = {(s, r): load_cell(base, s, r, args.num_runs, modes) for s in SEQS for r in RATES}

    for group_name, specs in DISPLAY_GROUPS:
        print(f"\n=== {group_name} ===")
        for spec in specs:
            key = (spec.path, spec.field)
            print(f"\n  {spec.label}")
            header = f"{'':15s}" + "".join(f"  rate={r:<3d}{'':12s}" for r in RATES)
            print(header)
            for s in SEQS:
                for mode in modes:
                    line = f"seqs={s:<3d} {mode[:11]:11s}"
                    for r in RATES:
                        cell_val = cells[(s, r)][key][mode]
                        line += "  " + fmt(cell_val, spec.scale, spec.fmt) + "  "
                    print(line)
                print()


if __name__ == "__main__":
    main()
