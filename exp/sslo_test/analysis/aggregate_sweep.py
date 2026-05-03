#!/usr/bin/env python3
"""Aggregate full sweep: max_num_seqs × request_rate × N trials × 3 modes.

Reads ${base_output}/rate_${rate}_seqs_${seqs}/run_{i}/seqs_${seqs}/summary.json
for every cell × run, then prints per-metric 2D tables (rate × seqs) showing
mean ± stddev for each mode (baseline / sslo / sslo_adaptive).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODES = ("baseline", "sslo", "sslo_adaptive")
RATES = (0, 4, 8, 16)
SEQS = (32, 64, 128)

METRICS = (
    ("ttft_p50",        "h2_ttft_p50_{mode}",         1.0,  "{:5.2f}", "TTFT p50 (s)"),
    ("queue_stall_p50", "queue_stall_p50_{mode}",     1.0,  "{:5.2f}", "queue stall p50 (s)"),
    ("tpot_p50",        "h4_tpot_p50_{mode}",         1000, "{:5.1f}", "TPOT p50 (ms)"),
    ("neg_slack_ratio", "h3_slack_neg_ratio_{mode}",  100,  "{:5.3f}", "neg slack ratio (%)"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-output", default="exp/sslo_test/output_sweep")
    p.add_argument("--num-runs", type=int, required=True)
    return p.parse_args()


def load_cell(base: Path, rate: int, seqs: int, num_runs: int) -> dict:
    """Returns per-mode lists of metric values across N runs."""
    cell = {m: {key: [] for key, _, _, _, _ in METRICS} for m in MODES}
    for i in range(1, num_runs + 1):
        path = base / f"rate_{rate}_seqs_{seqs}" / f"run_{i}" / f"seqs_{seqs}" / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for mode in MODES:
            for key, tmpl, _scale, _fmt, _label in METRICS:
                v = s.get(tmpl.format(mode=mode))
                if v is not None:
                    cell[mode][key].append(float(v))
    return cell


def fmt(values: list[float], scale: float, fmt_s: str) -> str:
    if not values:
        return "    n/a    "
    if len(values) == 1:
        return fmt_s.format(values[0] * scale) + "  (n=1)"
    m = st.mean(values) * scale
    sd = st.stdev(values) * scale
    return f"{fmt_s.format(m)}±{fmt_s.format(sd).strip()}"


def main() -> None:
    args = parse_args()
    base = Path(args.base_output)

    # Load every cell
    cells = {(r, s): load_cell(base, r, s, args.num_runs) for r in RATES for s in SEQS}

    # Per-metric table: rows=seqs, cols=rate, with 3 sub-rows per cell (one per mode)
    for key, _tmpl, scale, fmt_s, label in METRICS:
        print(f"\n=== {label} ===")
        header = f"{'':10s}" + "".join(f"  rate={r:<3d}{'':14s}" for r in RATES)
        print(header)
        for s in SEQS:
            for mode in MODES:
                line = f"seqs={s:<3d} {mode[:11]:11s}"  # one row per (seqs, mode)
                for r in RATES:
                    cell = cells[(r, s)][mode][key]
                    line += "  " + fmt(cell, scale, fmt_s) + "  "
                print(line)
            print()  # blank line between seqs blocks


if __name__ == "__main__":
    main()
