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
import signal
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import extra_metric_fields


MODES = ("baseline", "sslo", "sslo_adaptive")
RATES = (0, 4, 8, 16)
SEQS = (32, 64, 128)

METRICS = (
    ("ttft_p50",        "h2_ttft_p50_{mode}",         1.0,  "{:5.2f}", "TTFT p50 (s)"),
    ("ttft_p90",        "h2_ttft_p90_{mode}",         1.0,  "{:5.2f}", "TTFT p90 (s)"),
    ("ttft_p99",        "h2_ttft_p99_{mode}",         1.0,  "{:5.2f}", "TTFT p99 (s)"),
    ("ttft_max",        "h2_ttft_max_{mode}",         1.0,  "{:5.2f}", "TTFT max (s)"),
    ("queue_stall_p50", "queue_stall_p50_{mode}",     1.0,  "{:5.2f}", "queue stall p50 (s)"),
    ("queue_stall_p90", "queue_stall_p90_{mode}",     1.0,  "{:5.2f}", "queue stall p90 (s)"),
    ("queue_stall_p99", "queue_stall_p99_{mode}",     1.0,  "{:5.2f}", "queue stall p99 (s)"),
    ("queue_stall_max", "queue_stall_max_{mode}",     1.0,  "{:5.2f}", "queue stall max (s)"),
    ("tpot_p50",        "h4_tpot_p50_{mode}",         1000, "{:5.1f}", "TPOT p50 (ms)"),
    ("neg_slack_ratio", "h3_slack_neg_ratio_{mode}",  100,  "{:5.3f}", "neg slack ratio (%)"),
    ("slo_compliance_rate", "slo_compliance_rate_{mode}", 100, "{:5.1f}", "SLO compliance (%)"),
    ("slo_compliance_count", "slo_compliance_count_{mode}", 1.0, "{:5.1f}", "SLO compliant reqs"),
    ("slo_total_requests", "slo_total_requests_{mode}", 1.0, "{:5.1f}", "SLO total reqs"),
    ("neg_slack_mag_p50", "neg_slack_mag_p50_{mode}", 1.0, "{:5.2f}", "neg slack mag p50 (s)"),
    ("neg_slack_mag_p90", "neg_slack_mag_p90_{mode}", 1.0, "{:5.2f}", "neg slack mag p90 (s)"),
    ("neg_slack_mag_p99", "neg_slack_mag_p99_{mode}", 1.0, "{:5.2f}", "neg slack mag p99 (s)"),
    ("neg_slack_mag_max", "neg_slack_mag_max_{mode}", 1.0, "{:5.2f}", "neg slack mag max (s)"),
    ("running_mean", "running_mean_{mode}", 1.0, "{:5.1f}", "running mean"),
    ("running_p50", "running_p50_{mode}", 1.0, "{:5.1f}", "running p50"),
    ("running_p99", "running_p99_{mode}", 1.0, "{:5.1f}", "running p99"),
    ("combined_mean", "combined_mean_{mode}", 1.0, "{:5.1f}", "combined mean"),
    ("combined_p50", "combined_p50_{mode}", 1.0, "{:5.1f}", "combined p50"),
    ("combined_p99", "combined_p99_{mode}", 1.0, "{:5.1f}", "combined p99"),
    ("pending_time_p50", "pending_time_p50_{mode}", 1.0, "{:5.2f}", "pending time p50 (s)"),
    ("pending_time_p90", "pending_time_p90_{mode}", 1.0, "{:5.2f}", "pending time p90 (s)"),
    ("pending_time_p99", "pending_time_p99_{mode}", 1.0, "{:5.2f}", "pending time p99 (s)"),
    (
        "pending_intervals_p50", "pending_intervals_p50_{mode}", 1.0,
        "{:5.1f}", "pending intervals p50",
    ),
    (
        "pending_intervals_p90", "pending_intervals_p90_{mode}", 1.0,
        "{:5.1f}", "pending intervals p90",
    ),
    (
        "inter_chunk_delay_p50", "inter_chunk_delay_p50_{mode}", 1.0,
        "{:5.2f}", "inter chunk delay p50 (s)",
    ),
    (
        "inter_chunk_delay_p90", "inter_chunk_delay_p90_{mode}", 1.0,
        "{:5.2f}", "inter chunk delay p90 (s)",
    ),
    (
        "inter_chunk_delay_p99", "inter_chunk_delay_p99_{mode}", 1.0,
        "{:5.2f}", "inter chunk delay p99 (s)",
    ),
    (
        "inter_chunk_delay_max", "inter_chunk_delay_max_{mode}", 1.0,
        "{:5.2f}", "inter chunk delay max (s)",
    ),
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
        run_dir = path.parent
        s = json.loads(path.read_text())
        s = {**s, **extra_metric_fields(run_dir, seqs)}
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
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
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
