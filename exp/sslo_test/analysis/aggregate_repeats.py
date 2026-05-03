#!/usr/bin/env python3
"""Aggregate metrics across N repeated runs of run_single.sh.

Reads ${output_root}/run_{i}/seqs_${max_num_seqs}/summary.json and the chunks
jsonl files for each run, then prints mean ± stddev for the key metrics across
all runs, for each of the 3 modes (baseline, sslo, sslo_adaptive).

Usage:
  python3 exp/sslo_test/analysis/aggregate_repeats.py \
      --output-root exp/sslo_test/output --max-num-seqs 64 --num-runs 3
"""
from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import extra_metric_fields
from jsonl_utils import read_jsonl


MODES = ("baseline", "sslo", "sslo_adaptive")
METRICS = (
    ("ttft_p50", "h2_ttft_p50_{mode}", 1.0, "{:.3f}", "TTFT p50 (s)"),
    ("ttft_p90", "h2_ttft_p90_{mode}", 1.0, "{:.3f}", "TTFT p90 (s)"),
    ("ttft_p99", "h2_ttft_p99_{mode}", 1.0, "{:.3f}", "TTFT p99 (s)"),
    ("ttft_max", "h2_ttft_max_{mode}", 1.0, "{:.3f}", "TTFT max (s)"),
    ("queue_stall_p50", "queue_stall_p50_{mode}", 1.0, "{:.3f}", "queue stall p50 (s)"),
    ("queue_stall_p90", "queue_stall_p90_{mode}", 1.0, "{:.3f}", "queue stall p90 (s)"),
    ("queue_stall_p99", "queue_stall_p99_{mode}", 1.0, "{:.3f}", "queue stall p99 (s)"),
    ("queue_stall_max", "queue_stall_max_{mode}", 1.0, "{:.3f}", "queue stall max (s)"),
    ("tpot_p50", "h4_tpot_p50_{mode}", 1000, "{:.2f}", "TPOT p50 (ms)"),
    ("neg_slack_ratio", "h3_slack_neg_ratio_{mode}", 100, "{:.4f}", "neg slack ratio (%)"),
    ("slo_compliance_rate", "slo_compliance_rate_{mode}", 100, "{:.2f}", "SLO compliance (%)"),
    ("slo_compliance_count", "slo_compliance_count_{mode}", 1.0, "{:.1f}", "SLO compliant reqs"),
    ("slo_total_requests", "slo_total_requests_{mode}", 1.0, "{:.1f}", "SLO total reqs"),
    ("neg_slack_mag_p50", "neg_slack_mag_p50_{mode}", 1.0, "{:.3f}", "neg slack mag p50 (s)"),
    ("neg_slack_mag_p90", "neg_slack_mag_p90_{mode}", 1.0, "{:.3f}", "neg slack mag p90 (s)"),
    ("neg_slack_mag_p99", "neg_slack_mag_p99_{mode}", 1.0, "{:.3f}", "neg slack mag p99 (s)"),
    ("neg_slack_mag_max", "neg_slack_mag_max_{mode}", 1.0, "{:.3f}", "neg slack mag max (s)"),
    ("running_mean", "running_mean_{mode}", 1.0, "{:.2f}", "running mean"),
    ("running_p50", "running_p50_{mode}", 1.0, "{:.2f}", "running p50"),
    ("running_p99", "running_p99_{mode}", 1.0, "{:.2f}", "running p99"),
    ("combined_mean", "combined_mean_{mode}", 1.0, "{:.2f}", "combined mean"),
    ("combined_p50", "combined_p50_{mode}", 1.0, "{:.2f}", "combined p50"),
    ("combined_p99", "combined_p99_{mode}", 1.0, "{:.2f}", "combined p99"),
    ("pending_time_p50", "pending_time_p50_{mode}", 1.0, "{:.3f}", "pending time p50 (s)"),
    ("pending_time_p90", "pending_time_p90_{mode}", 1.0, "{:.3f}", "pending time p90 (s)"),
    ("pending_time_p99", "pending_time_p99_{mode}", 1.0, "{:.3f}", "pending time p99 (s)"),
    (
        "pending_intervals_p50", "pending_intervals_p50_{mode}", 1.0,
        "{:.2f}", "pending intervals p50",
    ),
    (
        "pending_intervals_p90", "pending_intervals_p90_{mode}", 1.0,
        "{:.2f}", "pending intervals p90",
    ),
    (
        "inter_chunk_delay_p50", "inter_chunk_delay_p50_{mode}", 1.0,
        "{:.3f}", "inter chunk delay p50 (s)",
    ),
    (
        "inter_chunk_delay_p90", "inter_chunk_delay_p90_{mode}", 1.0,
        "{:.3f}", "inter chunk delay p90 (s)",
    ),
    (
        "inter_chunk_delay_p99", "inter_chunk_delay_p99_{mode}", 1.0,
        "{:.3f}", "inter chunk delay p99 (s)",
    ),
    (
        "inter_chunk_delay_max", "inter_chunk_delay_max_{mode}", 1.0,
        "{:.3f}", "inter chunk delay max (s)",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="exp/sslo_test/output")
    p.add_argument("--max-num-seqs", type=int, required=True)
    p.add_argument("--num-runs", type=int, required=True)
    return p.parse_args()


def neg_chunks(path: Path) -> tuple[int, int]:
    """Return (neg_count, total_count) from a chunks jsonl."""
    rows = read_jsonl(path)
    neg = sum(1 for r in rows if r.get("cumulative_slack") is not None
              and r["cumulative_slack"] < 0)
    return neg, len(rows)


def fmt_pair(values: list[float], scale: float = 1.0, fmt: str = "{:.4f}") -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return fmt.format(values[0] * scale) + "  (n=1)"
    mean = statistics.mean(values) * scale
    stdev = statistics.stdev(values) * scale
    return f"{fmt.format(mean)} ± {fmt.format(stdev)}"


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args()
    root = Path(args.output_root)
    cfg_dir = f"seqs_{args.max_num_seqs}"

    # Collect per-mode metrics across runs
    per_mode_metrics: dict[str, dict[str, list[float]]] = {
        m: {key: [] for key, *_ in METRICS} for m in MODES
    }
    per_mode_neg: dict[str, list[tuple[int, int]]] = {m: [] for m in MODES}

    for i in range(1, args.num_runs + 1):
        run_dir = root / f"run_{i}" / cfg_dir
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"missing: {summary_path}")
            continue
        s = json.loads(summary_path.read_text())
        s = {**s, **extra_metric_fields(run_dir, args.max_num_seqs)}
        for mode in MODES:
            for key, tmpl, _scale, _fmt, _label in METRICS:
                val = s.get(tmpl.format(mode=mode))
                if val is not None:
                    per_mode_metrics[mode][key].append(float(val))
            chunks_path = run_dir / f"{mode}_chunks.jsonl"
            if chunks_path.exists():
                per_mode_neg[mode].append(neg_chunks(chunks_path))

    # Print table
    print(f"\n=== {args.num_runs}-run aggregate (max_num_seqs={args.max_num_seqs}) ===\n")
    header = f"{'metric':<22s}" + "".join(f"{m:>26s}" for m in MODES)
    print(header)
    print("-" * len(header))

    for key, _tmpl, scale, fmt, label in METRICS:
        cells = [fmt_pair(per_mode_metrics[m][key], scale=scale, fmt=fmt) for m in MODES]
        print(f"{label:<22s}" + "".join(f"{c:>26s}" for c in cells))

    # Negative slack chunk absolute counts
    print()
    print(f"{'neg slack chunks':<22s}" + "".join(
        f"{('  '.join(f'{n}/{t}' for n, t in per_mode_neg[m]) or 'n/a'):>26s}"
        for m in MODES
    ))
    print()
    neg_total_cells = []
    for mode in MODES:
        if per_mode_neg[mode]:
            mean_neg = statistics.mean([n for n, _t in per_mode_neg[mode]])
            mean_total = statistics.mean([t for _n, t in per_mode_neg[mode]])
            neg_total_cells.append(f"{mean_neg:.1f} / {mean_total:.0f}")
        else:
            neg_total_cells.append("n/a")
    print(f"{'mean neg / total':<22s}" + "".join(
        f"{cell:>26s}" for cell in neg_total_cells
    ))


if __name__ == "__main__":
    main()
