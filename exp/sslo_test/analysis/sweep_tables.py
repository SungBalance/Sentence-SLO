#!/usr/bin/env python3
"""Print sweep-progress tables split by (chunk_unit × mode × metric).

Layout: for each chunk_unit detected under output_sweep/, and for each of the 4
metrics (TTFT p50, TTFT p99, chunk-violation, request-violation), emit one
mini-table per mode laid out as (max_num_seqs rows × request_rate columns).
Cells without data are " — ".  Averages across the repeated runs."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("exp/sslo_test/output_sweep")
MODES = ("baseline", "sslo", "sslo_adaptive")
SEQS = (16, 32, 64, 128)
RATES = (0, 4, 16, 32, 64, 128)
NUM_RUNS = 3


def collect(unit_dir: Path):
    """Return (data, complete_cells) for one chunk_unit subtree.
    data[(seqs, rate)][mode][metric] -> list of run values."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    complete = 0
    for seqs in SEQS:
        for rate in RATES:
            for run in range(1, NUM_RUNS + 1):
                p = unit_dir / f"seqs_{seqs}" / f"rate_{rate}" / f"run_{run}" / "summary.json"
                if not p.exists():
                    continue
                try:
                    d = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                complete += 1
                m = d.get("metrics", {})
                for mode in MODES:
                    tt = m.get("ttft", {}).get(mode, {}).get("all", {})
                    if tt.get("p50") is not None:
                        data[(seqs, rate)][mode]["ttft_p50"].append(tt["p50"])
                    if tt.get("p99") is not None:
                        data[(seqs, rate)][mode]["ttft_p99"].append(tt["p99"])
                    sl = m.get("slack", {}).get(mode, {})
                    if sl.get("neg_ratio") is not None:
                        data[(seqs, rate)][mode]["chunk_viol"].append(sl["neg_ratio"])
                    cm = m.get("slo_compliance", {}).get(mode, {})
                    if cm.get("rate") is not None:
                        data[(seqs, rate)][mode]["req_viol"].append(1 - cm["rate"])
    return data, complete


def avg(xs):
    return statistics.fmean(xs) if xs else None


def fmt(x, prec):
    return f"{x:.{prec}f}" if x is not None else "—"


def emit_matrix(data, mode, metric_key, prec, col_w=10):
    head = f"{'seqs\\rate':>10} " + "".join(f"{r:>{col_w}}" for r in RATES)
    print(head)
    print("-" * len(head))
    for seqs in SEQS:
        line = f"{seqs:>10} "
        for rate in RATES:
            v = avg(data[(seqs, rate)][mode][metric_key])
            line += f"{fmt(v, prec):>{col_w}}"
        print(line)


METRICS = (
    ("TTFT p50 (s)",                              "ttft_p50",   3),
    ("TTFT p99 (s)",                              "ttft_p99",   3),
    ("Chunk-level violation ratio (chunk_idx>=1)", "chunk_viol", 4),
    ("Request-level violation ratio",             "req_viol",   4),
)


def main():
    if not ROOT.exists():
        print(f"No output dir at {ROOT}")
        return
    units = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])
    if not units:
        print(f"No chunk_unit subdirectories under {ROOT}")
        return

    grand_total = len(units) * len(SEQS) * len(RATES) * NUM_RUNS
    grand_complete = 0
    per_unit = {}
    for unit in units:
        data, complete = collect(ROOT / unit)
        per_unit[unit] = (data, complete)
        grand_complete += complete

    print(f"Sweep progress: {grand_complete}/{grand_total} cells complete "
          f"({100 * grand_complete // grand_total}%)")

    for unit in units:
        data, complete = per_unit[unit]
        unit_total = len(SEQS) * len(RATES) * NUM_RUNS
        print(f"\n{'=' * 72}\nchunk_unit = {unit}  ({complete}/{unit_total})\n{'=' * 72}")
        for title, key, prec in METRICS:
            print(f"\n## {title}")
            for mode in MODES:
                print(f"\n### {mode}")
                emit_matrix(data, mode, key, prec)


if __name__ == "__main__":
    main()
