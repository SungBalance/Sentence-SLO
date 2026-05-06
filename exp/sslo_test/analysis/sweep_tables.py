#!/usr/bin/env python3
"""Print sweep-progress tables split by (label × chunk_unit × mode × metric).

Layout: for each label found under output_sweep/, for each chunk_unit, and
for each of the four headline metrics (TTFT p50, TTFT p99, chunk-violation,
request-violation), emit one mini-table per mode laid out as
(max_num_seqs rows × request_rate columns). Cells without data are " — ".
Averages across the repeated runs.

Discovers labels, units, seqs, rates, runs from the filesystem so it adapts
to whatever shape the sweep produced.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("exp/sslo_test/output_sweep")
DEFAULT_MODES = ("baseline", "sslo", "sslo_adaptive")
METRICS = (
    ("TTFT p50 (s)",                              "ttft_p50",   3),
    ("TTFT p99 (s)",                              "ttft_p99",   3),
    ("Chunk-level violation ratio (chunk_idx>=1)", "chunk_viol", 4),
    ("Request-level violation ratio",             "req_viol",   4),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", default=str(DEFAULT_ROOT))
    p.add_argument(
        "--modes", default=",".join(DEFAULT_MODES),
        help="Comma-separated mode list to render columns for.")
    return p.parse_args()


def parse_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def collect_unit(unit_dir: Path, modes: tuple[str, ...]):
    """data[(seqs, rate)][mode][metric] -> list of run values."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seqs_set: set[int] = set()
    rates_set: set[int] = set()
    complete = 0
    for path in sorted(unit_dir.glob("seqs_*/rate_*/run_*/summary.json")):
        seqs = parse_int_suffix(path.parents[2].name, "seqs_")
        rate = parse_int_suffix(path.parents[1].name, "rate_")
        if seqs is None or rate is None:
            continue
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        complete += 1
        seqs_set.add(seqs)
        rates_set.add(rate)
        m = d.get("metrics", {})
        for mode in modes:
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
    return data, complete, sorted(seqs_set), sorted(rates_set)


def avg(xs):
    return statistics.fmean(xs) if xs else None


def fmt(x, prec):
    return f"{x:.{prec}f}" if x is not None else "—"


def emit_matrix(data, mode, metric_key, prec, seqs_list, rates_list, col_w=10):
    head = f"{'seqs\\rate':>10} " + "".join(f"{r:>{col_w}}" for r in rates_list)
    print(head)
    print("-" * len(head))
    for seqs in seqs_list:
        line = f"{seqs:>10} "
        for rate in rates_list:
            v = avg(data[(seqs, rate)][mode][metric_key])
            line += f"{fmt(v, prec):>{col_w}}"
        print(line)


def main() -> None:
    args = parse_args()
    root = Path(args.sweep_root)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not root.exists():
        print(f"No sweep root at {root}")
        return
    labels = sorted(p for p in root.iterdir() if p.is_dir())
    if not labels:
        print(f"No label subdirectories under {root}")
        return
    for label_dir in labels:
        units = sorted(p for p in label_dir.iterdir() if p.is_dir())
        if not units:
            continue
        print(f"\n{'#' * 72}\nlabel = {label_dir.name}\n{'#' * 72}")
        for unit_dir in units:
            data, complete, seqs_list, rates_list = collect_unit(unit_dir, modes)
            if not seqs_list or not rates_list:
                continue
            print(f"\n{'=' * 72}\nchunk_unit = {unit_dir.name}  "
                  f"({complete} run summaries)\n{'=' * 72}")
            for title, key, prec in METRICS:
                print(f"\n## {title}")
                for mode in modes:
                    print(f"\n### {mode}")
                    emit_matrix(data, mode, key, prec, seqs_list, rates_list)


if __name__ == "__main__":
    main()
