#!/usr/bin/env python3
"""Aggregate per-chunk durations into per-(model, word_count) profile rows.

Reads one or more durations.jsonl files (output of measure_audio_duration.py),
groups by (model, word_count), and emits mean/variance/std/min/max for
conversion_time_s, audio_duration_s, real_time_factor.

Output: stats.{jsonl, csv} + summary.json under --output-dir.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SUMMARY_COLUMNS,
    read_jsonl,
    summary_output_paths,
    write_csv,
    write_json,
    write_jsonl,
)


METRICS = ("conversion_time_s", "audio_duration_s", "real_time_factor")

# Default binning rule: wc <= 52 stays at 1-unit resolution; wc > 52 is
# grouped into width-4 bins starting at 53. Set --bin-width-above-break 1
# to disable binning entirely (every wc value gets its own row).
DEFAULT_BIN_BREAK = 52
DEFAULT_BIN_WIDTH = 4


def bin_word_count(wc: int, *, bin_break: int, bin_width: int) -> tuple[int, int]:
    if wc <= bin_break or bin_width <= 1:
        return (wc, wc)
    low = (bin_break + 1) + ((wc - (bin_break + 1)) // bin_width) * bin_width
    return (low, low + bin_width - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl", action="append", required=True,
        help="Repeat to merge multiple model outputs into one table.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-break", type=int, default=DEFAULT_BIN_BREAK,
                        help="Above this wc threshold, group into width-N bins.")
    parser.add_argument("--bin-width-above-break", type=int,
                        default=DEFAULT_BIN_WIDTH,
                        help="Bin width applied to wc > bin-break. 1 = no binning.")
    # SSLO: percentile cutoff. For each (model, wc) bucket, drop the
    # lowest `trim_pct_low`% and highest `trim_pct_high`% of each metric's
    # values before computing mean/var/std/min/max. Mitigates the
    # word_count vs audio_duration coupling break caused by numeric
    # sequences (π digits, formulas) and duplicate template chunks.
    parser.add_argument("--trim-pct-low", type=float, default=5.0,
                        help="Bottom percentile to drop per metric. "
                             "0 = no trim from below.")
    parser.add_argument("--trim-pct-high", type=float, default=5.0,
                        help="Top percentile to drop per metric. "
                             "0 = no trim from above.")
    parser.add_argument("--min-rows-for-trim", type=int, default=20,
                        help="Buckets with row_count below this keep all "
                             "values (trimming N<20 buckets is too noisy).")
    return parser.parse_args()


# SSLO
def _trim_values(values: list[float], *,
                 pct_low: float, pct_high: float,
                 min_rows: int) -> list[float]:
    """Drop the bottom pct_low% and top pct_high% of values.

    For buckets with fewer than min_rows samples, return values unchanged
    — percentile trimming a tiny bucket throws away signal.
    """
    n = len(values)
    if n < min_rows or (pct_low <= 0 and pct_high <= 0):
        return values
    s = sorted(values)
    lo = int(round(n * pct_low / 100.0))
    hi = n - int(round(n * pct_high / 100.0))
    if hi <= lo:
        return values
    return s[lo:hi]


def _dist(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"row_count": 0,
                **{f"{m}_{s}": 0.0 for m in METRICS
                   for s in ("mean", "var", "std", "min", "max")}}
    out: dict[str, float | int] = {"row_count": len(values)}
    return out


def build_stats(rows: list[dict[str, Any]], *,
                bin_break: int, bin_width: int,
                trim_pct_low: float = 0.0,
                trim_pct_high: float = 0.0,
                min_rows_for_trim: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], dict[str, list[float]]] = (
        defaultdict(lambda: {m: [] for m in METRICS}))
    for r in rows:
        try:
            wc = int(r["word_count"])
            low, high = bin_word_count(
                wc, bin_break=bin_break, bin_width=bin_width)
            key = (str(r["model"]), str(r["tts_backend"]), low, high)
        except (KeyError, ValueError, TypeError):
            continue
        for m in METRICS:
            try:
                grouped[key][m].append(float(r[m]))
            except (KeyError, ValueError, TypeError):
                pass

    out: list[dict[str, Any]] = []
    for (model, backend, low, high), buckets in sorted(grouped.items()):
        any_metric = buckets[METRICS[0]]
        # SSLO: row_count reflects RAW samples; per-metric trim is reported
        # in the *_trimmed_count column so callers know how much was dropped.
        raw_count = len(any_metric)
        row: dict[str, Any] = {
            "model": model, "tts_backend": backend,
            "word_count_low": low, "word_count_high": high,
            "row_count": raw_count,
        }
        for m in METRICS:
            # SSLO: percentile trim per-metric.
            vs = _trim_values(
                buckets[m],
                pct_low=trim_pct_low, pct_high=trim_pct_high,
                min_rows=min_rows_for_trim)
            if vs:
                var = statistics.pvariance(vs) if len(vs) > 1 else 0.0
                row[f"{m}_mean"] = statistics.fmean(vs)
                row[f"{m}_var"] = var
                row[f"{m}_std"] = var ** 0.5
                row[f"{m}_min"] = min(vs)
                row[f"{m}_max"] = max(vs)
            else:
                for s in ("mean", "var", "std", "min", "max"):
                    row[f"{m}_{s}"] = 0.0
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for p in args.input_jsonl:
        rows.extend(read_jsonl(p))

    stats_rows = build_stats(
        rows, bin_break=args.bin_break, bin_width=args.bin_width_above_break,
        # SSLO
        trim_pct_low=args.trim_pct_low, trim_pct_high=args.trim_pct_high,
        # SSLO
        min_rows_for_trim=args.min_rows_for_trim)
    paths = summary_output_paths(args.output_dir)
    write_jsonl(paths.stats_jsonl, stats_rows)
    write_csv(paths.stats_csv, stats_rows, columns=SUMMARY_COLUMNS)
    write_json(paths.summary_json, {
        "num_duration_rows": len(rows),
        "num_summary_rows": len(stats_rows),
        "models": sorted({str(r["model"]) for r in rows if "model" in r}),
        "backends": sorted({str(r.get("tts_backend", "")) for r in rows}),
        "word_count_min": min((r["word_count_low"] for r in stats_rows),
                              default=None),
        "word_count_max": max((r["word_count_high"] for r in stats_rows),
                              default=None),
        "bin_break": args.bin_break,
        "bin_width_above_break": args.bin_width_above_break,
        # SSLO
        "trim_pct_low": args.trim_pct_low,
        # SSLO
        "trim_pct_high": args.trim_pct_high,
        # SSLO
        "min_rows_for_trim": args.min_rows_for_trim,
        "metrics": list(METRICS),
    })
    print(f"Wrote {len(stats_rows)} (model, word_count) profile rows "
          f"under {args.output_dir}")


if __name__ == "__main__":
    main()
