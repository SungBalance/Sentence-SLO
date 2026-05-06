#!/usr/bin/env python3
"""10-case violation breakdown for chunk-level SSLO violations.

Reads chunks.jsonl + scheduler_stats.jsonl from one or more sweep base
directories (e.g. exp/sslo_test/output_sweep_optA) and prints a per-case
count for the chunks whose cumulative_slack < 0.

Cases (mutually exclusive, first-match priority):
  1 warmup            chunk_idx < NUM_WARMUP
  2 post_warmup_first chunk_idx == NUM_WARMUP (just graduated to MEASURED)
  3 capacity_full     at gen_finish_ts, running >= CAPACITY_FRAC * cap
  4 critical_step     has_critical=True at gen_finish_ts
  5 high_avg_score    avg_score >= HIGH_AVG_SCORE at gen_finish_ts
  6 waiting_blocked   waiting > 0 at gen_finish_ts (admission gated)
  7 long_pending      pending_time / (gen_time + pending_time) >= LONG_PENDING_FRAC
  8 gen_rate_lag      gen_time > GEN_LAG_FACTOR * (word_count * seconds_per_word)
  9 initial_burst     within INITIAL_BURST_S of the first scheduler step
 10 other

Usage:
  python3 analyze_violations_10cases.py <sweep_dir> [<sweep_dir> ...]
  python3 analyze_violations_10cases.py \\
    exp/sslo_test/output_sweep_optA \\
    exp/sslo_test/output_sweep_optB
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

NUM_WARMUP = 4
SECONDS_PER_WORD = 0.28
CAPACITY_FRAC = 0.95
LONG_PENDING_FRAC = 0.30
GEN_LAG_FACTOR = 1.5
HIGH_AVG_SCORE = 0.5
INITIAL_BURST_S = 5.0


def load_stats(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("kind") == "step":
                rows.append(d)
    rows.sort(key=lambda r: r["ts"])
    return rows


def step_at(rows: list[dict], ts: float) -> dict | None:
    """Latest step row with rows[i]['ts'] <= ts (binary search)."""
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["ts"] <= ts:
            lo = mid + 1
        else:
            hi = mid
    return rows[lo - 1] if lo else None


def classify(chunk: dict, stats_row: dict | None, t0: float | None,
             cap: int) -> str:
    idx = chunk.get("chunk_idx", 0)
    if idx < NUM_WARMUP:
        return "1 warmup"
    if idx == NUM_WARMUP:
        return "2 post_warmup_first"
    if stats_row is not None:
        if stats_row.get("running", 0) >= CAPACITY_FRAC * cap:
            return "3 capacity_full"
        if stats_row.get("has_critical"):
            return "4 critical_step"
        avg = stats_row.get("avg_score")
        if avg is not None and avg >= HIGH_AVG_SCORE:
            return "5 high_avg_score"
        if stats_row.get("waiting", 0) > 0:
            return "6 waiting_blocked"
    pending = chunk.get("pending_time") or 0.0
    gen = chunk.get("gen_time") or 0.0
    total = pending + gen
    if total > 0 and pending / total >= LONG_PENDING_FRAC:
        return "7 long_pending"
    wc = chunk.get("word_count") or 0
    expected = wc * SECONDS_PER_WORD
    if expected > 0 and gen > GEN_LAG_FACTOR * expected:
        return "8 gen_rate_lag"
    end_ts = chunk.get("end_time_ts")
    if end_ts is not None and t0 is not None and (end_ts - t0) < INITIAL_BURST_S:
        return "9 initial_burst"
    return "10 other"


def analyze_run(run_dir: str, cap: int) -> tuple[int, int, Counter]:
    chunks_p = os.path.join(run_dir, "chunks.jsonl")
    if not os.path.exists(chunks_p):
        # Fallback to mode-suffixed name (older sweep layouts).
        for mode in ("sslo", "sslo_adaptive", "baseline"):
            alt = os.path.join(run_dir, f"chunks_{mode}.jsonl")
            if os.path.exists(alt):
                chunks_p = alt
                break
    stats_p = os.path.join(run_dir, "scheduler_stats.jsonl")
    rows = load_stats(stats_p)
    t0 = rows[0]["ts"] if rows else None

    counts: Counter = Counter()
    total = 0
    violated = 0
    if not os.path.exists(chunks_p):
        return total, violated, counts
    with open(chunks_p) as f:
        for line in f:
            try:
                c = json.loads(line)
            except Exception:
                continue
            total += 1
            slack = c.get("cumulative_slack")
            if slack is None or slack >= 0:
                continue
            violated += 1
            stats_row = step_at(rows, c.get("end_time_ts") or 0.0)
            counts[classify(c, stats_row, t0, cap)] += 1
    return total, violated, counts


def show(label: str, base_dir: str, cap: int) -> None:
    if not os.path.exists(base_dir):
        print(f"=== {label}: MISSING ({base_dir}) ===\n")
        return
    print(f"=== {label} ===")
    agg_total = 0
    agg_violated = 0
    agg_counts: Counter = Counter()
    # Walk run directories: <base>/<unit>/seqs_<n>/rate_<r>/run_<i>/
    run_dirs: list[str] = []
    for unit in sorted(os.listdir(base_dir)):
        unit_p = os.path.join(base_dir, unit)
        if not os.path.isdir(unit_p):
            continue
        for seqs_d in sorted(os.listdir(unit_p)):
            seqs_p = os.path.join(unit_p, seqs_d)
            if not os.path.isdir(seqs_p) or not seqs_d.startswith("seqs_"):
                continue
            try:
                cap_for_dir = int(seqs_d.split("_")[1])
            except (IndexError, ValueError):
                cap_for_dir = cap
            for rate_d in sorted(os.listdir(seqs_p)):
                rate_p = os.path.join(seqs_p, rate_d)
                if not os.path.isdir(rate_p):
                    continue
                for run_d in sorted(os.listdir(rate_p)):
                    run_p = os.path.join(rate_p, run_d)
                    if os.path.isdir(run_p) and run_d.startswith("run_"):
                        run_dirs.append((run_p, cap_for_dir))
    for run_p, cap_for_dir in run_dirs:
        total, violated, counts = analyze_run(run_p, cap_for_dir)
        agg_total += total
        agg_violated += violated
        agg_counts.update(counts)
        rel = os.path.relpath(run_p, base_dir)
        rate = (100 * violated / total) if total else 0.0
        print(f"  {rel}: chunks={total} violated={violated} ({rate:.2f}%)")
    rate_t = (100 * agg_violated / agg_total) if agg_total else 0.0
    print(f"  TOTAL : chunks={agg_total} violated={agg_violated} ({rate_t:.2f}%)")
    print(f"  Breakdown of violated chunks:")
    for case in sorted(agg_counts):
        n = agg_counts[case]
        pct = 100 * n / max(agg_violated, 1)
        print(f"    {case:22s}  {n:5d}  ({pct:5.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="10-case chunk violation breakdown.")
    parser.add_argument(
        "sweep_dirs", nargs="+",
        help="Sweep base dirs (e.g. exp/sslo_test/output_sweep_optA).")
    parser.add_argument(
        "--cap", type=int, default=64,
        help="Default running cap when seqs_* parsing fails (default: 64).")
    args = parser.parse_args()
    for d in args.sweep_dirs:
        show(d, d, args.cap)


if __name__ == "__main__":
    main()
