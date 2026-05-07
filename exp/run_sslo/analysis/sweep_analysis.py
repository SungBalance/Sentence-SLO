#!/usr/bin/env python3
"""Sweep-level analysis: CSV export, console tables, and per-cell or
sweep-wide aggregates over repeated runs.

Subcommands:
  csv         Write a flat summary.csv from every summary.json under a
              sweep root.
  tables      Print compact (max_num_seqs × request_rate) tables for
              TTFT p50/p99 and chunk/request-level violation ratios per
              label × chunk_unit × mode.
  agg-sweep   Print per-metric tables averaged across runs for one
              chunk_unit's seqs × rate grid.
  agg-repeat  Print per-metric mean ± stddev across the N repeats of a
              single (seqs, rate) cell.
"""
from __future__ import annotations

import argparse
import json
import signal
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics_utils import (  # noqa: E402
    DISPLAY_GROUPS,
    MODES_DEFAULT,
    fmt_pair,
    lookup,
    parse_modes_arg,
)


# ---------------------------------------------------------------------------
# csv subcommand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistMetric:
    """A distribution metric — mean/p50/p90/p99 will be emitted per mode."""
    name: str
    path: tuple[str, ...]
    scale: float = 1.0


# Distribution metrics: each yields name_mean, name_p50, name_p90, name_p99.
DIST_METRICS: tuple[DistMetric, ...] = (
    DistMetric("ttft_s",                ("ttft", "all"),           1.0),
    DistMetric("tpot_ms",               ("tpot",),                 1000.0),
    DistMetric("queue_stall_ms",        ("queue_stall",),          1000.0),
    DistMetric("slack_s",               ("slack",),                1.0),
    DistMetric("running",               ("scheduler", "running"),  1.0),
    DistMetric("combined",              ("scheduler", "combined"), 1.0),
    DistMetric("pending_time_s",        ("pending", "time"),       1.0),
    DistMetric("inter_chunk_delay_ms",  ("inter_chunk_delay",),    1000.0),
)
STATS = ("mean", "p50", "p90", "p99")

# Scalar (non-distribution) metrics — emit one column each.
SCALAR_METRICS: tuple[tuple[str, tuple[str, ...], str, float], ...] = (
    ("slack_neg_ratio",      ("slack",),          "neg_ratio", 1.0),
    ("slo_compliance_rate",  ("slo_compliance",), "rate",      1.0),
    ("slo_compliance_count", ("slo_compliance",), "count",     1.0),
    ("slo_total_requests",   ("slo_compliance",), "total_requests", 1.0),
)

CONTEXT_COLUMNS = (
    "Model", "Label", "mode", "chunk_unit", "req/s", "max_num_seqs",
    "generation_max_tokens", "max_model_len", "run_idx", "num_requests",
)


def _metric_node(summary: dict, path: tuple[str, ...], mode: str):
    """Return summary['metrics'][path[0]][mode][path[1]]... or None."""
    node = summary.get("metrics", {}).get(path[0], {}).get(mode)
    for segment in path[1:]:
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node


def _emit_rows(summary_path: Path) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config", {})
    # Path layout: <root>/<label>/<unit>/seqs_<n>/rate_<r>/run_<i>/summary.json
    path_label = summary_path.parents[4].name
    run_idx = parse_int_suffix(summary_path.parent.name, "run_")
    rows: list[dict] = []
    for mode in cfg.get("modes_run", []):
        ttft_node = _metric_node(summary, ("ttft", "all"), mode) or {}
        row: dict = {
            "Model": cfg.get("model"),
            "Label": cfg.get("label") or path_label,
            "mode": mode,
            "chunk_unit": cfg.get("chunk_unit"),
            "req/s": cfg.get("request_rate"),
            "max_num_seqs": cfg.get("max_num_seqs"),
            "generation_max_tokens": cfg.get("generation_max_tokens"),
            "max_model_len": cfg.get("max_model_len"),
            "run_idx": run_idx,
            "num_requests": ttft_node.get("count"),
        }
        for dm in DIST_METRICS:
            node = _metric_node(summary, dm.path, mode) or {}
            for stat in STATS:
                value = node.get(stat)
                row[f"{dm.name}_{stat}"] = (
                    None if value is None else float(value) * dm.scale)
        for name, path, field, scale in SCALAR_METRICS:
            node = _metric_node(summary, path, mode) or {}
            value = node.get(field)
            row[name] = None if value is None else float(value) * scale
        rows.append(row)
    return rows


def cmd_csv(args: argparse.Namespace) -> None:
    sweep_root = Path(args.sweep_root)
    output_path = Path(args.output) if args.output else (
        sweep_root / "summary.csv")
    summary_paths = (
        sorted(sweep_root.glob("*/*/seqs_*/rate_*/run_*/summary.json"))
        if sweep_root.exists() else [])
    rows: list[dict] = []
    for path in summary_paths:
        rows.extend(_emit_rows(path))
    columns = list(CONTEXT_COLUMNS)
    for dm in DIST_METRICS:
        columns.extend(f"{dm.name}_{stat}" for stat in STATS)
    columns.extend(name for name, *_ in SCALAR_METRICS)
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        print(f"  no summary.json found under {sweep_root}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  wrote {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# tables subcommand
# ---------------------------------------------------------------------------


TABLE_METRICS = (
    ("TTFT p50 (s)",                              "ttft_p50",   3),
    ("TTFT p99 (s)",                              "ttft_p99",   3),
    ("Chunk-level violation ratio (chunk_idx>=1)", "chunk_viol", 4),
    ("Request-level violation ratio",             "req_viol",   4),
)


def parse_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix):])
    except ValueError:
        return None


def _collect_unit(unit_dir: Path, modes: tuple[str, ...]):
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


def _avg(xs: list[float]) -> float | None:
    return st.fmean(xs) if xs else None


def _fmt_table_cell(x: float | None, prec: int) -> str:
    return f"{x:.{prec}f}" if x is not None else "—"


def _emit_matrix(data, mode, metric_key, prec, seqs_list, rates_list, col_w=10):
    head = f"{'seqs\\rate':>10} " + "".join(f"{r:>{col_w}}" for r in rates_list)
    print(head)
    print("-" * len(head))
    for seqs in seqs_list:
        line = f"{seqs:>10} "
        for rate in rates_list:
            v = _avg(data[(seqs, rate)][mode][metric_key])
            line += f"{_fmt_table_cell(v, prec):>{col_w}}"
        print(line)


def cmd_tables(args: argparse.Namespace) -> None:
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
            data, complete, seqs_list, rates_list = _collect_unit(unit_dir, modes)
            if not seqs_list or not rates_list:
                continue
            print(f"\n{'=' * 72}\nchunk_unit = {unit_dir.name}  "
                  f"({complete} run summaries)\n{'=' * 72}")
            for title, key, prec in TABLE_METRICS:
                print(f"\n## {title}")
                for mode in modes:
                    print(f"\n### {mode}")
                    _emit_matrix(data, mode, key, prec, seqs_list, rates_list)


# ---------------------------------------------------------------------------
# agg-sweep subcommand (per-chunk_unit seqs×rate aggregate)
# ---------------------------------------------------------------------------


def _discover_axes(base: Path) -> tuple[list[int], list[int]]:
    seqs_set: set[int] = set()
    rates_set: set[int] = set()
    if not base.exists():
        return [], []
    for seqs_dir in base.glob("seqs_*"):
        seqs = parse_int_suffix(seqs_dir.name, "seqs_")
        if seqs is None:
            continue
        for rate_dir in seqs_dir.glob("rate_*"):
            rate = parse_int_suffix(rate_dir.name, "rate_")
            if rate is None:
                continue
            seqs_set.add(seqs)
            rates_set.add(rate)
    return sorted(seqs_set), sorted(rates_set)


def _load_cell(base: Path, seqs: int, rate: int, num_runs: int,
               modes: tuple[str, ...]) -> dict:
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


def _fmt_agg_cell(values: list[float], scale: float, fmt_s: str) -> str:
    if not values:
        return "    n/a    "
    if len(values) == 1:
        return fmt_s.format(values[0] * scale) + "  (n=1)"
    m = st.mean(values) * scale
    sd = st.stdev(values) * scale
    return f"{fmt_s.format(m)}+/-{fmt_s.format(sd).strip()}"


def cmd_agg_sweep(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    modes = parse_modes_arg(args.modes)
    base = Path(args.base_output)

    seqs_list, rates_list = _discover_axes(base)
    if not seqs_list or not rates_list:
        print(f"No seqs_*/rate_* dirs found under {base}")
        return
    cells = {
        (s, r): _load_cell(base, s, r, args.num_runs, modes)
        for s in seqs_list for r in rates_list
    }

    for group_name, specs in DISPLAY_GROUPS:
        print(f"\n=== {group_name} ===")
        for spec in specs:
            key = (spec.path, spec.field)
            print(f"\n  {spec.label}")
            header = f"{'':15s}" + "".join(
                f"  rate={r:<3d}{'':12s}" for r in rates_list)
            print(header)
            for s in seqs_list:
                for mode in modes:
                    line = f"seqs={s:<3d} {mode[:11]:11s}"
                    for r in rates_list:
                        cell_val = cells[(s, r)][key][mode]
                        line += "  " + _fmt_agg_cell(
                            cell_val, spec.scale, spec.fmt) + "  "
                    print(line)
                print()


# ---------------------------------------------------------------------------
# agg-repeat subcommand (one cell, N repeats)
# ---------------------------------------------------------------------------


def cmd_agg_repeat(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
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

    print(f"\n=== {args.num_runs}-run aggregate "
          f"(max_num_seqs={args.max_num_seqs}) ===\n")

    for group_name, specs in DISPLAY_GROUPS:
        print(f"\n=== {group_name} ===")
        header = f"{'metric':<30s}" + "".join(f"{m:>26s}" for m in modes)
        print(header)
        print("-" * len(header))
        for spec in specs:
            key = (spec.path, spec.field)
            cells = [
                fmt_pair(per_mode_values[key][m],
                         scale=spec.scale, fmt=spec.fmt)
                for m in modes
            ]
            print(f"{spec.label:<30s}" + "".join(f"{c:>26s}" for c in cells))


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


DEFAULT_SWEEP_ROOT = "exp/run_sslo/output_sweep"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_csv = sub.add_parser("csv", help="write summary.csv from sweep root")
    p_csv.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    p_csv.add_argument("--output", default=None,
                       help="default: <sweep_root>/summary.csv")
    p_csv.set_defaults(func=cmd_csv)

    p_tab = sub.add_parser("tables", help="print headline tables")
    p_tab.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    p_tab.add_argument("--modes", default=",".join(MODES_DEFAULT))
    p_tab.set_defaults(func=cmd_tables)

    p_as = sub.add_parser("agg-sweep",
                          help="seqs×rate aggregate for one chunk_unit")
    p_as.add_argument("--base-output", default=DEFAULT_SWEEP_ROOT)
    p_as.add_argument("--num-runs", type=int, required=True)
    p_as.add_argument("--modes", default=",".join(MODES_DEFAULT))
    p_as.set_defaults(func=cmd_agg_sweep)

    p_ar = sub.add_parser("agg-repeat",
                          help="N-run mean ± stddev for one cell")
    p_ar.add_argument("--output-root",
                      default="exp/run_sslo/output")
    p_ar.add_argument("--max-num-seqs", type=int, required=True)
    p_ar.add_argument("--num-runs", type=int, required=True)
    p_ar.add_argument("--modes", default=",".join(MODES_DEFAULT))
    p_ar.set_defaults(func=cmd_agg_repeat)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
