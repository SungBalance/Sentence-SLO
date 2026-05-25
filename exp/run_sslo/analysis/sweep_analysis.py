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


# R2 flat per-mode distribution metrics. CSV columns omit the trailing storage
# suffix, e.g. `ttft_mean` reads `metrics[mode]["ttft_mean_s"]`.
DIST_METRICS = (
    "ttft",
    "ttfc",
    "tpot",
    "queue_stall",
    "pending_time",
    "completion_latency",
    "request_max_stall_s",
    "request_total_stall_s",
    "request_stall_fraction",
)
STATS = ("mean", "p50", "p90", "p95", "p99", "max")

# Scalar (non-distribution) metrics — emit one column each.
SCALAR_METRICS = (
    "mean_running",
    "mean_pending",
    "mean_waiting",
    "mean_handling_users",
    "corrected_processed_tokens_per_s",
    "drop_timeout_rate",
    "starvation_count",
    "unit_deadline_miss_rate",
)
REQUEST_CU_TAUS = ("0.5", "1", "2", "5")
VALIDITY_COLUMNS = (
    "validity_pass",
    "no_harm_pass",
    "invalid_reason",
)

CONTEXT_COLUMNS = (
    "run_id",
    "model",
    "policy",
    "seed",
    "request_rate",
    "max_num_seqs",
    "consume_mode",
    "consume_model",
    "chunk_unit",
    "generation_max_tokens",
    "max_model_len",
    "run_idx",
    "num_requests",
    "tau_s",
    "beta",
)


def _none_if_blank(value):
    return None if value in ("", None) else value


def _run_id_part(value) -> str:
    value = _none_if_blank(value)
    return "unknown" if value is None else str(value)


def _parse_rate_dir(name: str) -> str | None:
    return name[len("rate_"):] if name.startswith("rate_") else None


def _mode_sslo_config(summary: dict, mode: str) -> dict:
    cfg = ((summary.get("config") or {}).get("sslo_config") or {})
    if isinstance(cfg, dict) and isinstance(cfg.get(mode), dict):
        return cfg[mode]
    return cfg if isinstance(cfg, dict) else {}


def _read_sidecar_config(summary_path: Path, mode: str) -> dict:
    cfg_path = summary_path.parent / "sslo_config.json"
    if not cfg_path.exists():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(cfg, dict) and isinstance(cfg.get(mode), dict):
        return cfg[mode]
    return cfg if isinstance(cfg, dict) else {}


def _path_context(summary_path: Path, mode: str) -> dict[str, str | None]:
    """Pull path-level context from the R3 sweep layout.

    Expected layout:
      <root>/sentence/<model_slug>/<consume_mode>/<tts_slug>/cap<N>/<mode>/run_<i>/rate_<r>/summary.json
    """
    model_slug = consume_mode = consume_model = cap = request_rate = None
    try:
        request_rate = _parse_rate_dir(summary_path.parents[0].name)
        cap_dir = summary_path.parents[3].name
        cap = cap_dir[len("cap"):] if cap_dir.startswith("cap") else cap_dir
        tts_slug = summary_path.parents[4].name
        consume_mode = summary_path.parents[5].name
        model_slug = summary_path.parents[6].name
        consume_model = "" if tts_slug == "none" else tts_slug.replace("__", "/")
    except IndexError:
        pass
    sidecar = _read_sidecar_config(summary_path, mode)
    consume_mode = sidecar.get("consume_mode") or consume_mode
    if consume_mode == "tts":
        consume_model = sidecar.get("tts_model") or consume_model
    else:
        consume_model = ""
    return {
        "model_slug": model_slug,
        "consume_mode": consume_mode,
        "consume_model": consume_model,
        "cap": cap,
        "request_rate": request_rate,
    }


def _active_workload(summary: dict, mode: str) -> dict:
    return ((summary.get("metrics") or {}).get("workload") or {}).get(mode) or {}


def _context_row(summary: dict, summary_path: Path, mode: str) -> dict:
    cfg = summary.get("config", {})
    run_meta = summary.get("run_meta", {}) or {}
    workload = _active_workload(summary, mode)
    path_ctx = _path_context(summary_path, mode)
    sslo_cfg = _mode_sslo_config(summary, mode)
    cap = cfg.get("max_num_seqs") or path_ctx.get("cap")
    request_rate = (
        cfg.get("request_rate")
        if cfg.get("request_rate") is not None
        else run_meta.get("request_rate", path_ctx.get("request_rate"))
    )
    run_idx = parse_int_suffix(summary_path.parents[1].name, "run_")
    seed = (
        sslo_cfg.get("request_rate_seed")
        or run_meta.get("request_rate_seed")
        or run_meta.get("seed")
    )
    consume_mode = sslo_cfg.get("consume_mode") or path_ctx.get("consume_mode")
    consume_model = (
        sslo_cfg.get("tts_model") or path_ctx.get("consume_model") or ""
        if consume_mode == "tts"
        else ""
    )
    policy = mode
    return {
        "run_id": "_".join(_run_id_part(v) for v in (
            policy, cap, request_rate, seed, run_idx)),
        "model": path_ctx.get("model_slug"),
        "policy": policy,
        "seed": seed,
        "request_rate": request_rate,
        "max_num_seqs": cap,
        "consume_mode": consume_mode,
        "consume_model": consume_model,
        "chunk_unit": cfg.get("chunk_unit"),
        "generation_max_tokens": cfg.get("generation_max_tokens"),
        "max_model_len": cfg.get("max_model_len"),
        "run_idx": run_idx,
        "num_requests": (
            workload.get("num_requests_total")
            or run_meta.get("N")
            or summary.get("in_window_count")
        ),
        "tau_s": "",
        "beta": sslo_cfg.get("beta", 0.05),
    }


def _dist_source_key(metric: str, stat: str) -> str:
    if metric in (
        "request_max_stall_s",
        "request_total_stall_s",
        "request_stall_fraction",
    ):
        return f"{metric}_{stat}"
    return f"{metric}_{stat}_s"


def _dist_column_name(metric: str, stat: str) -> str:
    """CSV column name. Time-unit metrics keep `_s`; fractions/counts do not."""
    if metric == "request_stall_fraction":
        return f"{metric}_{stat}"  # unitless
    if metric in ("request_max_stall_s", "request_total_stall_s"):
        return f"{metric}_{stat}"  # already ends in `_s`
    return f"{metric}_{stat}_s"  # ttft/ttfc/tpot/queue_stall/pending_time/completion_latency


def _validity_row(summary: dict) -> dict:
    validity = summary.get("validity") or {}
    reasons = validity.get("invalid_reason", []) or []
    if isinstance(reasons, list):
        reasons = ";".join(str(r) for r in reasons)
    return {
        "validity_pass": validity.get("validity_pass"),
        "no_harm_pass": validity.get("no_harm_pass"),
        "invalid_reason": reasons,
    }


def _tau_column(tau: str) -> str:
    return f"request_cu_slo_violation_rate_tau_{tau}"


def _emit_rows(summary_path: Path) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config", {})
    metrics = summary.get("metrics", {}) or {}
    req_viol = metrics.get("request_cu_slo_violation", {}) or {}
    rows: list[dict] = []
    for mode in cfg.get("modes_run", []):
        row: dict = _context_row(summary, summary_path, mode)
        row.update(_validity_row(summary))
        mode_metrics = metrics.get(mode, {}) or {}
        for metric in DIST_METRICS:
            for stat in STATS:
                row[_dist_column_name(metric, stat)] = mode_metrics.get(
                    _dist_source_key(metric, stat))
        for name in SCALAR_METRICS:
            row[name] = mode_metrics.get(name)
        tau_metrics = req_viol.get(mode, {}) or {}
        for tau in REQUEST_CU_TAUS:
            row[_tau_column(tau)] = (
                tau_metrics.get(f"tau_{tau}", {}) or {}).get("rate")
        rows.append(row)
    return rows


def cmd_csv(args: argparse.Namespace) -> None:
    sweep_root = Path(args.sweep_root)
    output_path = Path(args.output) if args.output else (
        sweep_root / "summary.csv")
    # Layout: <root>/sentence/<model_slug>/<consume_mode>/<tts_slug>/cap<n>/<mode>/run_<i>/rate_<r>/summary.json
    summary_paths = (
        sorted(sweep_root.glob(
            "sentence/*/*/*/cap*/*/run_*/rate_*/summary.json"))
        if sweep_root.exists() else [])
    rows: list[dict] = []
    for path in summary_paths:
        rows.extend(_emit_rows(path))
    columns = list(CONTEXT_COLUMNS) + list(VALIDITY_COLUMNS)
    for metric in DIST_METRICS:
        columns.extend(_dist_column_name(metric, stat) for stat in STATS)
    columns.extend(SCALAR_METRICS)
    columns.extend(_tau_column(tau) for tau in REQUEST_CU_TAUS)
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        print(f"  no summary.json found under {sweep_root}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  wrote {len(df)} rows to {output_path}")

    # Slim measurement-window sidecar (summary.csv unchanged). One row per
    # (cell, mode), with the absolute monotonic ts of the 1st request finish
    # and the (num_requests − max_num_seqs)-th request finish, plus the
    # duration. Definition matches progress_metrics.measurement_window.
    warmup_cols = list(CONTEXT_COLUMNS) + [
        "measurement_start_ts", "measurement_end_ts", "measurement_duration_s",
    ]
    warmup_rows: list[dict] = []
    for path in summary_paths:
        summary = json.loads(path.read_text())
        cfg = summary.get("config", {})
        for mode in cfg.get("modes_run", []):
            mw = lookup(summary, ("measurement_window",), None, mode) or {}
            warmup_rows.append(
                _context_row(summary, path, mode) | {
                    "measurement_start_ts": mw.get("start_ts"),
                    "measurement_end_ts": mw.get("end_ts"),
                    "measurement_duration_s": mw.get("duration_s"),
                })
    warmup_path = output_path.parent / "summary_warmup.csv"
    pd.DataFrame(warmup_rows, columns=warmup_cols).to_csv(warmup_path, index=False)
    print(f"  wrote {len(warmup_rows)} rows to {warmup_path}")


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
    # New layout: seqs_*/rate_*/<mode>/run_*/summary.json
    for path in sorted(unit_dir.glob("seqs_*/rate_*/*/run_*/summary.json")):
        seqs = parse_int_suffix(path.parents[3].name, "seqs_")
        rate = parse_int_suffix(path.parents[2].name, "rate_")
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
        for mode in modes:
            path = (base / f"seqs_{seqs}" / f"rate_{rate}"
                    / mode / f"run_{i}" / "summary.json")
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            for _group_name, specs in DISPLAY_GROUPS:
                for spec in specs:
                    key = (spec.path, spec.field)
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
