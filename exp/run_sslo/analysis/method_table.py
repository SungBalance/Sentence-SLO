#!/usr/bin/env python3
"""4-method comparison table for an output_sweep_v2 sweep root.

Emits one row per (cap, mode, rate) with the metrics the SSLO judgment needs:
violation rate, throughput, TTFC, concurrent users, and the TB* / D-axis
diagnostics from scheduler_stats.jsonl. When a cell has several run_N
directories the runs are averaged and the spread is reported, which is the
only way to tell a real regression from run-to-run noise.

Usage:
  python3 exp/run_sslo/analysis/method_table.py exp/run_sslo/output_sweep_v2/phaseE
  python3 exp/run_sslo/analysis/method_table.py <root_a> <root_b>   # side by side
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

MODES = [
    ("baseline", "baseline"),
    ("progress_serve_offload", "offload"),
    ("progress_serve_token_budget", "tb"),
    ("progress_serve_offload_token_budget", "offl+tb"),
]
# TB* sits at its floor when it cannot afford more; the floor is a config knob
# but 512 is the only value used across these sweeps.
TB_FLOOR = 512


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def read_cell(cell: Path) -> dict | None:
    """Aggregate one .../run_N/rate_R directory, or None if it has no data."""
    meta_path = cell / "run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    start = meta["measurement_window_start_ts"]
    end = meta["measurement_window_end_ts"]

    requests = []
    for line in (cell / "requests.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done = row.get("completion_wall_ts")
        if (done is None or not start <= done <= end
                or row.get("request_cu_slo_violated_tau_1") is None):
            continue
        requests.append(row)
    if not requests:
        return None

    ttfc = sorted(r["ttfc"] for r in requests if r.get("ttfc") is not None)
    out = {
        "viol": 100 * sum(r["request_cu_slo_violated_tau_1"]
                          for r in requests) / len(requests),
        "tput": sum(r["num_output_tokens"] for r in requests) / (end - start),
        "ttfc": _median(ttfc),
    }

    stats_path = cell / "scheduler_stats.jsonl"
    if not stats_path.exists():
        return out
    users, budgets, kappas, d_defers = [], [], [], 0
    with stats_path.open() as f:
        for line in f:
            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line while the sweep is still running
            if step.get("num_handling_users") is not None:
                users.append(step["num_handling_users"])
            if step.get("token_budget_prefill") is not None:
                budgets.append(step["token_budget_prefill"])
            if step.get("prefill_kappa_ms_per_tok"):
                kappas.append(step["prefill_kappa_ms_per_tok"])
            d_defers += step.get("token_budget_d_defers") or 0
    out["users"] = _median(users)
    if budgets:
        out["tb"] = _median(budgets)
        out["floor_pct"] = 100 * sum(b <= TB_FLOOR for b in budgets) / len(budgets)
        out["kappa"] = _median(kappas)
        out["d_defers"] = d_defers
    return out


def collect(root: Path) -> dict[tuple[str, str, str], list[dict]]:
    """(cap, mode, rate) -> one entry per run_N that has data."""
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for meta in root.glob("sentence/*/*/*/cap*/*/run_*/rate_*/run_meta.json"):
        cell = meta.parent
        cap = cell.parents[2].name
        mode = cell.parents[1].name
        rate = cell.name.removeprefix("rate_")
        data = read_cell(cell)
        if data is not None:
            cells.setdefault((cap, mode, rate), []).append(data)
    return cells


def fmt(runs: list[dict], key: str, digits: int) -> str:
    """Mean across runs, with +/-half-spread when a cell was repeated."""
    values = [r[key] for r in runs if r.get(key) is not None]
    if not values:
        return "-"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{(max(values) - min(values)) / 2:.{digits}f}"


def report(root: Path) -> None:
    cells = collect(root)
    if not cells:
        print(f"no completed cells under {root}")
        return
    caps = sorted({c for c, _, _ in cells},
                  key=lambda c: int(c.removeprefix("cap")))
    print(f"\n########## {root} ##########")
    for cap in caps:
        rates = sorted({r for c, _, r in cells if c == cap}, key=float)
        print(f"\n=== {cap} ===")
        print(f"{'mode':>8s} {'rate':>5s} {'viol%':>7s} {'tput':>9s} "
              f"{'TTFC':>8s} {'users':>7s} {'TB*':>7s} {'floor%':>8s} "
              f"{'kappa':>7s} {'Ddef':>7s} {'runs':>4s}")
        for mode, label in MODES:
            for rate in rates:
                runs = cells.get((cap, mode, rate))
                if not runs:
                    continue
                print(f"{label:>8s} {rate:>5s} {fmt(runs, 'viol', 1):>7s} "
                      f"{fmt(runs, 'tput', 0):>9s} {fmt(runs, 'ttfc', 0):>8s} "
                      f"{fmt(runs, 'users', 0):>7s} {fmt(runs, 'tb', 0):>7s} "
                      f"{fmt(runs, 'floor_pct', 1):>8s} "
                      f"{fmt(runs, 'kappa', 3):>7s} "
                      f"{fmt(runs, 'd_defers', 0):>7s} {len(runs):>4d}")


def main() -> None:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        raise SystemExit(2)
    for root in roots:
        report(Path(root))


if __name__ == "__main__":
    main()
