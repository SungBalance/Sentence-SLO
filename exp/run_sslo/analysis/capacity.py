#!/usr/bin/env python3
"""Section 8 capacity metrics: max mean_handling_users under a (tau, beta)
budget, and the per-policy ratio relative to baseline."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = (
    "policy",
    "model",
    "consume_mode",
    "consume_model",
    "tau_s",
    "beta",
    "supported_handling_users",
    "baseline_supported_handling_users",
    "capacity_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _tau_label(tau: float) -> str:
    return f"{tau:g}"


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid(row: dict[str, str]) -> bool:
    return (
        row.get("validity_pass") == "True"
        and row.get("no_harm_pass") == "True"
    )


def _cell_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("model", ""),
        row.get("consume_mode", ""),
        row.get("consume_model", ""),
    )


def _supported(
    rows: list[dict[str, str]],
    *,
    policy: str,
    cell: tuple[str, str, str],
    violation_col: str,
    beta: float,
) -> float | None:
    values: list[float] = []
    for row in rows:
        if row.get("policy") != policy or _cell_key(row) != cell:
            continue
        violation = _safe_float(row.get(violation_col))
        handling_users = _safe_float(row.get("mean_handling_users"))
        if violation is None or handling_users is None:
            continue
        if violation <= beta:
            values.append(handling_users)
    return max(values) if values else None


def build_rows(
    rows: list[dict[str, str]],
    *,
    tau_label: str,
    beta: float,
) -> list[dict[str, object]]:
    violation_col = f"request_cu_slo_violation_rate_tau_{tau_label}"
    policies_and_cells = sorted({
        (row.get("policy", ""), *_cell_key(row))
        for row in rows
    })
    valid_rows = [row for row in rows if _valid(row)]

    out_rows: list[dict[str, object]] = []
    for policy, model, consume_mode, consume_model in policies_and_cells:
        cell = (model, consume_mode, consume_model)
        supported = _supported(
            valid_rows,
            policy=policy,
            cell=cell,
            violation_col=violation_col,
            beta=beta,
        )
        baseline = _supported(
            valid_rows,
            policy="baseline",
            cell=cell,
            violation_col=violation_col,
            beta=beta,
        )
        ratio = (
            supported / baseline
            if supported is not None and baseline not in (None, 0.0)
            else None
        )
        out_rows.append({
            "policy": policy,
            "model": model,
            "consume_mode": consume_mode,
            "consume_model": consume_model,
            "tau_s": tau_label,
            "beta": beta,
            "supported_handling_users": supported,
            "baseline_supported_handling_users": baseline,
            "capacity_ratio": ratio,
        })
    return out_rows


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.sweep_root) / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"missing summary.csv: {summary_csv}")

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = build_rows(rows, tau_label=_tau_label(args.tau), beta=args.beta)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
