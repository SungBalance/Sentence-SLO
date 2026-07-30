from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _unit_miss_diagnostics import plot_hierarchical_grouped_bar
from fig7_aa_unit_miss_violation_severity_boxplot import (
    load_violated_unit_severities,
)

_BASENAME = "fig7_5_unit_miss_violation_severity_p99"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(f"{_BASENAME}.png")


def _p99_cells(df: pd.DataFrame) -> pd.DataFrame:
    sample_df = df[df["row_kind"] == "unit"].dropna(subset=["violation_severity_s"])
    if sample_df.empty:
        raise ValueError("Figure 7.5 has no unit violation severity samples.")
    grouped = (
        sample_df.groupby(
            [
                "model",
                "consume_profile_label",
                "policy",
                "policy_label",
                "max_num_seqs",
                "lambda_req_s",
            ],
            as_index=False,
        )
        .agg(
            unit_violation_amount_p99_s=("violation_severity_s", lambda s: s.quantile(0.99)),
            violated_unit_count=("violation_severity_s", "size"),
        )
        .sort_values(
            [
                "consume_profile_label",
                "model",
                "policy",
                "max_num_seqs",
                "lambda_req_s",
            ]
        )
    )
    return grouped


def preprocess_fig7_5(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_violated_unit_severities(input_path)
    _p99_cells(df).to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig7_5(input_path: Path, output_path: Path) -> None:
    plot_hierarchical_grouped_bar(
        input_path,
        output_path,
        value_column="unit_violation_amount_p99_s",
        ylabel="P99 unit violation amount (s)",
        mark_missing=True,
        profile_columns=True,
        share_y=False,
        x_axis_label="",
        append_lambda_to_last_tick=True,
        boundary_color="#111111",
        boundary_linewidth=0.9,
        boundary_linestyle="-",
        bar_width=0.44,
        bar_inner_scale=1.0,
        batch_gap=0.6,
        x_tick_labelsize=7.0,
        use_global_grid=True,
        batch_label_y=-0.12,
        batch_axis_label_y=-0.215,
        lambda_x_offset=0.42,
        lambda_label_y=-0.035,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig7_5(args.data, args.out)


if __name__ == "__main__":
    main()
