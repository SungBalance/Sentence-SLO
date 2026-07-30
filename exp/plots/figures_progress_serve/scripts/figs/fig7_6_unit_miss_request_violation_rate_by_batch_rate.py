from __future__ import annotations

import argparse
from pathlib import Path

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _unit_miss_diagnostics import plot_hierarchical_grouped_bar, write_unit_miss_cells

_BASENAME = "fig7_6_unit_miss_request_violation_rate_by_batch_rate"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(f"{_BASENAME}.png")


def preprocess_fig7_6(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    write_unit_miss_cells(
        input_path,
        output_dir,
        basename=_BASENAME,
        include_request_any=True,
    )


def plot_fig7_6(input_path: Path, output_path: Path) -> None:
    plot_hierarchical_grouped_bar(
        input_path,
        output_path,
        value_column="request_any_unit_miss_rate",
        ylabel="Requests with violation (%)",
        value_scale=100.0,
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
    plot_fig7_6(args.data, args.out)


if __name__ == "__main__":
    main()
