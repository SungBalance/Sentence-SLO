from __future__ import annotations

import argparse
from pathlib import Path

from figure_paths import figure_output_path, processed_csv_path
from _fig6_common import (
    OUTPUT_SWEEP_DIR,
    PROCESSED_DIR,
    plot_line_grid,
    preprocess_summary_metric,
)

_BASENAME = "fig6_2_handling_users_by_rate"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)


def preprocess_fig6_2(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    preprocess_summary_metric(
        input_path,
        output_dir,
        basename=_BASENAME,
        metric_column="mean_handling_users",
        value_column="mean_handling_users",
    )


def plot_fig6_2(input_path: Path, output_path: Path) -> None:
    plot_line_grid(
        input_path,
        output_path,
        value_column="mean_handling_users",
        ylabel="# of In-flight Users",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig6_2(args.data, args.out)


if __name__ == "__main__":
    main()
