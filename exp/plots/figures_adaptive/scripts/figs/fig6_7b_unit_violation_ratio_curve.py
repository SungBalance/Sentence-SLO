from __future__ import annotations

import argparse
from pathlib import Path

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _unit_miss_diagnostics import plot_scatter_diagnostic, write_unit_miss_cells

_BASENAME = "fig6_7b_unit_violation_ratio_curve"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)


def preprocess_fig6_7b(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    write_unit_miss_cells(input_path, output_dir, basename=_BASENAME)


def plot_fig6_7b(input_path: Path, output_path: Path) -> None:
    plot_scatter_diagnostic(
        input_path,
        output_path,
        value_column="unit_deadline_miss_rate",
        ylabel="Unit deadline\nmiss rate (%)",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig6_7b(args.data, args.out)


if __name__ == "__main__":
    main()
