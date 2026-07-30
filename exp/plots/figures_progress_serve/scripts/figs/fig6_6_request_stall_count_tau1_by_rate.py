from __future__ import annotations

import argparse
from pathlib import Path

from figure_paths import figure_output_path, processed_csv_path
from _fig6_common import (
    OUTPUT_SWEEP_DIR,
    PROCESSED_DIR,
    plot_stall_count_grid,
    preprocess_tau_metrics,
)

_BASENAME = "fig6_6_request_stall_count_tau1_by_rate"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
TAU_S = 1.0


def preprocess_fig6_6(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    preprocess_tau_metrics(
        input_path,
        output_dir,
        basename=_BASENAME,
        tau_s=TAU_S,
        mode="stall_count",
    )


def plot_fig6_6(input_path: Path, output_path: Path) -> None:
    plot_stall_count_grid(
        input_path,
        output_path,
        ylabel="Stall Count per Request @ tau=1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig6_6(args.data, args.out)


if __name__ == "__main__":
    main()
