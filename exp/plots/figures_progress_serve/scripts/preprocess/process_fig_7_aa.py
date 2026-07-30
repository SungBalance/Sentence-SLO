from __future__ import annotations

import argparse
import sys
from pathlib import Path


FIGS_DIR = Path(__file__).resolve().parents[1] / "figs"
sys.path.insert(0, str(FIGS_DIR))

from fig7_aa_unit_miss_violation_severity_boxplot import (  # noqa: E402
    DEFAULT_RAW_DATA,
    PROCESSED_DIR,
    preprocess_fig7_aa,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    preprocess_fig7_aa(args.data, args.out_dir)


if __name__ == "__main__":
    main()
