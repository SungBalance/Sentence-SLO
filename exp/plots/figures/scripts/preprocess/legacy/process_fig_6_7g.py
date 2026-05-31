from __future__ import annotations

import argparse
import sys
from pathlib import Path


FIGS_DIR = Path(__file__).resolve().parents[1] / "figs"
sys.path.insert(0, str(FIGS_DIR))

from fig6_7g_best_inflight_under_0_5pct import (  # noqa: E402
    DEFAULT_SOURCE_DATA,
    PROCESSED_DIR,
    preprocess_fig6_7g,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    preprocess_fig6_7g(args.data, args.out_dir)


if __name__ == "__main__":
    main()
