from __future__ import annotations

import argparse
import sys
from pathlib import Path


FIGS_DIR = Path(__file__).resolve().parents[1] / "figs"
sys.path.insert(0, str(FIGS_DIR))

from fig5_0_tts_word_count_profiles import (  # noqa: E402
    PROCESSED_DIR,
    RAW_READING_DATA,
    RAW_TTS_DATA,
    preprocess_fig5_0,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RAW_TTS_DATA)
    parser.add_argument("--reading-data", type=Path, default=RAW_READING_DATA)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    preprocess_fig5_0(args.data, args.reading_data, args.out_dir)


if __name__ == "__main__":
    main()
