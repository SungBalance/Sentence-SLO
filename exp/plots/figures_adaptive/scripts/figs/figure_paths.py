"""Shared path helpers for Figure 5 plotting scripts."""

from __future__ import annotations

from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = FIGURES_DIR / "data"
OUTPUT_SWEEP_DIR = DATA_DIR / "output_sweep"
PROCESSED_DIR = FIGURES_DIR / "processed"


def figure_output_path(stem: str) -> Path:
    return FIGURES_DIR / stem


def processed_csv_path(stem: str) -> Path:
    return PROCESSED_DIR / f"{stem}.csv"
