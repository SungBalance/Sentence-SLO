"""Figure 5.0 - TTS profile by chunk word count."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_paths import DATA_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps


_BASENAME = "fig5_0_tts_word_count_profiles"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_READING_DATA = PROCESSED_DIR / "fig5_0_reading_profile.csv"
DEFAULT_OUT = figure_output_path(_BASENAME)
RAW_TTS_DATA = DATA_DIR / "word_count_duration_stats.csv"
RAW_READING_DATA = DATA_DIR / "word_count_duration_stats_reading.csv"
MAX_WORD_COUNT = 60
READING_LABEL = "Human Reading"
MODEL_ORDER = ("Supertone/supertonic-3", "hexgrad/Kokoro-82M")

METRICS = (
    (
        "conversion_time_s",
        "Conversion time (s)",
    ),
    (
        "audio_duration_s",
        "Consume time (s)",
    ),
)


def model_label(model: str) -> str:
    return pps.TTS_MODEL_LABELS.get(model, model)


def _ordered_models(df: pd.DataFrame) -> list[str]:
    models = list(df["model"].drop_duplicates())
    ordered = [model for model in MODEL_ORDER if model in models]
    ordered.extend(model for model in models if model not in ordered)
    return ordered


def _load_profile(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[df["word_count_low"] <= MAX_WORD_COUNT]
    return df.sort_values(["model", "word_count_low"]).reset_index(drop=True)


def _load_reading_profile(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "chunk_words" in df.columns:
        df = df[df["chunk_words"] <= MAX_WORD_COUNT].copy()
        df = df.rename(
            columns={
                "chunk_words": "word_count_low",
                "consume_time_mean": "consume_time_s_mean",
                "consume_time_variance_s2": "consume_time_s_var",
            }
        )
    else:
        df = df[df["word_count_low"] <= MAX_WORD_COUNT].copy()
    return df.sort_values("word_count_low").reset_index(drop=True)


def _fit_regression(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = min(2, len(x) - 1)
    if degree <= 0:
        return x, mean, std
    fit = np.polynomial.Polynomial.fit(x, mean, degree)
    grid = np.linspace(float(x.min()), float(x.max()), 240)
    fitted = np.maximum(0.0, fit(grid))
    std_grid = np.interp(grid, x, std)
    return grid, fitted, std_grid


def _draw_regression(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    alpha: float,
    linestyle: str = "-",
    linewidth: float = 1.8,
) -> None:
    grid, fitted, std_grid = _fit_regression(x, mean, std)
    ax.fill_between(
        grid,
        np.maximum(0.0, fitted - std_grid),
        fitted + std_grid,
        color=color,
        alpha=alpha,
        linewidth=0,
    )
    ax.plot(grid, fitted, color=color, linewidth=linewidth, linestyle=linestyle)


def _metric_max_y(
    df: pd.DataFrame,
    metric: str,
    reading_df: pd.DataFrame | None,
) -> float:
    max_y = 0.0
    for _, group in df.groupby("model", sort=False):
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        std = np.sqrt(
            np.maximum(0.0, group[f"{metric}_var"].to_numpy(dtype=float)))
        max_y = max(max_y, float(np.max(mean + std)))
    if metric == "audio_duration_s" and reading_df is not None:
        mean = reading_df["consume_time_s_mean"].to_numpy(dtype=float)
        std = np.sqrt(
            np.maximum(0.0, reading_df["consume_time_s_var"].to_numpy(dtype=float)))
        max_y = max(max_y, float(np.max(mean + std)))
    return max_y


def _draw_metric(
    ax,
    df: pd.DataFrame,
    reading_df: pd.DataFrame | None,
    *,
    metric: str,
    ylabel: str,
) -> None:
    for model in _ordered_models(df):
        group = df[df["model"] == model].sort_values("word_count_low")
        x = group["word_count_low"].to_numpy(dtype=float)
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        std = np.sqrt(
            np.maximum(0.0, group[f"{metric}_var"].to_numpy(dtype=float)))
        _draw_regression(
            ax,
            x,
            mean,
            std,
            color=pps.TTS_MODEL_COLORS.get(model, pps.MODEL_PALETTE[2]),
            alpha=0.16,
        )

    if reading_df is not None:
        reading_x = reading_df["word_count_low"].to_numpy(dtype=float)
        reading_mean = reading_df["consume_time_s_mean"].to_numpy(dtype=float)
        reading_std = np.sqrt(
            np.maximum(0.0, reading_df["consume_time_s_var"].to_numpy(dtype=float)))
        _draw_regression(
            ax,
            reading_x,
            reading_mean,
            reading_std,
            color=pps.READING_COLOR,
            alpha=0.12,
            linewidth=1.7,
        )

    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, _metric_max_y(df, metric, reading_df) * 1.08)
    ax.set_xlim(0, MAX_WORD_COUNT + 1)
    ax.set_xticks(np.arange(0, MAX_WORD_COUNT + 1, 10))
    pps.clean_axes(ax, legend=False)


def _plot_profile(
    df: pd.DataFrame,
    reading_df: pd.DataFrame,
    output_path: Path,
) -> None:
    pps.paper_theme()
    w, h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(w, h),
        sharex=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.28},
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.18,
        top=0.78,
        wspace=0.32,
    )

    for ax, (metric, ylabel) in zip(axes[0], METRICS, strict=True):
        _draw_metric(
            ax,
            df,
            reading_df if metric == "audio_duration_s" else None,
            metric=metric,
            ylabel=ylabel,
        )
        ax.set_xlabel("Sentence Length (# of Words)")
        ax.tick_params(labelbottom=True)

    handles = [
        Line2D(
            [0],
            [0],
            color=pps.READING_COLOR,
            linewidth=1.7,
            label=READING_LABEL,
        )
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=pps.TTS_MODEL_COLORS.get(model, pps.MODEL_PALETTE[2]),
            linewidth=1.8,
            label=model_label(model),
        )
        for model in _ordered_models(df)
    )
    legend = fig.legend(
        handles=handles,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(handles),
        borderpad=0.3,
        handlelength=1.5,
    )
    pps.frame_legend(legend)
    pps.savefig(fig, output_path)
    plt.close(fig)


def preprocess_fig5_0(
    input_csv: Path = RAW_TTS_DATA,
    reading_csv: Path = RAW_READING_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = _load_profile(input_csv)
    reading_df = _load_reading_profile(reading_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{_BASENAME}.csv", index=False)
    reading_df.to_csv(output_dir / "fig5_0_reading_profile.csv", index=False)


def plot_fig5_0(input_csv: Path, reading_csv: Path, output_path: Path) -> None:
    df = _load_profile(input_csv)
    reading_df = _load_reading_profile(reading_csv)
    _plot_profile(df, reading_df, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reading-data", type=Path, default=DEFAULT_READING_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_0(args.data, args.reading_data, args.out)


if __name__ == "__main__":
    main()
