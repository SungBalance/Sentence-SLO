#!/usr/bin/env python3
"""Cross-cutting slack analysis for measure_internal_slack experiment.

Reads data by traversing the folder structure:
  {data_dir}/{model_slug}/{dataset_slug}/batch_{N}/{chunk_unit}/chunks.jsonl

Charts:
  1. slack_by_batch_size    – mean slack vs all batch sizes, lines = model × chunk_type
  2. slack_by_chunk_index   – mean slack vs chunk_idx, 2×2 subplots for 4 batch sizes
  3. slack_by_chunk_type    – mean slack vs 4 batch sizes, subplots per model, lines = chunk_type
  4. slack_by_model         – mean slack vs 4 batch sizes, subplots per chunk_type, lines = model
  5. slack_distribution     – violin plot of slack distribution per condition (all batch sizes)

Seaborn-based styling. Smooth trend lines, no fill.
Chunk index 0 excluded (cumulative slack is always 0 by design).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

FOCUS_BATCH_SIZES = [16, 64, 128, 256]
MA_WINDOW = 5             # moving-average window for chunk-index smoothing
MIN_N_CHUNK_IDX = 16      # chart 2: drop chunk_idx bins with fewer than this many samples

# Wong colorblind-safe palette
BLUE      = "#0072B2"
VERMILLION = "#D55E00"
GREEN     = "#009E73"
PURPLE    = "#CC79A7"

MODEL_COLOR = {"35B-A3B": BLUE, "27B": VERMILLION}
CHUNK_COLOR = {"sentence": BLUE, "paragraph": VERMILLION}
CHUNK_LS    = {"sentence": "-", "paragraph": "--"}

LABEL_MAP = {
    "Qwen__Qwen3.5-35B-A3B": "35B-A3B",
    "Qwen__Qwen3.5-27B":     "27B",
}

# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------

def apply_paper_style() -> None:
    sns.set_theme(
        style="ticks",
        font_scale=0.9,
        rc={
            "font.family":          "sans-serif",
            "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titleweight":     "bold",
            "axes.linewidth":       0.7,
            "axes.axisbelow":       True,
            "axes.spines.top":      False,
            "axes.spines.right":    False,
            "grid.linewidth":       0.4,
            "grid.color":           "#d0d0d0",
            "lines.linewidth":      1.6,
            "lines.markersize":     4.5,
            "xtick.major.width":    0.7,
            "ytick.major.width":    0.7,
            "xtick.major.size":     3.5,
            "ytick.major.size":     3.5,
            "legend.frameon":       True,
            "legend.framealpha":    0.92,
            "legend.edgecolor":     "#cccccc",
            "legend.handlelength":  2.2,
            "savefig.dpi":          300,
        },
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_label(slug: str) -> str:
    return LABEL_MAP.get(slug, slug.replace("__", "/"))


def discover_and_load(data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "analysis":
            continue
        model = make_label(model_dir.name)
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for batch_dir in sorted(dataset_dir.iterdir()):
                if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                    continue
                try:
                    batch_size = int(batch_dir.name[6:])
                except ValueError:
                    continue
                for chunk_dir in sorted(batch_dir.iterdir()):
                    if not chunk_dir.is_dir():
                        continue
                    chunk_type = chunk_dir.name
                    fpath = chunk_dir / "chunks.jsonl"
                    if not fpath.exists():
                        continue
                    for row in read_jsonl(fpath):
                        if int(row["chunk_idx"]) == 0:
                            continue
                        rows.append({
                            "model":      model,
                            "chunk_type": chunk_type,
                            "batch_size": batch_size,
                            "chunk_idx":  int(row["chunk_idx"]),
                            "slack":      float(row["cumulative_slack"]),
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def ma_smooth(y: np.ndarray, w: int = MA_WINDOW) -> np.ndarray:
    """Centered moving average."""
    w = min(w, len(y))
    w = max(w, 1)
    kernel = np.ones(w) / w
    padded = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def iqr_filter(g: pd.DataFrame, col: str = "slack", mult: float = 1.5) -> pd.DataFrame:
    """Drop rows outside [Q1 - mult*IQR, Q3 + mult*IQR] for one chunk_idx group."""
    vals = g[col].values
    if len(vals) < 4:
        return g
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    if iqr == 0:
        return g
    return g[(vals >= q1 - mult * iqr) & (vals <= q3 + mult * iqr)]


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    """Add bold panel letter in upper-left corner."""
    ax.text(
        -0.12, 1.06, f"({letter})",
        transform=ax.transAxes,
        fontsize=10, fontweight="bold",
        va="bottom", ha="left",
    )


def _batch_xticks(ax: plt.Axes, batch_sizes: list[int]) -> None:
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])


# ---------------------------------------------------------------------------
# Chart 1: mean slack vs ALL batch sizes — lines = model × chunk_type
# ---------------------------------------------------------------------------

def plot_by_batch_size(df: pd.DataFrame, output_path: Path) -> None:
    means = (
        df.groupby(["model", "chunk_type", "batch_size"])["slack"]
        .mean().reset_index()
    )
    all_bs = sorted(means["batch_size"].unique())
    combos = sorted(set(zip(means["model"], means["chunk_type"])))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for model, ct in combos:
        sub = means[(means["model"] == model) & (means["chunk_type"] == ct)]
        sub = sub.sort_values("batch_size")
        x = sub["batch_size"].values
        y = sub["slack"].values
        label = f"{model} / {ct}"
        ax.plot(x, y,
                color=MODEL_COLOR.get(model, "#333"),
                linestyle=CHUNK_LS.get(ct, "-"),
                marker="o", markersize=4,
                linewidth=1.6, label=label, zorder=3)

    ax.set_xlabel("Batch Size (max_num_seqs)")
    ax.set_ylabel("Mean Cumulative Slack (s)")
    ax.set_title("Cumulative Slack vs. Batch Size")
    _batch_xticks(ax, all_bs)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.grid(axis="y", alpha=0.6)
    ax.legend(loc="upper right", ncol=1)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 2: mean slack vs chunk_idx — 2×2 subplots for 4 focus batch sizes
# ---------------------------------------------------------------------------

def plot_by_chunk_index(df: pd.DataFrame, output_path: Path) -> None:
    combos = sorted(set(zip(df["model"], df["chunk_type"])))
    focus = [bs for bs in FOCUS_BATCH_SIZES if bs in df["batch_size"].unique()]
    letters = "abcd"

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharey=False, sharex=False)
    axes_flat = axes.flatten()

    for ax_i, bs in enumerate(focus):
        ax = axes_flat[ax_i]
        sub_df = df[df["batch_size"] == bs]

        for model, ct in combos:
            grp = sub_df[(sub_df["model"] == model) & (sub_df["chunk_type"] == ct)]
            if grp.empty:
                continue

            # IQR outlier removal per chunk_idx, then drop low-n bins
            parts = [iqr_filter(kg) for _, kg in grp.groupby("chunk_idx")]
            filtered = pd.concat(parts) if parts else grp.iloc[:0]
            counts = filtered.groupby("chunk_idx")["slack"].count()
            valid_idx = counts[counts >= MIN_N_CHUNK_IDX].index
            filtered = filtered[filtered["chunk_idx"].isin(valid_idx)]
            if filtered.empty:
                continue

            means = filtered.groupby("chunk_idx")["slack"].mean().sort_index()
            x = means.index.values
            y = ma_smooth(means.values)
            ax.plot(x, y,
                    color=MODEL_COLOR.get(model, "#333"),
                    linestyle=CHUNK_LS.get(ct, "-"),
                    linewidth=1.5,
                    label=f"{model} / {ct}")

        ax.axhline(0, color="#888", linewidth=0.7, linestyle=":", zorder=0)
        ax.set_title(f"batch = {bs}")
        ax.grid(axis="y", alpha=0.5)
        _panel_label(ax, letters[ax_i])

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Cumulative Slack (s)")
    for ax in axes[1, :]:
        ax.set_xlabel("Chunk Index")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(combos),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor="#cccccc")
    fig.suptitle("Cumulative Slack by Chunk Index", fontsize=11, fontweight="bold", y=1.01)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 3: mean slack vs 4 batch sizes — subplots per model, lines = chunk_type
# ---------------------------------------------------------------------------

def plot_by_chunk_type(df: pd.DataFrame, output_path: Path) -> None:
    df = df[df["batch_size"].isin(FOCUS_BATCH_SIZES)]
    means = (
        df.groupby(["model", "chunk_type", "batch_size"])["slack"]
        .mean().reset_index()
    )
    model_list = sorted(means["model"].unique())
    chunk_types = sorted(means["chunk_type"].unique())
    letters = "ab"

    fig, axes = plt.subplots(1, len(model_list), figsize=(7.0, 3.5), sharey=True)
    if len(model_list) == 1:
        axes = [axes]

    for ax_i, (ax, model) in enumerate(zip(axes, model_list)):
        for ct in chunk_types:
            sub = means[(means["model"] == model) & (means["chunk_type"] == ct)]
            sub = sub.sort_values("batch_size")
            ax.plot(sub["batch_size"].values, sub["slack"].values,
                    color=CHUNK_COLOR.get(ct, "#333"),
                    linestyle=CHUNK_LS.get(ct, "-"),
                    marker="o", markersize=4.5,
                    linewidth=1.6, label=ct.capitalize())

        ax.set_title(model)
        ax.set_xlabel("Batch Size (max_num_seqs)")
        ax.grid(axis="y", alpha=0.6)
        _batch_xticks(ax, FOCUS_BATCH_SIZES)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.legend(loc="upper right", title="Chunk Type")
        _panel_label(ax, letters[ax_i])

    axes[0].set_ylabel("Mean Cumulative Slack (s)")
    fig.suptitle("Sentence vs. Paragraph: Cumulative Slack by Batch Size",
                 fontsize=11, fontweight="bold")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 4: mean slack vs 4 batch sizes — subplots per chunk_type, lines = model
# ---------------------------------------------------------------------------

def plot_by_model(df: pd.DataFrame, output_path: Path) -> None:
    df = df[df["batch_size"].isin(FOCUS_BATCH_SIZES)]
    means = (
        df.groupby(["model", "chunk_type", "batch_size"])["slack"]
        .mean().reset_index()
    )
    chunk_types = sorted(means["chunk_type"].unique())
    model_list  = sorted(means["model"].unique())
    letters = "ab"

    fig, axes = plt.subplots(1, len(chunk_types), figsize=(7.0, 3.5), sharey=True)
    if len(chunk_types) == 1:
        axes = [axes]

    for ax_i, (ax, ct) in enumerate(zip(axes, chunk_types)):
        for model in model_list:
            sub = means[(means["model"] == model) & (means["chunk_type"] == ct)]
            sub = sub.sort_values("batch_size")
            ax.plot(sub["batch_size"].values, sub["slack"].values,
                    color=MODEL_COLOR.get(model, "#333"),
                    marker="o", markersize=4.5,
                    linewidth=1.6, label=model)

        ax.set_title(ct.capitalize())
        ax.set_xlabel("Batch Size (max_num_seqs)")
        ax.grid(axis="y", alpha=0.6)
        _batch_xticks(ax, FOCUS_BATCH_SIZES)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.legend(loc="upper right", title="Model")
        _panel_label(ax, letters[ax_i])

    axes[0].set_ylabel("Mean Cumulative Slack (s)")
    fig.suptitle("35B-A3B vs. 27B: Cumulative Slack by Batch Size",
                 fontsize=11, fontweight="bold")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Chart 5: violin distribution per condition
# ---------------------------------------------------------------------------

def plot_slack_distribution(df: pd.DataFrame, output_path: Path) -> None:
    model_list  = sorted(df["model"].unique())
    chunk_types = sorted(df["chunk_type"].unique())
    all_bs      = sorted(df["batch_size"].unique())
    letters = "abcd"

    fig, axes = plt.subplots(
        len(model_list), len(chunk_types),
        figsize=(12.0, 7.0), sharey=False,
    )

    for ri, model in enumerate(model_list):
        for ci, ct in enumerate(chunk_types):
            ax = axes[ri][ci]
            sub = df[(df["model"] == model) & (df["chunk_type"] == ct)].copy()
            sub["batch_size"] = sub["batch_size"].astype(str)

            # clip display range to 1st–99th percentile to prevent tail distortion
            lo = sub["slack"].quantile(0.01)
            hi = sub["slack"].quantile(0.99)

            sns.violinplot(
                data=sub, x="batch_size", y="slack",
                order=[str(b) for b in all_bs],
                color=MODEL_COLOR.get(model, "#333"),
                inner="quartile",
                cut=0,
                linewidth=0.8,
                ax=ax,
            )

            ax.axhline(0, color="#555", linewidth=0.8, linestyle="--", zorder=3)
            ax.set_ylim(lo * 1.1 if lo < 0 else lo * 0.5, hi * 1.1)
            ax.set_title(f"{model} / {ct}")
            ax.set_xlabel("Batch Size" if ri == len(model_list) - 1 else "")
            ax.set_ylabel("Cumulative Slack (s)" if ci == 0 else "")
            ax.tick_params(axis="x", labelsize=7.5)
            ax.grid(axis="y", alpha=0.5)
            sns.despine(ax=ax)
            letter = letters[ri * len(chunk_types) + ci]
            _panel_label(ax, letter)

    fig.suptitle("Cumulative Slack Distribution by Condition",
                 fontsize=11, fontweight="bold")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = q * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - idx + lo) + sv[hi] * (idx - lo)


def build_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (model, ct, bs), grp in df.groupby(["model", "chunk_type", "batch_size"]):
        vals = grp["slack"].tolist()
        neg  = sum(1 for v in vals if v < 0)
        rows.append({
            "model_label": model, "chunk_type": ct, "batch_size": bs,
            "count": len(vals),
            "mean":  sum(vals) / len(vals),
            "p05":   percentile(vals, 0.05),
            "p50":   percentile(vals, 0.50),
            "p95":   percentile(vals, 0.95),
            "min":   min(vals), "max": max(vals),
            "negative_fraction": neg / len(vals),
        })
    return sorted(rows, key=lambda r: (r["model_label"], r["chunk_type"], r["batch_size"]))


SUMMARY_COLS = ["model_label", "chunk_type", "batch_size", "count",
                "mean", "p05", "p50", "p95", "min", "max", "negative_fraction"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True,
                   help="Root outputs directory (contains model subdirs).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write PNG files and summary.csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()

    print("Loading data from folder structure...")
    df = discover_and_load(data_dir)
    if df.empty:
        raise SystemExit(f"No data found under {data_dir}")
    print(f"Loaded {len(df):,} chunk rows — "
          f"{df['model'].nunique()} models, "
          f"{df['chunk_type'].nunique()} chunk types, "
          f"{df['batch_size'].nunique()} batch sizes.")

    print("Plotting...")
    plot_by_batch_size(df,      output_dir / "slack_by_batch_size.png")
    plot_by_chunk_index(df,     output_dir / "slack_by_chunk_index.png")
    plot_by_chunk_type(df,      output_dir / "slack_by_chunk_type.png")
    plot_by_model(df,           output_dir / "slack_by_model.png")
    plot_slack_distribution(df, output_dir / "slack_distribution.png")

    summary = build_summary(df)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_COLS)

    print(f"Wrote analysis output to {output_dir}")
    for row in summary:
        print(
            f"  {row['model_label']}:{row['chunk_type']}:batch={row['batch_size']}  "
            f"n={row['count']}  mean={row['mean']:.1f}s  p50={row['p50']:.1f}s  "
            f"neg={row['negative_fraction']:.1%}"
        )


if __name__ == "__main__":
    main()
