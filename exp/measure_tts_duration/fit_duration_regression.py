#!/usr/bin/env python3
"""Fit duration regression functions from word-count duration rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit word-count to duration regression per model and chunk unit."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="Repeat to combine multiple duration CSV files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--max-scatter-points", type=int, default=2000)
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9, 0.95],
        help="Quantiles to fit for reviewer-friendly duration bands.",
    )
    return parser.parse_args()


def group_key(row: pd.Series) -> tuple[str, str]:
    return str(row["model"]), str(row["chunk_unit"])


def fit_group(group_df: pd.DataFrame, degree: int) -> dict:
    x = group_df["word_count"].to_numpy(dtype=float)
    y = group_df["duration_seconds"].to_numpy(dtype=float)
    coefficients = np.polyfit(x, y, deg=degree)
    predictor = np.poly1d(coefficients)
    predictions = predictor(x)

    residual = y - predictions
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    mae = float(np.mean(np.abs(residual)))
    rmse = math.sqrt(float(np.mean(residual**2)))

    return {
        "coefficients": [float(value) for value in coefficients],
        "intercept_last": float(coefficients[-1]),
        "r2": float(r2),
        "mae_seconds": mae,
        "rmse_seconds": rmse,
        "row_count": int(len(group_df)),
        "word_count_min": int(group_df["word_count"].min()),
        "word_count_max": int(group_df["word_count"].max()),
    }


def fit_quantiles(
    group_df: pd.DataFrame,
    *,
    degree: int,
    quantiles: list[float],
) -> dict[float, dict]:
    x = group_df["word_count"].to_numpy(dtype=float).reshape(-1, 1)
    y = group_df["duration_seconds"].to_numpy(dtype=float)
    features = PolynomialFeatures(degree=degree, include_bias=True)
    design = features.fit_transform(x)

    quantile_fits: dict[float, dict] = {}
    for quantile in quantiles:
        model = QuantileRegressor(
            quantile=quantile,
            alpha=0.0,
            solver="highs",
        )
        model.fit(design, y)
        predictions = model.predict(design)
        residual = y - predictions
        quantile_fits[quantile] = {
            "coefficients_low_to_high": [float(value) for value in model.coef_],
            "pinball_loss": float(
                np.mean(
                    np.maximum(
                        quantile * residual,
                        (quantile - 1.0) * residual,
                    )
                )
            ),
        }
    return quantile_fits


def sample_for_scatter(group_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group_df) <= max_points:
        return group_df
    return group_df.sample(n=max_points, random_state=0).sort_values("word_count")


def plot_group(
    *,
    group_df: pd.DataFrame,
    coefficients: list[float],
    output_path: Path,
    max_scatter_points: int,
) -> None:
    predictor = np.poly1d(coefficients)
    sampled = sample_for_scatter(group_df, max_scatter_points)
    x_line = np.linspace(
        float(group_df["word_count"].min()),
        float(group_df["word_count"].max()),
        200,
    )
    y_line = predictor(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        sampled["word_count"],
        sampled["duration_seconds"],
        s=8,
        alpha=0.35,
        label="samples",
    )
    plt.plot(x_line, y_line, color="crimson", linewidth=2, label="fit")
    plt.xlabel("word_count")
    plt.ylabel("duration_seconds")
    plt.tight_layout()
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_quantile_group(
    *,
    group_df: pd.DataFrame,
    quantile_fits: dict[float, dict],
    degree: int,
    output_path: Path,
    max_scatter_points: int,
) -> None:
    sampled = sample_for_scatter(group_df, max_scatter_points)
    x_line = np.linspace(
        float(group_df["word_count"].min()),
        float(group_df["word_count"].max()),
        200,
    )
    x_line_2d = x_line.reshape(-1, 1)
    features = PolynomialFeatures(degree=degree, include_bias=True)
    line_design = features.fit_transform(x_line_2d)

    quantile_predictions: dict[float, np.ndarray] = {}
    for quantile, fit in quantile_fits.items():
        coeffs = np.array(fit["coefficients_low_to_high"], dtype=float)
        quantile_predictions[quantile] = line_design @ coeffs

    plt.figure(figsize=(8, 5))
    plt.scatter(
        sampled["word_count"],
        sampled["duration_seconds"],
        s=8,
        alpha=0.20,
        color="gray",
        label="samples",
    )
    if 0.1 in quantile_predictions and 0.9 in quantile_predictions:
        plt.fill_between(
            x_line,
            quantile_predictions[0.1],
            quantile_predictions[0.9],
            color="steelblue",
            alpha=0.18,
            label="p10-p90 band",
        )
    if 0.5 in quantile_predictions:
        plt.plot(
            x_line,
            quantile_predictions[0.5],
            color="crimson",
            linewidth=2.2,
            label="p50",
        )
    if 0.9 in quantile_predictions:
        plt.plot(
            x_line,
            quantile_predictions[0.9],
            color="navy",
            linewidth=1.8,
            linestyle="--",
            label="p90",
        )
    if 0.95 in quantile_predictions:
        plt.plot(
            x_line,
            quantile_predictions[0.95],
            color="darkgreen",
            linewidth=1.8,
            linestyle=":",
            label="p95",
        )

    plt.xlabel("word_count")
    plt.ylabel("duration_seconds")
    plt.tight_layout()
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.degree < 1:
        raise ValueError("--degree must be at least 1.")
    for quantile in args.quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError("--quantiles values must lie strictly between 0 and 1.")

    dataframes = [pd.read_csv(path) for path in args.input_csv]
    frame = pd.concat(dataframes, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    quantile_summary_rows: list[dict] = []
    for (model, chunk_unit), group_df in frame.groupby(["model", "chunk_unit"], sort=True):
        fit = fit_group(group_df, degree=args.degree)
        quantile_fits = fit_quantiles(
            group_df,
            degree=args.degree,
            quantiles=args.quantiles,
        )
        group_slug = f"{slugify(model)}__{chunk_unit}"
        group_dir = output_dir / group_slug
        plot_group(
            group_df=group_df,
            coefficients=fit["coefficients"],
            output_path=group_dir / "regression_fit.png",
            max_scatter_points=args.max_scatter_points,
        )
        plot_quantile_group(
            group_df=group_df,
            quantile_fits=quantile_fits,
            degree=args.degree,
            output_path=group_dir / "quantile_regression_fit.png",
            max_scatter_points=args.max_scatter_points,
        )
        row = {
            "model": model,
            "chunk_unit": chunk_unit,
            "degree": args.degree,
            **fit,
            "figure_path": str(group_dir / "regression_fit.png"),
            "quantile_figure_path": str(group_dir / "quantile_regression_fit.png"),
        }
        summary_rows.append(row)
        for quantile, quantile_fit in sorted(quantile_fits.items()):
            quantile_summary_rows.append(
                {
                    "model": model,
                    "chunk_unit": chunk_unit,
                    "degree": args.degree,
                    "quantile": quantile,
                    "coefficients_low_to_high": quantile_fit[
                        "coefficients_low_to_high"
                    ],
                    "pinball_loss": quantile_fit["pinball_loss"],
                    "row_count": fit["row_count"],
                }
            )
        write_json(
            group_dir / "fit_summary.json",
            {
                **row,
                "quantile_fits": {
                    str(quantile): quantile_fit
                    for quantile, quantile_fit in sorted(quantile_fits.items())
                },
            },
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "regression_summary.csv", index=False)
    pd.DataFrame(quantile_summary_rows).to_csv(
        output_dir / "quantile_regression_summary.csv",
        index=False,
    )
    write_json(
        output_dir / "regression_summary.json",
        {
            "degree": args.degree,
            "num_groups": len(summary_rows),
            "groups": summary_rows,
        },
    )
    write_json(
        output_dir / "quantile_regression_summary.json",
        {
            "degree": args.degree,
            "quantiles": args.quantiles,
            "num_groups": len(summary_rows),
            "groups": quantile_summary_rows,
        },
    )
    print(
        f"Wrote {len(summary_rows)} regression summaries under {output_dir}"
    )


if __name__ == "__main__":
    main()
