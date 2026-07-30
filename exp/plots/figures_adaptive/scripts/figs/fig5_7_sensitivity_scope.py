"""Figure 5.7 - Chunk length estimation error from measured outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps
from _outputs_data import (
    load_summary_from_outputs,
    normalize_request_id,
    ordered_consume_profiles,
    ordered_models,
    read_jsonl,
    run_dir_from_summary_row,
)
from _policy_style import normalize_policy

_BASENAME = "fig5_7_sensitivity_scope"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)

CONSUME_PROFILE_COLORS = {
    "Reading": pps.READING_COLOR,
    "TTS (Supertonic-3)": pps.TTS_MODEL_COLORS["Supertone/supertonic-3"],
    "TTS (Kokoro-82M)": pps.TTS_MODEL_COLORS["hexgrad/Kokoro-82M"],
}
ESTIMATE_COLUMN = "chunk length estimation"
ACTUAL_COLUMN = "actual chunk length"
ERROR_COLUMN = "error ratio"


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _middle_value(values: pd.Series) -> object:
    ordered = sorted(values.dropna().unique().tolist())
    if not ordered:
        raise ValueError("Cannot choose a middle value from an empty series.")
    return ordered[(len(ordered) - 1) // 2]


def _candidate_rows_for_model(df_model: pd.DataFrame) -> pd.DataFrame:
    target_cap = _middle_value(df_model["max_num_seqs"])
    cap_rows = df_model[df_model["max_num_seqs"] == target_cap]
    target_rate = _middle_value(cap_rows["lambda_req_s"])
    rate_rows = cap_rows[cap_rows["lambda_req_s"] == target_rate]
    target_seed = _middle_value(rate_rows["seed"])

    ranked = df_model.copy()
    ranked["cap_distance"] = (ranked["max_num_seqs"] - target_cap).abs()
    ranked["rate_distance"] = (ranked["lambda_req_s"] - target_rate).abs()
    ranked["seed_distance"] = (ranked["seed"] - target_seed).abs()
    return ranked.sort_values(
        ["cap_distance", "rate_distance", "seed_distance", "max_num_seqs", "lambda_req_s", "seed"]
    )


def _rows_from_run(run_dir: Path, summary_row: pd.Series) -> list[dict]:
    chunks_path = run_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return []

    rows: list[dict] = []
    for chunk in read_jsonl(chunks_path):
        estimation = chunk.get("expected_len", chunk.get("word_count"))
        actual = chunk.get("num_token")
        try:
            estimation_value = float(estimation)
            actual_value = float(actual)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(estimation_value) or not np.isfinite(actual_value):
            continue
        if actual_value <= 0:
            continue
        rows.append(
            {
                "model": str(summary_row["model"]),
                "model_label": _short_model_label(str(summary_row["model"])),
                "consume_profile_label": str(summary_row["consume_profile_label"]),
                "max_num_seqs": int(summary_row["max_num_seqs"]),
                "lambda_req_s": float(summary_row["lambda_req_s"]),
                "seed": int(summary_row["seed"]),
                "request_id": normalize_request_id(chunk.get("request_id", "")),
                "chunk_idx": int(chunk.get("chunk_idx", chunk.get("unit_index", 0))),
                ESTIMATE_COLUMN: estimation_value,
                ACTUAL_COLUMN: actual_value,
                ERROR_COLUMN: (estimation_value - actual_value) / actual_value,
            }
        )
    return rows


def preprocess_chunk_length_errors(input_path: Path) -> pd.DataFrame:
    summary = load_summary_from_outputs(input_path).copy()
    required = {
        "model",
        "consume_profile_label",
        "policy",
        "max_num_seqs",
        "lambda_req_s",
        "seed",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"outputs summary data missing required columns: {sorted(missing)}")

    summary["policy_key"] = summary["policy"].map(normalize_policy)
    sslo = summary[summary["policy_key"] == "sslo"].copy()
    if sslo.empty:
        raise ValueError("Figure 5.7 needs at least one sslo row in outputs summary data.")

    rows: list[dict] = []
    missing_models: list[str] = []
    for model in sorted(sslo["model"].unique().tolist()):
        for profile in ordered_consume_profiles(sslo[sslo["model"] == model]):
            df_model = sslo[
                (sslo["model"] == model)
                & (sslo["consume_profile_label"] == profile)
            ]
            model_rows: list[dict] = []
            for _, summary_row in _candidate_rows_for_model(df_model).iterrows():
                run_dir = run_dir_from_summary_row(input_path, summary_row)
                model_rows = _rows_from_run(run_dir, summary_row)
                if model_rows:
                    break
            if model_rows:
                rows.extend(model_rows)
            else:
                missing_models.append(f"{_short_model_label(str(model))} | {profile}")

    if missing_models:
        raise ValueError(
            "Figure 5.7 needs chunks.jsonl rows with non-null expected_len and "
            f"positive num_token for models: {missing_models}."
        )
    if not rows:
        raise ValueError(
            "Figure 5.7 needs chunk rows with expected_len and num_token under sslo runs."
        )
    return pd.DataFrame(rows)


def _ordered_model_labels(df: pd.DataFrame) -> list[str]:
    labels = sorted(df["model_label"].unique().tolist())
    if "Qwen3.5-9B" in labels:
        labels.remove("Qwen3.5-9B")
        labels.insert(0, "Qwen3.5-9B")
    return labels


def _consume_profile_color(profile: str) -> str:
    return CONSUME_PROFILE_COLORS.get(profile, pps.MODEL_PALETTE[4])


def _central_bounds(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("Figure 5.7 has no finite values to plot.")

    x_min = float(finite.quantile(0.01))
    x_max = float(finite.quantile(0.99))
    if x_max <= x_min:
        x_min = float(finite.min())
        x_max = float(finite.max())
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    return x_min, x_max


def _smooth_density(
    values: pd.Series,
    *,
    x_min: float,
    x_max: float,
    min_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    data = data[(data >= x_min) & (data <= x_max)]
    if data.size == 0:
        return np.array([]), np.array([])

    grid = np.linspace(x_min, x_max, 256)
    if data.size == 1:
        center = float(data[0])
        bandwidth = max(min_bandwidth, (x_max - x_min) / 80.0)
        density = np.exp(-0.5 * ((grid - center) / bandwidth) ** 2)
        density /= bandwidth * np.sqrt(2.0 * np.pi)
        return grid, density

    std = float(np.std(data, ddof=1))
    iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
    robust_std = iqr / 1.349 if iqr > 0 else std
    scale = min(std, robust_std) if robust_std > 0 else std
    bandwidth = 0.9 * scale * (data.size ** -0.2) if scale > 0 else 0.0
    bandwidth = max(bandwidth, min_bandwidth, (x_max - x_min) / 120.0, 1e-6)

    scaled = (grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).mean(axis=1)
    density /= bandwidth * np.sqrt(2.0 * np.pi)
    return grid, density


def _plot_distribution_panel(
    ax,
    df: pd.DataFrame,
    *,
    column: str,
    xlabel: str,
    color: str,
    min_bandwidth: float,
) -> None:
    x_min, x_max = _central_bounds(df[column])

    sub = df[(df[column] >= x_min) & (df[column] <= x_max)]
    grid, density = _smooth_density(
        sub[column],
        x_min=x_min,
        x_max=x_max,
        min_bandwidth=min_bandwidth,
    )
    ax.fill_between(grid, density, color=color, alpha=0.16, linewidth=0)
    ax.plot(grid, density, color=color, linewidth=1.6)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    pps.clean_axes(ax, legend=False)


def preprocess_fig5_7(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = preprocess_chunk_length_errors(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig5_7(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)
    output_path = Path(output_path)
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    w, h = pps.fig_size("double", ratio=0.48)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * 2,
        figsize=(w * 1.45, h * max(1, len(profiles))),
        gridspec_kw={"wspace": 0.18},
        squeeze=False,
    )

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            model_label = _short_model_label(model)
            panel = df[
                (df["model"] == model)
                & (df["consume_profile_label"] == profile)
            ]
            color = _consume_profile_color(profile)
            left_ax = axes[profile_idx, model_idx * 2]
            right_ax = axes[profile_idx, model_idx * 2 + 1]
            _plot_distribution_panel(
                left_ax,
                panel,
                column=ACTUAL_COLUMN,
                xlabel="Actual Chunk Length (tokens)" if profile_idx == len(profiles) - 1 else "",
                color=color,
                min_bandwidth=2.0,
            )
            _plot_distribution_panel(
                right_ax,
                panel,
                column=ERROR_COLUMN,
                xlabel="Estimation Error Ratio" if profile_idx == len(profiles) - 1 else "",
                color=color,
                min_bandwidth=0.04,
            )
            right_ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
            if model_idx == 0:
                left_ax.set_ylabel(f"{profile}\nDensity")
            if profile_idx == 0:
                left_ax.set_title(f"{model_label}\nActual Length")
                right_ax.set_title(f"{model_label}\nError Ratio")

    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed Figure 5.7 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_7(args.data, args.out)


if __name__ == "__main__":
    main()
