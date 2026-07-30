from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps
from _outputs_data import (
    chunk_stall_duration,
    chunks_by_request,
    load_summary_from_outputs,
    normalize_request_id,
    ordered_consume_profiles,
    ordered_models,
    request_map,
    run_dir_from_summary_row,
)
from _policy_style import POLICY_COLORS, POLICY_ORDER, POLICY_PLOT_LABELS, normalize_policy

_BASENAME = "fig5_9_request_max_stall_distribution"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)

LARGE_BATCH_SIZES = (128, 256)
PREFERRED_REQUEST_RATES = (2.0, 8.0, 20.0)
Y_TICKS = (0.0, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0)


def _selected_request_rates(df: pd.DataFrame) -> tuple[float, ...]:
    available = sorted(
        float(value)
        for value in pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique()
    )
    selected = [rate for rate in PREFERRED_REQUEST_RATES if rate in available]
    if selected:
        return tuple(selected)
    return tuple(available[:3])


def load_data(path: Path) -> pd.DataFrame:
    df = load_summary_from_outputs(path)
    required = {
        "model",
        "consume_profile_label",
        "policy",
        "max_num_seqs",
        "lambda_req_s",
        "seed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"outputs summary data missing required columns: {sorted(missing)}")
    df = df.copy()
    df["policy_raw"] = df["policy"]
    df["policy"] = df["policy"].map(normalize_policy)
    return df


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.038,
            profile,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        pps.bold_text(text)


def _selected_summary_rows(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    request_rates = _selected_request_rates(df[df["max_num_seqs"] == batch_size])
    selected = df[
        (df["max_num_seqs"] == batch_size)
        & (df["lambda_req_s"].isin(request_rates))
        & (df["policy"].isin(POLICY_ORDER))
    ].copy()
    if selected.empty:
        raise ValueError(
            "Figure 5.9 needs rows at "
            f"max_num_seqs={batch_size}."
        )

    return selected


def _request_stall_rows(outputs_root: Path, summary_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, summary_row in summary_rows.iterrows():
        path_row = summary_row.copy()
        path_row["policy"] = summary_row["policy_raw"]
        run_dir = run_dir_from_summary_row(outputs_root, path_row)
        requests = request_map(run_dir)
        chunks = chunks_by_request(run_dir)
        if not chunks:
            raise ValueError(f"Figure 5.9 needs chunk rows at {run_dir / 'chunks.jsonl'}.")

        for request_id, request_chunks in chunks.items():
            request = requests.get(normalize_request_id(request_id), {})
            if request.get("terminal_outcome", "completed") != "completed":
                continue
            if request.get("in_window", True) is False:
                continue

            stall_values = []
            for chunk in request_chunks:
                stall_duration = chunk_stall_duration(chunk)
                if np.isfinite(stall_duration):
                    stall_values.append(stall_duration)
            rows.append(
                {
                    "model": summary_row["model"],
                    "consume_profile_label": summary_row["consume_profile_label"],
                    "policy": summary_row["policy"],
                    "max_num_seqs": int(summary_row["max_num_seqs"]),
                    "lambda_req_s": float(summary_row["lambda_req_s"]),
                    "seed": int(summary_row["seed"]),
                    "request_id": normalize_request_id(request_id),
                    "request_idx": request.get("request_idx"),
                    "num_chunks": len(request_chunks),
                    "max_stall_s": max(stall_values) if stall_values else 0.0,
                }
            )

    if not rows:
        raise ValueError("Figure 5.9 needs request-level stall rows.")
    return pd.DataFrame(rows)


def preprocess_fig5_9(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_data(input_path)
    candidate_batch_sizes = [
        batch_size
        for batch_size in LARGE_BATCH_SIZES
        if batch_size in set(pd.to_numeric(df["max_num_seqs"], errors="coerce").dropna().astype(int))
    ]
    if not candidate_batch_sizes:
        raise ValueError(
            "Figure 5.9 needs at least one large batch size from "
            f"{list(LARGE_BATCH_SIZES)}."
        )

    best: tuple[tuple[int, int], pd.DataFrame] | None = None
    for batch_size in candidate_batch_sizes:
        selected = _selected_summary_rows(df, batch_size)
        candidate_rows = _request_stall_rows(input_path, selected)
        positive_count = int((candidate_rows["max_stall_s"] > 0.0).sum())
        score = (positive_count, batch_size)
        if best is None or score > best[0]:
            best = (score, candidate_rows)

    assert best is not None
    stall_rows = best[1]
    stall_rows = stall_rows[stall_rows["max_stall_s"] > 0.0].copy()
    if stall_rows.empty:
        raise ValueError("Figure 5.9 found no positive max-stall requests.")
    output_dir.mkdir(parents=True, exist_ok=True)
    stall_rows.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def _draw_distribution(ax, df_panel: pd.DataFrame) -> None:
    if df_panel.empty:
        positions = np.arange(len(POLICY_ORDER), dtype=float)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Base", "Prog."], fontsize=8)
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.75)
        pps.clean_axes(ax, legend=False)
        return

    positions = np.arange(len(POLICY_ORDER), dtype=float)
    box_data = [
        df_panel[df_panel["policy"] == policy]["max_stall_s"].astype(float)
        for policy in POLICY_ORDER
    ]
    box_data = [values if not values.empty else pd.Series([np.nan]) for values in box_data]
    box = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    for idx, policy in enumerate(POLICY_ORDER):
        color = POLICY_COLORS[policy]
        box["boxes"][idx].set_facecolor(color)
        box["boxes"][idx].set_alpha(0.32)
        box["boxes"][idx].set_edgecolor(color)

    for artist_key in ("whiskers", "caps", "medians"):
        for artist in box[artist_key]:
            artist.set_linewidth(1.0)
            artist.set_color("#333333")

    ax.set_yscale("symlog", linthresh=0.01, linscale=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Base", "Prog."], fontsize=8)
    pps.clean_axes(ax, legend=False)


def plot_fig5_9(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    request_rates = tuple(sorted(df["lambda_req_s"].dropna().unique().tolist()))
    batch_sizes = sorted(pd.to_numeric(df["max_num_seqs"], errors="coerce").dropna().astype(int).unique().tolist())
    if not models or not profiles:
        raise ValueError("Figure 5.9 needs at least one model in processed data.")
    if not request_rates:
        raise ValueError("Figure 5.9 needs at least one request rate.")
    if not batch_sizes:
        raise ValueError("Figure 5.9 needs a selected batch size.")

    y_max = max(0.1, float(df["max_stall_s"].max()) * 1.25)
    y_ticks = [tick for tick in Y_TICKS if tick <= y_max]

    w, h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * len(request_rates),
        figsize=(w * 1.65, h * 1.12 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.075,
        top=0.86,
        hspace=0.58,
        wspace=0.28,
    )

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            for rate_idx, rate in enumerate(request_rates):
                col_idx = model_idx * len(request_rates) + rate_idx
                ax = axes[profile_idx, col_idx]
                panel = df[
                    (df["model"] == model)
                    & (df["consume_profile_label"] == profile)
                    & (df["lambda_req_s"] == rate)
                ]
                _draw_distribution(ax, panel)
                ax.set_ylim(0.0, y_max)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f"{tick:g}" for tick in y_ticks])
                if profile_idx == 0:
                    ax.set_title(
                        f"{_short_model_label(model)}\n{rate:g} req/s",
                        fontsize=9,
                        fontweight="bold",
                        pad=4,
                    )
                ax.set_xlabel("Policy" if profile_idx == len(profiles) - 1 else "")
                ax.set_ylabel(
                    "Max Stall Time (s)" if model_idx == 0 and rate_idx == 0 else ""
                )
                if col_idx > 0:
                    ax.tick_params(labelleft=False)

    _add_row_titles(fig, axes, profiles)
    legend_handles = [
        Patch(facecolor=POLICY_COLORS[policy], label=POLICY_PLOT_LABELS[policy])
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=11,
        title=f"Batch size: {batch_sizes[0]}",
        title_fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
        labelspacing=0.3,
    )
    legend.get_title().set_fontweight("bold")
    pps.frame_legend(legend)

    output_path = Path(output_path)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed Figure 5.9 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_9(args.data, args.out)


if __name__ == "__main__":
    main()
