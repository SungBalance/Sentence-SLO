from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps
from _outputs_data import load_summary_from_outputs, ordered_consume_profiles, ordered_models
from _policy_style import (
    POLICY_COLORS,
    POLICY_ORDER,
    POLICY_PLOT_LABELS,
    normalize_policy,
)

_BASENAME = "fig5_3_stall_capacity_frontier"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)

TAU_S = 1.0
PREFERRED_BATCH_SIZES = (16, 32)
BATCH_LINESTYLES = ("--", "-")


def _selected_batch_sizes(df: pd.DataFrame) -> tuple[int, ...]:
    available = sorted(
        int(value)
        for value in pd.to_numeric(df["max_num_seqs"], errors="coerce").dropna().unique()
    )
    selected = [batch_size for batch_size in PREFERRED_BATCH_SIZES if batch_size in available]
    for batch_size in available:
        if batch_size not in selected:
            selected.append(batch_size)
    return tuple(selected[:2])


def _batch_linestyle(batch_sizes: tuple[int, ...], batch_size: int) -> str:
    return BATCH_LINESTYLES[batch_sizes.index(batch_size) % len(BATCH_LINESTYLES)]


def load_data(path: Path) -> pd.DataFrame:
    df = load_summary_from_outputs(path)
    required = {
        "model",
        "consume_profile_label",
        "policy",
        "max_num_seqs",
        "lambda_req_s",
        "mean_handling_users",
        "ttfc_p95_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"outputs summary data missing required columns: {sorted(missing)}")
    df = df.copy()
    df["policy"] = df["policy"].map(normalize_policy)

    batch_sizes = _selected_batch_sizes(df)
    df = df[df["max_num_seqs"].isin(batch_sizes)]
    if df.empty:
        raise ValueError(
            "Figure 5.3 needs rows with at least one finite max_num_seqs."
        )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["model", "consume_profile_label", "policy", "max_num_seqs", "lambda_req_s"],
            as_index=False,
        )
        .agg(
            mean_handling_users=("mean_handling_users", "mean"),
            ttfc_p95_s=("ttfc_p95_s", "mean"),
        )
        .sort_values(
            ["model", "consume_profile_label", "policy", "max_num_seqs", "lambda_req_s"]
        )
    )


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y0 = min(ax.get_position().y0 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        fig.text(
            0.018,
            (y0 + y1) / 2,
            profile,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


def _set_arrival_rate_axis(ax, rates: list[float]) -> None:
    if not rates:
        return
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{rate:g}" for rate in rates], rotation=90, ha="center")
    ax.tick_params(axis="x", labelsize=9)
    if len(rates) == 1:
        pad = max(0.25, abs(rates[0]) * 0.08)
    else:
        pad = max(0.25, (max(rates) - min(rates)) * 0.02)
    ax.set_xlim(min(rates) - pad, max(rates) + pad)


def _set_y_range(ax, values: pd.Series, *, floor_zero: bool = True) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    y_min = float(finite.min())
    y_max = float(finite.max())
    if y_max <= y_min:
        pad = max(0.05, abs(y_max) * 0.08)
    else:
        pad = max((y_max - y_min) * 0.08, abs(y_max) * 0.015, 0.02)
    lower = y_min - pad
    upper = y_max + pad
    if floor_zero:
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + pad
    ax.set_ylim(lower, upper)


def _legend_handles(batch_sizes: tuple[int, ...]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=POLICY_COLORS[policy],
            linestyle=_batch_linestyle(batch_sizes, batch_size),
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=f"{POLICY_PLOT_LABELS[policy]} (Batch size {batch_size})",
        )
        for policy in POLICY_ORDER
        for batch_size in batch_sizes
    ]


def preprocess_fig5_3(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_data(input_path)
    agg = aggregate(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig5_3(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    agg = pd.read_csv(input_path)
    models = ordered_models(agg)
    profiles = ordered_consume_profiles(agg)
    batch_sizes = _selected_batch_sizes(agg)
    arrival_rates = sorted(agg["lambda_req_s"].unique().tolist())
    w, h = pps.fig_size("double", ratio=0.62)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * 2,
        figsize=(w * 1.45, h * max(1, len(profiles))),
        gridspec_kw={
            "hspace": 0.34,
            "wspace": 0.34,
        },
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.07,
        top=0.88,
        hspace=0.62,
        wspace=0.38,
    )

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            ax_capacity = axes[profile_idx, model_idx * 2]
            ax_stall = axes[profile_idx, model_idx * 2 + 1]
            agg_m = agg[
                (agg["model"] == model)
                & (agg["consume_profile_label"] == profile)
            ]

            for policy in POLICY_ORDER:
                for batch_size in batch_sizes:
                    sub = agg_m[
                        (agg_m["policy"] == policy)
                        & (agg_m["max_num_seqs"] == batch_size)
                    ]
                    if sub.empty:
                        continue
                    color = POLICY_COLORS[policy]
                    linestyle = _batch_linestyle(batch_sizes, batch_size)

                    ax_capacity.plot(
                        sub["lambda_req_s"],
                        sub["mean_handling_users"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.4,
                        marker="o",
                        markersize=3,
                        markerfacecolor=color,
                    )
                    ax_stall.plot(
                        sub["lambda_req_s"],
                        sub["ttfc_p95_s"],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.4,
                        marker="o",
                        markersize=3,
                        markerfacecolor=color,
                    )
            ax_capacity.set_ylabel("# of In-flight Users" if model_idx == 0 else "")
            ax_stall.set_ylabel(
                "TTFC p95 (s)" if model_idx == 0 else ""
            )
            if profile_idx == len(profiles) - 1:
                ax_stall.set_xlabel("Arrival rate (req/s)")
                ax_capacity.set_xlabel("Arrival rate (req/s)")
            ax_capacity.set_title(
                f"{_short_model_label(model)}\n# of In-flight Users",
                pad=4,
            )
            ax_stall.set_title(
                f"{_short_model_label(model)}\nTTFC p95",
                pad=4,
            )
            _set_arrival_rate_axis(ax_capacity, arrival_rates)
            _set_arrival_rate_axis(ax_stall, arrival_rates)
            _set_y_range(ax_capacity, agg_m["mean_handling_users"])
            _set_y_range(ax_stall, agg_m["ttfc_p95_s"])
            pps.clean_axes(ax_capacity, legend=False)
            pps.clean_axes(ax_stall, legend=False)

    _add_row_titles(fig, axes, profiles)
    legend = fig.legend(
        handles=_legend_handles(batch_sizes),
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        borderpad=0.35,
        handlelength=1.4,
        columnspacing=0.9,
        labelspacing=0.35,
    )
    pps.frame_legend(legend)

    output_path = Path(output_path)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed Figure 5.3 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_3(args.data, args.out)


if __name__ == "__main__":
    main()
