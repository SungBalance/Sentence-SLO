from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps
from _outputs_data import (
    chunk_stall_duration,
    chunks_by_request,
    load_summary_from_outputs,
    ordered_consume_profiles,
    ordered_models,
    request_map,
    run_dir_from_summary_row,
)
from _policy_style import (
    POLICY_COLORS,
    POLICY_ORDER,
    POLICY_PLOT_LABELS,
    normalize_policy,
)

_BASENAME = "fig5_8_handling_tradeoffs"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)

PREFERRED_BATCH_SIZES = (64, 128)
BATCH_STYLES = (
    {
        "markerfacecolor": "white",
    },
    {
        "markerfacecolor": None,
    },
)
TRADEOFF_MARKERSIZE = 4.8


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


def _batch_style(batch_sizes: tuple[int, ...], batch_size: int) -> dict:
    return BATCH_STYLES[batch_sizes.index(batch_size) % len(BATCH_STYLES)]


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
    df["policy_raw"] = df["policy"]
    df["policy"] = df["policy"].map(normalize_policy)

    batch_sizes = _selected_batch_sizes(df)
    df = df[df["max_num_seqs"].isin(batch_sizes)]
    if df.empty:
        raise ValueError(
            "Figure 5.8 needs rows with at least one finite max_num_seqs."
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
            mean_request_stall_count=("mean_request_stall_count", "mean"),
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
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.036,
            profile,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        pps.bold_text(text)


def _legend_handles(batch_sizes: tuple[int, ...]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=POLICY_COLORS[policy],
            linestyle="None",
            marker="o",
            markersize=5.2,
            markerfacecolor=(
                POLICY_COLORS[policy]
                if _batch_style(batch_sizes, batch_size)["markerfacecolor"] is None
                else _batch_style(batch_sizes, batch_size)["markerfacecolor"]
            ),
            markeredgecolor=POLICY_COLORS[policy],
            label=f"{POLICY_PLOT_LABELS[policy]} (Batch size {batch_size})",
        )
        for policy in POLICY_ORDER
        for batch_size in batch_sizes
    ]


def _set_x_range(ax, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    x_max = float(finite.max())
    if x_max <= 0.0:
        ax.set_xlim(0.0, 0.001)
    else:
        pad = max(x_max * 0.08, 0.0005)
        ax.set_xlim(0.0, x_max + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_count_tick))


def _format_count_tick(value: float, _position: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


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


def _draw_tradeoff(
    ax,
    df_model: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str | None,
    batch_sizes: tuple[int, ...],
) -> None:
    for policy in POLICY_ORDER:
        for batch_size in batch_sizes:
            sub = df_model[
                (df_model["policy"] == policy)
                & (df_model["max_num_seqs"] == batch_size)
            ].sort_values("lambda_req_s")
            if sub.empty:
                continue
            style = _batch_style(batch_sizes, batch_size)
            color = POLICY_COLORS[policy]
            markerfacecolor = (
                color
                if style["markerfacecolor"] is None
                else style["markerfacecolor"]
            )
            ax.plot(
                sub[x_column],
                sub[y_column],
                color=color,
                linestyle="None",
                marker="o",
                markersize=TRADEOFF_MARKERSIZE,
                markerfacecolor=markerfacecolor,
                markeredgecolor=color,
                markeredgewidth=0.9,
                alpha=0.9,
            )

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    pps.clean_axes(ax, legend=False)
    ax.tick_params(labelbottom=True)


def preprocess_fig5_8(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_data(input_path)
    stall_counts: list[float] = []
    for _, row in df.iterrows():
        path_row = row.copy()
        path_row["policy"] = row["policy_raw"]
        run_dir = run_dir_from_summary_row(input_path, path_row)
        chunks = chunks_by_request(run_dir)
        requests = request_map(run_dir)
        request_counts = []
        for request_id, request_chunks in chunks.items():
            request = requests.get(request_id, {})
            if request.get("terminal_outcome", "completed") != "completed":
                continue
            if request.get("in_window", True) is False:
                continue
            request_counts.append(
                sum(1 for chunk in request_chunks if chunk_stall_duration(chunk) > 0.0)
            )
        stall_counts.append(
            float(pd.Series(request_counts).mean()) if request_counts else 0.0
        )
    df = df.assign(mean_request_stall_count=stall_counts)
    agg = aggregate(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig5_8(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    agg = pd.read_csv(input_path)
    models = ordered_models(agg)
    profiles = ordered_consume_profiles(agg)
    batch_sizes = _selected_batch_sizes(agg)
    w, h = pps.fig_size("double", ratio=0.46)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * 2,
        figsize=(w * 1.45, h * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.07,
        top=0.84,
        hspace=0.76,
        wspace=0.22,
    )

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            df_model = agg[
                (agg["model"] == model)
                & (agg["consume_profile_label"] == profile)
            ]
            left_ax = axes[profile_idx, model_idx * 2]
            right_ax = axes[profile_idx, model_idx * 2 + 1]
            _draw_tradeoff(
                left_ax,
                df_model,
                x_column="mean_request_stall_count",
                y_column="mean_handling_users",
                xlabel="Avg. Stall Count per Request" if profile_idx == len(profiles) - 1 else "",
                ylabel="# of In-flight Users" if model_idx == 0 else None,
                batch_sizes=batch_sizes,
            )
            _draw_tradeoff(
                right_ax,
                df_model,
                x_column="mean_request_stall_count",
                y_column="ttfc_p95_s",
                xlabel="Avg. Stall Count per Request" if profile_idx == len(profiles) - 1 else "",
                ylabel="TTFC p95 (s)" if model_idx == 0 else None,
                batch_sizes=batch_sizes,
            )
            _set_y_range(left_ax, df_model["mean_handling_users"])
            _set_y_range(right_ax, df_model["ttfc_p95_s"])
            _set_x_range(left_ax, agg["mean_request_stall_count"])
            _set_x_range(right_ax, agg["mean_request_stall_count"])
            if profile_idx == 0:
                left_ax.set_title(f"{_short_model_label(model)}\n# of In-flight Users")
                right_ax.set_title(f"{_short_model_label(model)}\nTTFC p95")

    for ax in axes.ravel():
        ax.tick_params(labelbottom=True)

    _add_row_titles(fig, axes, profiles)
    legend = fig.legend(
        handles=_legend_handles(batch_sizes),
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
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
                        help="Processed Figure 5.8 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_8(args.data, args.out)


if __name__ == "__main__":
    main()
