from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _outputs_data import ordered_consume_profiles, ordered_models, short_model_label
from _policy_style import POLICY_ORDER
from _unit_miss_diagnostics import POLICY_LABELS, write_unit_miss_cells
import paper_plot_style as pps

_BASENAME = "fig6_7c_batch_size_operating_heatmap"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
CONTOUR_LEVELS = (
    (1.0, "#222222", "1% unit miss"),
    (5.0, "#e6e6e6", "5% unit miss"),
)


def preprocess_fig6_7c(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    write_unit_miss_cells(input_path, output_dir, basename=_BASENAME)


def _pivot_matrix(
    panel: pd.DataFrame,
    *,
    value_column: str,
    batch_sizes: list[int],
    rates: list[float],
) -> np.ndarray:
    pivot = panel.pivot_table(
        index="max_num_seqs",
        columns="lambda_req_s",
        values=value_column,
        aggfunc="mean",
    )
    return pivot.reindex(index=batch_sizes, columns=rates).to_numpy(dtype=float)


def _pivot_validity(
    panel: pd.DataFrame,
    *,
    batch_sizes: list[int],
    rates: list[float],
) -> np.ndarray:
    pivot = panel.pivot_table(
        index="max_num_seqs",
        columns="lambda_req_s",
        values="cell_valid",
        aggfunc=lambda values: bool(pd.Series(values).all()),
    )
    return pivot.reindex(index=batch_sizes, columns=rates).to_numpy()


def _annotate_cells(
    ax,
    values: np.ndarray,
    annotations: np.ndarray,
    validity: np.ndarray,
    *,
    color_vmax: float,
) -> None:
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            annotation = annotations[row_idx, col_idx]
            valid_value = validity[row_idx, col_idx]
            if np.isfinite(value) and pd.notna(valid_value) and not bool(valid_value):
                ax.add_patch(
                    Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1.0,
                        1.0,
                        facecolor=(1.0, 1.0, 1.0, 0.0),
                        edgecolor="#777777",
                        hatch="///",
                        linewidth=0.0,
                        alpha=0.55,
                    )
                )
            if not np.isfinite(annotation):
                continue
            text_color = "white" if np.isfinite(value) and value > color_vmax * 0.55 else "black"
            if pd.notna(valid_value) and not bool(valid_value):
                text_color = "black"
            ax.text(
                col_idx,
                row_idx,
                f"{annotation:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )


def _draw_contours(ax, values: np.ndarray) -> None:
    if values.shape[0] < 2 or values.shape[1] < 2:
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    x = np.arange(values.shape[1])
    y = np.arange(values.shape[0])
    masked = np.ma.masked_invalid(values)
    for level, color, _label in CONTOUR_LEVELS:
        if float(finite.min()) < level < float(finite.max()):
            ax.contour(
                x,
                y,
                masked,
                levels=[level],
                colors=[color],
                linewidths=1.15,
            )


def _row_keys(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (profile, model)
        for profile in ordered_consume_profiles(df)
        for model in ordered_models(df[df["consume_profile_label"] == profile])
    ]


def plot_fig6_7c(input_path: Path, output_path: Path) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    df = df.copy()
    df["lambda_req_s"] = pd.to_numeric(df["lambda_req_s"], errors="coerce")
    df["max_num_seqs"] = pd.to_numeric(df["max_num_seqs"], errors="coerce").astype(int)
    df["unit_deadline_miss_rate_pct"] = (
        pd.to_numeric(df["unit_deadline_miss_rate"], errors="coerce") * 100.0
    )
    df["mean_admitted_inflight_requests"] = pd.to_numeric(
        df["mean_admitted_inflight_requests"],
        errors="coerce",
    )
    rows = _row_keys(df)
    if not rows:
        raise ValueError("Heatmap diagnostic needs model/profile data.")

    finite = df["unit_deadline_miss_rate_pct"].dropna()
    if finite.empty:
        raise ValueError("Heatmap diagnostic has no finite unit miss values.")
    color_vmax = max(5.0, float(finite.quantile(0.95)))
    norm = Normalize(vmin=0.0, vmax=color_vmax)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#f2f2f2")

    w, _h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        len(rows),
        len(POLICY_ORDER),
        figsize=(w * 1.58, 1.75 * len(rows) + 0.85),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.145,
        right=0.91,
        bottom=0.06,
        top=0.94,
        hspace=0.30,
        wspace=0.20,
    )

    image = None
    for row_idx, (profile, model) in enumerate(rows):
        row_df = df[
            (df["consume_profile_label"] == profile)
            & (df["model"] == model)
        ]
        rates = sorted(row_df["lambda_req_s"].dropna().unique().tolist())
        batch_sizes = sorted(row_df["max_num_seqs"].dropna().astype(int).unique().tolist())
        for col_idx, policy in enumerate(POLICY_ORDER):
            ax = axes[row_idx, col_idx]
            panel = row_df[row_df["policy"] == policy]
            values = _pivot_matrix(
                panel,
                value_column="unit_deadline_miss_rate_pct",
                batch_sizes=batch_sizes,
                rates=rates,
            )
            annotations = _pivot_matrix(
                panel,
                value_column="mean_admitted_inflight_requests",
                batch_sizes=batch_sizes,
                rates=rates,
            )
            validity = _pivot_validity(panel, batch_sizes=batch_sizes, rates=rates)
            image = ax.imshow(
                np.ma.masked_invalid(values),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )
            _annotate_cells(
                ax,
                values,
                annotations,
                validity,
                color_vmax=color_vmax,
            )
            _draw_contours(ax, values)
            ax.set_title(
                POLICY_LABELS.get(policy, policy),
                fontsize=11,
                fontweight="bold",
                pad=2,
            )
            ax.set_xticks(np.arange(len(rates)))
            ax.set_xticklabels([f"{rate:g}" for rate in rates], rotation=90)
            ax.set_yticks(np.arange(len(batch_sizes)))
            ax.set_yticklabels([str(batch) for batch in batch_sizes])
            ax.set_xlabel("Request rate (req/s)" if row_idx == len(rows) - 1 else "")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{profile}\n{short_model_label(model)}\nBatch size",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.grid(False)
            pps.clean_axes(ax, legend=False)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
        cbar.set_label("Unit deadline miss rate (%)", fontweight="bold")
    contour_handles = [
        Line2D([0], [0], color=color, linewidth=1.15, label=label)
        for _level, color, label in CONTOUR_LEVELS
    ]
    mask_handle = Patch(
        facecolor="white",
        edgecolor="#777777",
        hatch="///",
        label="Validity/no-harm fail",
    )
    legend = fig.legend(
        handles=contour_handles + [mask_handle],
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.5,
    )
    pps.frame_legend(legend)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig6_7c(args.data, args.out)


if __name__ == "__main__":
    main()
