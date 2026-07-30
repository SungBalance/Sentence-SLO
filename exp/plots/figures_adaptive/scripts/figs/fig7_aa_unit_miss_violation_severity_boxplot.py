from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from _outputs_data import (
    chunk_stall_duration,
    chunks_by_request,
    display_consume_profile_label,
    iter_run_dirs,
    ordered_consume_profiles,
    ordered_models,
    request_map,
    short_model_label,
)
from _policy_style import POLICY_COLORS, POLICY_ORDER, normalize_policy
from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps

_BASENAME = "fig7_aa_unit_miss_violation_severity_boxplot"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(f"{_BASENAME}.png")

POLICY_LABELS = {
    "baseline": "Baseline",
    "sslo": "ProgressServe",
}


def _run_context(run) -> dict | None:
    summary_path = run.path / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    config = summary.get("config") or {}
    raw_policy = str(config.get("run_kind") or run.policy)
    policy = normalize_policy(raw_policy)
    if policy not in POLICY_ORDER:
        return None
    return {
        "model": config.get("model") or run.model_dir,
        "consume_profile_label": run.consume_profile_label,
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "max_num_seqs": int(run.max_num_seqs),
        "lambda_req_s": float(run.lambda_req_s),
        "seed": int(run.seed),
    }


def load_violated_unit_severities(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run in iter_run_dirs(outputs_root):
        context = _run_context(run)
        if context is None:
            continue

        requests = request_map(run.path)
        chunks = chunks_by_request(run.path)
        cell_row = dict(context)
        cell_row.update(
            {
                "row_kind": "cell",
                "request_id": "",
                "unit_index": None,
                "violation_severity_s": None,
            }
        )
        rows.append(cell_row)
        for request_id, request_chunks in chunks.items():
            request = requests.get(request_id, {})
            if request.get("terminal_outcome", "completed") != "completed":
                continue
            if request.get("in_window", True) is False:
                continue
            for chunk in request_chunks:
                severity_s = chunk_stall_duration(chunk)
                if severity_s <= 0.0:
                    continue
                row = dict(context)
                row.update(
                    {
                        "row_kind": "unit",
                        "request_id": request_id,
                        "unit_index": int(chunk.get("unit_index", chunk.get("chunk_idx", 0)) or 0),
                        "violation_severity_s": float(severity_s),
                    }
                )
                rows.append(row)

    if not rows:
        raise ValueError("No unit-miss severity rows were found.")
    return pd.DataFrame(rows).sort_values(
        [
            "consume_profile_label",
            "model",
            "policy",
            "max_num_seqs",
            "lambda_req_s",
            "seed",
            "request_id",
            "unit_index",
        ]
    )


def preprocess_fig7_aa(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    load_violated_unit_severities(input_path).to_csv(
        output_dir / f"{_BASENAME}.csv",
        index=False,
    )


def _hierarchical_positions(
    batch_sizes: list[int],
    rates: list[float],
    *,
    gap: float = 0.85,
) -> tuple[dict[tuple[int, float], float], list[float], list[str], dict[int, float]]:
    positions: dict[tuple[int, float], float] = {}
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    batch_centers: dict[int, float] = {}
    cursor = 0.0
    for batch_size in batch_sizes:
        batch_positions: list[float] = []
        for rate in rates:
            positions[(batch_size, rate)] = cursor
            tick_positions.append(cursor)
            tick_labels.append(f"{rate:g}")
            batch_positions.append(cursor)
            cursor += 1.0
        batch_centers[batch_size] = sum(batch_positions) / len(batch_positions)
        cursor += gap
    return positions, tick_positions, tick_labels, batch_centers


def _draw_batch_labels(
    ax,
    batch_centers: dict[int, float],
    rates_per_batch: int,
    *,
    gap: float,
) -> None:
    for batch_size, center in batch_centers.items():
        ax.text(
            center,
            -0.23,
            f"{batch_size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    for center in batch_centers.values():
        right_edge = center + rates_per_batch / 2 + gap / 2 - 0.5
        ax.axvline(right_edge, color="#d5d5d5", linewidth=0.7, zorder=0)


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.026,
            profile,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        pps.bold_text(text)


def _add_model_row_titles(fig, axes, models: list[str]) -> None:
    for model_idx, model in enumerate(models):
        row_axes = [ax for ax in axes[model_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        x0 = min(ax.get_position().x0 for ax in row_axes)
        x1 = max(ax.get_position().x1 for ax in row_axes)
        y1 = max(ax.get_position().y1 for ax in row_axes)
        title_offset = 0.055 if model_idx == 0 else 0.043
        text = fig.text(
            (x0 + x1) / 2,
            y1 + title_offset,
            short_model_label(model),
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )
        pps.bold_text(text)


def _set_y_range(ax, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        ax.set_ylim(0.0, 0.1)
        return
    y_max = float(finite.max())
    ax.set_ylim(0.0, max(0.1, y_max * 1.12))


def _boxplot_visible_upper(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return None
    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    upper_fence = q3 + 1.5 * (q3 - q1)
    visible = finite[finite <= upper_fence]
    if visible.empty:
        return float(finite.max())
    return float(visible.max())


def plot_fig7_aa(input_path: Path, output_path: Path) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    df = df.copy()
    if "row_kind" not in df.columns:
        df["row_kind"] = "unit"
    df["violation_severity_s"] = pd.to_numeric(
        df["violation_severity_s"],
        errors="coerce",
    )
    sample_df = df[df["row_kind"] == "unit"].dropna(subset=["violation_severity_s"])
    if sample_df.empty:
        raise ValueError("Figure 7.aa has no violated unit severity samples.")

    profiles = ordered_consume_profiles(df)
    models = ordered_models(df)
    if not profiles or not models:
        raise ValueError("Figure 7.aa needs model/profile data.")

    w, h = pps.fig_size("double", ratio=0.44)
    fig, axes = plt.subplots(
        len(models),
        len(profiles),
        figsize=(w * 1.68, h * 1.18 * max(1, len(models))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.12,
        top=0.84,
        hspace=0.82,
        wspace=0.34,
    )

    policies = [policy for policy in POLICY_ORDER if policy in set(df["policy"])]
    width = 0.30
    gap = 0.85
    offsets = {
        policy: (idx - (len(policies) - 1) / 2.0) * width * 1.25
        for idx, policy in enumerate(policies)
    }

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            ax = axes[model_idx, profile_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
            ]
            sample_panel = sample_df[
                (sample_df["consume_profile_label"] == profile)
                & (sample_df["model"] == model)
            ]
            batch_sizes = sorted(
                pd.to_numeric(panel["max_num_seqs"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            rates = sorted(
                pd.to_numeric(panel["lambda_req_s"], errors="coerce").dropna().unique()
            )
            if not batch_sizes or not rates:
                ax.text(
                    0.5,
                    0.5,
                    "No cells",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )
                pps.clean_axes(ax, legend=False)
                continue

            positions, tick_positions, tick_labels, batch_centers = _hierarchical_positions(
                batch_sizes,
                rates,
                gap=gap,
            )
            visible_uppers = []
            p99_values = []
            for policy in policies:
                box_data = []
                box_positions = []
                p99_positions = []
                policy_p99_values = []
                sub = sample_panel[sample_panel["policy"] == policy]
                for batch_size in batch_sizes:
                    for rate in rates:
                        values = sub[
                            (sub["max_num_seqs"].astype(int) == int(batch_size))
                            & (sub["lambda_req_s"].astype(float) == float(rate))
                        ]["violation_severity_s"]
                        values = pd.to_numeric(values, errors="coerce").dropna()
                        if values.empty:
                            continue
                        box_data.append(values)
                        visible_upper = _boxplot_visible_upper(values)
                        if visible_upper is not None:
                            visible_uppers.append(visible_upper)
                        p99_value = float(values.quantile(0.99))
                        p99_values.append(p99_value)
                        policy_p99_values.append(p99_value)
                        box_positions.append(
                            positions[(int(batch_size), float(rate))] + offsets[policy]
                        )
                        p99_positions.append(
                            positions[(int(batch_size), float(rate))] + offsets[policy]
                        )
                if not box_data:
                    continue
                box = ax.boxplot(
                    box_data,
                    positions=box_positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                color = POLICY_COLORS[policy]
                for patch in box["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.35)
                    patch.set_edgecolor(color)
                for key in ("whiskers", "caps", "medians"):
                    for artist in box[key]:
                        artist.set_color(color)
                        artist.set_linewidth(1.1)
                ax.scatter(
                    p99_positions,
                    policy_p99_values,
                    marker="D",
                    s=16,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.9,
                    zorder=4,
                )

            _set_y_range(ax, pd.Series(visible_uppers + p99_values))
            _draw_batch_labels(ax, batch_centers, len(rates), gap=gap)
            ax.set_title(
                display_consume_profile_label(profile),
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0)
            ax.tick_params(axis="x", labelsize=7, pad=1)
            ax.set_xlabel(
                "Request rate within batch size" if model_idx == len(models) - 1 else ""
            )
            ax.set_ylabel("Unit violation amount (s)" if profile_idx == 0 else "")
            ax.text(
                0.5,
                -0.34,
                "Batch size",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            pps.clean_axes(ax, legend=False)

    _add_model_row_titles(fig, axes, models)
    handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            label=POLICY_LABELS.get(policy, policy),
        )
        for policy in policies
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            color="#333333",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=5,
            label="P99",
        )
    )
    legend = fig.legend(
        handles=handles,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
    )
    pps.frame_legend(legend)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig7_aa(args.data, args.out)


if __name__ == "__main__":
    main()
