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
    finite_float,
    load_summary_from_outputs,
    normalize_request_id,
    ordered_consume_profiles,
    ordered_models,
    read_jsonl,
    run_dir_from_summary_row,
)
from _policy_style import POLICY_COLORS, POLICY_ORDER, POLICY_PLOT_LABELS, normalize_policy

_BASENAME = "fig5_6_policy_behavior_no_harm"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)

MID_BATCH_SIZE = 64
PREFERRED_REQUEST_RATES = (2.0, 8.0, 20.0)
METRICS = (
    ("handling_users", "# of In-flight Users"),
    ("queue_stall_s", "Queue Stall"),
    ("ttfc_s", "TTFC"),
    ("pending_time_s", "Pending Time"),
)


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


def _selected_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    request_rates = _selected_request_rates(df[df["max_num_seqs"] == MID_BATCH_SIZE])
    selected = df[
        (df["max_num_seqs"] == MID_BATCH_SIZE)
        & (df["lambda_req_s"].isin(request_rates))
        & (df["policy"].isin(POLICY_ORDER))
    ].copy()
    if selected.empty:
        raise ValueError(
            "Figure 5.6 needs rows at "
            f"max_num_seqs={MID_BATCH_SIZE}."
        )

    return selected


def _metric_rows(outputs_root: Path, summary_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, summary_row in summary_rows.iterrows():
        path_row = summary_row.copy()
        path_row["policy"] = summary_row["policy_raw"]
        run_dir = run_dir_from_summary_row(outputs_root, path_row)
        requests_path = run_dir / "requests.jsonl"
        if not requests_path.exists():
            raise ValueError(f"Figure 5.6 needs request metrics at {requests_path}.")

        for request in read_jsonl(requests_path):
            if request.get("terminal_outcome", "completed") != "completed":
                continue
            if request.get("in_window", True) is False:
                continue

            metric_values = {
                "queue_stall_s": finite_float(request.get("queue_stall_s")),
                "ttfc_s": finite_float(request.get("ttfc") or request.get("TTFC")),
            }
            if summary_row["policy"] == "sslo":
                metric_values["pending_time_s"] = finite_float(
                    request.get("total_pending_time_s")
                )
            for metric, value in metric_values.items():
                if value is None or not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "model": summary_row["model"],
                        "consume_profile_label": summary_row["consume_profile_label"],
                        "policy": summary_row["policy"],
                        "max_num_seqs": int(summary_row["max_num_seqs"]),
                        "lambda_req_s": float(summary_row["lambda_req_s"]),
                        "seed": int(summary_row["seed"]),
                        "sample_id": normalize_request_id(request.get("request_id", "")),
                        "sample_kind": "request",
                        "metric": metric,
                        "value_s": max(0.0, float(value)),
                    }
                )

        stats_path = run_dir / "scheduler_stats.jsonl"
        if not stats_path.exists():
            raise ValueError(f"Figure 5.6 needs scheduler stats at {stats_path}.")
        for step_idx, step in enumerate(read_jsonl(stats_path)):
            if step.get("kind") != "step":
                continue
            value = finite_float(step.get("num_handling_users"))
            if value is None or not np.isfinite(value):
                continue
            rows.append(
                {
                    "model": summary_row["model"],
                    "consume_profile_label": summary_row["consume_profile_label"],
                    "policy": summary_row["policy"],
                    "max_num_seqs": int(summary_row["max_num_seqs"]),
                    "lambda_req_s": float(summary_row["lambda_req_s"]),
                    "seed": int(summary_row["seed"]),
                    "sample_id": f"step_{step_idx}",
                    "sample_kind": "scheduler_step",
                    "metric": "handling_users",
                    "value_s": max(0.0, float(value)),
                }
            )

    if not rows:
        raise ValueError("Figure 5.6 needs request metrics and scheduler stats.")
    return pd.DataFrame(rows)


def _set_y_range(ax, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    y_min = float(finite.min())
    y_max = float(finite.max())
    if y_max <= y_min:
        pad = max(0.05, abs(y_max) * 0.08)
    else:
        pad = max((y_max - y_min) * 0.08, abs(y_max) * 0.015, 0.02)
    lower = max(0.0, y_min - pad)
    upper = y_max + pad
    if upper <= lower:
        upper = lower + pad
    ax.set_ylim(lower, upper)


def _available_metrics(df: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    available = []
    for metric, label in METRICS:
        values = pd.to_numeric(
            df[df["metric"] == metric]["value_s"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            available.append((metric, label))
    return tuple(available)


def _draw_metric_boxplots(
    ax,
    df_model: pd.DataFrame,
    metric: str,
    request_rates: tuple[float, ...],
) -> None:
    x = np.arange(len(request_rates), dtype=float)
    width = 0.28
    offsets = {
        "baseline": -width / 1.7,
        "sslo": width / 1.7,
    }
    policies = ("sslo",) if metric == "pending_time_s" else POLICY_ORDER

    for policy in policies:
        box_data = []
        positions = []
        for rate_idx, rate in enumerate(request_rates):
            values = df_model[
                (df_model["policy"] == policy)
                & (df_model["lambda_req_s"] == rate)
                & (df_model["metric"] == metric)
            ]["value_s"]
            box_data.append(values if not values.empty else pd.Series([np.nan]))
            offset = 0.0 if len(policies) == 1 else offsets[policy]
            positions.append(x[rate_idx] + offset)

        box = ax.boxplot(
            box_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        color = POLICY_COLORS[policy]
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.34)
            patch.set_edgecolor(color)
        for key in ("whiskers", "caps", "medians"):
            for artist in box[key]:
                artist.set_color(color)
                artist.set_linewidth(1.1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{rate:g}" for rate in request_rates])
    _set_y_range(ax, df_model[df_model["metric"] == metric]["value_s"])
    pps.clean_axes(ax, legend=False)


def preprocess_fig5_6(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_data(input_path)
    selected = _selected_summary_rows(df)
    request_metrics = _metric_rows(input_path, selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_metrics.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig5_6(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    request_rates = tuple(sorted(df["lambda_req_s"].dropna().unique().tolist()))
    metrics = _available_metrics(df)
    if not models or not profiles:
        raise ValueError("Figure 5.6 needs at least one model in outputs data.")
    if not request_rates or not metrics:
        raise ValueError("Figure 5.6 needs at least one request rate and metric.")

    w, h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * len(metrics),
        figsize=(w * 1.7, h * 1.15 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.065,
        top=0.87,
        hspace=0.44,
        wspace=0.42,
    )

    for profile_index, profile in enumerate(profiles):
        for model_index, model in enumerate(models):
            df_model = df[
                (df["model"] == model)
                & (df["consume_profile_label"] == profile)
            ]
            for metric_index, (metric, ylabel) in enumerate(metrics):
                ax = axes[profile_index, model_index * len(metrics) + metric_index]
                _draw_metric_boxplots(ax, df_model, metric, request_rates)
                if model_index == 0 and metric_index == 0:
                    ax.set_ylabel("# of In-flight Users")
                elif model_index == 0 and metric_index == 1:
                    ax.set_ylabel("Time (sec.)")
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("Rate (req/s)" if profile_index == len(profiles) - 1 else "")
                if profile_index == 0:
                    ax.set_title(
                        f"{_short_model_label(model)}\n{ylabel}",
                        fontsize=10,
                        fontweight="bold",
                        pad=5,
                    )

    _add_row_titles(fig, axes, profiles)
    legend_handles = [
        Patch(facecolor=POLICY_COLORS[policy], label=POLICY_PLOT_LABELS[policy])
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=11,
        title=f"Batch size: {MID_BATCH_SIZE}",
        title_fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
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
                        help="Processed Figure 5.6 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_6(args.data, args.out)


if __name__ == "__main__":
    main()
