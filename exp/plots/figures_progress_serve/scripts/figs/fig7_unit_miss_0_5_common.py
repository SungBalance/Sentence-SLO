from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _outputs_data import (
    display_consume_profile_label,
    finite_float,
    load_summary_from_outputs,
    ordered_consume_profiles,
    ordered_models,
    read_jsonl,
    run_dir_from_summary_row,
    short_model_label,
)
from _policy_style import POLICY_COLORS, POLICY_ORDER, normalize_policy
import paper_plot_style as pps

SOURCE_BASENAME = "fig6_7a_unit_miss_budget_supported_users"
DEFAULT_SOURCE_DATA = processed_csv_path(SOURCE_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
THRESHOLD_CONFIGS = {
    "strict": {
        "suffix": "0",
        "label": "=0% unit miss",
    },
    "conservative": {
        "suffix": "0.5",
        "label": "<0.5% unit miss",
    },
}
DEFAULT_THRESHOLD_KEY = "conservative"
POLICY_LABELS = {
    "baseline": "Baseline",
    "sslo": "ProgressServe",
}
MISSING_RATE_OFFSETS = {
    "baseline": 0.0,
    "sslo": 0.0,
}


def _threshold_suffix(threshold_key: str) -> str:
    return THRESHOLD_CONFIGS[threshold_key]["suffix"]


def _fig7_csv_name(fig_id: str, metric_slug: str, threshold_key: str) -> str:
    return f"fig7_{fig_id}_unit_miss_{metric_slug}_{_threshold_suffix(threshold_key)}"


def _load_threshold_points(
    input_path: Path,
    *,
    threshold_key: str,
    max_rank: int,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["threshold_key"] == threshold_key].copy()
    if df.empty:
        label = THRESHOLD_CONFIGS[threshold_key]["label"]
        raise ValueError(f"Figure 7 needs {label} rows from Figure 6.7a.")
    if "selection_rank" not in df.columns:
        df["selection_rank"] = 1
    numeric_columns = [
        "lambda_req_s",
        "mean_admitted_inflight_requests",
        "selected_max_num_seqs",
        "unit_deadline_miss_rate",
        "valid_consumable_unit_count",
        "violated_unit_count",
        "run_count",
        "selection_rank",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["selection_rank"] = df["selection_rank"].fillna(1).astype(int)
    df = df[df["selection_rank"] <= max_rank].copy()
    df = df.dropna(subset=["lambda_req_s", "selected_max_num_seqs"])
    if df.empty:
        raise ValueError("Figure 7 threshold source has no numeric rows.")
    df["selected_max_num_seqs"] = df["selected_max_num_seqs"].astype(int)
    df["run_count"] = df["run_count"].astype(int)
    return df.sort_values(
        [
            "consume_profile_label",
            "model",
            "policy",
            "selection_rank",
            "lambda_req_s",
        ]
    )


def preprocess_fig7_1(
    input_path: Path = DEFAULT_SOURCE_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for threshold_key in THRESHOLD_CONFIGS:
        _load_threshold_points(input_path, threshold_key=threshold_key, max_rank=1).to_csv(
            output_dir
            / f"{_fig7_csv_name('1', 'in-flight_requests', threshold_key)}.csv",
            index=False,
        )


def _summary_with_normalized_policy(outputs_root: Path) -> pd.DataFrame:
    summary = load_summary_from_outputs(outputs_root).copy()
    summary["policy_raw"] = summary["policy"]
    summary["policy"] = summary["policy"].map(normalize_policy)
    return summary


def _matching_summary_rows(selected: pd.DataFrame, outputs_root: Path) -> pd.DataFrame:
    summary = _summary_with_normalized_policy(outputs_root)
    rows = []
    for _, point in selected.iterrows():
        matched = summary[
            (summary["model"] == point["model"])
            & (summary["consume_profile_label"] == point["consume_profile_label"])
            & (summary["policy"] == point["policy"])
            & (summary["max_num_seqs"].astype(int) == int(point["selected_max_num_seqs"]))
            & (summary["lambda_req_s"].astype(float) == float(point["lambda_req_s"]))
        ].copy()
        if matched.empty:
            continue
        for column in (
            "threshold_key",
            "threshold_label",
            "threshold_value",
            "threshold_mode",
            "selection_rank",
            "unit_deadline_miss_rate",
            "valid_consumable_unit_count",
            "violated_unit_count",
            "run_count",
        ):
            matched[column] = point[column]
        matched["selected_max_num_seqs"] = int(point["selected_max_num_seqs"])
        matched["selected_mean_admitted_inflight_requests"] = point[
            "mean_admitted_inflight_requests"
        ]
        rows.append(matched)
    if not rows:
        raise ValueError("Figure 7 could not match selected threshold points to raw runs.")
    return pd.concat(rows, ignore_index=True, sort=False)


def preprocess_fig7_2(
    selected_path: Path = DEFAULT_SOURCE_DATA,
    outputs_root: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for threshold_key in THRESHOLD_CONFIGS:
        selected = _load_threshold_points(
            selected_path,
            threshold_key=threshold_key,
            max_rank=1,
        )
        matched = _matching_summary_rows(selected, outputs_root)
        rows = []
        for _, run_row in matched.iterrows():
            path_row = run_row.copy()
            path_row["policy"] = run_row["policy_raw"]
            run_dir = run_dir_from_summary_row(outputs_root, path_row)
            requests_path = run_dir / "requests.jsonl"
            if not requests_path.exists():
                continue
            for request in read_jsonl(requests_path):
                if request.get("terminal_outcome", "completed") != "completed":
                    continue
                if request.get("in_window", True) is False:
                    continue
                value = finite_float(request.get("queue_stall_s"))
                if value is None:
                    continue
                rows.append(
                    {
                        "model": run_row["model"],
                        "consume_profile_label": run_row["consume_profile_label"],
                        "policy": run_row["policy"],
                        "policy_label": POLICY_LABELS.get(
                            run_row["policy"],
                            run_row["policy"],
                        ),
                        "lambda_req_s": float(run_row["lambda_req_s"]),
                        "selected_max_num_seqs": int(run_row["selected_max_num_seqs"]),
                        "seed": int(run_row["seed"]),
                        "request_id": request.get("request_id"),
                        "queue_stall_s": max(0.0, float(value)),
                    }
                )
        if not rows:
            raise ValueError("Figure 7.2 found no queue_stall_s samples.")
        pd.DataFrame(rows).to_csv(
            output_dir
            / f"{_fig7_csv_name('2', 'in-queue_stall', threshold_key)}.csv",
            index=False,
        )


def preprocess_fig7_3(
    selected_path: Path = DEFAULT_SOURCE_DATA,
    outputs_root: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for threshold_key in THRESHOLD_CONFIGS:
        selected = _load_threshold_points(
            selected_path,
            threshold_key=threshold_key,
            max_rank=1,
        )
        matched = _matching_summary_rows(selected, outputs_root)
        matched["ttft_p99_s"] = pd.to_numeric(matched["ttft_p99_s"], errors="coerce")
        matched = matched.dropna(subset=["ttft_p99_s"])
        if matched.empty:
            raise ValueError("Figure 7.3 found no ttft_p99_s values.")
        grouped = (
            matched.groupby(
                [
                    "model",
                    "consume_profile_label",
                    "policy",
                    "lambda_req_s",
                    "selected_max_num_seqs",
                ],
                as_index=False,
            )
            .agg(ttft_p99_s=("ttft_p99_s", "mean"), run_count=("seed", "nunique"))
            .sort_values(["consume_profile_label", "model", "policy", "lambda_req_s"])
        )
        grouped["policy_label"] = grouped["policy"].map(POLICY_LABELS).fillna(
            grouped["policy"]
        )
        grouped.to_csv(
            output_dir / f"{_fig7_csv_name('3', 'TTFT', threshold_key)}.csv",
            index=False,
        )


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.032,
            profile,
            ha="center",
            va="bottom",
            fontsize=15,
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


def _set_rate_axis(ax, rates: list[float]) -> None:
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{rate:g}" for rate in rates], rotation=0, ha="center")
    if rates:
        ax.set_xlim(-0.5, len(rates) - 0.5)


def _set_y_range(ax, values: pd.Series, *, floor_zero: bool = True) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    y_min = float(finite.min())
    y_max = float(finite.max())
    pad = max((y_max - y_min) * 0.10, abs(y_max) * 0.02, 0.05)
    lower = y_min - pad
    if floor_zero:
        lower = max(0.0, lower)
    ax.set_ylim(lower, y_max + pad)


def _model_y_upper(df: pd.DataFrame, y_column: str) -> dict[str, float]:
    uppers: dict[str, float] = {}
    for model, sub in df.groupby("model"):
        finite = pd.to_numeric(sub[y_column], errors="coerce").dropna()
        if finite.empty:
            continue
        y_max = float(finite.max())
        pad = max(y_max * 0.08, 0.05)
        uppers[str(model)] = max(0.1, y_max + pad)
    return uppers


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


def _model_boxplot_y_upper(
    df: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
) -> dict[str, float]:
    uppers: dict[str, float] = {}
    for model, model_df in df.groupby("model"):
        visible_uppers = []
        for _, sub in model_df.groupby(group_columns):
            visible_upper = _boxplot_visible_upper(sub[value_column])
            if visible_upper is not None:
                visible_uppers.append(visible_upper)
        if visible_uppers:
            y_max = max(visible_uppers)
            uppers[str(model)] = max(0.1, y_max * 1.12)
    return uppers


def _base_grid(df: pd.DataFrame):
    profiles = ordered_consume_profiles(df)
    models = ordered_models(df)
    if not profiles or not models:
        raise ValueError("Figure 7 needs non-empty model/profile data.")
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
        bottom=0.085,
        top=0.84,
        hspace=0.56,
        wspace=0.34,
    )
    return fig, axes, profiles, models


def _legend(fig, policies: list[str]) -> None:
    handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            label=POLICY_LABELS.get(policy, policy),
        )
        for policy in policies
    ]
    legend = fig.legend(
        handles=handles,
        fontsize=13,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)


def plot_line_metric(
    input_path: Path,
    output_path: Path,
    *,
    y_column: str,
    ylabel: str,
    share_y_by_model: bool = False,
    annotate_extrema: bool = False,
    extrema_filter_column: str | None = None,
    extrema_filter_lt: float | None = None,
) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    if "selection_rank" not in df.columns:
        df["selection_rank"] = 1
    df["selection_rank"] = pd.to_numeric(
        df["selection_rank"],
        errors="coerce",
    ).fillna(1).astype(int)
    df[y_column] = pd.to_numeric(df[y_column], errors="coerce")
    df = df.dropna(subset=[y_column])
    fig, axes, profiles, models = _base_grid(df)
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique().tolist())
    policies = [policy for policy in POLICY_ORDER if policy in set(df["policy"])]
    model_y_upper = _model_y_upper(df, y_column) if share_y_by_model else {}

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            ax = axes[model_idx, profile_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
            ]
            for policy in policies:
                policy_rows = panel[panel["policy"] == policy]
                if policy_rows.empty:
                    continue
                ranks = sorted(policy_rows["selection_rank"].dropna().astype(int).unique())
                for rank in ranks:
                    sub = policy_rows[
                        policy_rows["selection_rank"].astype(int) == rank
                    ].sort_values("lambda_req_s")
                    if sub.empty:
                        continue
                    xs = [rates.index(float(rate)) for rate in sub["lambda_req_s"]]
                    is_primary = rank == 1
                    ax.plot(
                        xs,
                        sub[y_column],
                        color=POLICY_COLORS[policy],
                        marker="o",
                        markersize=4 if is_primary else 3.6,
                        linewidth=1.4 if is_primary else 1.0,
                        linestyle="-" if is_primary else ":",
                        alpha=0.95 if is_primary else 0.62,
                    )
            _set_rate_axis(ax, rates)
            if share_y_by_model and str(model) in model_y_upper:
                ax.set_ylim(0.0, model_y_upper[str(model)])
            else:
                _set_y_range(ax, panel[y_column])
            y0, y1 = ax.get_ylim()
            marker_y = y0 + (y1 - y0) * 0.04
            present = {
                (str(row["policy"]), float(row["lambda_req_s"]))
                for _, row in panel[panel["selection_rank"].astype(int) == 1].iterrows()
            }
            for policy in policies:
                missing_rates = [
                    rate
                    for rate in rates
                    if (policy, float(rate)) not in present
                ]
                if not missing_rates:
                    continue
                ax.plot(
                    [rates.index(rate) + MISSING_RATE_OFFSETS.get(policy, 0.0) for rate in missing_rates],
                    [marker_y] * len(missing_rates),
                    linestyle="",
                    marker="x",
                    markersize=5,
                    markeredgewidth=1.2,
                    color=POLICY_COLORS[policy],
                )
            if annotate_extrema:
                _annotate_extrema(
                    ax,
                    panel,
                    y_column=y_column,
                    rates=rates,
                    policies=policies,
                    filter_column=extrema_filter_column,
                    filter_lt=extrema_filter_lt,
                )
            ax.set_title(
                display_consume_profile_label(profile),
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel(ylabel if profile_idx == 0 else "")
            pps.clean_axes(ax, legend=False)

    _add_model_row_titles(fig, axes, models)
    _legend(fig, policies)
    pps.savefig(fig, output_path)
    plt.close(fig)


def _annotate_extrema(
    ax,
    panel: pd.DataFrame,
    *,
    y_column: str,
    rates: list[float],
    policies: list[str],
    filter_column: str | None,
    filter_lt: float | None,
) -> None:
    for policy in policies:
        sub = panel[panel["policy"] == policy].copy()
        if sub.empty:
            continue
        if filter_column and filter_column in sub.columns and filter_lt is not None:
            sub[filter_column] = pd.to_numeric(sub[filter_column], errors="coerce")
            sub = sub[sub[filter_column] < filter_lt]
        sub[y_column] = pd.to_numeric(sub[y_column], errors="coerce")
        sub = sub.dropna(subset=[y_column, "lambda_req_s"])
        if sub.empty:
            continue
        min_idx = sub[y_column].idxmin()
        max_idx = sub[y_column].idxmax()
        if float(sub.loc[min_idx, y_column]) == float(sub.loc[max_idx, y_column]):
            continue
        extrema = [("min", sub.loc[min_idx]), ("max", sub.loc[max_idx])]
        for label, row in extrema:
            rate = float(row["lambda_req_s"])
            if rate not in rates:
                continue
            ax.annotate(
                label,
                (rates.index(rate), float(row[y_column])),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                fontweight="bold",
                color=POLICY_COLORS[policy],
                alpha=0.95,
            )


def plot_queue_stall_boxplot(input_path: Path, output_path: Path) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    df["queue_stall_s"] = pd.to_numeric(df["queue_stall_s"], errors="coerce")
    df = df.dropna(subset=["queue_stall_s"])
    fig, axes, profiles, models = _base_grid(df)
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique().tolist())
    policies = [policy for policy in POLICY_ORDER if policy in set(df["policy"])]
    model_y_upper = _model_boxplot_y_upper(
        df,
        value_column="queue_stall_s",
        group_columns=["consume_profile_label", "policy", "lambda_req_s"],
    )
    width = 0.30
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
            for policy in policies:
                box_data = []
                positions = []
                for rate_idx, rate in enumerate(rates):
                    values = panel[
                        (panel["policy"] == policy)
                        & (panel["lambda_req_s"] == rate)
                    ]["queue_stall_s"]
                    values = pd.to_numeric(values, errors="coerce").dropna()
                    if values.empty:
                        continue
                    box_data.append(values)
                    positions.append(rate_idx + offsets[policy])
                if not box_data:
                    continue
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
                    patch.set_alpha(0.35)
                    patch.set_edgecolor(color)
                for key in ("whiskers", "caps", "medians"):
                    for artist in box[key]:
                        artist.set_color(color)
                        artist.set_linewidth(1.1)
            _set_rate_axis(ax, rates)
            if str(model) in model_y_upper:
                ax.set_ylim(0.0, model_y_upper[str(model)])
            else:
                _set_y_range(ax, panel["queue_stall_s"])
            ax.set_title(
                display_consume_profile_label(profile),
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel("Queue stall (s)" if profile_idx == 0 else "")
            pps.clean_axes(ax, legend=False)

    _add_model_row_titles(fig, axes, models)
    _legend(fig, policies)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main_fig7_1() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=processed_csv_path(
            _fig7_csv_name("1", "in-flight_requests", DEFAULT_THRESHOLD_KEY)
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=figure_output_path(
            f"{_fig7_csv_name('1', 'in-flight_requests', DEFAULT_THRESHOLD_KEY)}.png"
        ),
    )
    args = parser.parse_args()
    plot_line_metric(
        args.data,
        args.out,
        y_column="mean_admitted_inflight_requests",
        ylabel="# of in-flight requests",
        share_y_by_model=True,
    )


def main_fig7_2() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=processed_csv_path(
            _fig7_csv_name("2", "in-queue_stall", DEFAULT_THRESHOLD_KEY)
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=figure_output_path(
            f"{_fig7_csv_name('2', 'in-queue_stall', DEFAULT_THRESHOLD_KEY)}.png"
        ),
    )
    args = parser.parse_args()
    plot_queue_stall_boxplot(args.data, args.out)


def main_fig7_3() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=processed_csv_path(_fig7_csv_name("3", "TTFT", DEFAULT_THRESHOLD_KEY)),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=figure_output_path(
            f"{_fig7_csv_name('3', 'TTFT', DEFAULT_THRESHOLD_KEY)}.png"
        ),
    )
    args = parser.parse_args()
    plot_line_metric(
        args.data,
        args.out,
        y_column="ttft_p99_s",
        ylabel="TTFT p99 (s)",
        share_y_by_model=True,
    )

