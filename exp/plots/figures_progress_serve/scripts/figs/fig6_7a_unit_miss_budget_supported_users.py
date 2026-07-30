from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from _outputs_data import iter_run_dirs, ordered_consume_profiles, ordered_models, short_model_label
from _policy_style import POLICY_COLORS, POLICY_ORDER, normalize_policy
import paper_plot_style as pps

_BASENAME = "fig6_7a_unit_miss_budget_supported_users"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
MIN_RUN_COUNT = 1
POLICY_LABELS = {
    "baseline": "Baseline",
    "sslo": "ProgressServe",
}

THRESHOLDS = (
    ("strict", "Strict (=0)", 0.0, "eq"),
    ("conservative", "Conservative (<0.5%)", 0.005, "lt"),
    ("relaxed", "Relaxed (<1%)", 0.01, "lt"),
)
POLICY_MARKERS = {
    "baseline": "o",
    "sslo": "s",
}
ANNOTATION_OFFSETS = {
    "baseline": (3, 4),
    "sslo": (3, -10),
}
MISSING_RATE_OFFSETS = {
    "baseline": -0.18,
    "sslo": 0.18,
}


def _run_row(run) -> dict | None:
    summary = json.loads((run.path / "summary.json").read_text())
    config = summary.get("config") or {}
    raw_policy = str(config.get("run_kind") or run.policy)
    policy = normalize_policy(raw_policy)
    if policy not in POLICY_ORDER:
        return None

    metrics = summary.get("metrics") or {}
    policy_metrics = metrics.get(raw_policy) or metrics.get(policy) or {}
    stall_time = (metrics.get("stall_time") or {}).get(raw_policy) or {}
    workload = (metrics.get("workload") or {}).get(raw_policy) or {}

    mean_admitted = policy_metrics.get("mean_admitted_inflight_requests")
    if mean_admitted is None:
        mean_admitted = policy_metrics.get("mean_handling_users")
    unit_count = stall_time.get("count")
    if unit_count is None:
        unit_count = workload.get("num_consumable_units_total")
    violated_count = stall_time.get("violated_count")
    if violated_count is None:
        workload_rate = workload.get("unit_deadline_miss_rate")
        violated_count = (
            float(workload_rate) * float(unit_count)
            if workload_rate is not None and unit_count
            else None
        )
    if mean_admitted is None or unit_count is None or not unit_count:
        return None
    if violated_count is None:
        return None

    return {
        "model": config.get("model") or run.model_dir,
        "consume_profile_label": run.consume_profile_label,
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "max_num_seqs": run.max_num_seqs,
        "lambda_req_s": run.lambda_req_s,
        "seed": run.seed,
        "mean_admitted_inflight_requests": float(mean_admitted),
        "valid_consumable_unit_count": float(unit_count),
        "violated_unit_count": float(violated_count),
    }


def _load_run_rows(outputs_root: Path) -> pd.DataFrame:
    rows = []
    for run in iter_run_dirs(outputs_root):
        summary_path = run.path / "summary.json"
        if not summary_path.exists():
            continue
        row = _run_row(run)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("Figure 6.7a found no policy rows with unit miss data.")
    return pd.DataFrame(rows)


def _aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["model", "consume_profile_label", "policy", "policy_label", "max_num_seqs", "lambda_req_s"],
            as_index=False,
        )
        .agg(
            mean_admitted_inflight_requests=("mean_admitted_inflight_requests", "mean"),
            valid_consumable_unit_count=("valid_consumable_unit_count", "sum"),
            violated_unit_count=("violated_unit_count", "sum"),
            run_count=("seed", "nunique"),
        )
        .sort_values(["model", "consume_profile_label", "max_num_seqs", "lambda_req_s"])
    )
    grouped["unit_deadline_miss_rate"] = (
        grouped["violated_unit_count"] / grouped["valid_consumable_unit_count"]
    )
    grouped = grouped[grouped["run_count"] >= MIN_RUN_COUNT].copy()
    if grouped.empty:
        raise ValueError(
            f"Figure 6.7a needs at least {MIN_RUN_COUNT} run per aggregate, "
            "but none were found."
        )
    return grouped


def _satisfies_threshold(values: pd.Series, threshold: float, mode: str) -> pd.Series:
    if mode == "eq":
        return values == threshold
    if mode == "lt":
        return values < threshold
    raise ValueError(f"Unknown threshold mode: {mode}")


def _select_supported_points(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = [
        "model",
        "consume_profile_label",
        "policy",
        "policy_label",
        "lambda_req_s",
    ]
    for (model, profile, policy, policy_label, rate), group in df.groupby(group_keys):
        for threshold_key, threshold_label, threshold, threshold_mode in THRESHOLDS:
            candidates = group[
                _satisfies_threshold(
                    group["unit_deadline_miss_rate"],
                    threshold,
                    threshold_mode,
                )
            ].copy()
            if candidates.empty:
                continue
            candidates = candidates.sort_values(
                ["mean_admitted_inflight_requests", "max_num_seqs"],
                ascending=[False, False],
            )
            for rank, (_, selected) in enumerate(candidates.head(2).iterrows(), start=1):
                rows.append(
                    {
                        "model": model,
                        "consume_profile_label": profile,
                        "policy": policy,
                        "policy_label": policy_label,
                        "lambda_req_s": rate,
                        "threshold_key": threshold_key,
                        "threshold_label": threshold_label,
                        "threshold_value": threshold,
                        "threshold_mode": threshold_mode,
                        "selection_rank": rank,
                        "mean_admitted_inflight_requests": selected[
                            "mean_admitted_inflight_requests"
                        ],
                        "selected_max_num_seqs": int(selected["max_num_seqs"]),
                        "unit_deadline_miss_rate": selected["unit_deadline_miss_rate"],
                        "valid_consumable_unit_count": selected[
                            "valid_consumable_unit_count"
                        ],
                        "violated_unit_count": selected["violated_unit_count"],
                        "run_count": int(selected["run_count"]),
                    }
                )
    if not rows:
        raise ValueError("Figure 6.7a has no points satisfying the configured thresholds.")
    return pd.DataFrame(rows).sort_values(
        ["consume_profile_label", "model", "threshold_key", "policy", "lambda_req_s"]
    )


def preprocess_fig6_7a(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    selected = _select_supported_points(_aggregate_runs(_load_run_rows(input_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def _set_rate_axis(ax, rates: list[float]) -> None:
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{rate:g}" for rate in rates], rotation=0, ha="center")
    if rates:
        pad = max(0.25, (max(rates) - min(rates)) * 0.025)
        ax.set_xlim(min(rates) - pad, max(rates) + pad)


def _set_y_range(ax, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    y_min = float(finite.min())
    y_max = float(finite.max())
    pad = max((y_max - y_min) * 0.10, abs(y_max) * 0.02, 1.0)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.040,
            profile,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        pps.bold_text(text)


def plot_fig6_7a(input_path: Path, output_path: Path) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    if "selection_rank" not in df.columns:
        df["selection_rank"] = 1
    df["selection_rank"] = pd.to_numeric(df["selection_rank"], errors="coerce").fillna(1)
    df = df[df["selection_rank"] == 1].copy()
    profiles = ordered_consume_profiles(df)
    models = ordered_models(df)
    if not profiles or not models:
        raise ValueError("Figure 6.7a needs non-empty model/profile data.")
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique().tolist())
    columns = [
        (model, threshold_key, threshold_label)
        for model in models
        for threshold_key, threshold_label, _threshold, _mode in THRESHOLDS
    ]

    w, h = pps.fig_size("double", ratio=0.45)
    width_scale = 0.35 * len(columns) + 0.25
    fig, axes = plt.subplots(
        len(profiles),
        len(columns),
        figsize=(w * width_scale, h * 1.18 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.075,
        top=0.865,
        hspace=0.92,
        wspace=0.34,
    )

    for profile_idx, profile in enumerate(profiles):
        for column_idx, (model, threshold_key, threshold_label) in enumerate(columns):
            ax = axes[profile_idx, column_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
                & (df["threshold_key"] == threshold_key)
            ]
            ax.set_title(
                f"{short_model_label(model)}\n{threshold_label}",
                fontsize=10,
                fontweight="bold",
                pad=2,
            )
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel(
                "# of in-flight\nrequests" if column_idx == 0 else ""
            )
            if panel.empty:
                ax.set_ylim(0.0, 1.0)
                for policy in POLICY_ORDER:
                    ax.plot(
                        [rate + MISSING_RATE_OFFSETS[policy] for rate in rates],
                        [0.05] * len(rates),
                        linestyle="",
                        marker="x",
                        markersize=5,
                        markeredgewidth=1.2,
                        color=POLICY_COLORS[policy],
                    )
                _set_rate_axis(ax, rates)
                pps.clean_axes(ax, legend=False)
                continue
            for policy in POLICY_ORDER:
                sub = panel[panel["policy"] == policy].sort_values("lambda_req_s")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lambda_req_s"],
                    sub["mean_admitted_inflight_requests"],
                    color=POLICY_COLORS[policy],
                    marker=POLICY_MARKERS[policy],
                    markersize=4,
                    linewidth=1.4,
                )
                for _, row in sub.iterrows():
                    ax.annotate(
                        f"{int(row['selected_max_num_seqs'])}",
                        (
                            row["lambda_req_s"],
                            row["mean_admitted_inflight_requests"],
                        ),
                        textcoords="offset points",
                        xytext=(
                            ANNOTATION_OFFSETS[policy][0],
                            ANNOTATION_OFFSETS[policy][1] * 0.6,
                        ),
                        fontsize=9,
                        fontweight="bold",
                        color=POLICY_COLORS[policy],
                    )
            _set_rate_axis(ax, rates)
            _set_y_range(ax, panel["mean_admitted_inflight_requests"])
            y0, y1 = ax.get_ylim()
            marker_y = y0 + (y1 - y0) * 0.04
            present = {
                (str(row["policy"]), float(row["lambda_req_s"]))
                for _, row in panel.iterrows()
            }
            for policy in POLICY_ORDER:
                missing_rates = [
                    rate
                    for rate in rates
                    if (policy, float(rate)) not in present
                ]
                if not missing_rates:
                    continue
                ax.plot(
                    [rate + MISSING_RATE_OFFSETS[policy] for rate in missing_rates],
                    [marker_y] * len(missing_rates),
                    linestyle="",
                    marker="x",
                    markersize=5,
                    markeredgewidth=1.2,
                    color=POLICY_COLORS[policy],
                )
            pps.clean_axes(ax, legend=False)

    _add_row_titles(fig, axes, profiles)
    handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            label=POLICY_LABELS.get(policy, policy),
        )
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        fontsize=13,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)
    panel_label = fig.text(
        0.012,
        0.985,
        "A",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    pps.bold_text(panel_label)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig6_7a(args.data, args.out)


if __name__ == "__main__":
    main()
