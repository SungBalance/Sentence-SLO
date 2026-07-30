from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from figure_paths import PROCESSED_DIR, figure_output_path, processed_csv_path
from _outputs_data import ordered_consume_profiles, ordered_models, short_model_label
from _policy_style import POLICY_COLORS, POLICY_ORDER
import paper_plot_style as pps

_BASENAME = "fig6_7g_best_inflight_under_0_5pct"
SOURCE_BASENAME = "fig6_7a_unit_miss_budget_supported_users"
DEFAULT_SOURCE_DATA = processed_csv_path(SOURCE_BASENAME)
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_OUT = figure_output_path(_BASENAME)
THRESHOLD_KEY = "conservative"
THRESHOLD_LABEL = "<0.5% unit miss"
POLICY_LABELS = {
    "baseline": "Baseline",
    "sslo": "ProgressServe",
}


def _conservative_points_by_rate(df: pd.DataFrame) -> pd.DataFrame:
    conservative = df[df["threshold_key"] == THRESHOLD_KEY].copy()
    if conservative.empty:
        raise ValueError("Figure 6.7g needs conservative (<0.5%) rows from Figure 6.7a.")

    numeric_columns = [
        "lambda_req_s",
        "mean_admitted_inflight_requests",
        "selected_max_num_seqs",
        "unit_deadline_miss_rate",
        "valid_consumable_unit_count",
        "violated_unit_count",
        "run_count",
    ]
    if "selection_rank" not in conservative.columns:
        conservative["selection_rank"] = 1
    for column in numeric_columns:
        conservative[column] = pd.to_numeric(conservative[column], errors="coerce")
    conservative["selection_rank"] = pd.to_numeric(
        conservative["selection_rank"],
        errors="coerce",
    ).fillna(1)
    conservative = conservative[conservative["selection_rank"] == 1].copy()
    conservative = conservative.dropna(
        subset=[
            "lambda_req_s",
            "mean_admitted_inflight_requests",
            "selected_max_num_seqs",
        ]
    )
    if conservative.empty:
        raise ValueError("Figure 6.7g found no numeric conservative rows.")

    selected = conservative.rename(columns={"lambda_req_s": "request_rate"}).sort_values(
        ["consume_profile_label", "model", "request_rate", "policy"],
    )
    selected["selected_max_num_seqs"] = selected["selected_max_num_seqs"].astype(int)
    selected["run_count"] = selected["run_count"].astype(int)
    ordered_columns = [
        "model",
        "consume_profile_label",
        "policy",
        "policy_label",
        "threshold_key",
        "threshold_label",
        "threshold_value",
        "threshold_mode",
        "request_rate",
        "selected_max_num_seqs",
        "mean_admitted_inflight_requests",
        "unit_deadline_miss_rate",
        "valid_consumable_unit_count",
        "violated_unit_count",
        "run_count",
    ]
    return selected[ordered_columns]


def preprocess_fig6_7g(
    input_path: Path = DEFAULT_SOURCE_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    selected = _conservative_points_by_rate(pd.read_csv(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


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


def _bar_label(row: pd.Series) -> str:
    batch = int(row["selected_max_num_seqs"])
    return f"B={batch}"


def plot_fig6_7g(input_path: Path, output_path: Path) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    df = df.copy()
    df["mean_admitted_inflight_requests"] = pd.to_numeric(
        df["mean_admitted_inflight_requests"],
        errors="coerce",
    )
    df["request_rate"] = pd.to_numeric(
        df["request_rate"],
        errors="coerce",
    )
    df["selected_max_num_seqs"] = pd.to_numeric(
        df["selected_max_num_seqs"],
        errors="coerce",
    )
    df = df.dropna(
        subset=[
            "mean_admitted_inflight_requests",
            "request_rate",
            "selected_max_num_seqs",
        ]
    )
    profiles = ordered_consume_profiles(df)
    models = ordered_models(df)
    policies = [policy for policy in POLICY_ORDER if policy in set(df["policy"])]
    if not profiles or not models:
        raise ValueError("Figure 6.7g needs non-empty model/profile data.")

    global_values = df["mean_admitted_inflight_requests"].dropna()
    y_max = float(global_values.max()) if not global_values.empty else 1.0
    if y_max <= 0.0:
        y_max = 1.0
    request_rates = sorted(df["request_rate"].dropna().unique().tolist())

    w, h = pps.fig_size("double", ratio=0.44)
    fig, axes = plt.subplots(
        len(profiles),
        len(models),
        figsize=(w * 1.25, h * 1.18 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.085,
        top=0.875,
        hspace=0.74,
        wspace=0.30,
    )

    x_positions = list(range(len(request_rates)))
    bar_width = 0.34
    offsets = {
        policy: (idx - (len(policies) - 1) / 2.0) * bar_width * 1.18
        for idx, policy in enumerate(policies)
    }
    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            ax = axes[profile_idx, model_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
            ]
            if panel.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No conservative\nfeasible point",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                pps.clean_axes(ax, legend=False)
                continue

            for policy_idx, policy in enumerate(policies):
                sub = panel[panel["policy"] == policy]
                if sub.empty:
                    continue
                for rate_idx, rate in enumerate(request_rates):
                    rate_rows = sub[sub["request_rate"] == rate]
                    if rate_rows.empty:
                        continue
                    row = rate_rows.iloc[0]
                    value = float(row["mean_admitted_inflight_requests"])
                    x = x_positions[rate_idx] + offsets[policy]
                    ax.bar(
                        x,
                        value,
                        width=bar_width,
                        color=POLICY_COLORS[policy],
                        alpha=0.92,
                    )
                    ax.annotate(
                        _bar_label(row),
                        (x, value),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color=POLICY_COLORS[policy],
                        rotation=0,
                    )

            ax.set_title(short_model_label(model), fontsize=11, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [f"{float(rate):g}" for rate in request_rates],
                rotation=0,
                ha="center",
            )
            ax.set_xlabel(
                "Request rate (req/s)" if profile_idx == len(profiles) - 1 else ""
            )
            ax.set_ylabel(
                f"In-flight users\n({THRESHOLD_LABEL})" if model_idx == 0 else ""
            )
            ax.set_ylim(0.0, y_max * 1.24)
            if request_rates:
                ax.set_xlim(-0.55, len(request_rates) - 0.45)
            pps.clean_axes(ax, legend=False)

    _add_row_titles(fig, axes, profiles)
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
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
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
    plot_fig6_7g(args.data, args.out)


if __name__ == "__main__":
    main()
