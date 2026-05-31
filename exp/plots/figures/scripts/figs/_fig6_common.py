from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR
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
from _policy_style import POLICY_COLORS, POLICY_ORDER, POLICY_PLOT_LABELS, normalize_policy

_BASE_FONT_SIZE = 10
_ROW_TITLE_FONT_SIZE = _BASE_FONT_SIZE * 1.5
_LEGEND_FONT_SIZE = _BASE_FONT_SIZE * 1.4
_PANEL_TITLE_FONT_SIZE = 9
_SEMIBOLD_FONT = pps.PRETENDARD_FONTS_DIR / "Pretendard-SemiBold.ttf"


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _rate_label(rate: float) -> str:
    return f"{float(rate):g}"


def _load_summary(path: Path) -> pd.DataFrame:
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
    df = df[df["policy"].isin(POLICY_ORDER)]
    return df


def preprocess_summary_metric(
    input_path: Path,
    output_dir: Path,
    *,
    basename: str,
    metric_column: str,
    value_column: str,
) -> None:
    df = _load_summary(input_path)
    if metric_column not in df.columns:
        raise ValueError(f"outputs summary data missing required column: {metric_column}")
    rows = df[
        [
            "model",
            "consume_profile_label",
            "policy",
            "max_num_seqs",
            "lambda_req_s",
            "seed",
            metric_column,
        ]
    ].copy()
    rows = rows.rename(columns={metric_column: value_column})
    rows[value_column] = pd.to_numeric(rows[value_column], errors="coerce")
    rows = rows.dropna(subset=[value_column])
    if rows.empty:
        raise ValueError(f"Figure 6 metric {metric_column} has no finite values.")
    rows = (
        rows.groupby(
            ["model", "consume_profile_label", "policy", "max_num_seqs", "lambda_req_s"],
            as_index=False,
        )
        .agg(
            **{
                value_column: (value_column, "mean"),
                "run_count": ("seed", "nunique"),
            }
        )
        .sort_values(["model", "consume_profile_label", "max_num_seqs", "policy", "lambda_req_s"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / f"{basename}.csv", index=False)


def _request_allowed(request: dict) -> bool:
    return (
        request.get("terminal_outcome", "completed") == "completed"
        and request.get("in_window", True) is not False
    )


def preprocess_tau_metrics(
    input_path: Path,
    output_dir: Path,
    *,
    basename: str,
    tau_s: float,
    mode: str,
) -> None:
    summary = _load_summary(input_path)
    rows: list[dict] = []
    for _, summary_row in summary.iterrows():
        path_row = summary_row.copy()
        path_row["policy"] = summary_row["policy_raw"]
        run_dir = run_dir_from_summary_row(input_path, path_row)
        chunks = chunks_by_request(run_dir)
        requests = request_map(run_dir)
        if not chunks:
            continue

        total_chunks = 0
        violated_chunks = 0
        request_rows: list[dict] = []
        for request_id, request_chunks in chunks.items():
            request = requests.get(request_id, {})
            if not _request_allowed(request):
                continue
            stall_count = sum(
                1
                for chunk in request_chunks
                if chunk_stall_duration(chunk) > tau_s
            )
            total_chunks += len(request_chunks)
            violated_chunks += stall_count
            request_rows.append(
                {
                    "model": summary_row["model"],
                    "consume_profile_label": summary_row["consume_profile_label"],
                    "policy": summary_row["policy"],
                    "max_num_seqs": int(summary_row["max_num_seqs"]),
                    "lambda_req_s": float(summary_row["lambda_req_s"]),
                    "seed": int(summary_row["seed"]),
                    "request_id": request_id,
                    "request_idx": request.get("request_idx"),
                    "num_chunks": len(request_chunks),
                    "tau_s": tau_s,
                    "stall_count": int(stall_count),
                }
            )

        if mode == "violation":
            if total_chunks <= 0:
                continue
            rows.append(
                {
                    "model": summary_row["model"],
                    "consume_profile_label": summary_row["consume_profile_label"],
                    "policy": summary_row["policy"],
                    "max_num_seqs": int(summary_row["max_num_seqs"]),
                    "lambda_req_s": float(summary_row["lambda_req_s"]),
                    "seed": int(summary_row["seed"]),
                    "tau_s": tau_s,
                    "num_chunks": int(total_chunks),
                    "violated_chunks": int(violated_chunks),
                    "violation_rate_pct": 100.0 * violated_chunks / total_chunks,
                }
            )
        elif mode == "stall_count":
            rows.extend(request_rows)
        else:
            raise ValueError(f"Unknown Figure 6 tau metric mode: {mode}")

    if not rows:
        raise ValueError(f"Figure 6 {mode} data has no rows for tau={tau_s}.")
    frame = pd.DataFrame(rows)
    if mode == "violation":
        frame = (
            frame.groupby(
                ["model", "consume_profile_label", "policy", "max_num_seqs", "lambda_req_s", "tau_s"],
                as_index=False,
            )
            .agg(
                num_chunks=("num_chunks", "mean"),
                violated_chunks=("violated_chunks", "mean"),
                violation_rate_pct=("violation_rate_pct", "mean"),
                run_count=("seed", "nunique"),
            )
            .sort_values(["model", "consume_profile_label", "max_num_seqs", "policy", "lambda_req_s"])
        )
    else:
        frame = frame.sort_values(
            ["model", "consume_profile_label", "max_num_seqs", "policy", "lambda_req_s", "seed"]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{basename}.csv", index=False)


def _aggregate_line(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.groupby(
            ["model", "consume_profile_label", "policy", "max_num_seqs", "lambda_req_s"],
            as_index=False,
        )[value_column]
        .mean()
        .sort_values(["model", "consume_profile_label", "max_num_seqs", "policy", "lambda_req_s"])
    )


def _set_rate_axis(ax, rates: list[float]) -> None:
    ax.set_xticks(rates)
    ax.set_xticklabels([_rate_label(rate) for rate in rates], rotation=90, ha="center")
    ax.tick_params(axis="x", labelsize=8)
    if not rates:
        return
    if len(rates) == 1:
        pad = max(0.25, abs(rates[0]) * 0.08)
    else:
        pad = max(0.25, (max(rates) - min(rates)) * 0.025)
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
        pad = max((y_max - y_min) * 0.10, abs(y_max) * 0.02, 0.05)
    lower = y_min - pad
    if floor_zero:
        lower = max(0.0, lower)
    upper = y_max + pad
    if upper <= lower:
        upper = lower + pad
    ax.set_ylim(lower, upper)


def _set_panel_title(ax, model: str, cap: int) -> None:
    title = ax.set_title(
        f"{_short_model_label(model)}\nBatch size: {cap}",
        fontsize=_PANEL_TITLE_FONT_SIZE,
        fontweight="semibold",
        pad=5,
    )
    if _SEMIBOLD_FONT.exists():
        title.set_fontproperties(
            font_manager.FontProperties(
                fname=str(_SEMIBOLD_FONT),
                size=_PANEL_TITLE_FONT_SIZE,
            )
        )


def _add_row_titles(fig, axes, profiles: list[str]) -> None:
    for profile_idx, profile in enumerate(profiles):
        row_axes = [ax for ax in axes[profile_idx, :] if ax.get_visible()]
        if not row_axes:
            continue
        y1 = max(ax.get_position().y1 for ax in row_axes)
        text = fig.text(
            0.5,
            y1 + 0.052,
            profile,
            ha="center",
            va="bottom",
            fontsize=_ROW_TITLE_FONT_SIZE,
            fontweight="bold",
        )
        pps.bold_text(text)


def _column_keys(df: pd.DataFrame) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for model in ordered_models(df):
        model_df = df[df["model"] == model]
        for cap in sorted(pd.to_numeric(model_df["max_num_seqs"], errors="coerce").dropna().astype(int).unique()):
            keys.append((model, int(cap)))
    return keys


def _base_grid(df: pd.DataFrame):
    profiles = ordered_consume_profiles(df)
    columns = _column_keys(df)
    if not profiles or not columns:
        raise ValueError("Figure 6 needs non-empty consume profile and model/cap data.")
    w, h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        len(profiles),
        len(columns),
        figsize=(w * 2.25, h * 1.35 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.075,
        top=0.875,
        hspace=0.95,
        wspace=0.46,
    )
    return fig, axes, profiles, columns


def plot_line_grid(
    input_path: Path,
    output_path: Path,
    *,
    value_column: str,
    ylabel: str,
) -> None:
    pps.paper_theme(font_size=_BASE_FONT_SIZE)
    raw = pd.read_csv(input_path)
    df = _aggregate_line(raw, value_column)
    fig, axes, profiles, columns = _base_grid(df)
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique().tolist())

    for profile_idx, profile in enumerate(profiles):
        for column_idx, (model, cap) in enumerate(columns):
            ax = axes[profile_idx, column_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
                & (df["max_num_seqs"] == cap)
            ]
            for policy in POLICY_ORDER:
                sub = panel[panel["policy"] == policy].sort_values("lambda_req_s")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lambda_req_s"],
                    sub[value_column],
                    color=POLICY_COLORS[policy],
                    marker="o",
                    markersize=3.0,
                    linewidth=1.2,
                    label=POLICY_PLOT_LABELS[policy],
                )
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel(ylabel)
            _set_rate_axis(ax, rates)
            _set_y_range(ax, panel[value_column])
            pps.clean_axes(ax, legend=False)
            _set_panel_title(ax, model, cap)

    _add_row_titles(fig, axes, profiles)
    legend_handles = [
        Patch(facecolor=POLICY_COLORS[policy], label=POLICY_PLOT_LABELS[policy])
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=_LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.3,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)
    pps.savefig(fig, output_path)
    plt.close(fig)


def plot_stall_count_grid(
    input_path: Path,
    output_path: Path,
    *,
    ylabel: str,
) -> None:
    pps.paper_theme(font_size=_BASE_FONT_SIZE)
    df = pd.read_csv(input_path)
    fig, axes, profiles, columns = _base_grid(df)
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique().tolist())
    x = np.arange(len(rates), dtype=float)
    width = 0.28
    offsets = {"baseline": -width / 1.7, "sslo": width / 1.7}
    rng = np.random.default_rng(6)

    for profile_idx, profile in enumerate(profiles):
        for column_idx, (model, cap) in enumerate(columns):
            ax = axes[profile_idx, column_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
                & (df["max_num_seqs"] == cap)
            ]
            for policy in POLICY_ORDER:
                box_data = []
                positions = []
                for rate_idx, rate in enumerate(rates):
                    values = panel[
                        (panel["policy"] == policy)
                        & (panel["lambda_req_s"] == rate)
                    ]["stall_count"]
                    values = pd.to_numeric(values, errors="coerce").dropna()
                    values = values[values > 0]
                    if values.empty:
                        continue
                    box_data.append(values)
                    positions.append(x[rate_idx] + offsets[policy])
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
                    patch.set_alpha(0.34)
                    patch.set_edgecolor(color)
                for key in ("whiskers", "caps", "medians"):
                    for artist in box[key]:
                        artist.set_color(color)
                        artist.set_linewidth(0.9)
                for values, position in zip(box_data, positions):
                    if values.empty or values.isna().all():
                        continue
                    ax.scatter(
                        position + rng.uniform(-width * 0.28, width * 0.28, size=len(values)),
                        values,
                        s=4,
                        color=color,
                        alpha=0.16,
                        linewidths=0,
                        rasterized=True,
                        zorder=3,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([_rate_label(rate) for rate in rates], rotation=90, ha="center")
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("Request rate (req/s)")
            ax.set_ylabel(ylabel)
            positive_stall_counts = pd.to_numeric(panel["stall_count"], errors="coerce")
            _set_y_range(ax, positive_stall_counts[positive_stall_counts > 0])
            pps.clean_axes(ax, legend=False)
            _set_panel_title(ax, model, cap)

    _add_row_titles(fig, axes, profiles)
    legend_handles = [
        Patch(facecolor=POLICY_COLORS[policy], label=POLICY_PLOT_LABELS[policy])
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=_LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.3,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)
    pps.savefig(fig, output_path)
    plt.close(fig)


__all__ = [
    "OUTPUT_SWEEP_DIR",
    "PROCESSED_DIR",
    "plot_line_grid",
    "plot_stall_count_grid",
    "preprocess_summary_metric",
    "preprocess_tau_metrics",
]
