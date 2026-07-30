from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
import paper_plot_style as pps

MIN_RUN_COUNT = 1
POLICY_LABELS = {
    "baseline": "vLLM",
    "sslo": "ProgressServe",
}
BAR_POLICY_LABELS = {
    "baseline": "Baseline",
    "sslo": "ProgressServe",
}
POLICY_MARKERS = {
    "baseline": "o",
    "sslo": "s",
}


def _policy_metric(metrics: dict, raw_policy: str, policy: str, name: str) -> dict:
    node = metrics.get(name) or {}
    return node.get(raw_policy) or node.get(policy) or {}


def _bool_or_default(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    return bool(value)


def _chunk_has_unit_miss(chunk: dict) -> bool:
    missed = chunk.get("unit_deadline_missed")
    if missed is not None:
        return bool(int(missed))
    return chunk_stall_duration(chunk) > 0.0


def _request_any_unit_miss_counts(run_dir: Path) -> tuple[int, int]:
    chunks = chunks_by_request(run_dir)
    requests = request_map(run_dir)
    missed_requests = 0
    measured_requests = 0
    for request_id, request_chunks in chunks.items():
        request = requests.get(request_id, {})
        if request.get("terminal_outcome", "completed") != "completed":
            continue
        if request.get("in_window", True) is False:
            continue
        if not request_chunks:
            continue
        measured_requests += 1
        if any(_chunk_has_unit_miss(chunk) for chunk in request_chunks):
            missed_requests += 1
    return missed_requests, measured_requests


def _run_row(run, *, include_request_any: bool = False) -> dict | None:
    summary = json.loads((run.path / "summary.json").read_text())
    config = summary.get("config") or {}
    raw_policy = str(config.get("run_kind") or run.policy)
    policy = normalize_policy(raw_policy)
    if policy not in POLICY_ORDER:
        return None

    metrics = summary.get("metrics") or {}
    policy_metrics = metrics.get(raw_policy) or metrics.get(policy) or {}
    stall_time = _policy_metric(metrics, raw_policy, policy, "stall_time")
    workload = _policy_metric(metrics, raw_policy, policy, "workload")

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

    validity = summary.get("validity") or {}
    row = {
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
        "validity_pass": _bool_or_default(validity.get("validity_pass")),
        "no_harm_pass": _bool_or_default(validity.get("no_harm_pass")),
    }
    if include_request_any:
        missed_requests, measured_requests = _request_any_unit_miss_counts(run.path)
        if measured_requests <= 0:
            return None
        row["request_any_unit_missed_count"] = int(missed_requests)
        row["measured_request_count"] = int(measured_requests)
    return row


def load_unit_miss_run_rows(
    outputs_root: Path,
    *,
    include_request_any: bool = False,
) -> pd.DataFrame:
    rows = []
    for run in iter_run_dirs(outputs_root):
        summary_path = run.path / "summary.json"
        if not summary_path.exists():
            continue
        row = _run_row(run, include_request_any=include_request_any)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No unit-miss diagnostic rows found.")
    return pd.DataFrame(rows)


def aggregate_unit_miss_cells(
    df: pd.DataFrame,
    *,
    min_run_count: int = MIN_RUN_COUNT,
) -> pd.DataFrame:
    agg_spec = {
        "mean_admitted_inflight_requests": ("mean_admitted_inflight_requests", "mean"),
        "valid_consumable_unit_count": ("valid_consumable_unit_count", "sum"),
        "violated_unit_count": ("violated_unit_count", "sum"),
        "run_count": ("seed", "nunique"),
        "validity_pass": ("validity_pass", "all"),
        "no_harm_pass": ("no_harm_pass", "all"),
    }
    if "request_any_unit_missed_count" in df.columns:
        agg_spec["request_any_unit_missed_count"] = (
            "request_any_unit_missed_count",
            "sum",
        )
        agg_spec["measured_request_count"] = ("measured_request_count", "sum")

    grouped = (
        df.groupby(
            [
                "model",
                "consume_profile_label",
                "policy",
                "policy_label",
                "max_num_seqs",
                "lambda_req_s",
            ],
            as_index=False,
        )
        .agg(**agg_spec)
        .sort_values(
            [
                "consume_profile_label",
                "model",
                "policy",
                "max_num_seqs",
                "lambda_req_s",
            ]
        )
    )
    grouped["unit_deadline_miss_rate"] = (
        grouped["violated_unit_count"] / grouped["valid_consumable_unit_count"]
    )
    if "request_any_unit_missed_count" in grouped.columns:
        grouped["request_any_unit_miss_rate"] = (
            grouped["request_any_unit_missed_count"]
            / grouped["measured_request_count"]
        )
    grouped["cell_valid"] = grouped["validity_pass"] & grouped["no_harm_pass"]
    grouped = grouped[grouped["run_count"] >= min_run_count].copy()
    if grouped.empty:
        raise ValueError(
            f"No unit-miss cells have at least {min_run_count} run after aggregation."
        )
    return grouped


def write_unit_miss_cells(
    input_path: Path,
    output_dir: Path,
    *,
    basename: str,
    include_request_any: bool = False,
) -> None:
    df = load_unit_miss_run_rows(
        input_path,
        include_request_any=include_request_any,
    )
    cells = aggregate_unit_miss_cells(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / f"{basename}.csv", index=False)


def _batch_marker_size(batch_size: int, batch_sizes: list[int]) -> float:
    if len(batch_sizes) <= 1:
        return 34.0
    idx = batch_sizes.index(batch_size)
    return 24.0 + 12.0 * idx


def _set_range(ax, values: pd.Series, *, axis: str, floor_zero: bool = True) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    lo = float(finite.min())
    hi = float(finite.max())
    pad = max((hi - lo) * 0.08, abs(hi) * 0.025, 0.05)
    if hi <= lo:
        pad = max(abs(hi) * 0.08, 0.05)
    lower = lo - pad
    if floor_zero:
        lower = max(0.0, lower)
    upper = hi + pad
    if upper <= lower:
        upper = lower + pad
    if axis == "x":
        ax.set_xlim(lower, upper)
    else:
        ax.set_ylim(lower, upper)


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


def plot_scatter_diagnostic(
    input_path: Path,
    output_path: Path,
    *,
    value_column: str,
    ylabel: str,
) -> None:
    pps.paper_theme(font_size=10)
    df = pd.read_csv(input_path)
    profiles = ordered_consume_profiles(df)
    models = ordered_models(df)
    if not profiles or not models:
        raise ValueError("Scatter diagnostic needs model/profile data.")

    df = df.copy()
    df["plot_value"] = pd.to_numeric(df[value_column], errors="coerce") * 100.0
    df["mean_admitted_inflight_requests"] = pd.to_numeric(
        df["mean_admitted_inflight_requests"],
        errors="coerce",
    )
    df = df.dropna(subset=["plot_value", "mean_admitted_inflight_requests"])
    rates = sorted(pd.to_numeric(df["lambda_req_s"], errors="coerce").dropna().unique())
    rate_norm = Normalize(vmin=min(rates), vmax=max(rates))
    cmap = plt.get_cmap("viridis")
    batch_sizes = sorted(
        pd.to_numeric(df["max_num_seqs"], errors="coerce").dropna().astype(int).unique()
    )

    w, h = pps.fig_size("double", ratio=0.44)
    fig, axes = plt.subplots(
        len(profiles),
        len(models),
        figsize=(w * 1.35, h * 1.18 * max(1, len(profiles))),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.91,
        bottom=0.085,
        top=0.875,
        hspace=0.74,
        wspace=0.34,
    )
    x_values = df["mean_admitted_inflight_requests"]
    y_values = df["plot_value"]

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
                    "No 3-run cell",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )
            for _, row in panel.iterrows():
                policy = str(row["policy"])
                batch_size = int(row["max_num_seqs"])
                ax.scatter(
                    row["mean_admitted_inflight_requests"],
                    row["plot_value"],
                    c=[float(row["lambda_req_s"])],
                    cmap=cmap,
                    norm=rate_norm,
                    marker=POLICY_MARKERS.get(policy, "o"),
                    s=_batch_marker_size(batch_size, batch_sizes),
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.88,
                )
            ax.set_title(short_model_label(model), fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean admitted in-flight requests")
            ax.set_ylabel(ylabel if model_idx == 0 else "")
            _set_range(ax, x_values, axis="x")
            _set_range(ax, y_values, axis="y")
            pps.clean_axes(ax, legend=False)

    _add_row_titles(fig, axes, profiles)
    policy_handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            label=POLICY_LABELS.get(policy, policy),
        )
        for policy in POLICY_ORDER
    ]
    legend_batch_sizes = batch_sizes
    if len(batch_sizes) > 4:
        mid = batch_sizes[len(batch_sizes) // 2]
        legend_batch_sizes = [batch_sizes[0], mid, batch_sizes[-1]]
    batch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#333333",
            markerfacecolor="#999999",
            markersize=max(4.0, _batch_marker_size(batch, batch_sizes) ** 0.5),
            label=f"{batch}",
        )
        for batch in legend_batch_sizes
    ]
    legend = fig.legend(
        handles=policy_handles + batch_handles,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(policy_handles) + len(batch_handles),
        borderpad=0.35,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    pps.frame_legend(legend)
    cbar = fig.colorbar(
        ScalarMappable(norm=rate_norm, cmap=cmap),
        ax=axes,
        fraction=0.025,
        pad=0.015,
    )
    cbar.set_label("Request rate (req/s)", fontweight="bold")
    pps.savefig(fig, output_path)
    plt.close(fig)


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
    boundary_color: str = "#d5d5d5",
    boundary_linewidth: float = 0.7,
    boundary_linestyle: str = "-",
    batch_label_y: float = -0.23,
) -> None:
    for batch_size, center in batch_centers.items():
        ax.text(
            center,
            batch_label_y,
            f"{batch_size}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    for center in batch_centers.values():
        right_edge = center + rates_per_batch / 2 + gap / 2
        ax.axvline(
            right_edge,
            color=boundary_color,
            linewidth=boundary_linewidth,
            linestyle=boundary_linestyle,
            zorder=0,
        )


def plot_hierarchical_grouped_bar(
    input_path: Path,
    output_path: Path,
    *,
    value_column: str,
    ylabel: str,
    value_scale: float = 1.0,
    mark_missing: bool = False,
    profile_columns: bool = False,
    share_y: bool = True,
    x_axis_label: str = "Request rate within batch size",
    append_lambda_to_last_tick: bool = False,
    boundary_color: str = "#d5d5d5",
    boundary_linewidth: float = 0.7,
    boundary_linestyle: str = "-",
    bar_width: float = 0.34,
    bar_inner_scale: float = 0.92,
    batch_gap: float = 0.85,
    x_tick_labelsize: float = 7.0,
    use_global_grid: bool = False,
    batch_label_y: float = -0.23,
    batch_axis_label_y: float = -0.34,
    lambda_x_offset: float = 0.26,
    lambda_label_y: float = -0.055,
) -> None:
    pps.paper_theme(font_size=10)
    grid_df = pd.read_csv(input_path).copy()
    grid_df[value_column] = pd.to_numeric(grid_df[value_column], errors="coerce")
    df = grid_df.dropna(subset=[value_column]).copy()
    df[value_column] = df[value_column] * value_scale
    layout_df = grid_df if mark_missing else df
    profiles = ordered_consume_profiles(layout_df)
    models = ordered_models(layout_df)
    if not profiles or not models:
        raise ValueError("Grouped bar diagnostic needs model/profile data.")

    w, h = pps.fig_size("double", ratio=0.44)
    nrows = len(models) if profile_columns else len(profiles)
    ncols = len(profiles) if profile_columns else len(models)
    width_scale = 1.68 if profile_columns else 1.55
    height_scale = 1.18 if profile_columns else 1.28
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(w * width_scale, h * height_scale * max(1, nrows)),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.12,
        top=0.84 if profile_columns else 0.875,
        hspace=0.82 if profile_columns else 0.60,
        wspace=0.34 if profile_columns else 0.30,
    )
    global_values = pd.to_numeric(df[value_column], errors="coerce").dropna()
    y_max = float(global_values.max()) if not global_values.empty else 1.0
    if y_max <= 0:
        y_max = 1.0

    global_batch_sizes = sorted(
        pd.to_numeric(layout_df["max_num_seqs"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    global_rates = sorted(
        pd.to_numeric(layout_df["lambda_req_s"], errors="coerce").dropna().unique()
    )
    gap = batch_gap
    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            ax = axes[model_idx, profile_idx] if profile_columns else axes[profile_idx, model_idx]
            panel = df[
                (df["consume_profile_label"] == profile)
                & (df["model"] == model)
            ]
            grid_panel = layout_df[
                (layout_df["consume_profile_label"] == profile)
                & (layout_df["model"] == model)
            ]
            if use_global_grid:
                batch_sizes = global_batch_sizes
                rates = global_rates
            else:
                batch_sizes = sorted(
                    pd.to_numeric(grid_panel["max_num_seqs"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                )
                rates = sorted(
                    pd.to_numeric(grid_panel["lambda_req_s"], errors="coerce")
                    .dropna()
                    .unique()
                )
            if not batch_sizes or not rates:
                ax.text(
                    0.5,
                    0.5,
                    "No 3-run cell",
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
            present_by_policy: dict[str, set[tuple[int, float]]] = {}
            for policy_idx, policy in enumerate(POLICY_ORDER):
                offset = (policy_idx - (len(POLICY_ORDER) - 1) / 2) * bar_width
                sub = panel[panel["policy"] == policy]
                values = {
                    (int(row["max_num_seqs"]), float(row["lambda_req_s"])): float(row[value_column])
                    for _, row in sub.iterrows()
                }
                present_by_policy[policy] = set(values)
                xs = []
                ys = []
                for batch_size in batch_sizes:
                    for rate in rates:
                        key = (batch_size, float(rate))
                        if key not in values:
                            continue
                        xs.append(positions[key] + offset)
                        ys.append(values[key])
                ax.bar(
                    xs,
                    ys,
                    width=bar_width * bar_inner_scale,
                    color=POLICY_COLORS[policy],
                    label=BAR_POLICY_LABELS.get(policy, policy),
                    alpha=0.92,
                )
            panel_values = pd.to_numeric(panel[value_column], errors="coerce").dropna()
            panel_y_max = float(panel_values.max()) if not panel_values.empty else 1.0
            if panel_y_max <= 0:
                panel_y_max = 1.0
            _draw_batch_labels(
                ax,
                batch_centers,
                len(rates),
                gap=gap,
                boundary_color=boundary_color,
                boundary_linewidth=boundary_linewidth,
                boundary_linestyle=boundary_linestyle,
                batch_label_y=batch_label_y,
            )
            title = (
                display_consume_profile_label(profile)
                if profile_columns
                else short_model_label(model)
            )
            if profile_columns:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            else:
                ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(tick_positions)
            display_tick_labels = list(tick_labels)
            ax.set_xticklabels(display_tick_labels, rotation=0)
            if tick_positions:
                right_pad = max(1.0, lambda_x_offset + 0.55) if append_lambda_to_last_tick else 0.65
                ax.set_xlim(tick_positions[0] - 0.65, tick_positions[-1] + right_pad)
            if append_lambda_to_last_tick and tick_positions:
                ax.text(
                    tick_positions[-1] + lambda_x_offset,
                    lambda_label_y,
                    "=λ",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="top",
                    fontsize=x_tick_labelsize,
                    fontweight="medium",
                )
            if profile_columns:
                ax.tick_params(axis="x", labelsize=x_tick_labelsize, pad=1)
                for label in ax.get_xticklabels():
                    label.set_fontweight("medium")
            show_xlabel = model_idx == len(models) - 1 if profile_columns else profile_idx == len(profiles) - 1
            ax.set_xlabel(x_axis_label if show_xlabel else "")
            ax.set_ylabel(ylabel if (profile_idx == 0 if profile_columns else model_idx == 0) else "")
            ax.set_ylim(0.0, (y_max if share_y else panel_y_max) * 1.10)
            if mark_missing:
                y0, y1 = ax.get_ylim()
                marker_y = y0 + (y1 - y0) * 0.035
                for policy_idx, policy in enumerate(POLICY_ORDER):
                    offset = (policy_idx - (len(POLICY_ORDER) - 1) / 2) * bar_width
                    missing_x = []
                    present = present_by_policy.get(policy, set())
                    for batch_size in batch_sizes:
                        for rate in rates:
                            key = (batch_size, float(rate))
                            if key in present:
                                continue
                            missing_x.append(positions[key] + offset)
                    if not missing_x:
                        continue
                    ax.plot(
                        missing_x,
                        [marker_y] * len(missing_x),
                        linestyle="",
                        marker="x",
                        markersize=4.8,
                        markeredgewidth=1.1,
                        color=POLICY_COLORS[policy],
                    )
            ax.text(
                0.5,
                batch_axis_label_y,
                "Batch size",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            pps.clean_axes(ax, legend=False)

    if profile_columns:
        _add_model_row_titles(fig, axes, models)
    else:
        _add_row_titles(fig, axes, profiles)
    handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            label=BAR_POLICY_LABELS.get(policy, policy),
        )
        for policy in POLICY_ORDER
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
