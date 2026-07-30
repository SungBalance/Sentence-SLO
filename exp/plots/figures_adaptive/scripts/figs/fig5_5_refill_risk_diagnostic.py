"""Figure 5.5 - Refill-risk pressure diagnostic from measured outputs."""
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
    chunk_stall_duration,
    finite_float,
    load_summary_from_outputs,
    normalize_request_id,
    ordered_consume_profiles,
    ordered_models,
    read_jsonl,
    run_dir_from_summary_row,
    run_metadata_from_path,
)
from _policy_style import POLICY_COLORS, POLICY_ORDER, POLICY_PLOT_LABELS, normalize_policy

_BASENAME = "fig5_5_refill_risk_diagnostic"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
PRESSURE_BINS = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
PRESSURE_BIN_LABELS = ("0-0.25", "0.25-0.5", "0.5-1", "1-2", ">2")


def _candidate_runs(outputs_root: Path) -> list[Path]:
    df = load_summary_from_outputs(outputs_root)
    df = df[df["policy"].isin(["baseline", "sslo_mlp"])].copy()
    df["stall_rank"] = pd.to_numeric(
        df.get("max_stall_interval_s_p99", 0.0),
        errors="coerce",
    ).fillna(0.0)
    df = df.sort_values(
        ["stall_rank", "lambda_req_s", "max_num_seqs"],
        ascending=False,
    )

    selected: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    for _, row in df.iterrows():
        model_dir = str(row["model"]).rsplit("/", 1)[-1]
        key = (model_dir, str(row["consume_profile_label"]), str(row["policy"]))
        if key in seen:
            continue
        run_dir = run_dir_from_summary_row(outputs_root, row)
        if (run_dir / "decisions.jsonl").exists() and (run_dir / "chunks.jsonl").exists():
            selected.append(run_dir)
            seen.add(key)
    return selected


def _pressure_value(row: dict) -> float | None:
    for key in ("defer_pressure", "serve_pressure", "depletion_pressure"):
        value = finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _short_model_label(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _pressure_rows_by_request(run_dir: Path) -> tuple[list[dict], dict[str, list[tuple[float, float]]]]:
    rows: list[dict] = []
    by_request: dict[str, list[tuple[float, float]]] = {}
    metadata = run_metadata_from_path(run_dir)
    model = metadata["model_dir"]
    consume_profile_label = metadata["consume_profile_label"]
    policy = normalize_policy(metadata["policy"])
    for decision in read_jsonl(run_dir / "decisions.jsonl"):
        if not decision.get("pressure_available"):
            continue
        if decision.get("phase") != "MEASURED":
            continue
        pressure = _pressure_value(decision)
        ts = finite_float(decision.get("ts"))
        if pressure is None or ts is None:
            continue
        if not np.isfinite(pressure):
            pressure = 3.0
        request_id = normalize_request_id(decision["request_id"])
        rows.append(
            {
                "model": model,
                "consume_profile_label": consume_profile_label,
                "policy": policy,
                "sample_type": "pressure_decision",
                "request_id": request_id,
                "chunk_idx": np.nan,
                "pressure": float(pressure),
                "chunk_stall_time_s": np.nan,
                "num_pressure_decisions": 1,
            }
        )
        by_request.setdefault(request_id, []).append((float(ts), float(pressure)))

    for decisions in by_request.values():
        decisions.sort(key=lambda item: item[0])
    return rows, by_request


def _chunk_pressure_stall_rows(
    run_dir: Path,
    pressure_by_request: dict[str, list[tuple[float, float]]],
) -> list[dict]:
    rows: list[dict] = []
    metadata = run_metadata_from_path(run_dir)
    model = metadata["model_dir"]
    consume_profile_label = metadata["consume_profile_label"]
    policy = normalize_policy(metadata["policy"])
    for chunk in read_jsonl(run_dir / "chunks.jsonl"):
        request_id = normalize_request_id(chunk.get("request_id", ""))
        decisions = pressure_by_request.get(request_id, [])
        if not decisions:
            continue
        start_ts = finite_float(
            chunk.get("chunk_generation_start_ts", chunk.get("text_generation_end_time"))
        )
        end_ts = finite_float(
            chunk.get("demand_window_start_ts", chunk.get("consumer_ready_time"))
        )
        if end_ts is None:
            end_ts = finite_float(chunk.get("chunk_generation_end_ts", chunk.get("text_generation_end_time")))
        if start_ts is None or end_ts is None:
            continue
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        pressures = [
            pressure
            for ts, pressure in decisions
            if start_ts <= ts <= end_ts
        ]
        if not pressures:
            continue
        stall_time = chunk_stall_duration(chunk)
        if stall_time is None:
            continue
        rows.append(
            {
                "model": model,
                "consume_profile_label": consume_profile_label,
                "policy": policy,
                "sample_type": "chunk_pressure_stall",
                "request_id": request_id,
                "chunk_idx": int(chunk.get("chunk_idx", chunk.get("unit_index", 0))),
                "pressure": float(np.mean(pressures)),
                "chunk_stall_time_s": max(0.0, float(stall_time)),
                "num_pressure_decisions": len(pressures),
            }
        )
    return rows


def load_pressure_violation_correlation(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in _candidate_runs(outputs_root):
        pressure_rows, pressure_by_request = _pressure_rows_by_request(run_dir)
        rows.extend(pressure_rows)
        rows.extend(_chunk_pressure_stall_rows(run_dir, pressure_by_request))

    if not rows:
        raise ValueError(
            "Figure 5.5 needs pressure decisions and chunk stall rows."
        )
    return pd.DataFrame(rows)


def preprocess_fig5_5(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_pressure_violation_correlation(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def _draw_pressure_distribution(ax, model_df: pd.DataFrame) -> None:
    finite = (
        model_df[model_df["sample_type"] == "pressure_decision"][["policy", "pressure"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    finite = finite[finite["pressure"] > 0]
    if len(finite) < 2:
        return

    x_min = float(finite["pressure"].min())
    x_max = float(finite["pressure"].max())
    if x_max <= x_min:
        return

    bins = np.geomspace(x_min, x_max * 1.001, num=24)
    for policy in POLICY_ORDER:
        values = finite[finite["policy"] == policy]["pressure"]
        if values.empty:
            continue
        weights = np.full(len(values), 1.0 / len(values))
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            weights=weights,
            color=POLICY_COLORS[policy],
            linewidth=1.3,
        )

    ax.set_xscale("log")
    ax.set_ylabel("Share")
    pps.clean_axes(ax, legend=False)


def _draw_pressure_stall_boxplots(ax, model_df: pd.DataFrame) -> None:
    chunk_df = (
        model_df[model_df["sample_type"] == "chunk_pressure_stall"][
            ["policy", "pressure", "chunk_stall_time_s"]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    chunk_df = chunk_df[
        (chunk_df["pressure"] > 0)
        & (chunk_df["chunk_stall_time_s"] >= 0)
    ].copy()
    if chunk_df.empty:
        return
    chunk_df["pressure_bin"] = pd.cut(
        chunk_df["pressure"],
        bins=PRESSURE_BINS,
        labels=PRESSURE_BIN_LABELS,
        include_lowest=True,
    )

    x = np.arange(len(PRESSURE_BIN_LABELS), dtype=float)
    width = 0.30
    offsets = {
        "baseline": -width / 1.7,
        "sslo": width / 1.7,
    }
    for policy in POLICY_ORDER:
        box_data = []
        positions = []
        for bin_idx, label in enumerate(PRESSURE_BIN_LABELS):
            values = chunk_df[
                (chunk_df["policy"] == policy)
                & (chunk_df["pressure_bin"] == label)
            ]["chunk_stall_time_s"]
            box_data.append(values.to_numpy() if not values.empty else np.array([np.nan]))
            positions.append(x[bin_idx] + offsets[policy])
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
                artist.set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(PRESSURE_BIN_LABELS)
    ax.tick_params(axis="x", labelsize=9, pad=1)
    ax.set_ylabel("Chunk Stall Time (sec.)")
    pps.clean_axes(ax, legend=False)


def plot_fig5_5(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    w, h = pps.fig_size("double", ratio=0.42)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * 2,
        figsize=(w * 1.45, h * max(1, len(profiles))),
        squeeze=False,
        gridspec_kw={"hspace": 0.34, "wspace": 0.30},
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.09,
        top=0.86,
        hspace=0.48,
        wspace=0.28,
    )

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            model_df = df[
                (df["model"] == model)
                & (df["consume_profile_label"] == profile)
            ]
            ax_dist = axes[profile_idx, model_idx * 2]
            ax_stall = axes[profile_idx, model_idx * 2 + 1]
            _draw_pressure_distribution(ax_dist, model_df)
            _draw_pressure_stall_boxplots(ax_stall, model_df)
            if profile_idx == 0:
                ax_dist.set_title(
                    f"{_short_model_label(model)}\nPressure Dist",
                    fontsize=11,
                    fontweight="bold",
                    pad=5,
                )
                ax_stall.set_title(
                    f"{_short_model_label(model)}\nPressure vs Violation",
                    fontsize=11,
                    fontweight="bold",
                    pad=5,
                )
            if profile_idx == len(profiles) - 1:
                ax_dist.set_xlabel("Pressure")
                ax_stall.set_xlabel("Mean Chunk Pressure")
            else:
                ax_dist.set_xlabel("")
                ax_stall.set_xlabel("")
            if model_idx == 0:
                ax_dist.set_ylabel(f"{profile}\nShare")

    legend_handles = [
        Patch(
            facecolor=POLICY_COLORS[policy],
            edgecolor=POLICY_COLORS[policy],
            alpha=0.68,
            label=POLICY_PLOT_LABELS[policy],
        )
        for policy in POLICY_ORDER
    ]
    legend = fig.legend(
        handles=legend_handles,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        borderpad=0.35,
        columnspacing=1.0,
        handlelength=1.4,
        labelspacing=0.3,
    )
    pps.frame_legend(legend)

    output_path = Path(output_path)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed Figure 5.5 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_5(args.data, args.out)


if __name__ == "__main__":
    main()
