from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
import paper_plot_style as pps
from _outputs_data import load_summary_from_outputs, ordered_consume_profiles, ordered_models
from _policy_style import POLICY_ORDER, POLICY_PLOT_LABELS, normalize_policy

_BASENAME = "fig5_4_operating_map"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)


def load_data(path: Path) -> pd.DataFrame:
    df = load_summary_from_outputs(path)
    required = {
        "model",
        "consume_profile_label",
        "policy",
        "max_num_seqs",
        "lambda_req_s",
        "mean_handling_users",
        "chunk_slo_violation_rate_tau_1s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"outputs summary data missing required columns: {sorted(missing)}")
    df = df.copy()
    df["policy"] = df["policy"].map(normalize_policy)
    return df


def grid_matrices(df_policy: pd.DataFrame, rates: list[float], caps: list[int]):
    viol = np.full((len(caps), len(rates)), np.nan)
    handling = np.full((len(caps), len(rates)), np.nan)
    for row_idx, cap in enumerate(caps):
        for col_idx, rate in enumerate(rates):
            sub = df_policy[
                (df_policy["max_num_seqs"] == cap)
                & (df_policy["lambda_req_s"] == rate)
            ]
            if not sub.empty:
                viol[row_idx, col_idx] = sub["chunk_slo_violation_rate_tau_1s"].mean()
                handling[row_idx, col_idx] = sub["mean_handling_users"].mean()
    return viol, handling


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
            y1 + 0.032,
            profile,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        pps.bold_text(text)


def _violation_vmax(df: pd.DataFrame) -> float:
    observed = float(pd.to_numeric(
        df["chunk_slo_violation_rate_tau_1s"],
        errors="coerce",
    ).max())
    if not np.isfinite(observed) or observed <= 0:
        return 0.5
    return float(observed * 1.05)


def _processed_rows(df: pd.DataFrame) -> pd.DataFrame:
    rates = sorted(df["lambda_req_s"].unique().tolist())
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    if not rates or not models or not profiles:
        raise ValueError("Figure 5.4 needs non-empty model, cap, and arrival-rate data.")

    flat_rows: list[dict] = []
    for profile in profiles:
        for model in models:
            df_m = df[
                (df["model"] == model)
                & (df["consume_profile_label"] == profile)
            ]
            for policy in POLICY_ORDER:
                df_mp = df_m[df_m["policy"] == policy]
                if df_mp.empty:
                    continue
                caps = sorted(df_mp["max_num_seqs"].dropna().unique().tolist())
                if not caps:
                    continue
                viol, handling = grid_matrices(df_mp, rates, caps)
                for row_idx in range(viol.shape[0]):
                    for col_idx in range(viol.shape[1]):
                        flat_rows.append(
                            {
                                "model": model,
                                "consume_profile_label": profile,
                                "policy": policy,
                                "max_num_seqs": caps[row_idx],
                                "lambda_req_s": rates[col_idx],
                                "chunk_slo_violation_rate_tau_1s": viol[row_idx, col_idx],
                                "mean_handling_users": handling[row_idx, col_idx],
                            }
                        )
    return pd.DataFrame(flat_rows)


def preprocess_fig5_4(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    df = load_data(input_path)
    processed = _processed_rows(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_dir / f"{_BASENAME}.csv", index=False)


def plot_fig5_4(input_path: Path, output_path: Path) -> None:
    pps.paper_theme()
    df = pd.read_csv(input_path)

    rates = sorted(df["lambda_req_s"].unique().tolist())
    models = ordered_models(df)
    profiles = ordered_consume_profiles(df)
    if not rates or not models or not profiles:
        raise ValueError("Figure 5.4 needs non-empty model, cap, and arrival-rate data.")

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")

    w, h = pps.fig_size("double", ratio=0.48)
    fig, axes = plt.subplots(
        len(profiles),
        len(models) * len(POLICY_ORDER),
        figsize=(w * 1.45, h * max(1, len(profiles)) * 1.10),
        gridspec_kw={"hspace": 0.68, "wspace": 0.08},
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.92,
        bottom=0.08,
        top=0.875,
        hspace=0.82,
        wspace=0.10,
    )

    shared_vmax = _violation_vmax(df)
    for profile_idx, profile in enumerate(profiles):
        row_image = None
        for model_idx, model in enumerate(models):
            df_m = df[
                (df["model"] == model)
                & (df["consume_profile_label"] == profile)
            ]
            for policy_idx, policy in enumerate(POLICY_ORDER):
                ax = axes[profile_idx, model_idx * len(POLICY_ORDER) + policy_idx]
                df_mp = df_m[df_m["policy"] == policy]
                if df_mp.empty:
                    ax.set_visible(False)
                    continue

                panel_points = df_mp.dropna(
                    subset=[
                        "chunk_slo_violation_rate_tau_1s",
                        "mean_handling_users",
                    ],
                    how="all",
                )
                caps = sorted(panel_points["max_num_seqs"].dropna().unique().tolist())
                panel_rates = sorted(panel_points["lambda_req_s"].dropna().unique().tolist())
                if not caps or not panel_rates:
                    ax.set_visible(False)
                    continue

                viol, handling = grid_matrices(panel_points, panel_rates, caps)
                im = ax.imshow(
                    np.ma.masked_invalid(viol),
                    cmap=cmap,
                    vmin=0.0,
                    vmax=shared_vmax,
                    aspect="auto",
                    origin="lower",
                )
                row_image = im

                ax.set_xticks(range(len(panel_rates)))
                ax.set_xticklabels([f"{rate:g}" for rate in panel_rates], rotation=90, ha="center")
                ax.tick_params(axis="x", labelbottom=True, labelsize=9)
                ax.set_yticks(range(len(caps)))
                show_y_axis = model_idx == 0 and policy_idx == 0
                ax.set_yticklabels(caps if show_y_axis else [])
                ax.tick_params(axis="y", left=show_y_axis, labelleft=show_y_axis)
                ax.set_xlabel("Arrival rate (req/s)")
                ax.set_ylabel("Batch size" if show_y_axis else "")
                title = ax.set_title(
                    f"{_short_model_label(model)}\n{POLICY_PLOT_LABELS[policy]}",
                    fontsize=10,
                    fontweight="bold",
                    pad=5,
                )
                pps.bold_text(title)
                pps.bold_axis_labels(ax)

        if row_image is not None:
            cbar = fig.colorbar(
                row_image,
                ax=axes[profile_idx, :].tolist(),
                fraction=0.025,
                pad=0.02,
                label="CU-SLO Violation Rate (ratio)",
            )
            pps.bold_text(cbar.ax.yaxis.label)

    _add_row_titles(fig, axes, profiles)
    output_path = Path(output_path)
    pps.savefig(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Processed Figure 5.4 CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_4(args.data, args.out)


if __name__ == "__main__":
    main()
