"""Figure 5.2 - CU-SLO reconstruction from measured outputs."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
import pandas as pd

from figure_paths import OUTPUT_SWEEP_DIR, PROCESSED_DIR, figure_output_path, processed_csv_path
from fig5_1_token_metric_mismatch import (
    MAX_CHUNKS,
    MAX_OUTPUT_TOKENS,
    MIN_CHUNKS,
    MIN_OUTPUT_TOKENS,
    _is_filtered_text,
    _request_text,
)
import paper_plot_style as pps
from _outputs_data import (
    build_request_trace,
    chunks_by_request,
    chunk_stall_duration,
    load_summary_from_outputs,
    ordered_consume_profiles,
    ordered_models,
    request_map,
    run_dir_from_summary_row,
    stall_intervals,
)

_BASENAME = "fig5_2_cu_slo_reconstruction"
DEFAULT_DATA = processed_csv_path(_BASENAME)
DEFAULT_STALLS_DATA = PROCESSED_DIR / f"{_BASENAME}_stalls.csv"
DEFAULT_RAW_DATA = OUTPUT_SWEEP_DIR
DEFAULT_OUT = figure_output_path(_BASENAME)
TAU_S = 1.0
X_MAX_S = 30.0
X_TICKS_S = [0, 10, 20, 30]
Y_TICK_STEP = 100
GEN_CU_COLOR = "#0072B2"
CONSUMED_COLOR = "#D55E00"
STALL_COLOR = "red"
STALL_ALPHA = 0.15
TARGET_MODEL_FRAGMENT = "35B"
TARGET_REQUEST_ID = "1023"


def _candidate_rows(outputs_root: Path) -> pd.DataFrame:
    df = load_summary_from_outputs(outputs_root)
    df = df[df["policy"] == "baseline"].copy()
    df["stall_rank"] = pd.to_numeric(
        df.get("max_stall_interval_s_p99", 0.0),
        errors="coerce",
    ).fillna(0.0)
    df = df.sort_values(
        ["stall_rank", "lambda_req_s", "max_num_seqs"],
        ascending=False,
    )
    return df


def _run_metadata(summary_row: pd.Series) -> dict:
    model = str(summary_row["model"])
    return {
        "model": model,
        "model_label": model.rsplit("/", 1)[-1],
        "consume_profile_label": str(summary_row["consume_profile_label"]),
        "max_batch_size": int(summary_row["max_num_seqs"]),
        "request_rate": float(summary_row["lambda_req_s"]),
    }


def _select_request_from_rows(
    outputs_root: Path,
    rows: pd.DataFrame,
) -> tuple[pd.Series, Path, str, list[dict], dict, dict] | None:
    if rows.empty:
        return None
    max_batch_size = int(pd.to_numeric(rows["max_num_seqs"], errors="coerce").max())
    rows = rows[pd.to_numeric(rows["max_num_seqs"], errors="coerce") == max_batch_size].copy()
    best: tuple[tuple[float, ...], pd.Series, Path, str, list[dict], dict] | None = None
    fallback: tuple[tuple[float, ...], pd.Series, Path, str, list[dict], dict] | None = None
    for _, summary_row in rows.iterrows():
        run_dir = run_dir_from_summary_row(outputs_root, summary_row)
        if not (run_dir / "chunks.jsonl").exists():
            continue
        chunks = chunks_by_request(run_dir)
        requests = request_map(run_dir)
        for request_id, request_chunks in chunks.items():
            request_chunks = sorted(
                request_chunks,
                key=lambda item: int(item.get("chunk_idx", item.get("unit_index", 0))),
            )
            if _is_filtered_text(_request_text(request_chunks)):
                continue
            max_stall = max(chunk_stall_duration(item) for item in request_chunks)
            total_tokens = float(
                request_chunks[-1].get("cumulative_tokens_at_end")
                or request_chunks[-1].get("token_end")
                or 0.0
            )
            stall_count = sum(
                1 for item in request_chunks
                if chunk_stall_duration(item) > 0.0
            )
            total_stall = sum(chunk_stall_duration(item) for item in request_chunks)
            if stall_count > 0:
                fallback_score = (
                    stall_count,
                    total_stall,
                    max_stall,
                    -abs(len(request_chunks) - 8),
                    -abs(total_tokens - 600) / 1000.0,
                )
                if fallback is None or fallback_score > fallback[0]:
                    fallback = (
                        fallback_score,
                        summary_row,
                        run_dir,
                        request_id,
                        request_chunks,
                        requests.get(request_id, {}),
                    )
            if not MIN_CHUNKS <= len(request_chunks) <= MAX_CHUNKS:
                continue
            if not MIN_OUTPUT_TOKENS <= total_tokens <= MAX_OUTPUT_TOKENS:
                continue
            if stall_count <= 0:
                continue
            score = (
                stall_count,
                total_stall,
                max_stall,
                -abs(len(request_chunks) - 8),
                -abs(total_tokens - 600) / 1000.0,
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    summary_row,
                    run_dir,
                    request_id,
                    request_chunks,
                    requests.get(request_id, {}),
                )

    if best is not None:
        _, summary_row, run_dir, request_id, request_chunks, request_info = best
        return (
            summary_row,
            run_dir,
            request_id,
            request_chunks,
            request_info,
            _run_metadata(summary_row),
        )
    if fallback is not None:
        _, summary_row, run_dir, request_id, request_chunks, request_info = fallback
        return (
            summary_row,
            run_dir,
            request_id,
            request_chunks,
            request_info,
            _run_metadata(summary_row),
        )
    return None


def _select_requests(
    outputs_root: Path,
) -> list[tuple[pd.Series, Path, str, list[dict], dict, dict]]:
    rows = _candidate_rows(outputs_root)
    target_rows = rows[rows["model"].astype(str).str.contains(TARGET_MODEL_FRAGMENT)].copy()
    best: tuple[tuple[int, float, int], pd.Series, Path, str, list[dict], dict] | None = None
    for _, summary_row in target_rows.iterrows():
        run_dir = run_dir_from_summary_row(outputs_root, summary_row)
        if not (run_dir / "chunks.jsonl").exists():
            continue
        chunks = chunks_by_request(run_dir)
        if TARGET_REQUEST_ID not in chunks:
            continue
        request_chunks = chunks[TARGET_REQUEST_ID]
        requests = request_map(run_dir)
        stall_count = sum(1 for item in request_chunks if chunk_stall_duration(item) > 0.0)
        total_stall = sum(chunk_stall_duration(item) for item in request_chunks)
        score = (
            stall_count,
            total_stall,
            int(summary_row["max_num_seqs"]),
        )
        if best is None or score > best[0]:
            best = (
                score,
                summary_row,
                run_dir,
                TARGET_REQUEST_ID,
                request_chunks,
                requests.get(TARGET_REQUEST_ID, {}),
            )
    if best is None:
        raise ValueError(
            f"Figure 5.2 needs request {TARGET_REQUEST_ID} in {TARGET_MODEL_FRAGMENT} "
            "baseline outputs."
        )
    _, summary_row, run_dir, request_id, request_chunks, request_info = best
    return [
        (
            summary_row,
            run_dir,
            request_id,
            request_chunks,
            request_info,
            _run_metadata(summary_row),
        )
    ]


def preprocess_fig5_2(
    input_path: Path = DEFAULT_RAW_DATA,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    trace_rows = []
    stall_rows = []
    for _summary_row, _run_dir, request_id, chunks, _request_info, metadata in _select_requests(input_path):
        trace = build_request_trace(chunks, samples=1200)
        stalls = stall_intervals(chunks)
        if stalls.empty:
            continue
        trace_rows.append(
            trace.assign(
                request_id=request_id,
                tau_s=TAU_S,
                model=metadata["model"],
                model_label=metadata["model_label"],
                consume_profile_label=metadata["consume_profile_label"],
                max_batch_size=metadata["max_batch_size"],
                request_rate=metadata["request_rate"],
            )
        )
        stall_rows.append(
            stalls.assign(
                request_id=request_id,
                model=metadata["model"],
                model_label=metadata["model_label"],
                consume_profile_label=metadata["consume_profile_label"],
            )
        )
    if not trace_rows:
        raise ValueError("Figure 5.2 selected requests have no measured stall intervals.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(trace_rows, ignore_index=True).to_csv(
        output_dir / f"{_BASENAME}.csv",
        index=False,
    )
    pd.concat(stall_rows, ignore_index=True).to_csv(
        output_dir / f"{_BASENAME}_stalls.csv",
        index=False,
    )


def _format_rate(value: float) -> str:
    return f"{float(value):g}"


def _draw_request_title(ax, request_id: str, metadata: pd.Series) -> None:
    title_props = {"ha": "center", "va": "center", "fontsize": 10}
    metric_props = {"ha": "center", "va": "center", "fontsize": 9}
    metric_lines = VPacker(
        children=[
            TextArea(
                f"Batch size: {int(metadata['max_batch_size'])}, "
                f"Request Rate: {_format_rate(metadata['request_rate'])} req/s",
                textprops=metric_props,
            ),
        ],
        align="center",
        pad=0.0,
        sep=1.0,
    )
    title = VPacker(
        children=[
            TextArea(
                str(metadata["model_label"]),
                textprops={**title_props, "fontsize": 9, "fontweight": "bold"},
            ),
            TextArea(
                str(metadata["consume_profile_label"]),
                textprops={**title_props, "fontsize": 9, "fontweight": "bold"},
            ),
            TextArea(
                f"Request {request_id}",
                textprops={**title_props, "fontweight": "bold"},
            ),
            metric_lines,
        ],
        align="center",
        pad=0.0,
        sep=3.0,
    )
    box = AnchoredOffsetbox(
        loc="lower center",
        child=title,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=offset_copy(
            ax.transAxes,
            fig=ax.figure,
            x=0.0,
            y=4.0,
            units="points",
        ),
        borderpad=0.0,
    )
    ax.add_artist(box)


def _visible_y_max(trace: pd.DataFrame) -> float:
    visible = trace[trace["time_s"] <= X_MAX_S]
    if visible.empty:
        visible = trace
    max_tokens = float(
        visible[
            [
                "generated_tokens",
                "available_tokens",
                "consumed_tokens",
                "buffer_tokens",
            ]
        ].max().max()
    )
    return float(
        max(
            Y_TICK_STEP,
            math.ceil((max_tokens * 1.08) / Y_TICK_STEP) * Y_TICK_STEP,
        )
    )


def plot_fig5_2(
    input_path: Path,
    output_path: Path,
    stalls_path: Path = DEFAULT_STALLS_DATA,
) -> None:
    pps.paper_theme()
    trace = pd.read_csv(input_path)
    trace["request_id"] = trace["request_id"].astype(str)
    stalls = pd.read_csv(stalls_path)
    if stalls.empty:
        raise ValueError("Figure 5.2 processed stall CSV is empty.")
    models = ordered_models(trace)
    profiles = ordered_consume_profiles(trace)

    w, h = pps.fig_size("double", ratio=0.75)
    fig = plt.figure(figsize=(w * 1.35, h * max(1, len(profiles))), constrained_layout=False)
    outer_grid = fig.add_gridspec(len(profiles), len(models), hspace=0.72, wspace=0.24)

    for profile_idx, profile in enumerate(profiles):
        for model_idx, model in enumerate(models):
            panel_trace = trace[
                (trace["model"] == model)
                & (trace["consume_profile_label"] == profile)
            ]
            if panel_trace.empty:
                continue
            request_id = str(panel_trace["request_id"].iloc[0])
            panel_stalls = stalls[
                (stalls["model"] == model)
                & (stalls["consume_profile_label"] == profile)
                & (stalls["request_id"].astype(str) == request_id)
            ]
            metadata = panel_trace.iloc[0]
            y_max_tokens = _visible_y_max(panel_trace)
            inner_grid = outer_grid[profile_idx, model_idx].subgridspec(2, 1, hspace=0.08)
            ax_top = fig.add_subplot(inner_grid[0, 0])
            ax_bottom = fig.add_subplot(inner_grid[1, 0], sharex=ax_top)

            ax_top.plot(
                panel_trace["time_s"],
                panel_trace["generated_tokens"],
                color=GEN_CU_COLOR,
                linewidth=1.5,
                linestyle="-",
                label="Generated Tokens",
            )
            ax_top.step(
                panel_trace["time_s"],
                panel_trace["available_tokens"],
                where="post",
                color=GEN_CU_COLOR,
                linewidth=1.3,
                linestyle=":",
                label="Available Consumable Unit",
            )
            ax_top.plot(
                panel_trace["time_s"],
                panel_trace["consumed_tokens"],
                color=CONSUMED_COLOR,
                linewidth=1.5,
                label="Consumed",
            )
            for _, row in panel_stalls.iterrows():
                ax_top.axvspan(
                    row["stall_start_s"],
                    row["stall_end_s"],
                    color=STALL_COLOR,
                    alpha=STALL_ALPHA,
                )
            _draw_request_title(ax_top, request_id, metadata)
            ax_top.set_ylabel("# Tokens" if model_idx == 0 else "")
            ax_top.tick_params(axis="x", labelbottom=False)
            pps.clean_axes(ax_top)

            ax_bottom.plot(
                panel_trace["time_s"],
                panel_trace["buffer_tokens"],
                color=GEN_CU_COLOR,
                linewidth=1.2,
            )
            ax_bottom.axhline(0, color="black", linewidth=0.8)
            for _, row in panel_stalls.iterrows():
                ax_bottom.axvspan(
                    row["stall_start_s"],
                    row["stall_end_s"],
                    color=STALL_COLOR,
                    alpha=STALL_ALPHA,
                    zorder=0,
                )

            for ax in (ax_top, ax_bottom):
                ax.set_xlim(0.0, X_MAX_S)
                ax.set_xticks(X_TICKS_S)
                ax.set_ylim(0.0, y_max_tokens)
                ax.set_yticks(range(0, int(y_max_tokens) + 1, Y_TICK_STEP))
            if profile_idx == len(profiles) - 1:
                ax_bottom.set_xlabel("Time (s)")
            else:
                ax_bottom.tick_params(axis="x", labelbottom=False)
            ax_bottom.set_ylabel("Rem. CU" if model_idx == 0 else "")
            pps.clean_axes(ax_bottom)
            tick_labels = ax_bottom.get_xticklabels()
            if tick_labels:
                tick_labels[0].set_ha("left")
                tick_labels[-1].set_ha("right")
                for label in tick_labels:
                    label.set_clip_on(False)

    handles, labels = [], []
    seen = set()
    for ax in (ax_top, ax_bottom):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    handles.append(
        Patch(
            facecolor=STALL_COLOR,
            edgecolor=STALL_COLOR,
            alpha=STALL_ALPHA,
            label="Stall Time",
        )
    )
    labels.append("Stall Time")
    legend = fig.legend(
        handles,
        labels,
        fontsize=11,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
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
                        help="Processed trace CSV.")
    parser.add_argument("--stalls-data", type=Path, default=DEFAULT_STALLS_DATA,
                        help="Processed stall interval CSV.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot_fig5_2(args.data, args.out, args.stalls_data)


if __name__ == "__main__":
    main()
