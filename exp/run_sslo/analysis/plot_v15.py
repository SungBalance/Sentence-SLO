"""v15 grid plots: per-model panel grid (cap × metric), rate-vs-metric lines,
baseline vs sslo_mlp.

Outputs PNGs into exp/run_sslo/output/v15_grid/{model}/plots/.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# (column_name, axis_label, log_y)
METRICS = [
    ("request_slo_violation_rate_tau_1s", "req SLO viol @1s (%)", False),
    ("chunk_slo_violation_rate_tau_1s",   "chunk SLO viol @1s (%)", False),
    ("ttfc_p95_s",                        "TTFC p95 (s)",           True),
    ("queue_stall_p95_s",                 "Queue stall p95 (s)",    True),
    ("mean_handling_users",               "Mean handling users",    False),
    ("urgent_mode_fraction_pct",          "Urgent-mode %",          False),
    ("total_tokens_per_second",           "Total tokens/s",         False),
    ("max_stall_interval_s_p99",          "Max stall p99 (s)",      True),
]


def load_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def cell_avg(rows: list[dict], col: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v == "" or v is None:
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return statistics.mean(vals) if vals else None


def plot_model(csv_path: Path, model_label: str, out_dir: Path) -> None:
    rows = load_csv(csv_path)
    if not rows:
        print(f"skip {model_label}: no rows")
        return
    # Group by (cap, rate, mode), average over repeats
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    caps_set: set[int] = set()
    rates_set: set[float] = set()
    for r in rows:
        try:
            cap = int(r["max_num_seqs"])
            rate = float(r["lambda_req_s"])
            mode = r["policy"]
        except (KeyError, ValueError):
            continue
        grouped[(cap, rate, mode)].append(r)
        caps_set.add(cap)
        rates_set.add(rate)

    caps = sorted(caps_set)
    rates = sorted(rates_set)
    modes = ["baseline", "sslo_mlp"]
    colors = {"baseline": "tab:blue", "sslo_mlp": "tab:red"}
    markers = {"baseline": "o", "sslo_mlp": "s"}

    out_dir.mkdir(parents=True, exist_ok=True)

    # One figure per metric: rows = caps, cols = empty (1 col), each panel
    # has rate-vs-metric for both modes. Then a multi-metric figure.

    # Combined multi-panel figure: rows=metrics, cols=caps
    n_rows = len(METRICS)
    n_cols = len(caps)
    fig_w = max(3 * n_cols, 12)
    fig_h = 2.4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              squeeze=False)
    for mi, (col, ylabel, log_y) in enumerate(METRICS):
        for ci, cap in enumerate(caps):
            ax = axes[mi][ci]
            for mode in modes:
                ys = []
                for rate in rates:
                    cell = grouped.get((cap, rate, mode), [])
                    ys.append(cell_avg(cell, col) if cell else None)
                xs_valid = [r for r, y in zip(rates, ys) if y is not None]
                ys_valid = [y for y in ys if y is not None]
                if not xs_valid:
                    continue
                ax.plot(xs_valid, ys_valid,
                        marker=markers[mode],
                        color=colors[mode],
                        label=mode, linewidth=1.2, markersize=4)
            if log_y:
                ax.set_yscale("log")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if mi == 0:
                ax.set_title(f"cap={cap}", fontsize=10)
            if mi == n_rows - 1:
                ax.set_xlabel("request rate (req/s)", fontsize=9)
            ax.set_xscale("log")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, which="both")
            if mi == 0 and ci == n_cols - 1:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"v15 grid: {model_label} (mean over 3 repeats)",
                 fontsize=12, y=1.001)
    fig.tight_layout()
    out_path = out_dir / "panel_all_metrics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    grid_root = Path("/workspace/mlsys/exp/run_sslo/output/v15_grid")
    for model_dir in sorted(grid_root.glob("Qwen*")):
        csv = model_dir / "summary.csv"
        if not csv.exists():
            continue
        plot_model(csv, model_dir.name, model_dir / "plots")


if __name__ == "__main__":
    main()
