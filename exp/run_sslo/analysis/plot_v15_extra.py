"""DEPRECATED (2026-05): this script uses v15-era column names that no longer
exist after the R3 metric refactor. Do not run on current sweep output.
Kept only as historical reference."""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRID_ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v15_grid")

# (col, label) tuples for heatmap & trade-off panels
RATIO_METRICS = [
    ("request_slo_violation_rate_tau_1s", "req SLO viol@1s"),
    ("ttfc_p95_s",                        "TTFC p95"),
    ("queue_stall_p95_s",                 "queue stall p95"),
    ("mean_handling_users",               "handling users"),
    ("total_tokens_per_second",           "total tokens/s"),
    ("max_stall_interval_s_p99",          "max stall p99"),
]


def load(csv_path: Path) -> list[dict]:
    return list(csv.DictReader(csv_path.open()))


def cell_avg(rows: list[dict], col: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v == "" or v is None: continue
        try: vals.append(float(v))
        except ValueError: continue
    return statistics.mean(vals) if vals else None


def group_avg(csv_path: Path) -> tuple[dict, list[int], list[float]]:
    rows = load(csv_path)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    caps_set, rates_set = set(), set()
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
    return grouped, sorted(caps_set), sorted(rates_set)


# 1. Ratio heatmap (sslo/baseline)
def plot_ratio_heatmap(csv_path: Path, model_label: str, out_dir: Path) -> None:
    grouped, caps, rates = group_avg(csv_path)
    n = len(RATIO_METRICS)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2),
                              squeeze=False)
    for mi, (col, label) in enumerate(RATIO_METRICS):
        ax = axes[mi // cols][mi % cols]
        mat = np.full((len(caps), len(rates)), np.nan)
        for ci, cap in enumerate(caps):
            for ri, rate in enumerate(rates):
                b = cell_avg(grouped.get((cap, rate, "baseline"), []), col)
                s = cell_avg(grouped.get((cap, rate, "sslo_mlp"), []), col)
                if b is None or s is None or b == 0:
                    continue
                # log2 ratio centred at 0 (= sslo equal to baseline)
                mat[ci, ri] = math.log2(max(s, 1e-9) / max(b, 1e-9))
        # Diverging colormap: blue=sslo better (negative log ratio), red=sslo worse
        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        vmax = max(vmax, 0.1)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="auto")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates], fontsize=8)
        ax.set_yticks(range(len(caps)))
        ax.set_yticklabels([f"{c}" for c in caps], fontsize=8)
        ax.set_xlabel("rate (req/s)", fontsize=9)
        ax.set_ylabel("cap", fontsize=9)
        ax.set_title(f"log2(sslo / baseline)  —  {label}", fontsize=10)
        # Annotate each cell with value
        for ci in range(len(caps)):
            for ri in range(len(rates)):
                v = mat[ci, ri]
                if np.isnan(v): continue
                ax.text(ri, ci, f"{v:+.1f}", ha="center", va="center",
                         fontsize=7,
                         color="white" if abs(v) > vmax / 2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for mi in range(n, rows * cols):
        axes[mi // cols][mi % cols].axis("off")
    fig.suptitle(f"v15 sslo/baseline ratio heatmap — {model_label}\n"
                  "blue = sslo better (smaller), red = sslo worse (larger)",
                  fontsize=11, y=1.001)
    fig.tight_layout()
    out = out_dir / "ratio_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# 2. Throughput trade-off: handling users vs total throughput per cap-mode
def plot_tradeoff(csv_path: Path, model_label: str, out_dir: Path) -> None:
    grouped, caps, rates = group_avg(csv_path)
    n_cols = len(caps)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.5, 4),
                              squeeze=False, sharey=False)
    for ci, cap in enumerate(caps):
        ax = axes[0][ci]
        for mode, color, marker in (("baseline", "tab:blue", "o"),
                                       ("sslo_mlp", "tab:red", "s")):
            xs, ys, ls = [], [], []
            for rate in rates:
                cell = grouped.get((cap, rate, mode), [])
                tp = cell_avg(cell, "total_tokens_per_second")
                hu = cell_avg(cell, "mean_handling_users")
                if tp is None or hu is None: continue
                xs.append(tp); ys.append(hu); ls.append(rate)
            if not xs: continue
            ax.plot(xs, ys, marker=marker, color=color, label=mode,
                     linewidth=1.2, markersize=5)
            # annotate rates
            for x, y, r in zip(xs, ys, ls):
                ax.annotate(f"r={r:g}", (x, y), fontsize=6, alpha=0.6,
                              xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("total tokens/s", fontsize=9)
        if ci == 0:
            ax.set_ylabel("mean handling users", fontsize=9)
        ax.set_title(f"cap={cap}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"v15 trade-off: handling users vs total throughput — {model_label}",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    out = out_dir / "tradeoff_users_vs_throughput.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# 3. SLO violation tau-curves: tau on x, viol_rate on y, per (cap, rate, mode)
TAUS = [0.5, 1, 2]
CHUNK_VIOL_COLS = {tau: f"chunk_slo_violation_rate_tau_{str(tau).replace('.','_')}s"
                    for tau in TAUS}
REQ_VIOL_COLS   = {tau: f"request_slo_violation_rate_tau_{str(tau).replace('.','_')}s"
                    for tau in TAUS}


def plot_viol_tau_curves(csv_path: Path, model_label: str, out_dir: Path) -> None:
    grouped, caps, rates = group_avg(csv_path)
    # 2 columns (chunk-viol, req-viol), one row per cap.
    n_caps = len(caps)
    fig, axes = plt.subplots(n_caps, 2, figsize=(10, n_caps * 2.5),
                              squeeze=False)
    # color per rate, line per mode (solid baseline, dashed sslo)
    cmap = plt.colormaps.get_cmap("viridis")
    rate_colors = {r: cmap(i / max(1, len(rates) - 1))
                    for i, r in enumerate(rates)}

    for ci, cap in enumerate(caps):
        for col_idx, (viol_cols, title) in enumerate((
                (CHUNK_VIOL_COLS, "chunk SLO viol rate (%)"),
                (REQ_VIOL_COLS,   "request SLO viol rate (%)"),
            )):
            ax = axes[ci][col_idx]
            for rate in rates:
                for mode, linestyle, alpha in (
                        ("baseline", "-",  0.8),
                        ("sslo_mlp", "--", 0.9)):
                    cell = grouped.get((cap, rate, mode), [])
                    ys = [cell_avg(cell, viol_cols[tau]) for tau in TAUS]
                    if any(y is None for y in ys): continue
                    ax.plot(TAUS, ys, linestyle=linestyle,
                             color=rate_colors[rate], alpha=alpha,
                             marker="o" if mode == "baseline" else "s",
                             markersize=4)
            ax.set_xlabel("tau (s)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"cap={cap}\n{title}", fontsize=9)
            else:
                ax.set_ylabel(title, fontsize=9)
            if ci == 0:
                ax.set_title(title, fontsize=10)
            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")

    # legend (single)
    legend_handles = []
    for rate in rates:
        legend_handles.append(
            plt.Line2D([0], [0], color=rate_colors[rate],
                       label=f"r={rate:g}", linewidth=2))
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linestyle="-",
                   marker="o", label="baseline"))
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linestyle="--",
                   marker="s", label="sslo_mlp"))
    fig.legend(handles=legend_handles, loc="center right",
                bbox_to_anchor=(1.08, 0.5), fontsize=8)
    fig.suptitle(f"v15 SLO violation rate vs tau — {model_label}",
                  fontsize=11, y=1.001)
    fig.tight_layout()
    out = out_dir / "viol_tau_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# 4. Saturation curves: rate vs throughput per cap (one panel per mode)
def plot_saturation(csv_path: Path, model_label: str, out_dir: Path) -> None:
    grouped, caps, rates = group_avg(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    cmap = plt.colormaps.get_cmap("plasma")
    cap_colors = {c: cmap(i / max(1, len(caps) - 1))
                   for i, c in enumerate(caps)}
    for mi, mode in enumerate(("baseline", "sslo_mlp")):
        ax = axes[0][mi]
        for cap in caps:
            xs, ys = [], []
            for rate in rates:
                cell = grouped.get((cap, rate, mode), [])
                tp = cell_avg(cell, "total_tokens_per_second")
                if tp is None: continue
                xs.append(rate); ys.append(tp)
            if not xs: continue
            ax.plot(xs, ys, marker="o", color=cap_colors[cap],
                     label=f"cap={cap}", linewidth=1.5, markersize=5)
        ax.set_xscale("log")
        ax.set_xlabel("request rate (req/s)", fontsize=10)
        ax.set_ylabel("total tokens/s", fontsize=10)
        ax.set_title(f"{mode}", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"v15 saturation curves — {model_label}",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    out = out_dir / "saturation_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    for model_dir in sorted(GRID_ROOT.glob("Qwen*")):
        csv = model_dir / "summary.csv"
        if not csv.exists(): continue
        out_dir = model_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_ratio_heatmap(csv, model_dir.name, out_dir)
        plot_tradeoff(csv, model_dir.name, out_dir)
        plot_viol_tau_curves(csv, model_dir.name, out_dir)
        plot_saturation(csv, model_dir.name, out_dir)


if __name__ == "__main__":
    raise SystemExit(
        "plot_v15_extra.py is deprecated; see top-of-file docstring.")
