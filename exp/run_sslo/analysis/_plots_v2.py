"""Saturation curves with SHARED scales across all subplots.

Layout: 6 rows × 4 cols
  rows = (model, consume) ∈ {(9B,read), (9B,Kokoro), (9B,Supertone),
                              (35B,read), (35B,Kokoro), (35B,Supertone)}
  cols = cap ∈ {32, 64, 128, 256}
Each panel: x = rate, lines = baseline (red) vs sslo_mlp (blue).

All panels share x-axis (rate) and y-axis (per-metric global range).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = list(csv.DictReader(
    open("/workspace/mlsys/exp/run_sslo/output_sweep/summary.csv")))


def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None


key_of = lambda r: (
    r["model"], r["consume_mode"], r["consume_model"] or "-",
    int(r["max_num_seqs"]), float(r["request_rate"]), r["policy"])
data = {key_of(r): r for r in ROWS}

MODELS = ["Qwen3.5-9B", "Qwen3.5-35B-A3B"]
CONSUMES = [
    ("read", "-"),
    ("tts", "hexgrad/Kokoro-82M"),
    ("tts", "Supertone/supertonic-3"),
]
ROW_CELLS = [(m, c, t) for m in MODELS for c, t in CONSUMES]
CAPS = [32, 64, 128, 256]
RATES = [4, 8, 12, 16, 20, 24]
MODE_STYLE = {
    "baseline": dict(color="tab:red", linestyle="-", marker="o"),
    "sslo_mlp": dict(color="tab:blue", linestyle="-", marker="s"),
}


def panel_series(model, consume, tts, cap, mode, col):
    xs, ys = [], []
    for r in RATES:
        row = data.get((model, consume, tts, cap, float(r), mode))
        xs.append(r)
        ys.append(num(row.get(col)) if row else None)
    return xs, ys


def make_plot(col, title, ylabel, ymin=0, ymax=None, log=False):
    # No sharex/sharey — we MANUALLY apply the same x/ylim to every panel
    # so tick labels remain on every subplot while limits stay aligned.
    fig, axes = plt.subplots(len(ROW_CELLS), len(CAPS), figsize=(18, 16))
    for ri, (model, consume, tts) in enumerate(ROW_CELLS):
        for ci, cap in enumerate(CAPS):
            ax = axes[ri][ci]
            for mode, style in MODE_STYLE.items():
                xs, ys = panel_series(model, consume, tts, cap, mode, col)
                ax.plot(xs, ys, markersize=4, linewidth=1.2,
                        label=mode if (ri == 0 and ci == 0) else None, **style)
            ax.grid(True, alpha=0.3, linewidth=0.4)
            if log:
                ax.set_yscale("log")
            else:
                if ymax is not None:
                    ax.set_ylim(ymin, ymax)
            ax.set_xlim(min(RATES) - 1, max(RATES) + 1)
            ax.set_xticks(RATES)
            ax.tick_params(axis="both", labelsize=7,
                            labelbottom=True, labelleft=True)
            if ri == 0:
                ax.set_title(f"cap={cap}", fontsize=10)
            if ci == 0:
                tts_lbl = "" if consume == "read" else f"\n{tts.split('/')[-1]}"
                ax.set_ylabel(
                    f"{model.replace('Qwen3.5-', '')}\n{consume}{tts_lbl}",
                    fontsize=9)
            if ri == len(ROW_CELLS) - 1:
                ax.set_xlabel("rate (req/s)", fontsize=8)
    axes[0][0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"{title}  —  shared scale (tick labels on every panel)",
                  fontsize=12, y=0.998)
    fig.supylabel(ylabel, fontsize=11, x=0.003)
    fig.tight_layout()
    return fig


# Compute global y-max for each metric
def global_max(col):
    vals = []
    for r in ROWS:
        v = num(r.get(col))
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


out_dir = Path("/workspace/mlsys/exp/run_sslo/output_sweep/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# TTFC p95
ttfc_max = global_max("ttfc_p95_s")
print(f"TTFC max across all cells: {ttfc_max:.1f}s")
f = make_plot("ttfc_p95_s", "TTFC p95 (s)", "TTFC p95 (s)",
                ymin=0, ymax=ttfc_max * 1.05)
f.savefig(out_dir / "saturation_ttfc.png", dpi=130)
print(f"wrote {out_dir / 'saturation_ttfc.png'}")
plt.close(f)

# viol@1
viol_max = global_max("request_cu_slo_violation_rate_tau_1")
print(f"viol@1 max: {viol_max:.3f}")
f = make_plot("request_cu_slo_violation_rate_tau_1",
                "CU-SLO viol@τ=1s", "viol@1 (rate)",
                ymin=0, ymax=viol_max * 1.05)
f.savefig(out_dir / "saturation_viol.png", dpi=130)
print(f"wrote {out_dir / 'saturation_viol.png'}")
plt.close(f)

# mean_HU
hu_max = global_max("mean_handling_users")
print(f"mean_HU max: {hu_max:.0f}")
f = make_plot("mean_handling_users",
                "mean handling users", "mean_HU",
                ymin=0, ymax=hu_max * 1.05)
f.savefig(out_dir / "saturation_hu.png", dpi=130)
print(f"wrote {out_dir / 'saturation_hu.png'}")
plt.close(f)

# viol@0.5 (request-level, τ=0.5s)
viol05_max = global_max("request_cu_slo_violation_rate_tau_0.5")
print(f"viol@0.5 max: {viol05_max:.3f}")
f = make_plot("request_cu_slo_violation_rate_tau_0.5",
                "CU-SLO viol@τ=0.5s", "viol@0.5 (rate)",
                ymin=0, ymax=viol05_max * 1.05)
f.savefig(out_dir / "saturation_viol05.png", dpi=130)
print(f"wrote {out_dir / 'saturation_viol05.png'}")
plt.close(f)

# unit-level miss rate (= viol@τ=0 at unit level: fraction of units
# with unit_deadline_miss_s > 0). Closest column to "mean viol@0".
unit_miss_max = global_max("unit_deadline_miss_rate")
print(f"unit_deadline_miss_rate max: {unit_miss_max:.3f}")
f = make_plot("unit_deadline_miss_rate",
                "unit_deadline_miss_rate  (≈ viol@τ=0 per-unit)",
                "unit_miss_rate",
                ymin=0, ymax=unit_miss_max * 1.05)
f.savefig(out_dir / "saturation_unit_miss.png", dpi=130)
print(f"wrote {out_dir / 'saturation_unit_miss.png'}")
plt.close(f)

# tokens/s
tps_max = global_max("corrected_processed_tokens_per_s")
print(f"tps max: {tps_max:.0f}")
f = make_plot("corrected_processed_tokens_per_s",
                "output tokens / s", "tps",
                ymin=0, ymax=tps_max * 1.05)
f.savefig(out_dir / "saturation_tps.png", dpi=130)
print(f"wrote {out_dir / 'saturation_tps.png'}")
plt.close(f)
