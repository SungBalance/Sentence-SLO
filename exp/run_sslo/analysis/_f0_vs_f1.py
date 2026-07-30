"""F0 (pre-fix) vs F1 (scale floor=1.0) comparison on cap=256.

Reads summary.csv.bak_pref1 (F0) and summary.csv (F1), prints per-cell
viol@1, HU, TTFC for both baseline and sslo_mlp at cap=256.
"""
import csv
from pathlib import Path

ROOT = Path("/workspace/mlsys/exp/run_sslo/output_sweep")
F0 = list(csv.DictReader(open(ROOT / "summary.csv.bak_pref1")))
F1 = list(csv.DictReader(open(ROOT / "summary.csv")))

def num(v):
    try: return float(v)
    except: return None

def index(rows):
    out = {}
    for r in rows:
        if int(r["max_num_seqs"]) != 256: continue
        key = (r["model"], r["consume_mode"], r["consume_model"] or "-",
                float(r["request_rate"]), r["policy"])
        out[key] = r
    return out

f0 = index(F0)
f1 = index(F1)

CELLS = [
    ("Qwen3.5-9B", "read", "-"),
    ("Qwen3.5-9B", "tts", "hexgrad/Kokoro-82M"),
    ("Qwen3.5-9B", "tts", "Supertone/supertonic-3"),
    ("Qwen3.5-35B-A3B", "read", "-"),
    ("Qwen3.5-35B-A3B", "tts", "hexgrad/Kokoro-82M"),
    ("Qwen3.5-35B-A3B", "tts", "Supertone/supertonic-3"),
]
RATES = [4, 8, 12, 16, 20, 24]

print("F0 = pre-fix (scale = N/cap, no floor)")
print("F1 = post-fix (scale = max(1.0, N/cap))")
print()
print("=" * 130)
print(f"{'cell':<42s} {'rate':>4s}  "
      f"{'BASE viol':>9s} {'BASE HU':>7s}  "
      f"{'F0  viol':>9s} {'F0  HU':>7s}  "
      f"{'F1  viol':>9s} {'F1  HU':>7s}  "
      f"{'Δviol':>7s} {'ΔHU':>5s}")
print("=" * 130)
total_f0_viol = total_f1_viol = total_f0_hu = total_f1_hu = 0.0
total_base_viol = total_base_hu = 0.0
n_cells = 0
regressions_f0 = regressions_f1 = 0

for cell in CELLS:
    model, consume, tts = cell
    label = f"{model.replace('Qwen3.5-','')}/{consume}" + ("" if consume=="read"
                                                          else f"/{tts.split('/')[-1]}")
    print(f"-- {label}")
    for rate in RATES:
        base = f1.get((model, consume, tts, float(rate), "baseline"))
        f0_row = f0.get((model, consume, tts, float(rate), "sslo_mlp"))
        f1_row = f1.get((model, consume, tts, float(rate), "sslo_mlp"))
        if not (base and f0_row and f1_row): continue
        bv = num(base["request_cu_slo_violation_rate_tau_1"]) or 0
        bh = num(base["mean_handling_users"]) or 0
        v0 = num(f0_row["request_cu_slo_violation_rate_tau_1"]) or 0
        h0 = num(f0_row["mean_handling_users"]) or 0
        v1 = num(f1_row["request_cu_slo_violation_rate_tau_1"]) or 0
        h1 = num(f1_row["mean_handling_users"]) or 0
        dv = v1 - v0
        dh = h1 - h0
        # mark cells where F0/F1 sslo > baseline
        f0_reg = "*" if v0 > bv + 0.001 else " "
        f1_reg = "*" if v1 > bv + 0.001 else " "
        print(f"   {'':<39s} {rate:>4d}  "
              f"{bv:>9.3f} {bh:>7.0f}  "
              f"{v0:>9.3f}{f0_reg} {h0:>7.0f}  "
              f"{v1:>9.3f}{f1_reg} {h1:>7.0f}  "
              f"{dv:>+7.3f} {dh:>+5.0f}")
        if v0 > bv + 0.001: regressions_f0 += 1
        if v1 > bv + 0.001: regressions_f1 += 1
        total_base_viol += bv; total_base_hu += bh
        total_f0_viol += v0; total_f0_hu += h0
        total_f1_viol += v1; total_f1_hu += h1
        n_cells += 1

print("=" * 130)
print(f"mean over {n_cells} cells:  "
      f"BASE viol={total_base_viol/n_cells:.3f} HU={total_base_hu/n_cells:.0f}  "
      f"F0 viol={total_f0_viol/n_cells:.3f} HU={total_f0_hu/n_cells:.0f}  "
      f"F1 viol={total_f1_viol/n_cells:.3f} HU={total_f1_hu/n_cells:.0f}")
print(f"sslo > baseline viol regressions: F0={regressions_f0}/{n_cells}  "
      f"F1={regressions_f1}/{n_cells}")
