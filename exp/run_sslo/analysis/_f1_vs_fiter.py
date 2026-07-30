"""Compare 9B cap=256 sslo_mlp: F1 (scale floor) vs F-iter (iter-quant pressure).

F1 data: exp/run_sslo/output_sweep/summary.csv.bak_f1 (cap=256 sslo_mlp rows)
F-iter data: exp/run_sslo/output_smoke/summary.csv (cap=256 sslo_mlp rows)
Baseline: exp/run_sslo/output_sweep/summary.csv (current — cap=256 baseline)
"""
import csv

def num(v):
    try: return float(v)
    except: return None

def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        if int(r["max_num_seqs"]) != 256: continue
        key = (r["model"], r["consume_mode"], r["consume_model"] or "-",
                float(r["request_rate"]), r["policy"])
        out[key] = r
    return out

F1 = load("/workspace/mlsys/exp/run_sslo/output_sweep/summary.csv.bak_f1")
F_ITER = load("/workspace/mlsys/exp/run_sslo/output_smoke/aggregated.csv")
BASE = load("/workspace/mlsys/exp/run_sslo/output_sweep/summary.csv")

CELLS = [
    ("Qwen3.5-9B", "read", "-"),
    ("Qwen3.5-9B", "tts", "hexgrad/Kokoro-82M"),
    ("Qwen3.5-9B", "tts", "Supertone/supertonic-3"),
]
RATES = [4, 8, 12, 16, 20, 24]

print("9B cap=256 sslo_mlp:  F1 (scale-floor) vs F-iter (iteration-quantized)")
print("=" * 130)
print(f"{'cell':<32s} {'rate':>4s}  "
      f"{'BASE viol':>9s} {'BASE HU':>7s}  "
      f"{'F1 viol':>8s} {'F1 HU':>6s} {'F1 mxStall':>10s}  "
      f"{'FIT viol':>8s} {'FIT HU':>6s} {'FIT mxStall':>11s}  "
      f"{'Δviol':>7s} {'ΔHU':>5s}")
print("=" * 130)

f1_total_viol = fit_total_viol = 0.0
f1_total_hu = fit_total_hu = 0.0
base_total_viol = base_total_hu = 0.0
n = 0
f1_regr = fit_regr = 0
fit_better_than_f1 = fit_worse_than_f1 = 0

for cell in CELLS:
    model, consume, tts = cell
    label = f"{model.replace('Qwen3.5-','')}/{consume}" + (
        "" if consume=="read" else f"/{tts.split('/')[-1]}")
    print(f"-- {label}")
    for rate in RATES:
        base = BASE.get((model, consume, tts, float(rate), "baseline"))
        f1 = F1.get((model, consume, tts, float(rate), "sslo_mlp"))
        fit = F_ITER.get((model, consume, tts, float(rate), "sslo_mlp"))
        if not (base and f1 and fit): continue
        bv = num(base["request_cu_slo_violation_rate_tau_1"]) or 0
        bh = num(base["mean_handling_users"]) or 0
        v1 = num(f1["request_cu_slo_violation_rate_tau_1"]) or 0
        h1 = num(f1["mean_handling_users"]) or 0
        s1 = num(f1["request_max_stall_s_max"]) or 0
        vi = num(fit["request_cu_slo_violation_rate_tau_1"]) or 0
        hi = num(fit["mean_handling_users"]) or 0
        si = num(fit["request_max_stall_s_max"]) or 0
        dv = vi - v1
        dh = hi - h1
        f1_mark = "*" if v1 > bv + 0.001 else " "
        fit_mark = "*" if vi > bv + 0.001 else " "
        print(f"   {'':<29s} {rate:>4d}  "
              f"{bv:>9.3f} {bh:>7.0f}  "
              f"{v1:>8.3f}{f1_mark} {h1:>6.0f} {s1:>10.2f}  "
              f"{vi:>8.3f}{fit_mark} {hi:>6.0f} {si:>11.2f}  "
              f"{dv:>+7.3f} {dh:>+5.0f}")
        if v1 > bv + 0.001: f1_regr += 1
        if vi > bv + 0.001: fit_regr += 1
        if vi < v1 - 0.001: fit_better_than_f1 += 1
        if vi > v1 + 0.001: fit_worse_than_f1 += 1
        f1_total_viol += v1; fit_total_viol += vi
        f1_total_hu += h1; fit_total_hu += hi
        base_total_viol += bv; base_total_hu += bh
        n += 1

print("=" * 130)
print(f"mean over {n} cells: BASE viol={base_total_viol/n:.3f} HU={base_total_hu/n:.0f}  "
      f"F1 viol={f1_total_viol/n:.3f} HU={f1_total_hu/n:.0f}  "
      f"FIT viol={fit_total_viol/n:.3f} HU={fit_total_hu/n:.0f}")
print(f"sslo > baseline regressions:  F1={f1_regr}/{n}  F-iter={fit_regr}/{n}")
print(f"FIT vs F1: better in {fit_better_than_f1} cells, worse in {fit_worse_than_f1} cells")
