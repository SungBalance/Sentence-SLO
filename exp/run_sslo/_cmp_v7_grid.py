"""9B grid 4 cells × 3 modes."""
import json
from pathlib import Path
import statistics

CELLS = ["cap64_r2", "cap64_r16", "cap256_r2", "cap256_r16"]
MODES = [
    ("baseline",      "baseline"),
    ("sslo abatch=T", "sslo_mlp"),
    ("sslo abatch=F", "sslo_mlp_noabatch_s"),
]
ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v7_9b_grid")

def load(cell, mode_dir, mode_key):
    if "_s" in mode_dir:
        p = ROOT / cell / mode_dir / "sslo_mlp" / "run_1"
    else:
        p = ROOT / cell / mode_dir / "run_1"
    sp = p / "summary.json"
    if not sp.exists(): return None
    s = json.loads(sp.read_text())
    m = s.get("metrics", {})
    mode_key_lookup = "sslo_mlp" if "sslo" in mode_dir else "baseline"
    cps = m.get("cpslo", {}).get(mode_key_lookup, {})
    viol = cps.get("cp_slo_violation_rates_by_tau", {})
    ms = cps.get("max_stall_interval_distribution", {})
    ttfc = m.get("ttfc", {}).get(mode_key_lookup, {})
    tp = m.get("throughput", {}).get(mode_key_lookup, {})
    sp2 = p / "scheduler_stats.jsonl"
    runs = []; waits = []; n_total=n_crit=0
    with sp2.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "step": continue
            n_total += 1
            if r.get("has_critical"): n_crit += 1
            runs.append(r.get("running",0) or 0)
            waits.append(r.get("waiting",0) or 0)
    def rate(t): v = viol.get(t, {}); return (v.get("rate", 0) or 0) * 100
    return dict(
        inj=s.get("injected_count",0), win=s.get("in_window_count",0),
        v05=rate("tau_0.5"), v1=rate("tau_1"), v2=rate("tau_2"),
        p99=ms.get("p99",0) or 0, mx=ms.get("max",0) or 0,
        ttfc_m=ttfc.get("mean",0) or 0,
        tps=tp.get("tokens_per_second",0) or 0,
        crit=(n_crit/max(1,n_total)*100),
        run_m=statistics.mean(runs) if runs else 0,
        wait_m=statistics.mean(waits) if waits else 0,
    )

for cell in CELLS:
    print(f"\n========== {cell} ==========")
    print(f"{'mode':<16} | {'inj':>5} {'win':>5} {'v.5':>6} {'v1':>6} {'v2':>6} {'p99':>5} {'mx':>6} {'ttfc':>7} {'tps':>6} {'crit%':>6} {'run':>5} {'wait':>6}")
    print('-' * 116)
    for label, mode_dir in MODES:
        d = load(cell, mode_dir, label)
        if d is None: print(f"{label:<16} | MISSING"); continue
        print(f"{label:<16} | {d['inj']:>5} {d['win']:>5} "
              f"{d['v05']:>5.2f}% {d['v1']:>5.2f}% {d['v2']:>5.2f}% "
              f"{d['p99']:>5.2f} {d['mx']:>6.2f} {d['ttfc_m']:>7.2f} "
              f"{d['tps']:>6.0f} {d['crit']:>5.1f}% {d['run_m']:>5.1f} {d['wait_m']:>6.1f}")
