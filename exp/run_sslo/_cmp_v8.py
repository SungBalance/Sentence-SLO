import json
import statistics
from pathlib import Path

ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v8_9b_lowcrit")

def load(cell, sub, mkey):
    if mkey == "baseline":
        p = ROOT / cell / sub / "run_1"
    else:
        if sub.endswith("_s"):
            p = ROOT / cell / sub / "sslo_mlp" / "run_1"
        else:
            p = ROOT / cell / sub / "run_1"
    s = json.loads((p / "summary.json").read_text())
    m = s.get("metrics", {})
    cps = m.get("cpslo", {}).get(mkey, {})
    v = cps.get("cp_slo_violation_rates_by_tau", {})
    def r(t):
        vv = v.get(t, {})
        return (vv.get("rate", 0) or 0) * 100
    ms = cps.get("max_stall_interval_distribution", {})
    ttfc = m.get("ttfc", {}).get(mkey, {})
    tp = m.get("throughput", {}).get(mkey, {})
    sp = p / "scheduler_stats.jsonl"
    runs = []; pends = []; n_total = n_crit = 0
    with sp.open() as f:
        for line in f:
            r2 = json.loads(line)
            if r2.get("kind") != "step":
                continue
            n_total += 1
            if r2.get("has_critical"):
                n_crit += 1
            runs.append(r2.get("running", 0) or 0)
            pends.append(r2.get("pending", 0) or 0)
    return dict(
        win=s.get("in_window_count", 0),
        v05=r("tau_0.5"), v1=r("tau_1"), v2=r("tau_2"),
        p99=ms.get("p99", 0) or 0, mx=ms.get("max", 0) or 0,
        ttfc_m=ttfc.get("mean", 0) or 0,
        ttfc_p99=ttfc.get("p99", 0) or 0,
        tps=tp.get("tokens_per_second", 0) or 0,
        crit=n_crit / max(1, n_total) * 100,
        run_m=statistics.mean(runs) if runs else 0,
        pend_m=statistics.mean(pends) if pends else 0,
    )

print(f"{'cell':<12} {'mode':<12} | {'win':>5} {'v05':>6} {'v1':>6} {'v2':>6} {'p99':>5} {'mx':>6} {'ttfc_m':>7} {'ttfc99':>7} {'tps':>6} {'crit%':>6} {'run_m':>6} {'pend_m':>6} {'adm_m':>6}")
print("-" * 132)
for cell in ["cap256_r4", "cap256_r8"]:
    for label, sub, mkey in [("baseline", "baseline", "baseline"),
                              ("abatch=T", "sslo_mlp", "sslo_mlp"),
                              ("abatch=F", "sslo_mlp_noabatch_s", "sslo_mlp")]:
        d = load(cell, sub, mkey)
        adm = d['run_m'] + d['pend_m']
        print(f"{cell:<12} {label:<12} | {d['win']:>5} "
              f"{d['v05']:>5.2f}% {d['v1']:>5.2f}% {d['v2']:>5.2f}% "
              f"{d['p99']:>5.2f} {d['mx']:>6.2f} "
              f"{d['ttfc_m']:>7.2f} {d['ttfc_p99']:>7.2f} {d['tps']:>6.0f} "
              f"{d['crit']:>5.1f}% {d['run_m']:>6.1f} {d['pend_m']:>6.1f} {adm:>6.1f}")
    print()
