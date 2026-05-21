"""5-way 9B comparison: baseline + 4 combos of (adaptive_batching x allow_admit_critical)."""
import json
from pathlib import Path

CONFIGS = [
    ("baseline",            "/workspace/mlsys/exp/run_sslo/output/v6_9b_5way/baseline/run_1", "baseline"),
    ("abatch=T allow=F",    "/workspace/mlsys/exp/run_sslo/output/v6_9b_5way/abatchT_allowF_default_s/sslo_mlp/run_1", "sslo_mlp"),
    ("abatch=F allow=F",    "/workspace/mlsys/exp/run_sslo/output/v6_9b_5way/abatchF_allowF_s/sslo_mlp/run_1", "sslo_mlp"),
    ("abatch=T allow=T",    "/workspace/mlsys/exp/run_sslo/output/v6_9b_5way/abatchT_allowT_s/sslo_mlp/run_1", "sslo_mlp"),
    ("abatch=F allow=T",    "/workspace/mlsys/exp/run_sslo/output/v6_9b_5way/abatchF_allowT_s/sslo_mlp/run_1", "sslo_mlp"),
]
print(f"{'variant':<20} | {'inj':>5} {'win':>5} {'viol.5':>7} {'viol1':>7} {'viol2':>7} {'p99st':>6} {'maxst':>6} {'ttfc_m':>7} {'tp/s':>7} {'crit%':>6} {'run_m':>6} {'wait_m':>7}")
print('-' * 130)
for label, path, mode in CONFIGS:
    p = Path(path) / "summary.json"
    if not p.exists():
        print(f"{label:<20} | MISSING ({p})"); continue
    s = json.loads(p.read_text())
    m = s.get("metrics", {})
    cps = m.get("cpslo", {}).get(mode, {})
    viol = cps.get("cp_slo_violation_rates_by_tau", {})
    def r(t): v = viol.get(t, {}); return (v.get("rate", 0) or 0) * 100
    ms = cps.get("max_stall_interval_distribution", {})
    ttfc = m.get("ttfc", {}).get(mode, {})
    tp = m.get("throughput", {}).get(mode, {})
    sched_p = Path(path) / "scheduler_stats.jsonl"
    n_total = n_crit = 0; runs = []; waits = []
    with sched_p.open() as f:
        for line in f:
            r2 = json.loads(line)
            if r2.get("kind") != "step": continue
            n_total += 1
            if r2.get("has_critical"): n_crit += 1
            runs.append(r2.get("running",0) or 0)
            waits.append(r2.get("waiting",0) or 0)
    crit_share = n_crit / max(1, n_total) * 100
    run_m = sum(runs) / max(1, len(runs))
    wait_m = sum(waits) / max(1, len(waits))
    print(f"{label:<20} | {s.get('injected_count',0):>5} {s.get('in_window_count',0):>5} "
          f"{r('tau_0.5'):>6.2f}% {r('tau_1'):>6.2f}% {r('tau_2'):>6.2f}% "
          f"{ms.get('p99',0):>6.2f} {ms.get('max',0):>6.2f} "
          f"{ttfc.get('mean',0):>7.2f} "
          f"{tp.get('tokens_per_second',0) or 0:>7.0f} "
          f"{crit_share:>5.1f}% {run_m:>6.1f} {wait_m:>7.1f}")
