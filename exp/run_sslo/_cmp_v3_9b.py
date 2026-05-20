"""Compare 9B baseline vs sslo_mlp (cap=256 r=16 window=180s)."""
import json
from pathlib import Path

ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v3_9b_256_16")
for d in ["baseline", "sslo_mlp"]:
    p = ROOT / d / "run_1" / "summary.json"
    s = json.loads(p.read_text())
    m = s.get("metrics", {})
    mode_key = "baseline" if d == "baseline" else "sslo_mlp"
    cps = m.get("cpslo", {}).get(mode_key, {})
    print(f"\n=== {d} ===")
    print(f"  injected={s.get('injected_count')} in_window={s.get('in_window_count')} out={s.get('out_of_window_count')}")
    viol = cps.get("cp_slo_violation_rates_by_tau", {})
    for t in ["tau_0.5","tau_1","tau_2","tau_5"]:
        v = viol.get(t, {})
        rate = v.get("rate"); n = v.get("violated"); tot = v.get("total")
        print(f"  viol@{t}: {rate*100 if rate is not None else 0:.2f}% ({n}/{tot})")
    ms = cps.get("max_stall_interval_distribution", {})
    print(f"  max_stall: mean={ms.get('mean',0):.3f} p95={ms.get('p95',0):.3f} p99={ms.get('p99',0):.3f} max={ms.get('max',0):.3f}")
    ttfc = m.get("ttfc", {}).get(mode_key, {})
    print(f"  ttfc: mean={ttfc.get('mean',0):.2f} p95={ttfc.get('p95',0):.2f} p99={ttfc.get('p99',0):.2f} max={ttfc.get('max',0):.2f}")
    ttft = m.get("ttft", {}).get(mode_key, {}).get("all", {})
    print(f"  ttft: mean={ttft.get('mean',0):.2f} p99={ttft.get('p99',0):.2f}")
    tp = m.get("throughput", {}).get(mode_key, {})
    print(f"  throughput: tokens/s={tp.get('tokens_per_second')} req/s={tp.get('completed_req_per_s')}")
    hu = cps.get("mean_handling_users_time_weighted")
    print(f"  handling_users mean: {hu}")
