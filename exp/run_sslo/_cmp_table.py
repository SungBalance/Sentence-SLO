import json
import os

ROOT = "exp/run_sslo/output_cmp_ps"
MODES = ["baseline", "progress_serve"]
RATES = [8, 16, 24]


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


hdr = (f"{'rate':>4} {'mode':<14} {'HU_mean':>8} {'viol%':>7} "
       f"{'ttfc_p50':>8} {'ttfc_p99':>8} {'out_tok/s':>9} "
       f"{'dec_tok/s':>9} {'KV%':>6} {'stall_p99':>9}")
print(hdr)
print("-" * len(hdr))
for r in RATES:
    for mode in MODES:
        p = f"{ROOT}/{mode}/run_1/rate_{r}/summary.json"
        if not os.path.exists(p):
            print(f"{r:>4} {mode:<14} (missing)")
            continue
        m = json.load(open(p))["metrics"]
        mm = m.get(mode, {})
        wl = m.get("workload", {}).get(mode, {})
        sch = m.get("scheduler", {}).get(mode, {})
        hu = mm.get("mean_handling_users")
        if hu is None:
            nhu = sch.get("num_handling_users")
            hu = nhu.get("mean") if isinstance(nhu, dict) else None
        viol = wl.get("unit_deadline_miss_rate")
        kvu, kvt = sch.get("kv_blocks_used_mean"), sch.get("kv_blocks_total")
        kvpct = (100 * kvu / kvt) if (kvu and kvt) else None
        print(f"{r:>4} {mode:<14} {f(hu,1):>8} "
              f"{f((viol or 0)*100,3):>7} "
              f"{f(mm.get('ttfc_p50_s'),2):>8} {f(mm.get('ttfc_p99_s'),2):>8} "
              f"{f(wl.get('tokens_per_second'),0):>9} "
              f"{f(sch.get('decode_tokens_per_second'),0):>9} "
              f"{f(kvpct,1):>6} {f(mm.get('request_max_stall_s_p99'),2):>9}")
    print()
