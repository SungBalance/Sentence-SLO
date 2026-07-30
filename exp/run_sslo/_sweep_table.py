import json
import os
from collections import defaultdict

ROOT = "exp/run_sslo/output_sweep_progress_serve/sentence"
MODES = ["baseline", "progress_serve"]
VIOL_OK = 0.5  # % threshold for "limit rate"


def metric(p, mode):
    try:
        m = json.load(open(p))["metrics"]
    except Exception:
        return None
    mm = m.get(mode, {})
    wl = m.get("workload", {}).get(mode, {})
    sch = m.get("scheduler", {}).get(mode, {})
    hu = mm.get("mean_handling_users")
    if hu is None:
        nhu = sch.get("num_handling_users")
        hu = nhu.get("mean") if isinstance(nhu, dict) else None
    viol = wl.get("unit_deadline_miss_rate")
    kvu, kvt = sch.get("kv_blocks_used_mean"), sch.get("kv_blocks_total")
    return {
        "hu": hu,
        "viol": (viol * 100) if viol is not None else None,
        "ttfc99": mm.get("ttfc_p99_s"),
        "toks": wl.get("tokens_per_second"),
        "kv": (100 * kvu / kvt) if (kvu and kvt) else None,
    }


# group[(model, consume_label, cap)][rate][mode] = metrics
group = defaultdict(lambda: defaultdict(dict))
for dp, _dn, fns in os.walk(ROOT):
    if "summary.json" not in fns:
        continue
    parts = dp.split("/")
    try:
        i = parts.index("sentence")
        model, consume, tts, capd, mode, rund, rated = parts[i + 1:i + 8]
    except (ValueError, IndexError):
        continue
    if mode not in MODES or not rated.startswith("rate_"):
        continue
    cap = int(capd[3:])
    rate = float(rated[5:])
    label = consume if consume == "read" else tts.replace("__", "/")
    d = metric(os.path.join(dp, "summary.json"), mode)
    if d:
        group[(model, label, cap)][rate][mode] = d


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


limits = []
for key in sorted(group):
    model, label, cap = key
    rates = sorted(group[key])
    print(f"\n### {model} | {label} | cap={cap}")
    print(f"{'rate':>4} | {'HU base':>7} {'HU PS':>7} | "
          f"{'viol% base':>10} {'viol% PS':>9} | "
          f"{'ttfc99 b':>8} {'ttfc99 PS':>9} | {'KV% PS':>6}")
    ps_limit = None
    for r in rates:
        b = group[key][r].get("baseline", {})
        p = group[key][r].get("progress_serve", {})
        if p.get("viol") is not None and p["viol"] < VIOL_OK:
            ps_limit = r
        print(f"{r:>4.0f} | {f(b.get('hu')):>7} {f(p.get('hu')):>7} | "
              f"{f(b.get('viol'),3):>10} {f(p.get('viol'),3):>9} | "
              f"{f(b.get('ttfc99')):>8} {f(p.get('ttfc99')):>9} | "
              f"{f(p.get('kv')):>6}")
    # HU at the PS limit rate vs baseline
    hu_b = hu_p = None
    if ps_limit is not None:
        hu_b = group[key][ps_limit].get("baseline", {}).get("hu")
        hu_p = group[key][ps_limit].get("progress_serve", {}).get("hu")
    limits.append((model, label, cap, ps_limit, hu_b, hu_p))

print("\n\n===== LIMIT RATE (max rate with progress_serve viol < "
      f"{VIOL_OK}%) =====")
print(f"{'model':<16} {'consume':<16} {'cap':>4} {'limit_rate':>10} "
      f"{'HU base@lim':>11} {'HU PS@lim':>10} {'PS/base':>7}")
for model, label, cap, lim, hb, hp in limits:
    ratio = f"{hp/hb:.1f}x" if (hb and hp) else "-"
    print(f"{model:<16} {label:<16} {cap:>4} {f(lim,0):>10} "
          f"{f(hb):>11} {f(hp):>10} {ratio:>7}")
