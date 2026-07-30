import collections
import json
import os
from collections import defaultdict

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
MODES = ["baseline", "progress_serve", "progress_serve_adaptive"]
SHORT = {"baseline": "base", "progress_serve": "PS",
         "progress_serve_adaptive": "PS+ad"}
VIOL_OK = 0.5


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


def shrink_pct(dirpath):
    """% of scheduler steps where cur_max_num_requests < the max seen."""
    sp = os.path.join(dirpath, "scheduler_stats.jsonl")
    if not os.path.exists(sp):
        return None
    caps = []
    for l in open(sp):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("kind") == "step":
            caps.append(r.get("cur_max_num_requests"))
    caps = [c for c in caps if c is not None]
    if not caps:
        return None
    mx = max(caps)
    return 100 * sum(1 for c in caps if c < mx) / len(caps)


# group[(model,label,cap)][rate][mode] = metrics ; shrink[(model,label,cap)][rate] = %
group = defaultdict(lambda: defaultdict(dict))
shrink = defaultdict(dict)
for dp, _dn, fns in os.walk(ROOT):
    if "summary.json" not in fns:
        continue
    parts = dp.split("/")
    try:
        i = parts.index("sentence")
        model, consume, tts, capd, mode, _run, rated = parts[i + 1:i + 8]
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
    if mode == "progress_serve_adaptive":
        shrink[(model, label, cap)][rate] = shrink_pct(dp)


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


for key in sorted(group):
    model, label, cap = key
    print(f"\n### {model} | {label} | cap={cap}")
    print(f"{'rate':>4} | "
          f"{'HU b':>6} {'HU PS':>6} {'HU+ad':>6} | "
          f"{'vi b':>5} {'vi PS':>5} {'vi+ad':>5} | "
          f"{'t99 b':>6} {'t99 PS':>6} {'t99ad':>6} | "
          f"{'KVad':>5} {'shr%':>5}")
    for r in sorted(group[key]):
        g = group[key][r]
        b = g.get("baseline", {})
        p = g.get("progress_serve", {})
        a = g.get("progress_serve_adaptive", {})
        print(f"{r:>4.0f} | "
              f"{f(b.get('hu'),0):>6} {f(p.get('hu'),0):>6} "
              f"{f(a.get('hu'),0):>6} | "
              f"{f(b.get('viol'),2):>5} {f(p.get('viol'),2):>5} "
              f"{f(a.get('viol'),2):>5} | "
              f"{f(b.get('ttfc99'),0):>6} {f(p.get('ttfc99'),0):>6} "
              f"{f(a.get('ttfc99'),0):>6} | "
              f"{f(a.get('kv'),0):>5} "
              f"{f(shrink[key].get(r),1):>5}")
