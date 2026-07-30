"""adaptive T/F ablation on worst-case cells: 3-run mean±std per
(model, mode, rate). Modes: progress_serve (adaptive=F) vs
progress_serve_adaptive (adaptive=T)."""
import collections
import json
import os
import statistics

ROOT = "exp/run_sslo/output_adaptive_ablation/sentence"
MODES = ["progress_serve", "progress_serve_adaptive"]
TAG = {"progress_serve": "F", "progress_serve_adaptive": "T"}


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
    return {
        "hu": hu,
        "viol": (viol * 100) if viol is not None else None,
        "ttfc99": mm.get("ttfc_p99_s"),
        "dec": sch.get("decode_tokens_per_second"),
    }


def shrink_pct(dirpath):
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


# acc[(model,cap,rate,mode)][field] = [values over runs]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
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
    d = metric(os.path.join(dp, "summary.json"), mode)
    if not d:
        continue
    key = (model, cap, rate, mode)
    for f in ("hu", "viol", "ttfc99", "dec"):
        if d[f] is not None:
            acc[key][f].append(d[f])
    sp = shrink_pct(dp)
    if sp is not None:
        acc[key]["shr"].append(sp)


def ms(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.mean(vals):.1f}±{statistics.pstdev(vals):.1f}"


models = sorted({k[0] for k in acc})
caps = sorted({k[1] for k in acc})
rates = sorted({k[2] for k in acc})
for model in models:
    for cap in caps:
        if not any(k[0] == model and k[1] == cap for k in acc):
            continue
        print(f"\n### {model} | read | cap={cap}  (3-run mean±std, ad=F vs ad=T)")
        print(f"{'rate':>4} | {'HU F':>9} {'HU T':>9} | "
              f"{'viol% F':>9} {'viol% T':>9} | "
              f"{'ttfc99 F':>10} {'ttfc99 T':>10} | "
              f"{'dec/s F':>9} {'dec/s T':>9} | {'shr%T':>7}")
        for rate in rates:
            f_ = acc.get((model, cap, rate, "progress_serve"), {})
            t_ = acc.get((model, cap, rate, "progress_serve_adaptive"), {})
            if not f_ and not t_:
                continue
            print(f"{rate:>4.0f} | {ms(f_.get('hu',[])):>9} {ms(t_.get('hu',[])):>9} | "
                  f"{ms(f_.get('viol',[])):>9} {ms(t_.get('viol',[])):>9} | "
                  f"{ms(f_.get('ttfc99',[])):>10} {ms(t_.get('ttfc99',[])):>10} | "
                  f"{ms(f_.get('dec',[])):>9} {ms(t_.get('dec',[])):>9} | "
                  f"{ms(t_.get('shr',[])):>7}")
