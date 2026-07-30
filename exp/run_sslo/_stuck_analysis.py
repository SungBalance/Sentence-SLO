"""Does small-batch lock-in actually hurt? For flagged cells, compare
baseline / progress_serve / progress_serve_adaptive on:
 - decode throughput (tokens/s), out tokens/s
 - backlog: time-weighted running / pending / waiting
 - violation, ttfc99, handling-users
 - adaptive: realized batch (cur_max distribution) + e_viol>=1 fraction
"""
import collections
import json
import os

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
CELLS = [
    ("Qwen3.5-35B-A3B", "read", "none", 128, 80),
    ("Qwen3.5-9B", "read", "none", 128, 8),
    ("Qwen3.5-35B-A3B", "read", "none", 256, 32),  # adaptive helps here
]
MODES = ["baseline", "progress_serve", "progress_serve_adaptive"]


def load_summary(d, mode):
    p = f"{d}/summary.json"
    if not os.path.exists(p):
        return {}
    m = json.load(open(p)).get("metrics", {})
    mm = m.get(mode, {})
    wl = m.get("workload", {}).get(mode, {})
    sch = m.get("scheduler", {}).get(mode, {})

    def dist(x):
        return x.get("mean") if isinstance(x, dict) else x
    return {
        "hu": mm.get("mean_handling_users"),
        "viol%": (wl.get("unit_deadline_miss_rate") or 0) * 100,
        "ttfc99": mm.get("ttfc_p99_s"),
        "out_tok/s": wl.get("tokens_per_second"),
        "dec_tok/s": sch.get("decode_tokens_per_second"),
        "run": dist(sch.get("running")),
        "pend": dist(sch.get("num_handling_users")),  # handling=run+pend proxy
        "kv%": (100 * sch.get("kv_blocks_used_mean") / sch["kv_blocks_total"])
        if sch.get("kv_blocks_used_mean") and sch.get("kv_blocks_total")
        else None,
    }


def realized_batch(d):
    sp = f"{d}/scheduler_stats.jsonl"
    if not os.path.exists(sp):
        return None, None, None
    caps, runs, waits = [], [], []
    for l in open(sp):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("kind") != "step":
            continue
        caps.append(r.get("cur_max_num_requests"))
        runs.append(r.get("running"))
        waits.append(r.get("waiting"))
    caps = [c for c in caps if c is not None]
    runs = [c for c in runs if c is not None]
    waits = [c for c in waits if c is not None]
    cap_mode = collections.Counter(caps).most_common(1)[0] if caps else None
    avg_run = sum(runs) / len(runs) if runs else None
    avg_wait = sum(waits) / len(waits) if waits else None
    return cap_mode, avg_run, avg_wait


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


for model, consume, tts, cap, rate in CELLS:
    print(f"\n==== {model} | {consume} | cap{cap} | rate{rate} ====")
    print(f"{'mode':<22} {'HU':>6} {'viol%':>6} {'ttfc99':>7} "
          f"{'out/s':>6} {'dec/s':>6} {'avgRun':>7} {'avgWait':>8} {'realB':>10}")
    for mode in MODES:
        d = f"{ROOT}/{model}/{consume}/{tts}/cap{cap}/{mode}/run_1/rate_{rate:g}"
        s = load_summary(d, mode)
        if not s:
            print(f"{mode:<22} (missing)")
            continue
        cap_mode, avg_run, avg_wait = realized_batch(d)
        rb = (f"{cap_mode[0]}x{round(100*cap_mode[1]/1,0):.0f}"
              if cap_mode else "-")
        # realB shown as "size" only (most common)
        rb = str(cap_mode[0]) if cap_mode else "-"
        print(f"{mode:<22} {f(s['hu'],0):>6} {f(s['viol%'],3):>6} "
              f"{f(s['ttfc99'],0):>7} {f(s['out_tok/s'],0):>6} "
              f"{f(s['dec_tok/s'],0):>6} {f(avg_run,1):>7} "
              f"{f(avg_wait,1):>8} {rb:>10}")
