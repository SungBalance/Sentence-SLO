"""Hypothesis: in 35B, a step's iteration time depends heavily on how many
prefills are mixed into the batch, but pick_adaptive_batch's Δ (profile +
wall_ema, prefill=0-preferring) only sees the decode-only latency. So the
picker underestimates real Δ → R_defer underestimated → it shrinks when it
shouldn't, and throughput drops more than the model predicts.

Measure, from a 35B/cap128 adaptive run's scheduler_stats, the actual
per-step wall time bucketed by (realized batch, num_prefills>0?).
"""
import collections
import json
import os
import statistics

ROOT = "exp/run_sslo/output_adaptive_ablation/sentence"
CASES = [
    ("Qwen3.5-35B-A3B", 128, 80, "progress_serve_adaptive"),
    ("Qwen3.5-35B-A3B", 128, 80, "progress_serve"),
    ("Qwen3.5-9B", 128, 80, "progress_serve_adaptive"),
]


def f(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


for model, cap, rate, mode in CASES:
    # run_1 dir
    d = f"{ROOT}/{model}/read/none/cap{cap}/{mode}/run_1/rate_{rate}"
    sp = f"{d}/scheduler_stats.jsonl"
    if not os.path.exists(sp):
        print("MISSING", sp)
        continue
    rows = [json.loads(l) for l in open(sp)
            if l.strip() and json.loads(l).get("kind") == "step"]
    rows.sort(key=lambda r: r["ts"])
    # bucket dt by (running, has_prefill). prefill proxy: num_prefill_reqs.
    dec_only = collections.defaultdict(list)   # running -> [dt] when prefills==0
    with_pref = collections.defaultdict(list)  # running -> [dt] when prefills>0
    pref_frac = []
    for a, b in zip(rows, rows[1:]):
        dt = b["ts"] - a["ts"]
        if dt <= 0 or dt > 5:
            continue
        run = a.get("running")
        npref = a.get("num_prefill_reqs")
        if npref is None:
            npref = a.get("num_prefills", 0)
        (with_pref if (npref and npref > 0) else dec_only)[run].append(dt)
        pref_frac.append(1 if (npref and npref > 0) else 0)
    print(f"\n==== {model}/cap{cap}/rate{rate}/{mode} ====")
    print(f"  steps with prefill mixed: "
          f"{sum(pref_frac)}/{len(pref_frac)} "
          f"({100*sum(pref_frac)/max(1,len(pref_frac)):.1f}%)")
    # dominant realized batch
    allruns = [r.get("running") for r in rows]
    dom = collections.Counter(allruns).most_common(1)[0][0]
    do = dec_only.get(dom, [])
    wp = with_pref.get(dom, [])
    print(f"  dominant running={dom}")
    if do:
        print(f"  Δ decode-only  : median={statistics.median(do):.4f}s "
              f"n={len(do)}  → thru={dom/statistics.median(do):.0f} tok/s")
    if wp:
        print(f"  Δ prefill-mixed: median={statistics.median(wp):.4f}s "
              f"n={len(wp)}  → thru={dom/statistics.median(wp):.0f} tok/s")
    if do and wp:
        print(f"  prefill-mix slowdown: "
              f"{statistics.median(wp)/statistics.median(do):.2f}x")
