"""Measure actual per-step time Δ(b) from step-stat timestamps, and the
realized decode throughput b/Δ(b), to see whether shrinking the batch
actually lowers throughput (→ R_defer should RISE → E_viol should rise →
hill-climb should stop). If it does, lock-in implies the Δ(b) estimate the
picker used was optimistic, not that the math is wrong.
"""
import json
import os
import statistics

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
CELLS = [
    ("Qwen3.5-35B-A3B", "read", "none", 128, 80),
    ("Qwen3.5-9B", "read", "none", 128, 8),
]
MODES = ["baseline", "progress_serve", "progress_serve_adaptive"]


def step_time_by_batch(d):
    sp = f"{d}/scheduler_stats.jsonl"
    if not os.path.exists(sp):
        return {}
    rows = [json.loads(l) for l in open(sp)
            if l.strip() and json.loads(l).get("kind") == "step"]
    rows.sort(key=lambda r: r["ts"])
    by_b = {}
    for a, b in zip(rows, rows[1:]):
        dt = b["ts"] - a["ts"]
        if dt <= 0 or dt > 5:  # drop gaps / pauses
            continue
        cap = a.get("cur_max_num_requests")
        run = a.get("running")
        # bucket by realized running count (actual decode batch)
        by_b.setdefault(run, []).append(dt)
    return by_b


def f(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


for model, consume, tts, cap, rate in CELLS:
    print(f"\n==== {model} | {consume} | cap{cap} | rate{rate} ====")
    print(f"{'mode':<22} {'realizedB':>9} {'Δ_med(s)':>9} "
          f"{'thru=b/Δ':>9} {'N_future∝thru':>13}")
    for mode in MODES:
        d = f"{ROOT}/{model}/{consume}/{tts}/cap{cap}/{mode}/run_1/rate_{rate:g}"
        by_b = step_time_by_batch(d)
        if not by_b:
            print(f"{mode:<22} (missing)")
            continue
        # dominant realized batch = the one with most steps
        dom = max(by_b, key=lambda k: len(by_b[k]))
        dt = statistics.median(by_b[dom])
        thru = dom / dt if dt else None
        print(f"{mode:<22} {dom:>9} {f(dt):>9} {f(thru,1):>9} "
              f"{f(thru,1):>13}")
