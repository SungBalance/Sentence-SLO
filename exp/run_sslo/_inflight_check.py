"""Key hypothesis: when the picker shrinks b, does it create deferred
requests (which carry the R_defer penalty)? If in-flight count (running)
at decision time is already <= the shrunk b, then decode_cap >= in-flight
=> deferred = 0 => NO R_defer penalty => E_viol only sees R_run (which
drops with smaller Δ) => unconditional shrink => lock-in.

Compare, on the locked-in adaptive cells, the realized running count vs
the chosen batch b across steps.
"""
import json
import os
import statistics

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
CELLS = [
    ("Qwen3.5-35B-A3B", "read", "none", 128, 80),
    ("Qwen3.5-9B", "read", "none", 128, 8),
]


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


for model, consume, tts, cap, rate in CELLS:
    d = (f"{ROOT}/{model}/{consume}/{tts}/cap{cap}/"
         f"progress_serve_adaptive/run_1/rate_{rate:g}")
    sp = f"{d}/scheduler_stats.jsonl"
    if not os.path.exists(sp):
        print("MISSING", sp)
        continue
    rows = [json.loads(l) for l in open(sp)
            if l.strip() and json.loads(l).get("kind") == "step"]
    cap_b = [r.get("cur_max_num_requests") for r in rows]
    run = [r.get("running") for r in rows]
    pend = [r.get("pending") for r in rows]
    wait = [r.get("waiting") for r in rows]
    # On steps where batch was shrunk (cap_b < cap), is running >= cap_b
    # (=> batch is the binding limit, deferred possible) or < cap_b
    # (=> in-flight under-fills batch, no deferred, no R_defer)?
    n = len(rows)
    run_ge_b = sum(1 for cb, rn in zip(cap_b, run)
                   if cb is not None and rn is not None and rn >= cb)
    print(f"\n==== {model}/{consume}/cap{cap}/rate{rate} (n={n}) ====")
    print(f"  median cur_max_num_requests = {statistics.median(cap_b)}")
    print(f"  median running = {statistics.median(run)}")
    print(f"  median pending = {statistics.median(pend)}")
    print(f"  median waiting = {statistics.median(wait)}")
    print(f"  steps with running >= batch (batch is binding) = "
          f"{run_ge_b}/{n} = {100*run_ge_b/n:.1f}%")
    # pending>0 means deferral actually happened
    pend_pos = sum(1 for p in pend if p and p > 0)
    print(f"  steps with pending>0 (deferral happened) = "
          f"{pend_pos}/{n} = {100*pend_pos/n:.1f}%")
