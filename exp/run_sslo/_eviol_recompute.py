"""Reconstruct E_viol(b=128) vs E_viol(b=64) from a representative locked-in
step, to see whether the picker's own math actually prefers the small batch
(=> the objective is the defect) or not (=> some wiring issue).

We approximate the in-flight tail with the global chunk-length empirical
distribution from the run's chunks.jsonl, and use measured Δ(128)/Δ(64).
"""
import json
import math
import os
from bisect import bisect_right

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
# 35B/read/cap128/rate80 locked at 64; measured deltas from _delta_check:
CASE = dict(
    model="Qwen3.5-35B-A3B", consume="read", tts="none", cap=128, rate=80,
    delta={128: 0.0591, 64: 0.0436},
    n_inflight=250,   # running(64)+pending(186) ~ candidate in-flight set
)


def load_hist(d):
    cp = f"{d}/chunks.jsonl"
    h = []
    for l in open(cp):
        if not l.strip():
            continue
        r = json.loads(l)
        nt = r.get("num_token")
        if nt:
            h.append(int(nt))
    return sorted(h)


def tail(sorted_h, x, c_q):
    n = len(sorted_h)
    denom = n - bisect_right(sorted_h, c_q)
    if denom <= 0:
        return 1.0
    return (n - bisect_right(sorted_h, x)) / denom


d = (f"{ROOT}/{CASE['model']}/{CASE['consume']}/{CASE['tts']}/"
     f"cap{CASE['cap']}/progress_serve_adaptive/run_1/rate_{CASE['rate']:g}")
H = load_hist(d)
print(f"global hist n={len(H)} median={H[len(H)//2]} "
      f"p90={H[int(0.9*len(H))]} p99={H[int(0.99*len(H))]}")

# Representative in-flight: assume each req has some c_q (tokens into current
# unit). Use a spread of c_q over the typical unit length.
import statistics
med = statistics.median(H)
c_qs = [0, med // 4, med // 2, med]  # mix of fresh..mid-unit requests
N = CASE["n_inflight"]
T_q = 1.0   # ~1s to deadline (tight, high-pressure regime); horizon scales

for b in (128, 64):
    delta = CASE["delta"][b]
    s = min(1.0, b / N)
    H_q = max(0, math.floor(T_q / delta))
    i_q = 1 if H_q >= 1 else 0
    n_future = math.floor(max(H_q - 1, 0) * s)
    n_run = i_q + n_future
    # decode_cap = b; deferred = max(0, N - b). Split c_qs evenly.
    # E_viol = sum over scheduled R_run + deferred R_defer.
    per_group = N / len(c_qs)
    sched = b
    e = 0.0
    rrun_ex = rdef_ex = None
    placed = 0
    for cq in c_qs:
        r_run = tail(H, cq + n_run, cq)
        r_def = tail(H, cq + n_future, cq)
        grp = per_group
        # schedule highest-benefit first; approximate: fill scheduled then defer
        sch_n = max(0, min(grp, sched - placed))
        def_n = grp - sch_n
        placed += sch_n
        e += sch_n * r_run + def_n * r_def
        rrun_ex, rdef_ex = r_run, r_def
    print(f"\nb={b}: Δ={delta} s={s:.3f} H_q={H_q} "
          f"N_run={n_run} N_future={n_future}")
    print(f"  sample R_run={rrun_ex:.4f} R_defer={rdef_ex:.4f} "
          f"(diff={rdef_ex-rrun_ex:.4f})")
    print(f"  E_viol≈{e:.1f}  (deferred={max(0,N-b)})")
