"""Trace what limits run_m from reaching cap on critical-admit (sslo +1+2+3).

If admit unfreezes waiting → why does running stay at ~96 instead of
climbing toward 256?

Check 1: How many step did waiting_admission_budget > 0 vs how many
         actually admitted (waiting → running delta)?
Check 2: Step-over-step running delta distribution.
Check 3: bf_kv_full / bf_too_few_tokens / waiting empty share.
"""
import json
from collections import Counter
import statistics
from pathlib import Path

ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v4_9b_256_16/sslo_mlp_123_s/sslo_mlp/run_1")
steps = []
with (ROOT / "scheduler_stats.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        if r.get("kind") != "step": continue
        steps.append(r)
print(f"total steps: {len(steps)}")

# Per-step delta of running and waiting.
run_delta = []
wait_delta = []
prev_run = None; prev_wait = None
for s in steps:
    run = s.get("running", 0) or 0
    wait = s.get("waiting", 0) or 0
    if prev_run is not None:
        run_delta.append(run - prev_run)
        wait_delta.append(wait - prev_wait)
    prev_run = run; prev_wait = wait

print(f"\nrunning delta step-over-step:")
positives = [d for d in run_delta if d > 0]
negatives = [d for d in run_delta if d < 0]
zeros = sum(1 for d in run_delta if d == 0)
print(f"  +ve: n={len(positives)} mean={statistics.mean(positives) if positives else 0:.2f} max={max(positives) if positives else 0}")
print(f"  -ve: n={len(negatives)} mean={statistics.mean(negatives) if negatives else 0:.2f} min={min(negatives) if negatives else 0}")
print(f"  zero: {zeros} ({zeros/len(run_delta)*100:.1f}%)")

# Backfill skip reasons
bf_reasons = Counter()
for s in steps:
    r = s.get("bf_skip_reason")
    if r is not None: bf_reasons[r] += 1
print(f"\nbf_skip_reason distribution:")
for r, ct in bf_reasons.most_common(6):
    print(f"  {r}: {ct} ({ct/len(steps)*100:.1f}%)")

# Critical steps with waiting>0 (room to admit) and next-step run delta
slack_admit_candidates = []
for i in range(len(steps)-1):
    s = steps[i]; n = steps[i+1]
    if not s.get("has_critical"): continue
    run = s.get("running",0) or 0
    wait = s.get("waiting",0) or 0
    slack = 256 - run
    if slack > 0 and wait > 0:
        next_run = n.get("running",0) or 0
        slack_admit_candidates.append((slack, wait, next_run - run))

print(f"\ncritical steps with admit room (slack>0 + waiting>0) n={len(slack_admit_candidates)}:")
if slack_admit_candidates:
    slacks = [s for s,_,_ in slack_admit_candidates]
    waits = [w for _,w,_ in slack_admit_candidates]
    deltas = [d for _,_,d in slack_admit_candidates]
    print(f"  slack mean={statistics.mean(slacks):.1f} max={max(slacks)}")
    print(f"  waiting mean={statistics.mean(waits):.1f}")
    print(f"  actual next-step run delta mean={statistics.mean(deltas):.2f} max={max(deltas)} min={min(deltas)}")
    print(f"  net positive admit: {sum(1 for d in deltas if d>0)} of {len(deltas)} ({sum(1 for d in deltas if d>0)/len(deltas)*100:.1f}%)")
    # Histogram of deltas
    pos_hist = Counter()
    for d in deltas:
        if d <= 0: bucket = "≤0"
        elif d <= 2: bucket = "1-2"
        elif d <= 5: bucket = "3-5"
        elif d <= 10: bucket = "6-10"
        else: bucket = ">10"
        pos_hist[bucket] += 1
    for b in ["≤0", "1-2", "3-5", "6-10", ">10"]:
        print(f"  delta {b}: {pos_hist[b]} ({pos_hist[b]/len(deltas)*100:.1f}%)")
