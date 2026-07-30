import collections
import json
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else (
    "exp/run_sslo/output_smoke_adaptive/progress_serve_adaptive/"
    "run_1/rate_40/scheduler_stats.jsonl")
if not os.path.exists(p):
    print("MISSING", p)
    raise SystemExit
steps = [json.loads(l) for l in open(p) if l.strip()]
steps = [s for s in steps if s.get("kind") == "step"]
caps = [s["cur_max_num_requests"] for s in steps]
shrunk = sum(1 for x in caps if x < max(caps))
c = collections.Counter(caps)
print("step_rows", len(steps))
print("cap_dist", c.most_common(10))
print("shrunk_pct", round(100 * shrunk / max(1, len(caps)), 1))
