"""For the 'adaptive looks worse' cells, show the actual cur_max_num_requests
distribution to tell logic-defect from run noise."""
import collections
import json
import os

ROOT = "exp/run_sslo/output_sweep_adaptive/sentence"
# (model_slug, consume_dir, tts_dir, cap, rate) cells flagged as adaptive-worse
FLAGGED = [
    ("Qwen3.5-35B-A3B", "read", "none", 128, 64),
    ("Qwen3.5-35B-A3B", "read", "none", 128, 80),
    ("Qwen3.5-35B-A3B", "read", "none", 256, 32),
    ("Qwen3.5-9B", "read", "none", 128, 8),
    ("Qwen3.5-9B", "tts", "hexgrad__Kokoro-82M", 256, 16),
]
for model, consume, tts, cap, rate in FLAGGED:
    d = (f"{ROOT}/{model}/{consume}/{tts}/cap{cap}/"
         f"progress_serve_adaptive/run_1/rate_{rate:g}")
    sp = f"{d}/scheduler_stats.jsonl"
    if not os.path.exists(sp):
        print(f"MISSING {sp}")
        continue
    caps = []
    evs = []
    for l in open(sp):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("kind") == "step":
            caps.append(r.get("cur_max_num_requests"))
            evs.append(r.get("e_viol", 0.0))
    c = collections.Counter(caps)
    trig = sum(1 for e in evs if e is not None and e >= 1.0)
    print(f"\n{model}/{consume}/cap{cap}/rate{rate}: steps={len(caps)}")
    print(f"  cap_dist={c.most_common(6)}")
    print(f"  steps with e_viol>=1 (trigger fired)={trig} "
          f"({100*trig/max(1,len(evs)):.1f}%)")
