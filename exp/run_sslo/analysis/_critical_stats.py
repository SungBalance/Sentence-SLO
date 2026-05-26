"""Critical-mode statistics per cell from latest smoke."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/workspace/mlsys/exp/run_sslo/output_smoke/sentence/Qwen3.5-9B")
CELLS = [
    ("read/baseline",       BASE / "read/none/cap256/baseline/run_1/rate_16"),
    ("read/sslo",           BASE / "read/none/cap256/sslo_mlp/run_1/rate_16"),
    ("Kokoro/baseline",     BASE / "tts/hexgrad__Kokoro-82M/cap256/baseline/run_1/rate_16"),
    ("Kokoro/sslo",         BASE / "tts/hexgrad__Kokoro-82M/cap256/sslo_mlp/run_1/rate_16"),
    ("Supertone/baseline",  BASE / "tts/Supertone__supertonic-3/cap256/baseline/run_1/rate_16"),
    ("Supertone/sslo",      BASE / "tts/Supertone__supertonic-3/cap256/sslo_mlp/run_1/rate_16"),
]

print(f"{'cell':<22s}  {'steps':>7s}  {'crit%':>6s}  "
      f"{'avg HU':>7s}  {'avg HU (crit)':>14s}  {'max max_score':>13s}")
print("=" * 90)

for label, d in CELLS:
    sp = d / "scheduler_stats.jsonl"
    if not sp.exists():
        print(f"{label}: missing"); continue
    n_steps = 0
    n_crit = 0
    sum_hu = 0
    sum_hu_crit = 0
    n_hu_crit = 0
    max_score_global = 0.0
    with sp.open() as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                continue
            if s.get("kind") != "step":
                continue
            n_steps += 1
            hu = s.get("num_handling_users") or 0
            sum_hu += hu
            ms = s.get("max_score") or 0
            if ms > max_score_global:
                max_score_global = ms
            if s.get("has_critical"):
                n_crit += 1
                sum_hu_crit += hu
                n_hu_crit += 1
    avg_hu = sum_hu / max(1, n_steps)
    avg_hu_crit = sum_hu_crit / max(1, n_hu_crit) if n_hu_crit else 0
    crit_pct = 100.0 * n_crit / max(1, n_steps)
    print(f"{label:<22s}  {n_steps:>7d}  {crit_pct:>5.1f}%  "
          f"{avg_hu:>7.0f}  {avg_hu_crit:>14.0f}  {max_score_global:>13.2f}")
