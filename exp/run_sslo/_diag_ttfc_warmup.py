"""Why did TTFC mean rise 132s → 197s with +1+2?

Per-req TTFC = first_chunk_end_ts - queued_ts (vLLM queue entry).
Warmup gating delays window start, but TTFC is per-request so warmup
shouldn't affect it directly. Hypothesis: longer warmup → queue gets
deeper by window start → in-window reqs face longer queue waits.

Compare in-window reqs of v3 vs +1+2:
  - queued_ts → ttfc latency distribution
  - admitted_ts → first_chunk latency (post-admission)
  - injection_ts → admitted_ts (queue wait)
"""
import json
import statistics
from pathlib import Path

CONFIGS = [
    ("v3 (1st-completion window)",
     "/workspace/mlsys/exp/run_sslo/output/v3_9b_256_16/sslo_mlp/run_1",
     1779255369.2113204, 1779255549.2113204),
    ("+1+2 (256-completion window)",
     "/workspace/mlsys/exp/run_sslo/output/v4_9b_256_16/sslo_mlp_12_s/sslo_mlp/run_1",
     None, None),  # read from run_meta
]

def pct(arr, q): return sorted(arr)[min(len(arr)-1, int(q*len(arr)))] if arr else 0

for label, root, mw0, mw1 in CONFIGS:
    root = Path(root)
    if mw0 is None:
        meta = json.loads((root / "run_meta.json").read_text())
        mw0 = meta["measurement_window_start_ts"]
        mw1 = meta["measurement_window_end_ts"]
    # Load chunks first.
    first_chunk_end = {}
    with (root / "chunks.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            if c.get("chunk_idx", 0) != 0: continue
            rid = str(c.get("request_id"))
            first_chunk_end[rid] = c.get("end_time_ts")
    inj_to_admit = []
    admit_to_first = []
    inj_to_first = []
    queued_to_first = []
    inj_offsets_in_window = []  # injection_ts - window_start_ts
    n_in_win = 0
    with (root / "requests.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            inj = r.get("injection_ts")
            adm = r.get("admitted_ts")
            queued = None  # vLLM internal, not directly in row
            rid = str(r.get("request_id"))
            fc = first_chunk_end.get(rid)
            if inj is None or fc is None or adm is None: continue
            if not (mw0 <= inj < mw1): continue
            n_in_win += 1
            inj_to_admit.append(adm - inj)
            admit_to_first.append(fc - adm)
            inj_to_first.append(fc - inj)
            inj_offsets_in_window.append(inj - mw0)
    print(f"\n=== {label} ===")
    print(f"  in-window n={n_in_win}, window_dur={mw1-mw0:.1f}s")
    print(f"  injection_offset within window: "
          f"min={min(inj_offsets_in_window):.1f} mean={statistics.mean(inj_offsets_in_window):.1f} max={max(inj_offsets_in_window):.1f}")
    print(f"  injection → admitted (vLLM queue wait):")
    print(f"    mean={statistics.mean(inj_to_admit):.2f}s p50={pct(inj_to_admit,0.5):.2f} p99={pct(inj_to_admit,0.99):.2f} max={max(inj_to_admit):.2f}")
    print(f"  admitted → first_chunk:")
    print(f"    mean={statistics.mean(admit_to_first):.2f}s p50={pct(admit_to_first,0.5):.2f} p99={pct(admit_to_first,0.99):.2f}")
    print(f"  injection → first_chunk (full TTFC analog):")
    print(f"    mean={statistics.mean(inj_to_first):.2f}s p50={pct(inj_to_first,0.5):.2f} p99={pct(inj_to_first,0.99):.2f}")
