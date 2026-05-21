#!/usr/bin/env python3
"""Print per-model throughput-vs-batch table + identify the 10% knee."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "output"

for model_dir in sorted(ROOT.glob("*")):
    print(f"\n=== {model_dir.name} ===")
    print(f"  {'batch':>6} {'tok/s':>8} {'req/s':>6} {'win_s':>7} {'tokens':>7} {'growth':>7}")
    print("  " + "-" * 56)
    rows = []
    for bsz_dir in sorted(model_dir.glob("bsz_*"), key=lambda p: int(p.name.split("_")[1])):
        rp = bsz_dir / "result.json"
        if not rp.exists():
            continue
        d = json.loads(rp.read_text())
        rows.append(d)
    prev = None
    knee_batch = None
    for d in rows:
        tps = d["throughput_tokens_per_second"]
        growth = (tps / prev - 1) * 100 if prev else None
        growth_str = f"{growth:+6.1f}%" if growth is not None else "    —"
        if prev is not None and tps < prev * 1.10 and knee_batch is None:
            knee_batch = d["max_num_seqs"]
        print(f"  {d['max_num_seqs']:>6} {tps:>8.1f} "
              f"{d['throughput_req_per_second']:>6.2f} "
              f"{d['window_duration_s']:>7.1f} "
              f"{d['window_total_output_tokens']:>7} "
              f"{growth_str}")
        prev = tps
    if rows:
        peak = max(rows, key=lambda r: r["throughput_tokens_per_second"])
        print(f"  → peak: batch={peak['max_num_seqs']} ({peak['throughput_tokens_per_second']:.1f} tok/s)"
              + (f"; knee at batch={knee_batch}" if knee_batch else ""))
