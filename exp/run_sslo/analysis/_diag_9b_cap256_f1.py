"""Diagnose remaining 9B/cap=256/F1 regressions vs baseline.

Reads chunks.jsonl + requests.jsonl + scheduler_stats.jsonl for a few
representative cells:
  (a) regression: 9B/Supertone r=16  (worst F1 regression)
  (b) regression: 9B/Kokoro r=20
  (c) NO regression: 9B/read r=16 (F1 fixed it)
  (d) NO regression: 9B/Supertone cap=128 r=16 (cap-bound, never regressed)

For each cell, report:
  - HU trajectory: peak vs avg
  - per-request max_stall distribution (p50/p95/p99/max + # > 1s)
  - violator stall vs queue_stall (was it queue stall or chunk stall?)
  - violator chunk index distribution (early/mid/late chunks?)
  - violator word_count distribution
  - predictor error: predicted_chunk_token vs actual_chunk_token (mean error %)
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("/workspace/mlsys/exp/run_sslo/output_sweep/sentence")

def num(v):
    try: return float(v)
    except: return None

CELLS = [
    ("REG-WORST", "Qwen3.5-9B", "tts/Supertone__supertonic-3", 256, "sslo_mlp", 16),
    ("REG-MID",   "Qwen3.5-9B", "tts/hexgrad__Kokoro-82M",   256, "sslo_mlp", 20),
    ("OK-READ",   "Qwen3.5-9B", "read/none",                  256, "sslo_mlp", 16),
    ("OK-CAP128", "Qwen3.5-9B", "tts/Supertone__supertonic-3", 128, "sslo_mlp", 16),
]

def cell_dir(model, cellpath, cap, mode, rate):
    return (BASE / model / cellpath / f"cap{cap}" / mode / "run_1" / f"rate_{rate}")

def percentile(xs, p):
    if not xs: return None
    s = sorted(xs); n = len(s)
    k = max(0, min(n - 1, int(p * (n - 1))))
    return s[k]

for label, model, cellpath, cap, mode, rate in CELLS:
    d = cell_dir(model, cellpath, cap, mode, rate)
    print(f"\n{'='*80}\n[{label}] {model}/{cellpath} cap={cap} mode={mode} rate={rate}")
    print(f"  dir: {d.relative_to(BASE.parent)}")

    chunks_path = d / "chunks.jsonl"
    reqs_path = d / "requests.jsonl"
    stats_path = d / "scheduler_stats.jsonl"
    if not chunks_path.exists():
        print("  MISSING chunks.jsonl"); continue

    # Per-request aggregates
    per_req_max_stall = defaultdict(float)
    per_req_first_chunk_violated = {}
    per_req_word_counts = defaultdict(list)
    per_req_chunk_count = defaultdict(int)
    per_req_violator_chunk_indices = defaultdict(list)
    predicted_vs_actual = []  # (predicted_tokens, actual_tokens)

    with chunks_path.open() as f:
        for line in f:
            r = json.loads(line)
            rid = r.get("request_id")
            miss = float(r.get("unit_deadline_miss_s") or 0)
            wc = int(r.get("word_count") or 0)
            cidx = int(r.get("unit_index", r.get("chunk_idx", 0)) or 0)
            if rid is None: continue
            per_req_chunk_count[rid] += 1
            per_req_word_counts[rid].append(wc)
            if miss > per_req_max_stall[rid]:
                per_req_max_stall[rid] = miss
            if miss > 0:
                per_req_violator_chunk_indices[rid].append((cidx, miss, wc))
            # predictor error (if recorded)
            p_tok = r.get("predicted_chunk_num_token")
            a_tok = r.get("num_token")
            if p_tok is not None and a_tok is not None:
                try:
                    predicted_vs_actual.append((float(p_tok), float(a_tok)))
                except (TypeError, ValueError):
                    pass

    stalls = list(per_req_max_stall.values())
    n_reqs = len(stalls)
    n_viol_1 = sum(1 for s in stalls if s > 1.0)
    n_viol_05 = sum(1 for s in stalls if s > 0.5)
    print(f"\n  requests={n_reqs}  viol@1={n_viol_1}/{n_reqs}={n_viol_1/n_reqs:.3%}"
          f"  viol@0.5={n_viol_05}/{n_reqs}={n_viol_05/n_reqs:.3%}")
    print(f"  max_stall: p50={percentile(stalls,0.5):.3f} "
          f"p95={percentile(stalls,0.95):.3f} "
          f"p99={percentile(stalls,0.99):.3f} "
          f"max={max(stalls):.2f}s")

    # Violators (max_stall > 1s): what chunk index? what word_count?
    bad_rids = [rid for rid, s in per_req_max_stall.items() if s > 1.0]
    if bad_rids:
        bad_chunk_indices = []
        bad_chunk_wcs = []
        for rid in bad_rids:
            for cidx, _, wc in per_req_violator_chunk_indices[rid]:
                bad_chunk_indices.append(cidx)
                bad_chunk_wcs.append(wc)
        print(f"  violators ({len(bad_rids)} reqs): "
              f"violating chunks={len(bad_chunk_indices)}")
        if bad_chunk_indices:
            print(f"    violating chunk_idx:  mean={statistics.mean(bad_chunk_indices):.1f} "
                  f"median={statistics.median(bad_chunk_indices):.1f} "
                  f"min={min(bad_chunk_indices)} max={max(bad_chunk_indices)}")
            print(f"    violating word_count: mean={statistics.mean(bad_chunk_wcs):.1f} "
                  f"median={statistics.median(bad_chunk_wcs):.1f} "
                  f"min={min(bad_chunk_wcs)} max={max(bad_chunk_wcs)}")

    # Request-level word_count compared to non-violators
    all_word_counts_first = [wc[0] if wc else 0 for wc in per_req_word_counts.values()]
    bad_word_counts_first = [per_req_word_counts[rid][0]
                              for rid in bad_rids if per_req_word_counts[rid]]
    if all_word_counts_first:
        print(f"  word_count of FIRST chunk: all reqs mean={statistics.mean(all_word_counts_first):.1f}, "
              f"violators mean={statistics.mean(bad_word_counts_first) if bad_word_counts_first else 0:.1f}")

    # Predictor error
    if predicted_vs_actual:
        errs = [(p - a) for p, a in predicted_vs_actual]
        rel_errs = [(p - a) / max(1, a) for p, a in predicted_vs_actual]
        print(f"  chunk_length predictor: n={len(predicted_vs_actual)}  "
              f"mean_err={statistics.mean(errs):+.1f} tokens  "
              f"mean_rel_err={100*statistics.mean(rel_errs):+.1f}%  "
              f"|rel_err|_p90={100*percentile([abs(e) for e in rel_errs], 0.9):.1f}%")

    # HU trajectory from scheduler_stats.jsonl
    if stats_path.exists():
        hus = []
        running_max = pending_max = 0
        with stats_path.open() as f:
            for line in f:
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                hu = s.get("num_handling_users") or 0
                hus.append(hu)
                running_max = max(running_max, s.get("running") or 0)
                pending_max = max(pending_max, s.get("pending") or 0)
        if hus:
            print(f"  HU trajectory: mean={statistics.mean(hus):.0f} "
                  f"p95={percentile(hus,0.95):.0f} max={max(hus)}  "
                  f"running_max={running_max} pending_max={pending_max}")
