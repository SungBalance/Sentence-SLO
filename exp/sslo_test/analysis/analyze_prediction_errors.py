#!/usr/bin/env python3
"""Diagnose which predictions failed for SSLO-only negative-slack chunks.

Approach: for each SSLO neg-slack chunk that did NOT exist in baseline (i.e.,
SSLO introduced the miss), compare the chunk's gen_time and word_count against
the EMA at that point (computed from prior chunks of the same request) and
flag where prediction was off.

Inputs:
  exp/sslo_test/output/seqs_64/{baseline,sslo}_chunks.jsonl

Excludes chunk_idx=0 from neg-slack ratio (slack always 0 by definition).
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def group_by_req(chunks):
    by = defaultdict(list)
    for c in chunks:
        by[c["request_id"]].append(c)
    for rid in by:
        by[rid].sort(key=lambda r: r["chunk_idx"])
    return by


def ema_after(samples, alpha=0.2):
    """EMA value after seeing the given list of samples."""
    if not samples:
        return None
    ema = samples[0]
    for s in samples[1:]:
        ema = alpha * s + (1 - alpha) * ema
    return ema


ROOT = Path("/workspace/mlsys/exp/sslo_test/output/seqs_64")
base = load(ROOT / "baseline_chunks.jsonl")
sslo = load(ROOT / "sslo_chunks.jsonl")

base_by = group_by_req(base)
sslo_by = group_by_req(sslo)

# === ratio with/without chunk 0 ===
def neg_ratio(chunks, exclude_chunk_0):
    pool = [c for c in chunks if not (exclude_chunk_0 and c["chunk_idx"] == 0)]
    neg = [c for c in pool if c["cumulative_slack"] < 0]
    return len(neg), len(pool), len(neg) / len(pool) * 100.0 if pool else 0.0

print("=== neg ratio: chunk_0 inclusion check ===")
for label, chunks in [("BASELINE", base), ("SSLO", sslo)]:
    n_with, t_with, r_with = neg_ratio(chunks, exclude_chunk_0=False)
    n_no, t_no, r_no = neg_ratio(chunks, exclude_chunk_0=True)
    print(f"{label}: incl chunk0 -> {n_with}/{t_with} ({r_with:.3f}%);  excl chunk0 -> {n_no}/{t_no} ({r_no:.3f}%)")

# === SSLO-only neg requests ===
base_neg_ids = {c["request_id"] for c in base if c["cumulative_slack"] < 0 and c["chunk_idx"] >= 1}
sslo_neg_ids = {c["request_id"] for c in sslo if c["cumulative_slack"] < 0 and c["chunk_idx"] >= 1}
sslo_only_ids = sslo_neg_ids - base_neg_ids
print(f"\nSSLO-only neg-affected requests: {len(sslo_only_ids)}")

# === per-affected-request: examine the neg chunk ===
# For each SSLO-only neg request, find its neg chunk(s), compare with prior chunks of same req
print("\n=== SSLO-only neg requests: per-chunk diagnosis ===")
print(f"{'req':<5} {'idx':<4} {'slack':>8} {'gen_t':>7} {'words':>5} {'prev_gen_p50':>13} {'prev_words_p50':>15} {'gen_x_med':>10} {'words_x_med':>12}")

stats_rows = []
for rid in sorted(sslo_only_ids, key=int):
    req_chunks = sslo_by[rid]
    neg_chunks = [c for c in req_chunks if c["cumulative_slack"] < 0 and c["chunk_idx"] >= 1]
    for nc in neg_chunks:
        idx = nc["chunk_idx"]
        prior = [c for c in req_chunks if c["chunk_idx"] < idx and c["chunk_idx"] >= 1]
        if not prior:
            prior_gen_med = None
            prior_words_med = None
            gen_x = None
            words_x = None
        else:
            prior_gen_med = statistics.median(c["gen_time"] for c in prior)
            prior_words_med = statistics.median(c["word_count"] for c in prior)
            gen_x = nc["gen_time"] / prior_gen_med if prior_gen_med > 0 else None
            words_x = nc["word_count"] / prior_words_med if prior_words_med > 0 else None
        stats_rows.append({
            "rid": rid, "idx": idx, "slack": nc["cumulative_slack"],
            "gen": nc["gen_time"], "words": nc["word_count"],
            "prior_gen": prior_gen_med, "prior_words": prior_words_med,
            "gen_x": gen_x, "words_x": words_x,
        })
        print(f"{rid:<5} {idx:<4} {nc['cumulative_slack']:>8.3f} {nc['gen_time']:>7.3f} {nc['word_count']:>5d} "
              f"{(prior_gen_med if prior_gen_med else 0):>13.3f} {(prior_words_med if prior_words_med else 0):>15.1f} "
              f"{(gen_x if gen_x else 0):>10.2f} {(words_x if words_x else 0):>12.2f}")

# === aggregate prediction-error patterns ===
print("\n=== aggregate ===")
gen_x_vals = [r["gen_x"] for r in stats_rows if r["gen_x"] is not None]
words_x_vals = [r["words_x"] for r in stats_rows if r["words_x"] is not None]
if gen_x_vals:
    gen_x_vals.sort()
    print(f"neg chunk gen_time / prior median:  p25={gen_x_vals[len(gen_x_vals)//4]:.2f}  p50={gen_x_vals[len(gen_x_vals)//2]:.2f}  p90={gen_x_vals[int(0.9*len(gen_x_vals))]:.2f}  max={gen_x_vals[-1]:.2f}")
if words_x_vals:
    words_x_vals.sort()
    print(f"neg chunk word_count / prior median: p25={words_x_vals[len(words_x_vals)//4]:.2f}  p50={words_x_vals[len(words_x_vals)//2]:.2f}  p90={words_x_vals[int(0.9*len(words_x_vals))]:.2f}  max={words_x_vals[-1]:.2f}")

# How many "gen_time spike" (gen_x >= 2.0) vs "word spike" (words_x >= 2.0)
spike_gen = sum(1 for r in stats_rows if r["gen_x"] and r["gen_x"] >= 2.0)
spike_words = sum(1 for r in stats_rows if r["words_x"] and r["words_x"] >= 2.0)
both = sum(1 for r in stats_rows if r["gen_x"] and r["words_x"] and r["gen_x"] >= 2.0 and r["words_x"] >= 2.0)
neither = sum(1 for r in stats_rows if r["gen_x"] is not None and r["words_x"] is not None
              and r["gen_x"] < 2.0 and r["words_x"] < 2.0)
print(f"gen_time >= 2x prior median: {spike_gen}/{len(stats_rows)} ({spike_gen/max(1,len(stats_rows))*100:.0f}%)")
print(f"word_count >= 2x prior median: {spike_words}/{len(stats_rows)} ({spike_words/max(1,len(stats_rows))*100:.0f}%)")
print(f"BOTH gen_time and words spike: {both}")
print(f"NEITHER spikes (load-induced?): {neither}")

# Pure gen-time inflation: gen_x > words_x means "took longer than expected for the size"
# If a chunk has 1.2x words but 3x gen_time → load (or pending) inflated wall-clock
inflation = []
for r in stats_rows:
    if r["gen_x"] is not None and r["words_x"] is not None and r["words_x"] > 0:
        inflation.append(r["gen_x"] / r["words_x"])
if inflation:
    inflation.sort()
    print(f"gen_x / words_x (inflation factor): p50={inflation[len(inflation)//2]:.2f}  p90={inflation[int(0.9*len(inflation))]:.2f}  max={inflation[-1]:.2f}")
    print(f"  high inflation (>=2.0) means wall-clock per-token grew much faster than chunk size")
