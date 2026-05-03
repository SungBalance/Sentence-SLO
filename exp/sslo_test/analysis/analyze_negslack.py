#!/usr/bin/env python3
import json
from collections import Counter

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

base_chunks = load("/workspace/mlsys/exp/sslo_test/output/seqs_64/baseline_chunks.jsonl")
sslo_chunks = load("/workspace/mlsys/exp/sslo_test/output/seqs_64/sslo_chunks.jsonl")
base_ttft = load("/workspace/mlsys/exp/sslo_test/output/seqs_64/baseline_ttft.jsonl")
sslo_ttft = load("/workspace/mlsys/exp/sslo_test/output/seqs_64/sslo_ttft.jsonl")


def analyze(label, chunks, ttft):
    print(f"\n========== {label} ==========")
    neg = [c for c in chunks if c["cumulative_slack"] < 0]
    total_reqs = len(set(c["request_id"] for c in chunks))
    print(f"Total chunks: {len(chunks)}, negative: {len(neg)} "
          f"({len(neg)/len(chunks)*100:.3f}%), affected reqs: "
          f"{len(set(c['request_id'] for c in neg))}/{total_reqs}")

    if neg:
        neg_vals = sorted(c["cumulative_slack"] for c in neg)
        print(f"Magnitude: min={neg_vals[0]:.2f}s  median={neg_vals[len(neg_vals)//2]:.3f}s  worst-close-to-0={neg_vals[-1]:.4f}s")
        bins = [(-100,-10),(-10,-5),(-5,-2),(-2,-1),(-1,-0.5),(-0.5,-0.1),(-0.1,0)]
        for lo, hi in bins:
            n = sum(1 for v in neg_vals if lo < v <= hi)
            if n:
                print(f"  ({lo}, {hi}]: {n}")

    req_neg = Counter(c["request_id"] for c in neg)
    if req_neg:
        top5 = req_neg.most_common(5)
        print(f"Top 5 reqs by neg-chunk count: {top5}")

    # chunk index position
    if neg:
        idx_dist = Counter(c["chunk_idx"] for c in neg)
        idx_sorted = sorted(idx_dist.items())
        print(f"By chunk_idx (first 10): {idx_sorted[:10]}")
        max_chunks_per_req = {}
        for c in chunks:
            rid = c["request_id"]
            max_chunks_per_req[rid] = max(max_chunks_per_req.get(rid, -1), c["chunk_idx"])
        end_neg = sum(1 for c in neg if c["chunk_idx"] == max_chunks_per_req[c["request_id"]])
        print(f"  At LAST chunk of their request: {end_neg}/{len(neg)} ({end_neg/len(neg)*100:.0f}%)")
        # Position as fraction of request length
        positions = []
        for c in neg:
            mx = max_chunks_per_req[c["request_id"]]
            if mx > 0:
                positions.append(c["chunk_idx"] / mx)
        if positions:
            positions.sort()
            print(f"  Position fraction (idx/max_idx): p25={positions[len(positions)//4]:.2f} p50={positions[len(positions)//2]:.2f} p75={positions[3*len(positions)//4]:.2f}")

    # cutoff at max_tokens
    cutoff_at_max = sum(1 for r in ttft if r["num_tokens"] == 512)
    print(f"Reqs at max_tokens (512): {cutoff_at_max}/{len(ttft)}")
    if cutoff_at_max and neg:
        cutoff_ids = {str(r["request_id"]) for r in ttft if r["num_tokens"] == 512}
        neg_in_cutoff = sum(1 for c in neg if c["request_id"] in cutoff_ids)
        print(f"  neg chunks in cutoff reqs: {neg_in_cutoff}/{len(neg)} ({neg_in_cutoff/len(neg)*100:.0f}%)")

    # gen_time of neg chunks vs all
    if neg and any(c.get("gen_time") for c in chunks):
        all_gen = sorted(c["gen_time"] for c in chunks if c.get("gen_time", 0) > 0)
        neg_gen = sorted(c["gen_time"] for c in neg if c.get("gen_time", 0) > 0)
        if all_gen and neg_gen:
            print(f"gen_time all p50={all_gen[len(all_gen)//2]:.3f}s p95={all_gen[int(0.95*len(all_gen))]:.3f}s")
            print(f"gen_time neg p50={neg_gen[len(neg_gen)//2]:.3f}s p95={neg_gen[int(0.95*len(neg_gen))]:.3f}s")

    # word_count of neg chunks vs all
    if neg:
        all_wc = sorted(c["word_count"] for c in chunks)
        neg_wc = sorted(c["word_count"] for c in neg)
        print(f"word_count all p50={all_wc[len(all_wc)//2]} p95={all_wc[int(0.95*len(all_wc))]}")
        print(f"word_count neg p50={neg_wc[len(neg_wc)//2]} p95={neg_wc[int(0.95*len(neg_wc))]}")


analyze("BASELINE", base_chunks, base_ttft)
analyze("SSLO (p99)", sslo_chunks, sslo_ttft)

print(f"\n========== OVERLAP ==========")
base_neg_ids = set(c["request_id"] for c in base_chunks if c["cumulative_slack"] < 0)
sslo_neg_ids = set(c["request_id"] for c in sslo_chunks if c["cumulative_slack"] < 0)
print(f"baseline neg-affected: {len(base_neg_ids)}")
print(f"sslo     neg-affected: {len(sslo_neg_ids)}")
print(f"intersection (same req has neg in both): {len(base_neg_ids & sslo_neg_ids)}")
only_base = base_neg_ids - sslo_neg_ids
only_sslo = sslo_neg_ids - base_neg_ids
print(f"only baseline (sslo fixed it): {len(only_base)}")
print(f"only sslo (sslo broke it):    {len(only_sslo)}")

print(f"\n========== SSLO PENDING CORRELATION ==========")
sslo_neg = [c for c in sslo_chunks if c["cumulative_slack"] < 0]
neg_with_pending = sum(1 for c in sslo_neg if c.get("pending_time", 0) > 0)
print(f"neg chunks WITH pending_time > 0: {neg_with_pending}/{len(sslo_neg)} ({neg_with_pending/len(sslo_neg)*100:.0f}%)")
total_pending = sum(c.get("pending_time", 0) for c in sslo_neg)
print(f"Sum of pending_time across neg chunks: {total_pending:.2f}s")
total_pending_all = sum(c.get("pending_time", 0) for c in sslo_chunks)
print(f"Sum of pending_time across ALL chunks: {total_pending_all:.2f}s")
print(f"  → fraction of total pending in neg chunks: {total_pending/total_pending_all*100:.1f}%")

# How does cumulative_slack of FIRST chunks compare?
print(f"\n========== EARLY-CHUNK SLACK ==========")
for label, chunks in [("BASELINE", base_chunks), ("SSLO", sslo_chunks)]:
    by_idx = {}
    for c in chunks:
        by_idx.setdefault(c["chunk_idx"], []).append(c["cumulative_slack"])
    print(f"{label}:")
    for i in [0, 1, 2, 3, 5, 10, 20]:
        if i in by_idx:
            vals = sorted(by_idx[i])
            print(f"  chunk_idx={i:3d}: n={len(vals):4d}  p50={vals[len(vals)//2]:7.2f}s  min={vals[0]:7.2f}s  neg_count={sum(1 for v in vals if v<0)}")
