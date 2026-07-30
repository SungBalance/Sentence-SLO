"""Why does SSLO improve read but regress TTS vs baseline?
Compare violator distribution + chunk patterns for read vs TTS at same rate."""
import json
from collections import defaultdict
from pathlib import Path

SMOKE = Path("/workspace/mlsys/exp/run_sslo/output_smoke/sentence/Qwen3.5-9B")
PREFULL_BASE = Path("/workspace/mlsys/exp/run_sslo/output_sweep_prefull/sentence/Qwen3.5-9B")

# read r=20 (SSLO wins): SSLO vs ... no baseline in smoke. Use prefull baseline if exists.
CELLS = [
    ("read SSLO r=16", SMOKE / "read/none/cap256/sslo_mlp/run_1/rate_16", "sslo"),
    ("Kokoro SSLO r=16", SMOKE / "tts/hexgrad__Kokoro-82M/cap256/sslo_mlp/run_1/rate_16", "sslo"),
    ("Supertone SSLO r=16", SMOKE / "tts/Supertone__supertonic-3/cap256/sslo_mlp/run_1/rate_16", "sslo"),
]


def pct(xs, p):
    if not xs: return None
    s = sorted(xs); n = len(s)
    return s[max(0, min(n-1, int(p*(n-1))))]


for label, d, _ in CELLS:
    cp = d / "chunks.jsonl"
    rp = d / "requests.jsonl"
    if not cp.exists():
        print(f"-- {label}: missing"); continue
    per_req_max_stall = defaultdict(float)
    per_req_chunks = defaultdict(list)
    per_req_tok = defaultdict(int)
    per_req_wc = defaultdict(int)
    with cp.open() as f:
        for line in f:
            r = json.loads(line)
            rid = r['request_id']
            miss = float(r.get('unit_deadline_miss_s') or 0)
            if miss > per_req_max_stall[rid]:
                per_req_max_stall[rid] = miss
            per_req_chunks[rid].append(r)
            per_req_tok[rid] += int(r.get('num_token') or 0)
            per_req_wc[rid] += int(r.get('word_count') or 0)

    stalls = list(per_req_max_stall.values())
    n = len(stalls)
    n_v1 = sum(1 for s in stalls if s > 1.0)
    n_v05 = sum(1 for s in stalls if s > 0.5)
    n_v2 = sum(1 for s in stalls if s > 2.0)
    n_v5 = sum(1 for s in stalls if s > 5.0)

    print(f"\n=== {label} ===")
    print(f"  reqs={n}  viol@0.5={n_v05}/{n}={n_v05/n:.2%}  "
          f"viol@1={n_v1}/{n}={n_v1/n:.2%}  viol@2={n_v2}/{n}={n_v2/n:.2%}  "
          f"viol@5={n_v5}/{n}")
    print(f"  max_stall p50={pct(stalls,0.5):.3f}  p90={pct(stalls,0.90):.3f}  "
          f"p95={pct(stalls,0.95):.3f}  p99={pct(stalls,0.99):.3f}  max={max(stalls):.2f}s")

    # Violators (max_stall > 1s) stats
    bad = [(rid, ms) for rid, ms in per_req_max_stall.items() if ms > 1.0]
    if bad:
        bad_tokens = [per_req_tok[rid] for rid, _ in bad]
        bad_wc = [per_req_wc[rid] for rid, _ in bad]
        bad_nchunks = [len(per_req_chunks[rid]) for rid, _ in bad]
        all_tok = [per_req_tok[rid] for rid in per_req_max_stall]
        all_wc = [per_req_wc[rid] for rid in per_req_max_stall]
        print(f"  violators: out_tokens mean={sum(bad_tokens)/len(bad_tokens):.0f} "
              f"(all {sum(all_tok)/len(all_tok):.0f})  "
              f"wc mean={sum(bad_wc)/len(bad_wc):.0f} "
              f"(all {sum(all_wc)/len(all_wc):.0f})  "
              f"n_chunks mean={sum(bad_nchunks)/len(bad_nchunks):.0f}")

        # WORST violating chunk: tok/wc ratio
        worst = max(bad, key=lambda kv: kv[1])
        rid, ms = worst
        chunks = sorted(per_req_chunks[rid], key=lambda c: c.get('unit_index') or 0)
        wc = max(chunks, key=lambda c: c.get('unit_deadline_miss_s', 0) or 0)
        wtok = int(wc.get('num_token') or 0)
        wwc = int(wc.get('word_count') or 0)
        cdur = float(wc.get('consume_duration') or 0)
        print(f"  WORST chunk in worst-stall req (rid={rid}, stall={ms:.2f}s):")
        print(f"    idx={wc.get('unit_index')} wc={wwc} tok={wtok} tok/wc={wtok/max(1,wwc):.2f} "
              f"consume_dur={cdur:.2f}s")
