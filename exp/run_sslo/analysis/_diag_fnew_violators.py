"""Inspect violating chunks/requests in F-new for 9B/cap=256/r=12 cells.
Focus: 9B/read r=12 (regression) vs 9B/Supertone r=12 (improvement)."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/workspace/mlsys/exp/run_sslo/output_smoke/sentence/Qwen3.5-9B")
CELLS = [
    # Worst-viol cells in F-new
    ("read r=12       ", BASE / "read/none/cap256/sslo_mlp/run_1/rate_12"),
    ("Kokoro r=16     ", BASE / "tts/hexgrad__Kokoro-82M/cap256/sslo_mlp/run_1/rate_16"),
    ("Kokoro r=20     ", BASE / "tts/hexgrad__Kokoro-82M/cap256/sslo_mlp/run_1/rate_20"),
    ("Supertone r=16  ", BASE / "tts/Supertone__supertonic-3/cap256/sslo_mlp/run_1/rate_16"),
    ("Supertone r=24  ", BASE / "tts/Supertone__supertonic-3/cap256/sslo_mlp/run_1/rate_24"),
]


def pct(xs, p):
    if not xs: return None
    s = sorted(xs); n = len(s)
    k = max(0, min(n - 1, int(p * (n - 1))))
    return s[k]


for label, d in CELLS:
    print(f"\n{'='*100}\n=== {label}  cap=256 r=12 ===")
    cp = d / "chunks.jsonl"
    rp = d / "requests.jsonl"

    # Per-request: max stall, word_counts of chunks, num_tokens, num_chunks
    per_req = defaultdict(lambda: {"max_stall": 0.0, "chunks": []})
    with cp.open() as f:
        for line in f:
            r = json.loads(line)
            rid = r["request_id"]
            miss = float(r.get("unit_deadline_miss_s") or 0)
            per_req[rid]["chunks"].append(r)
            if miss > per_req[rid]["max_stall"]:
                per_req[rid]["max_stall"] = miss

    # Get prompts
    prompts = {}
    output_tokens = {}
    with rp.open() as f:
        for line in f:
            req = json.loads(line)
            rid = req.get("request_id")
            prompts[rid] = req.get("prompt") or ""
            output_tokens[rid] = req.get("num_output_tokens") or 0

    # Cell-wide stats
    stalls = [v["max_stall"] for v in per_req.values()]
    n = len(stalls)
    n_v1 = sum(1 for s in stalls if s > 1.0)
    print(f"  reqs={n}  viol@1={n_v1}/{n}={n_v1/n:.3%}  "
          f"max_stall p95={pct(stalls,0.95):.2f}  p99={pct(stalls,0.99):.2f}  "
          f"max={max(stalls):.2f}s")

    n_tok_all = [output_tokens[rid] for rid in per_req]
    print(f"  output_tokens (all reqs): mean={sum(n_tok_all)/n:.0f}  "
          f"p95={pct(n_tok_all,0.95):.0f}  max={max(n_tok_all)}")

    # Violators
    bad = [(rid, v) for rid, v in per_req.items() if v["max_stall"] > 1.0]
    if not bad:
        print("  (no violators)")
        continue
    bad.sort(key=lambda kv: -kv[1]["max_stall"])

    bad_out = [output_tokens[rid] for rid, _ in bad]
    print(f"  violators ({len(bad)}): output_tokens mean={sum(bad_out)/len(bad_out):.0f}  "
          f"max={max(bad_out)}")

    # Top 3 violators: show prompt + worst-stall chunk
    print(f"\n  --- top 3 violators ---")
    for rid, info in bad[:3]:
        chunks = sorted(info["chunks"], key=lambda c: c["unit_index"])
        # find worst stall chunk
        worst = max(chunks, key=lambda c: c.get("unit_deadline_miss_s") or 0)
        prompt = prompts.get(rid, "").replace("\n", " ")[:200]
        print(f"  rid={rid}  max_stall={info['max_stall']:.2f}s  "
              f"n_chunks={len(chunks)}  out_tok={output_tokens[rid]}")
        print(f"    PROMPT: {prompt[:180]}")
        print(f"    WORST CHUNK idx={worst['unit_index']} wc={worst.get('word_count')} "
              f"tok={worst.get('num_token')} cons_dur={worst.get('consume_duration'):.2f}s "
              f"miss={worst.get('unit_deadline_miss_s'):.2f}s")
        print(f"      text: {(worst.get('text','') or '')[:140].replace(chr(10),' ')}")
