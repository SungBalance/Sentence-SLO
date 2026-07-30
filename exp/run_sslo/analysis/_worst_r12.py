"""Find worst-stall request at r=12 for each 9B cap=256 cell, dump prompt + chunks."""
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/workspace/mlsys/exp/run_sslo/output_smoke/sentence/Qwen3.5-9B")
CELLS = [
    ("read",       BASE / "read/none/cap256/sslo_mlp/run_1/rate_12"),
    ("Kokoro",     BASE / "tts/hexgrad__Kokoro-82M/cap256/sslo_mlp/run_1/rate_12"),
    ("Supertone",  BASE / "tts/Supertone__supertonic-3/cap256/sslo_mlp/run_1/rate_12"),
]

for label, d in CELLS:
    print(f"\n{'='*80}\n=== {label} ===")
    cp = d / "chunks.jsonl"
    rp = d / "requests.jsonl"
    # Per-request max stall
    per_req = defaultdict(lambda: {"max_stall": 0.0, "chunks": []})
    with cp.open() as f:
        for line in f:
            r = json.loads(line)
            rid = r["request_id"]
            miss = float(r.get("unit_deadline_miss_s") or 0)
            per_req[rid]["chunks"].append(r)
            if miss > per_req[rid]["max_stall"]:
                per_req[rid]["max_stall"] = miss
    # Top 3 violators
    top = sorted(per_req.items(), key=lambda kv: -kv[1]["max_stall"])[:3]
    # Get prompt text from requests.jsonl
    prompts = {}
    if rp.exists():
        with rp.open() as f:
            for line in f:
                req = json.loads(line)
                rid = req.get("request_id")
                prompts[rid] = req.get("prompt") or req.get("text") or ""
    for rid, info in top:
        chunks = sorted(info["chunks"], key=lambda c: c["unit_index"])
        prompt = prompts.get(rid, "(prompt not stored)")
        print(f"\n--- req={rid}  max_stall={info['max_stall']:.2f}s  n_chunks={len(chunks)}")
        print(f"  PROMPT ({len(prompt)} chars): {prompt[:300]}{'...' if len(prompt)>300 else ''}")
        print(f"  CHUNK TIMELINE:")
        for c in chunks:
            miss = c.get("unit_deadline_miss_s", 0)
            mark = " ★" if miss > 1.0 else "  "
            text = (c.get("text") or "").replace("\n", " ").strip()
            print(f"   {mark} idx={c['unit_index']:2d} wc={c.get('word_count'):3d} "
                  f"tok={c.get('num_token'):3d} cons={c.get('consume_duration'):.2f}s "
                  f"miss={miss:.2f}s  text={text[:120]}{'...' if len(text)>120 else ''}")
