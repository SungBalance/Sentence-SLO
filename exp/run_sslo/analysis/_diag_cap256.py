"""Find cap=256 cells where sslo_mlp > baseline in viol@1, and diagnose."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROWS = list(csv.DictReader(
    open("/workspace/mlsys/exp/run_sslo/output_sweep/summary.csv")))

# Index per (model, consume, tts, cap, rate, policy)
data = {}
for r in ROWS:
    consume = r["consume_mode"]; tts = r["consume_model"] or "-"
    key = (r["model"], consume, tts, int(r["max_num_seqs"]),
           float(r["request_rate"]), r["policy"])
    data[key] = r

def num(v):
    try: return float(v)
    except: return None

# Find cap=256 cells where sslo viol > baseline viol
print(f"=== cap=256 cells where sslo_mlp viol > baseline viol ===")
print(f"{'model':<20s} {'consume':>6s} {'tts':>22s} {'rate':>5s} "
      f"{'b_viol':>7s} {'s_viol':>7s} {'b_HU':>5s} {'s_HU':>5s} "
      f"{'b_TTFC':>7s} {'s_TTFC':>7s}")
regressions = []
for (model, consume, tts, cap, rate, _), b in [(k, v) for k, v in data.items()
                                                if k[3] == 256 and k[5] == "baseline"]:
    s = data.get((model, consume, tts, 256, rate, "sslo_mlp"))
    if not s: continue
    b_viol = num(b.get("request_cu_slo_violation_rate_tau_1")) or 0
    s_viol = num(s.get("request_cu_slo_violation_rate_tau_1")) or 0
    if s_viol > b_viol + 0.001:
        regressions.append((model, consume, tts, rate, b_viol, s_viol, b, s))

regressions.sort(key=lambda x: -(x[5] - x[4]))
for model, consume, tts, rate, bv, sv, b, s in regressions:
    print(f"{model:<20s} {consume:>6s} {tts:>22s} r={rate:<3.0f} "
          f"{bv:>7.3f} {sv:>7.3f} "
          f"{num(b.get('mean_handling_users')):>5.0f} {num(s.get('mean_handling_users')):>5.0f} "
          f"{num(b.get('ttfc_p95_s')):>7.2f} {num(s.get('ttfc_p95_s')):>7.2f}")
print()
print(f"total regressions: {len(regressions)}")

# Pick the WORST case and dive deeper
if regressions:
    worst = regressions[0]
    model, consume, tts, rate, _, _, _, _ = worst
    print(f"\n=== deepest dive: {model} / {consume}/{tts} / cap=256 / r={rate} ===")
    # Open chunks.jsonl for sslo_mlp cell
    tts_slug = "none" if consume == "read" else tts.replace("/", "__")
    base = Path("/workspace/mlsys/exp/run_sslo/output_sweep/sentence")
    sslo_dir = (base / model / consume / tts_slug / "cap256" / "sslo_mlp"
                / "run_1" / f"rate_{rate:g}")
    base_dir = (base / model / consume / tts_slug / "cap256" / "baseline"
                / "run_1" / f"rate_{rate:g}")
    for label, d in [("sslo_mlp", sslo_dir), ("baseline", base_dir)]:
        cp = d / "chunks.jsonl"
        if not cp.exists():
            print(f"  {label}: chunks.jsonl missing"); continue
        # Per-request max stall
        per_req = defaultdict(float)
        wc_max = defaultdict(int)
        with cp.open() as f:
            for line in f:
                row = json.loads(line)
                rid = row.get("request_id")
                miss = float(row.get("unit_deadline_miss_s") or 0)
                if rid is None: continue
                if miss > per_req[rid]: per_req[rid] = miss
                wc = int(row.get("word_count") or 0)
                if wc > wc_max[rid]: wc_max[rid] = wc
        stalls = list(per_req.values())
        n = len(stalls)
        n_viol = sum(1 for s_ in stalls if s_ > 1.0)
        if not stalls: continue
        s_sorted = sorted(stalls)
        p50, p95, p99, mx = (s_sorted[int(0.50*(n-1))], s_sorted[int(0.95*(n-1))],
                             s_sorted[int(0.99*(n-1))], max(stalls))
        print(f"  {label}: reqs={n}  viol={n_viol}/{n}={n_viol/n:.3%}  "
              f"max_stall p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} max={mx:.2f}s")
        # worst 5 violators with their wc
        worst5 = sorted(per_req.items(), key=lambda kv: -kv[1])[:5]
        for rid, ms in worst5:
            print(f"    req={rid} max_stall={ms:.2f}s, max_word_count={wc_max[rid]}")
