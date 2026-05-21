"""Per-cell deep dive: TTFC distribution + num_admitted breakdown."""
import json
import statistics
from pathlib import Path

CELLS = [
    ("cap64_r2",   "/workspace/mlsys/exp/run_sslo/output/v7_9b_grid/cap64_r2",   64),
    ("cap64_r16",  "/workspace/mlsys/exp/run_sslo/output/v7_9b_grid/cap64_r16",  64),
    ("cap256_r2",  "/workspace/mlsys/exp/run_sslo/output/v7_9b_grid/cap256_r2",  256),
    ("cap256_r16", "/workspace/mlsys/exp/run_sslo/output/v7_9b_grid/cap256_r16", 256),
]
MODES = [
    ("baseline",     "baseline",          "baseline"),
    ("abatch=T",     "sslo_mlp",          "sslo_mlp"),
    ("abatch=F",     "sslo_mlp_noabatch_s/sslo_mlp", "sslo_mlp"),
]

def pct(arr, q):
    return sorted(arr)[min(len(arr)-1, int(q*len(arr)))] if arr else 0

print(f"{'cell':<12} {'mode':<10} | {'win':>5} {'ttfc_m':>7} {'ttfc_p50':>9} {'ttfc_p95':>9} {'ttfc_p99':>9} {'ttfc_max':>9} | {'q_m':>5} {'q_max':>6} | {'tps':>6} {'run_m':>6}")
print('-' * 124)
for cell, root_str, cap in CELLS:
    root = Path(root_str)
    for label, sub, mkey in MODES:
        rp = root / sub / "run_1" / "requests.jsonl"
        sp = root / sub / "run_1" / "scheduler_stats.jsonl"
        mp = root / sub / "run_1" / "run_meta.json"
        if not rp.exists():
            print(f"{cell:<12} {label:<10} | MISSING")
            continue
        meta = json.loads(mp.read_text())
        mw0 = meta["measurement_window_start_ts"]; mw1 = meta["measurement_window_end_ts"]
        ttfcs = []
        with rp.open() as f:
            for line in f:
                r = json.loads(line)
                inj = r.get("injection_ts")
                if inj is None or not (mw0 <= inj < mw1): continue
                t = r.get("ttfc")
                if t is not None: ttfcs.append(t)
        # scheduler waiting / running
        runs = []; waits = []; n_total = 0
        with sp.open() as f:
            for line in f:
                rr = json.loads(line)
                if rr.get("kind") != "step": continue
                n_total += 1
                runs.append(rr.get("running",0) or 0)
                waits.append(rr.get("waiting",0) or 0)
        # throughput from summary
        sumr = Path(rp.parent) / "summary.json"
        tps = 0
        if sumr.exists():
            sm = json.loads(sumr.read_text())
            tp = sm.get("metrics", {}).get("throughput", {}).get(mkey, {})
            tps = tp.get("tokens_per_second", 0) or 0
        if not ttfcs:
            print(f"{cell:<12} {label:<10} | {0:>5}"); continue
        st = sorted(ttfcs)
        print(f"{cell:<12} {label:<10} | {len(ttfcs):>5} "
              f"{statistics.mean(ttfcs):>7.2f} {pct(st,0.5):>9.2f} {pct(st,0.95):>9.2f} {pct(st,0.99):>9.2f} {max(ttfcs):>9.2f} | "
              f"{statistics.mean(waits):>5.0f} {max(waits):>6.0f} | "
              f"{tps:>6.0f} {statistics.mean(runs):>6.1f}")
