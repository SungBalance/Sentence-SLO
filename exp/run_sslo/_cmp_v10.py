"""v10 grid: 9B + 35B-A3B × cap × rate × {baseline, sslo_mlp}.

For each cell prints all 6 priority metrics:
  1. adm_m (= run_m + pend_m) — admitted pool size
  2. viol@τ (0.5, 1, 2)
  3. ttfc (mean, p99)
  4. tokens/sec (throughput), in_window count
  5. out_tokens per req (window: mean, p99)
  6. chunk_tokens per chunk (window: mean, p99)

Also writes summary.csv next to the grid root for spreadsheet use.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/mlsys/exp/run_sslo/output/v10_grid")

def load_cell(model_dir, cap, rate, mode):
    if mode == "baseline":
        p = ROOT / model_dir / f"cap{cap}_r{rate}" / mode / "run_1"
    else:
        p = ROOT / model_dir / f"cap{cap}_r{rate}" / mode / "run_1"
    sp = p / "summary.json"
    if not sp.exists(): return None
    s = json.loads(sp.read_text())
    m = s.get("metrics", {})
    mkey = mode  # "baseline" or "sslo_mlp"
    cps = m.get("cpslo", {}).get(mkey, {})
    viol = cps.get("cp_slo_violation_rates_by_tau", {})
    def vr(t):
        vv = viol.get(t, {})
        return (vv.get("rate", 0) or 0) * 100
    ms = cps.get("max_stall_interval_distribution", {})
    ttfc = m.get("ttfc", {}).get(mkey, {})
    tp = m.get("throughput", {}).get(mkey, {})
    # scheduler stats
    runs = []; pends = []; n_total = n_crit = 0
    with (p / "scheduler_stats.jsonl").open() as f:
        for line in f:
            r2 = json.loads(line)
            if r2.get("kind") != "step": continue
            n_total += 1
            if r2.get("has_critical"): n_crit += 1
            runs.append(r2.get("running", 0) or 0)
            pends.append(r2.get("pending", 0) or 0)
    # window bounds
    meta = json.loads((p / "run_meta.json").read_text())
    mw0 = meta.get("measurement_window_start_ts"); mw1 = meta.get("measurement_window_end_ts")
    # per-req out tokens (window)
    out_tokens_in = []
    req_ids_in = set()
    with (p / "requests.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            inj = r.get("injection_ts")
            if inj is None or not (mw0 <= inj < mw1): continue
            n_out = r.get("num_output_tokens") or 0
            out_tokens_in.append(n_out)
            req_ids_in.add(str(r.get("request_id")))
    # chunk tokens (window reqs)
    chunk_tokens = []
    with (p / "chunks.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            if str(c.get("request_id")) not in req_ids_in: continue
            nt = c.get("num_token") or c.get("num_tokens") or 0
            chunk_tokens.append(nt)
    def pct(arr, q):
        if not arr: return 0
        s = sorted(arr); return s[min(len(s) - 1, int(q * len(s)))]
    return {
        "win": s.get("in_window_count", 0),
        "v05": vr("tau_0.5"), "v1": vr("tau_1"), "v2": vr("tau_2"),
        "p99st": ms.get("p99", 0) or 0, "mxst": ms.get("max", 0) or 0,
        "ttfc_m": ttfc.get("mean", 0) or 0,
        "ttfc99": ttfc.get("p99", 0) or 0,
        "tps": tp.get("tokens_per_second", 0) or 0,
        "crit": n_crit / max(1, n_total) * 100,
        "run_m": statistics.mean(runs) if runs else 0,
        "pend_m": statistics.mean(pends) if pends else 0,
        "out_m": statistics.mean(out_tokens_in) if out_tokens_in else 0,
        "out99": pct(out_tokens_in, 0.99),
        "ch_m": statistics.mean(chunk_tokens) if chunk_tokens else 0,
        "ch99": pct(chunk_tokens, 0.99),
    }


CSV_ROWS: list[dict] = []


def print_grid(model_dir, caps, rates):
    print(f"\n========== {model_dir} ==========")
    print(f"{'cap':>4} {'rate':>4} {'mode':<10} | "
          f"{'adm':>6} {'v05':>5} {'v1':>5} {'v2':>5} "
          f"{'p99st':>6} {'mxst':>6} {'ttfc_m':>7} {'ttfc99':>7} "
          f"{'tps':>6} {'win':>4} {'crit%':>5} "
          f"{'out_m':>6} {'out99':>6} {'ch_m':>5} {'ch99':>5}")
    print('-' * 152)
    for cap in caps:
        for rate in rates:
            for mode in ["baseline", "sslo_mlp"]:
                d = load_cell(model_dir, cap, rate, mode)
                if d is None:
                    print(f"{cap:>4} {rate:>4} {mode:<10} | MISSING")
                    continue
                adm = d["run_m"] + d["pend_m"]
                print(f"{cap:>4} {rate:>4} {mode:<10} | "
                      f"{adm:>6.1f} {d['v05']:>4.2f}% {d['v1']:>4.2f}% {d['v2']:>4.2f}% "
                      f"{d['p99st']:>6.2f} {d['mxst']:>6.2f} "
                      f"{d['ttfc_m']:>7.2f} {d['ttfc99']:>7.2f} "
                      f"{d['tps']:>6.0f} {d['win']:>4} {d['crit']:>4.0f}% "
                      f"{d['out_m']:>6.0f} {d['out99']:>6.0f} "
                      f"{d['ch_m']:>5.1f} {d['ch99']:>5.0f}")
                CSV_ROWS.append({
                    "model": model_dir, "cap": cap, "rate": rate, "mode": mode,
                    "adm_m": round(adm, 2),
                    "run_m": round(d["run_m"], 2),
                    "pend_m": round(d["pend_m"], 2),
                    "viol_tau_0.5_pct": round(d["v05"], 4),
                    "viol_tau_1_pct":   round(d["v1"], 4),
                    "viol_tau_2_pct":   round(d["v2"], 4),
                    "max_stall_p99": round(d["p99st"], 3),
                    "max_stall_max": round(d["mxst"], 3),
                    "ttfc_mean": round(d["ttfc_m"], 2),
                    "ttfc_p99":  round(d["ttfc99"], 2),
                    "tokens_per_second": round(d["tps"], 1),
                    "in_window_count": d["win"],
                    "critical_pct": round(d["crit"], 1),
                    "out_tokens_mean": round(d["out_m"], 1),
                    "out_tokens_p99": d["out99"],
                    "chunk_tokens_mean": round(d["ch_m"], 2),
                    "chunk_tokens_p99": d["ch99"],
                })
        print()


print_grid("Qwen3.5-9B",     [16, 32, 64, 128],   [0.5, 1, 2, 4, 8, 12])
print_grid("Qwen3.5-35B-A3B", [64, 128, 256, 512], [0.5, 1, 2, 4, 8, 12])

# CSV export
csv_path = ROOT / "summary.csv"
if CSV_ROWS:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(CSV_ROWS)
    print(f"\nWrote {len(CSV_ROWS)} rows to {csv_path}", file=sys.stderr)
