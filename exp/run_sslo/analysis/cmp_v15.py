"""DEPRECATED (2026-05): this script uses v15-era column names that no longer
exist after the R3 metric refactor. Do not run on current sweep output.
Kept only as historical reference."""
import csv
import statistics
from pathlib import Path
from collections import defaultdict


CSV = Path("/workspace/mlsys/exp/run_sslo/output/v15_grid/Qwen3.5-9B/summary.csv")


def main():
    rows = list(csv.DictReader(CSV.open()))
    print(f"loaded {len(rows)} rows from {CSV.name}")

    metric_cols = [
        "tokens_per_second", "urgent_mode_fraction_pct",
        "mean_running", "mean_pending", "mean_waiting", "mean_handling_users",
        "queue_stall_p95_s", "queue_stall_p99_s", "queue_stall_max_s",
        "ttfc_p95_s", "completion_latency_p95_s",
        "max_stall_interval_s_p99",
        "chunk_slo_violation_rate_tau_1s",
        "request_slo_violation_rate_tau_1s",
    ]

    # Group: (cap, rate, policy) -> list of dicts (one per repeat)
    grouped = defaultdict(list)
    for r in rows:
        key = (int(r["max_num_seqs"]), float(r["lambda_req_s"]), r["policy"])
        grouped[key].append(r)

    def avg(rs, k):
        vals = []
        for r in rs:
            try:
                vals.append(float(r[k]))
            except (KeyError, ValueError):
                pass
        return statistics.mean(vals) if vals else None

    # Print compact table: per cap, all rates side-by-side baseline | sslo_mlp
    for cap in (16, 32, 64, 128):
        print(f"\n{'='*145}")
        print(f"=== cap = {cap} (mean over 3 repeats) ===")
        print(f"{'='*145}")
        header = (
            f"{'rate':>5} | {'mode':<10} | "
            f"{'tps':>7} | {'crit%':>6} | "
            f"{'run':>5} {'pend':>5} {'wait':>6} {'adm':>5} | "
            f"{'q_p95':>6} {'q_p99':>6} | "
            f"{'ttfc_p95':>8} {'compL_p95':>9} | "
            f"{'stall_p99':>9} | "
            f"{'chunk_v@1':>9} {'req_v@1':>8}"
        )
        print(header)
        for rate in (0.5, 1, 2, 4, 8, 12, 16, 20):
            for mode in ("baseline", "sslo_mlp"):
                key = (cap, rate, mode)
                rs = grouped.get(key, [])
                if not rs:
                    print(f"{rate:>5} | {mode:<10} | (missing)")
                    continue
                vals = {k: avg(rs, k) for k in metric_cols}
                def f(k, ndig=2):
                    v = vals[k]
                    return f"{v:.{ndig}f}" if v is not None else "-"
                print(
                    f"{rate:>5} | {mode:<10} | "
                    f"{f('tokens_per_second', 0):>7} | {f('urgent_mode_fraction_pct'):>6} | "
                    f"{f('mean_running'):>5} {f('mean_pending'):>5} {f('mean_waiting'):>6} {f('mean_handling_users'):>5} | "
                    f"{f('queue_stall_p95_s'):>6} {f('queue_stall_p99_s'):>6} | "
                    f"{f('ttfc_p95_s'):>8} {f('completion_latency_p95_s'):>9} | "
                    f"{f('max_stall_interval_s_p99'):>9} | "
                    f"{f('chunk_slo_violation_rate_tau_1s', 4):>9} {f('request_slo_violation_rate_tau_1s', 4):>8}"
                )


if __name__ == "__main__":
    raise SystemExit("cmp_v15.py is deprecated; see top-of-file docstring.")
