#!/usr/bin/env python3
"""TTFT p50 trends across (rate, seqs) for SSLO and SSLO+Adaptive.

For each mode (sslo, sslo_adaptive):
  (a) TTFT vs request_rate, one curve per max_num_seqs.
  (b) TTFT vs max_num_seqs, one curve per request_rate.

Also reports baseline as reference, plus the *speedup* SSLO (and adaptive)
gives over baseline per cell so the operating regime where SSLO matters is
visible.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


MODES = ("baseline", "sslo", "sslo_adaptive")
RATES = (0, 4, 8, 16)
SEQS = (32, 64, 128)
NUM_RUNS_DEFAULT = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-output", default="exp/sslo_test/output_sweep")
    p.add_argument("--num-runs", type=int, default=NUM_RUNS_DEFAULT)
    return p.parse_args()


def load_ttft(base: Path, rate: int, seqs: int, num_runs: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {m: [] for m in MODES}
    for i in range(1, num_runs + 1):
        path = base / f"rate_{rate}_seqs_{seqs}" / f"run_{i}" / f"seqs_{seqs}" / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for m in MODES:
            v = s.get(f"h2_ttft_p50_{m}")
            if v is not None:
                out[m].append(float(v))
    return out


def stat(vs: list[float]) -> tuple[float | None, float]:
    if not vs:
        return None, 0.0
    return st.mean(vs), (st.stdev(vs) if len(vs) > 1 else 0.0)


def fmt(v: float | None, sigma: float = 0.0, fmt_s: str = "{:6.2f}") -> str:
    if v is None:
        return "   n/a   "
    if sigma > 0:
        return f"{fmt_s.format(v)}±{sigma:.2f}"
    return fmt_s.format(v)


def main() -> None:
    args = parse_args()
    base = Path(args.base_output)
    cells = {
        (r, s): load_ttft(base, r, s, args.num_runs)
        for r in RATES
        for s in SEQS
    }

    # ===== (a) TTFT vs rate, per seqs =====
    print("\n" + "=" * 72)
    print("(a) TTFT p50 vs request_rate, per max_num_seqs")
    print("=" * 72)
    for mode in MODES:
        print(f"\n--- {mode} ---")
        header = f"{'seqs':<6}" + "".join(f"  rate={r:<3d}{'':5s}" for r in RATES)
        print(header)
        for s in SEQS:
            row = f"{s:<6}"
            for r in RATES:
                m, sd = stat(cells[(r, s)][mode])
                row += "  " + fmt(m, sd) + " "
            print(row)

    # ===== (b) TTFT vs seqs, per rate =====
    print("\n" + "=" * 72)
    print("(b) TTFT p50 vs max_num_seqs, per request_rate")
    print("=" * 72)
    for mode in MODES:
        print(f"\n--- {mode} ---")
        header = f"{'rate':<6}" + "".join(f"  seqs={s:<3d}{'':5s}" for s in SEQS)
        print(header)
        for r in RATES:
            row = f"{r:<6}"
            for s in SEQS:
                m, sd = stat(cells[(r, s)][mode])
                row += "  " + fmt(m, sd) + " "
            print(row)

    # ===== (c) SSLO / adaptive speedup over baseline =====
    print("\n" + "=" * 72)
    print("(c) Speedup over baseline (TTFT_baseline / TTFT_mode)")
    print("    >1.0 = SSLO is faster.  '∞' = baseline >> ~0.")
    print("=" * 72)
    for mode in ("sslo", "sslo_adaptive"):
        print(f"\n--- {mode} vs baseline ---")
        header = f"{'seqs':<6}" + "".join(f"  rate={r:<3d}{'':5s}" for r in RATES)
        print(header)
        for s in SEQS:
            row = f"{s:<6}"
            for r in RATES:
                b, _ = stat(cells[(r, s)]["baseline"])
                v, _ = stat(cells[(r, s)][mode])
                if b is None or v is None:
                    cell = "  n/a   "
                elif v < 0.001:
                    cell = " >999×  "  # avoid div-by-zero
                else:
                    cell = f"{b / v:6.1f}×"
                row += "  " + cell + " "
            print(row)

    # ===== (d) sslo_adaptive vs sslo: where does adaptive help/hurt? =====
    print("\n" + "=" * 72)
    print("(d) Adaptive vs plain SSLO (TTFT_sslo / TTFT_adaptive)")
    print("    >1.0 = adaptive faster.  <1.0 = adaptive slower.")
    print("=" * 72)
    header = f"{'seqs':<6}" + "".join(f"  rate={r:<3d}{'':5s}" for r in RATES)
    print(header)
    for s in SEQS:
        row = f"{s:<6}"
        for r in RATES:
            ss, _ = stat(cells[(r, s)]["sslo"])
            ad, _ = stat(cells[(r, s)]["sslo_adaptive"])
            if ss is None or ad is None or ad < 0.001:
                cell = "  n/a   "
            else:
                ratio = ss / ad
                cell = f"{ratio:6.2f}×"
            row += "  " + cell + " "
        print(row)


if __name__ == "__main__":
    main()
