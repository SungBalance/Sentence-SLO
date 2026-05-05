#!/usr/bin/env python3
"""Print a compact sweep progress report from partial summary.json files."""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

BASE = "exp/sslo_test/output_sweep"
TOTAL_CELLS = 2 * 3 * 4 * 3  # chunk_units * seqs * rates * runs


def main() -> None:
    summaries = sorted(glob(f"{BASE}/*/seqs_*/rate_*/run_*/summary.json"))
    print(f"Progress: {len(summaries)}/{TOTAL_CELLS} cells completed")
    if not summaries:
        return

    agg: dict[str, list[tuple]] = {}
    for path in summaries:
        s = json.loads(Path(path).read_text())
        ttft = s.get("metrics", {}).get("ttft", {})
        slack = s.get("metrics", {}).get("slack", {})
        comp = s.get("metrics", {}).get("slo_compliance", {})
        for mode in ttft:
            post = ttft[mode].get("post_cap", {})
            sl = slack.get(mode, {})
            cm = comp.get(mode, {})
            rate = cm.get("rate")
            agg.setdefault(mode, []).append((
                post.get("p50"),
                post.get("p99"),
                sl.get("neg_ratio"),
                (1 - rate) if rate is not None else None,
            ))

    def avg(rows, idx):
        vals = [r[idx] for r in rows if r[idx] is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v, w, prec):
        return f"{v:>{w}.{prec}f}" if v is not None else " " * w

    print()
    print(f"{'mode':<22s} {'n':>3s} {'ttft_p50(s)':>11s} "
          f"{'ttft_p99(s)':>11s} {'chunk_viol':>11s} {'req_viol':>10s}")
    print("-" * 72)
    for mode, rows in sorted(agg.items()):
        print(
            f"{mode:<22s} {len(rows):>3d} "
            f"{fmt(avg(rows, 0), 11, 3)} "
            f"{fmt(avg(rows, 1), 11, 3)} "
            f"{fmt(avg(rows, 2), 11, 4)} "
            f"{fmt(avg(rows, 3), 10, 4)}"
        )


if __name__ == "__main__":
    main()
