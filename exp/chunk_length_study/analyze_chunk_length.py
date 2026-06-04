"""Chunk-length vs request-output-length variability analysis.

Reads each category's baseline run (chunks.jsonl + requests.jsonl), and for
per-chunk token length (num_token) vs per-request output length
(num_output_tokens) computes three scale-aware variability metrics:
  - CV        = std / mean           (relative spread)
  - p99/p50   = tail / median        (tail heaviness)
  - IQR/med   = (p75 - p25) / p50    (robust spread)
plus prompt-length correlations (Pearson + Spearman). Writes a JSON+CSV
breakdown and three plots.

Hypothesis: chunk-level variability < request-level variability ⇒ predicting
the next chunk is easier than predicting the whole request output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# Reuse run_sslo helpers (read-only import).
_RUN_SSLO = Path(__file__).resolve().parents[1] / "run_sslo"
sys.path.insert(0, str(_RUN_SSLO))
from jsonl_utils import read_jsonl  # noqa: E402
from metrics_utils import numeric_values, percentile  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def variability(values: list[float]) -> dict:
    """CV, p99/p50, IQR/median, plus the raw stats they derive from."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return {"count": n, "mean": None, "std": None, "cv": None,
                "p50": None, "p99": None, "p99_p50": None,
                "iqr_med": None}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals)
    p25 = percentile(vals, 25)
    p50 = percentile(vals, 50)
    p75 = percentile(vals, 75)
    p99 = percentile(vals, 99)
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "cv": (std / mean) if mean else None,
        "p50": p50,
        "p99": p99,
        "p99_p50": (p99 / p50) if p50 else None,
        "iqr_med": ((p75 - p25) / p50) if p50 else None,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    try:
        return statistics.correlation(xs, ys)
    except (statistics.StatisticsError, ValueError):
        return None


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None

    def rank(a: list[float]) -> list[float]:
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(rank(xs), rank(ys))


def analyze_category(run_dir: Path) -> dict | None:
    chunks = read_jsonl(run_dir / "chunks.jsonl")
    reqs = read_jsonl(run_dir / "requests.jsonl")
    if not chunks or not reqs:
        return None
    chunk_len = numeric_values(chunks, "num_token")
    req_out = numeric_values(reqs, "num_output_tokens")

    # Per-request mean chunk length, paired with prompt length (correlation).
    chunk_by_req: dict[str, list[float]] = {}
    for c in chunks:
        rid = c.get("request_id")
        nt = c.get("num_token")
        if rid is not None and nt is not None:
            chunk_by_req.setdefault(rid, []).append(float(nt))
    prompt_len, out_len, mean_chunk_len = [], [], []
    for r in reqs:
        pl = r.get("num_prompt_tokens")
        ol = r.get("num_output_tokens")
        rid = r.get("request_id")
        if pl is None or ol is None:
            continue
        prompt_len.append(float(pl))
        out_len.append(float(ol))
        cl = chunk_by_req.get(rid, [])
        mean_chunk_len.append(statistics.fmean(cl) if cl else math.nan)

    pl_ol = [(p, o) for p, o in zip(prompt_len, out_len)]
    pl_cl = [(p, c) for p, c in zip(prompt_len, mean_chunk_len)
             if not math.isnan(c)]
    return {
        "chunk_len": variability(chunk_len),
        "request_out_len": variability(req_out),
        "corr": {
            "prompt_vs_output": {
                "pearson": pearson([p for p, _ in pl_ol],
                                   [o for _, o in pl_ol]),
                "spearman": spearman([p for p, _ in pl_ol],
                                     [o for _, o in pl_ol]),
            },
            "prompt_vs_mean_chunk": {
                "pearson": pearson([p for p, _ in pl_cl],
                                   [c for _, c in pl_cl]),
                "spearman": spearman([p for p, _ in pl_cl],
                                     [c for _, c in pl_cl]),
            },
        },
        "_raw": {"chunk_len": chunk_len, "req_out": req_out,
                 "prompt_len": prompt_len, "out_len": out_len},
    }


def f(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "output"
    ap.add_argument("--runs-dir", type=Path, default=base / "runs")
    ap.add_argument("--out-dir", type=Path, default=base / "stats")
    args = ap.parse_args()

    def find_run_dir(cat_dir: Path) -> Path | None:
        # baseline/run_1/rate_<r>/chunks.jsonl
        run1 = cat_dir / "baseline" / "run_1"
        if not run1.is_dir():
            return None
        for rate_dir in sorted(run1.glob("rate_*")):
            if (rate_dir / "chunks.jsonl").exists():
                return rate_dir
        return None

    run_dirs = {}
    if args.runs_dir.exists():
        for p in sorted(args.runs_dir.iterdir()):
            if not p.is_dir():
                continue
            rd = find_run_dir(p)
            if rd is not None:
                run_dirs[p.name] = rd
    if not run_dirs:
        print(f"no category runs found under {args.runs_dir}")
        return

    results = {}
    for cat, run_dir in run_dirs.items():
        res = analyze_category(run_dir)
        if res:
            results[cat] = res
            print(f"loaded {cat}: "
                  f"{res['chunk_len']['count']} chunks, "
                  f"{res['request_out_len']['count']} requests")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # JSON (drop _raw for compactness).
    clean = {cat: {k: v for k, v in r.items() if k != "_raw"}
             for cat, r in results.items()}
    (args.out_dir / "chunk_vs_request_variability.json").write_text(
        json.dumps(clean, indent=2))

    # CSV: one row per (category, level) with the three variability metrics.
    csv_path = args.out_dir / "chunk_vs_request_variability.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "level", "count", "mean", "cv",
                    "p99_p50", "iqr_med"])
        for cat, r in results.items():
            for level in ("chunk_len", "request_out_len"):
                s = r[level]
                w.writerow([cat, level, s["count"], f(s["mean"], 1),
                            f(s["cv"]), f(s["p99_p50"]), f(s["iqr_med"])])

    # Console table.
    print("\n=== variability (chunk vs request) ===")
    print(f"{'category':<14} {'lvl':<8} {'n':>6} {'mean':>7} "
          f"{'CV':>6} {'p99/p50':>8} {'IQR/med':>8}")
    for cat, r in results.items():
        for level, tag in (("chunk_len", "chunk"),
                           ("request_out_len", "request")):
            s = r[level]
            print(f"{cat:<14} {tag:<8} {s['count']:>6} "
                  f"{f(s['mean'],1):>7} {f(s['cv']):>6} "
                  f"{f(s['p99_p50']):>8} {f(s['iqr_med']):>8}")
    print("\n=== prompt-length correlation (Pearson / Spearman) ===")
    print(f"{'category':<14} {'prompt→output':>20} {'prompt→meanChunk':>20}")
    for cat, r in results.items():
        po = r["corr"]["prompt_vs_output"]
        pc = r["corr"]["prompt_vs_mean_chunk"]
        print(f"{cat:<14} "
              f"{f(po['pearson'],2)+'/'+f(po['spearman'],2):>20} "
              f"{f(pc['pearson'],2)+'/'+f(pc['spearman'],2):>20}")

    _plots(results, args.out_dir)
    print(f"\nwrote stats + plots to {args.out_dir}")


def _plots(results: dict, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cats = list(results)

    # (a) length distribution overlay: chunk vs request, per category.
    n = len(cats)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for i, cat in enumerate(cats):
        ax = axes[0][i]
        raw = results[cat]["_raw"]
        ax.hist(raw["chunk_len"], bins=40, density=True, alpha=0.55,
                label="chunk", color="tab:blue")
        ax.hist(raw["req_out"], bins=40, density=True, alpha=0.55,
                label="request out", color="tab:orange")
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("tokens")
        if i == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
    fig.suptitle("Per-chunk vs per-request output length distribution")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plots / "length_distribution.png", dpi=140)
    plt.close(fig)

    # (b) variability grouped bars: CV / p99p50 / IQRmed, chunk vs request.
    metrics = [("cv", "CV"), ("p99_p50", "p99/p50"), ("iqr_med", "IQR/median")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), squeeze=False)
    x = range(len(cats))
    for j, (key, label) in enumerate(metrics):
        ax = axes[0][j]
        cv_chunk = [results[c]["chunk_len"][key] or 0 for c in cats]
        cv_req = [results[c]["request_out_len"][key] or 0 for c in cats]
        ax.bar([i - 0.2 for i in x], cv_chunk, 0.4, label="chunk",
               color="tab:blue")
        ax.bar([i + 0.2 for i in x], cv_req, 0.4, label="request",
               color="tab:orange")
        ax.set_xticks(list(x))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        if j == 0:
            ax.set_ylabel("variability")
            ax.legend(fontsize=8)
    fig.suptitle("Variability: chunk vs request (lower = more predictable)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plots / "variability_bars.png", dpi=140)
    plt.close(fig)

    # (c) prompt-length vs output-length scatter, per category.
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for i, cat in enumerate(cats):
        ax = axes[0][i]
        raw = results[cat]["_raw"]
        ax.scatter(raw["prompt_len"], raw["out_len"], s=6, alpha=0.4)
        po = results[cat]["corr"]["prompt_vs_output"]
        ax.set_title(f"{cat}\nr={f(po['pearson'],2)} "
                     f"ρ={f(po['spearman'],2)}", fontsize=8)
        ax.set_xlabel("prompt tokens")
        if i == 0:
            ax.set_ylabel("output tokens")
    fig.suptitle("Prompt length vs request output length")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots / "prompt_output_correlation.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
