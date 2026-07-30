"""Consumable-unit (chunk) length distribution + per-request divergence.

For each category run: the GLOBAL distribution of per-chunk token lengths
(`num_token`, the consumable unit the estimator predicts), and the requests
whose own chunk-length distribution diverges most from that global pool.

The engine's ProgressServe posterior is GLOBAL — pooled over all requests in
the run and reset per run — so "global" here = the run's full chunk pool. A
request that diverges hard from it is exactly one the global-posterior serves
worst; its characteristics show when per-request adaptation would matter.

Divergence = two-sample Kolmogorov–Smirnov over the integer length support
(D = max_x |F_req(x) - F_global(x)|, in [0,1], sample-size comparable). Top-5
is a single KS ranking across categories (KS is normalised). Requests with
fewer than MIN_CHUNKS units are dropped (their empirical CDF is too coarse).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Reuse run_sslo helpers (read-only import).
_RUN_SSLO = Path(__file__).resolve().parents[1] / "run_sslo"
sys.path.insert(0, str(_RUN_SSLO))
from jsonl_utils import read_jsonl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MIN_CHUNKS = 8  # below this a per-request distribution is too coarse to trust
TOP_N = 5       # top divergent requests to surface in detail
PCTS = (1, 10, 25, 50, 75, 90, 99)
# Chunks above this are runaway un-segmented outputs (no sentence boundary
# hit), not consumable units — excluded from the distribution and divergence.
# Real sentence chunks top out near p99.9 ≈ 55–82; >100 is ≤0.08% of data.
DEFAULT_MAX_CHUNK_LEN = 100


def _cdf(sample: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Empirical CDF of `sample` evaluated at every value in `support`."""
    s = np.sort(sample)
    return np.searchsorted(s, support, side="right") / len(s)


def _stats(lens: np.ndarray) -> dict:
    mean = float(lens.mean())
    std = float(lens.std())
    p50 = float(np.percentile(lens, 50))
    return {"n": int(lens.size), "mean": mean, "std": std,
            "cv": std / mean if mean else None,
            "p50": p50,
            "p99": float(np.percentile(lens, 99)),
            "max": int(lens.max())}


def analyze_category(run_dir: Path, max_len: int) -> dict | None:
    chunks = read_jsonl(run_dir / "chunks.jsonl")
    reqs = read_jsonl(run_dir / "requests.jsonl")
    if not chunks or not reqs:
        return None

    by_req: dict[str, list[int]] = {}
    n_dropped = 0
    for c in chunks:
        rid, nt = c.get("request_id"), c.get("num_token")
        if rid is None or nt is None:
            continue
        if max_len > 0 and int(nt) > max_len:
            n_dropped += 1  # runaway un-segmented chunk, not a consumable unit
            continue
        by_req.setdefault(rid, []).append(int(nt))
    glob = np.array([nt for lens in by_req.values() for nt in lens])
    if glob.size == 0:
        return None

    support = np.arange(glob.max() + 1)
    gcdf = _cdf(glob, support)
    gstats = _stats(glob)

    meta = {r.get("request_id"): r for r in reqs}
    per_req = []
    for rid, lens in by_req.items():
        arr = np.array(lens)
        if arr.size < MIN_CHUNKS:
            continue
        ks = float(np.abs(_cdf(arr, support) - gcdf).max())
        rs = _stats(arr)
        m = meta.get(rid, {})
        per_req.append({
            "request_id": rid, "ks": ks,
            "n_chunks": rs["n"],
            "mean": rs["mean"], "cv": rs["cv"],
            "median": rs["p50"], "p99": rs["p99"], "max": rs["max"],
            "median_shift": rs["p50"] - gstats["p50"],
            "cv_ratio": (rs["cv"] / gstats["cv"]
                         if gstats["cv"] and rs["cv"] is not None else None),
            "prompt_tokens": m.get("num_prompt_tokens"),
            "output_tokens": m.get("num_output_tokens"),
            "prompt": (m.get("prompt") or "")[:240],
            "lens": lens,
        })
    return {"global": gstats, "global_lens": glob,
            "global_pcts": {f"p{p}": float(np.percentile(glob, p))
                            for p in PCTS},
            "support": support, "gcdf": gcdf,
            "per_req": per_req, "n_dropped": n_dropped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "output"
    ap.add_argument("--runs-dir", type=Path, default=base / "runs")
    ap.add_argument("--out-dir", type=Path, default=base / "stats")
    ap.add_argument("--max-chunk-len", type=int,
                    default=DEFAULT_MAX_CHUNK_LEN,
                    help="exclude chunks longer than this (runaway "
                         "un-segmented outputs); 0 disables")
    args = ap.parse_args()

    cats: dict[str, dict] = {}
    for cat_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        run1 = cat_dir / "baseline" / "run_1"
        rate_dirs = sorted(run1.glob("rate_*")) if run1.is_dir() else []
        run_dir = next((rd for rd in rate_dirs
                        if (rd / "chunks.jsonl").exists()), None)
        if run_dir is None:
            continue
        res = analyze_category(run_dir, args.max_chunk_len)
        if res:
            cats[cat_dir.name] = res
            g = res["global"]
            print(f"loaded {cat_dir.name}: {g['n']} chunks "
                  f"({res['n_dropped']} dropped > {args.max_chunk_len}), "
                  f"{len(res['per_req'])} reqs (>= {MIN_CHUNKS} chunks)")
    if not cats:
        print(f"no category runs found under {args.runs_dir}")
        return

    # Single KS ranking across all categories.
    ranked = sorted(
        ((cat, r) for cat, res in cats.items() for r in res["per_req"]),
        key=lambda cr: cr[1]["ks"], reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Global distribution percentiles per category.
    cut = (f"chunks > {args.max_chunk_len} excluded"
           if args.max_chunk_len > 0 else "no outlier filter")
    print(f"\n=== global consumable-unit length distribution (tokens; "
          f"{cut}) ===")
    hdr = "  ".join(f"p{p}" for p in PCTS)
    print(f"{'category':<14} {'n':>7} {'drop':>5} {'mean':>6} {'cv':>5}   "
          f"{hdr}")
    gcsv = args.out_dir / "unit_distribution_global.csv"
    with gcsv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "n_chunks", "n_dropped", "max_chunk_len",
                    "mean", "cv"] + [f"p{p}" for p in PCTS])
        for cat, res in cats.items():
            g = res["global"]
            pj = res["global_pcts"]
            ps = "  ".join(f"{pj[f'p{p}']:.0f}" for p in PCTS)
            print(f"{cat:<14} {g['n']:>7} {res['n_dropped']:>5} "
                  f"{g['mean']:>6.1f} {g['cv']:>5.2f}   {ps}")
            w.writerow([cat, g["n"], res["n_dropped"], args.max_chunk_len,
                        f"{g['mean']:.2f}", f"{g['cv']:.3f}"]
                       + [f"{pj[f'p{p}']:.1f}" for p in PCTS])

    # Top-N divergent requests with characteristics.
    print(f"\n=== top {TOP_N} requests most divergent from category global "
          f"(KS) ===")
    top = ranked[:TOP_N]
    for i, (cat, r) in enumerate(top, 1):
        gl = cats[cat]["global"]
        print(f"\n#{i}  {cat}  req={r['request_id']}  KS={r['ks']:.3f}")
        print(f"    chunks={r['n_chunks']}  "
              f"median={r['median']:.0f} (global {gl['p50']:.0f}, "
              f"shift {r['median_shift']:+.0f})  "
              f"mean={r['mean']:.1f} (global {gl['mean']:.1f})  "
              f"CV={r['cv']:.2f} (×{r['cv_ratio']:.2f} global)  "
              f"max={r['max']}")
        print(f"    prompt_tokens={r['prompt_tokens']}  "
              f"output_tokens={r['output_tokens']}")
        print(f"    prompt: {r['prompt']!r}")

    # CSV: ranked top divergent (broader, for scanning).
    rcsv = args.out_dir / "top_divergent_requests.csv"
    with rcsv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "category", "request_id", "ks", "n_chunks",
                    "median", "global_median", "median_shift", "mean",
                    "cv", "cv_ratio", "max", "prompt_tokens",
                    "output_tokens", "prompt"])
        for i, (cat, r) in enumerate(ranked[:50], 1):
            gl = cats[cat]["global"]
            w.writerow([i, cat, r["request_id"], f"{r['ks']:.4f}",
                        r["n_chunks"], f"{r['median']:.0f}",
                        f"{gl['p50']:.0f}", f"{r['median_shift']:+.0f}",
                        f"{r['mean']:.1f}",
                        f"{r['cv']:.3f}" if r["cv"] is not None else "",
                        f"{r['cv_ratio']:.2f}"
                        if r["cv_ratio"] is not None else "",
                        r["max"], r["prompt_tokens"], r["output_tokens"],
                        r["prompt"].replace("\n", " ")])

    # JSON.
    clean = {"_meta": {"max_chunk_len": args.max_chunk_len,
                       "min_chunks": MIN_CHUNKS}}
    clean.update({cat: {"global": res["global"],
                        "global_pcts": res["global_pcts"],
                        "n_dropped": res["n_dropped"],
                        "n_reqs": len(res["per_req"])}
                  for cat, res in cats.items()})
    clean["top_divergent"] = [
        {"category": cat,
         **{k: v for k, v in r.items() if k != "lens"}}
        for cat, r in ranked[:TOP_N]]
    (args.out_dir / "unit_distribution.json").write_text(
        json.dumps(clean, indent=2))

    _plots(cats, top, args.out_dir)
    print(f"\nwrote stats + plots to {args.out_dir}")


def _plots(cats: dict, top: list, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    names = list(cats)
    n = len(names)

    # (a) global consumable-unit distribution per category (hist + CDF).
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.3), squeeze=False)
    for i, cat in enumerate(names):
        ax = axes[0][i]
        lens = cats[cat]["global_lens"]
        xmax = float(np.percentile(lens, 99.5))
        # Bin within the display window so a few extreme outliers don't
        # widen every bin into one flat block (range, not full data span).
        ax.hist(lens, bins=50, range=(0, xmax), density=True,
                color="tab:blue", alpha=0.6)
        ax.set_title(f"{cat}\n(median {cats[cat]['global']['p50']:.0f}, "
                     f"p99 {cats[cat]['global_pcts']['p99']:.0f}, "
                     f"max {cats[cat]['global']['max']})",
                     fontsize=8)
        ax.set_xlabel("chunk length (tokens)")
        ax.set_xlim(0, xmax)
        if i == 0:
            ax.set_ylabel("density")
    fig.suptitle("Global consumable-unit length distribution")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots / "unit_distribution_global.png", dpi=140)
    plt.close(fig)

    # (b) top-N divergent requests: their chunk-length CDF vs category global.
    k = len(top)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.4), squeeze=False)
    for i, (cat, r) in enumerate(top):
        ax = axes[0][i]
        support = cats[cat]["support"]
        ax.plot(support, cats[cat]["gcdf"], color="gray", lw=1.3,
                label="global")
        ax.plot(support, _cdf(np.array(r["lens"]), support),
                color="tab:red", lw=1.5, label=f"req {r['request_id']}")
        ax.set_title(f"{cat}  KS={r['ks']:.2f}\n"
                     f"med {r['median']:.0f} vs {cats[cat]['global']['p50']:.0f}"
                     f", n={r['n_chunks']}", fontsize=8)
        ax.set_xlabel("chunk length (tokens)")
        ax.set_xlim(0, max(r["max"],
                           float(np.percentile(cats[cat]["global_lens"], 99))))
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("CDF")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Top divergent requests: chunk-length CDF vs global")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots / "top_divergent_requests.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
