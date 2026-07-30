"""Oracle vs online-posterior chunk-length estimation replay.

Replays each category's baseline chunks.jsonl in chunk-completion-time order
(`text_generation_end_time`), feeding the SAME global estimator the engine
uses (`ChunkLengthPredictor`, deque maxlen 4096, loaded straight from
vllm/vllm/sslo/slo_state.py) with the SAME gating `length_tail_prob` applies
(warmup >= 128 samples, conditioning mass >= min_denom 4, else the analytic
cold-start step at c + 2048). This reproduces exactly the continuously
updated posterior ProgressServe would see at each moment — no GPU rerun.

Per chunk, two evaluation points:
progress fractions f ∈ {0, 25%, 50%, 75%}: at each f the chunk has emitted
c = floor(f·L) tokens, so we condition on {L > c} and predict the REMAINING
length L − c (conditional median) at the history available at that moment
(history = completions strictly before token c's emission time). As f grows
the conditional tail sharpens — this is the "tail tightens with progress"
path ProgressServe actually exercises. The online−oracle gap at each f is the
cost of online estimation under deeper (thinner-mass) conditioning.

Oracle = the same conditional-median predictor but holding the category
run's FULL empirical distribution (hindsight). Oracle per-chunk error is the
irreducible variability floor; online − oracle gap is the cost of online
estimation (cold start + finite/windowed history).

Also reports tail-probability calibration: predicted P(L > x | L > c=0) at
the oracle's {p25,p50,p75,p90,p99} cutoffs vs realized frequency (ECE),
split into early (first 1024 chunks) vs late.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

# Reuse run_sslo helpers (read-only import).
_RUN_SSLO = Path(__file__).resolve().parents[1] / "run_sslo"
sys.path.insert(0, str(_RUN_SSLO))
from jsonl_utils import read_jsonl  # noqa: E402
from metrics_utils import percentile  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Engine gating defaults (SsloConfig / progress_serve.DEFAULT_MIN_DENOM).
WARMUP_SAMPLES = 128
MIN_DENOM = 4
COLD_START_MAX = 2048

EARLY_EVENTS = 1024  # calibration split: first N chunk evals vs the rest
CONV_SAMPLE_EVERY = 32  # convergence-curve sampling stride (events)
# Progress points: predict remaining length after f-fraction of the chunk is
# emitted (c = floor(f·L)). 0 = predict the whole unit; >0 = remaining tail.
PROGRESS_FRACTIONS = (0.0, 0.25, 0.50, 0.75)


def _load_chunk_len_predictor():
    """Load ChunkLengthPredictor from slo_state.py WITHOUT importing the
    vllm package (slo_state only needs stdlib + numpy)."""
    src = (Path(__file__).resolve().parents[2]
           / "vllm" / "vllm" / "sslo" / "slo_state.py")
    spec = importlib.util.spec_from_file_location("_sslo_slo_state", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass processing needs sys.modules
    spec.loader.exec_module(mod)
    return mod.ChunkLengthPredictor


class OnlinePosterior:
    """ChunkLengthPredictor + the exact length_tail_prob gating."""

    def __init__(self, predictor_cls) -> None:
        self.pred = predictor_cls()  # engine defaults: p90, maxlen 4096

    def observe(self, num_token: int) -> None:
        self.pred.update(num_token)

    def is_warm(self, c_q: int) -> bool:
        return (self.pred.sample_count >= WARMUP_SAMPLES
                and self.pred.sample_count_above(c_q) >= MIN_DENOM)

    def tail(self, x: int, c_q: int) -> float:
        """P(L > x | L > c_q) exactly as RequestSLOState.length_tail_prob."""
        if self.is_warm(c_q):
            p = self.pred.tail_prob(x, c_q)
            if p is not None:
                return p
        return 1.0 if x < c_q + COLD_START_MAX else 0.0


class OracleTail:
    """Full-run (hindsight) empirical tail. No gating."""

    def __init__(self, lengths: list[int]) -> None:
        self.sorted = sorted(lengths)

    def tail(self, x: int, c_q: int) -> float:
        s = self.sorted
        denom = len(s) - bisect.bisect_right(s, c_q)
        if denom <= 0:
            return 0.0
        return (len(s) - bisect.bisect_right(s, x)) / denom


def quantile_from_tail(tail, c_q: int, q: float) -> float:
    """Smallest x >= c_q with P(L > x | L > c_q) <= 1 - q (binary search;
    tail is non-increasing in x)."""
    lo, hi = c_q, c_q + COLD_START_MAX + 1
    target = 1.0 - q
    while lo < hi:
        mid = (lo + hi) // 2
        if tail(mid, c_q) <= target:
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def replay_category(chunks: list[dict], predictor_cls) -> dict:
    """Completion-order replay; returns per-chunk eval records + curves."""
    recs = [c for c in chunks
            if c.get("num_token") is not None
            and c.get("text_generation_end_time") is not None]
    end_time = {(c["request_id"], c["unit_index"]):
                float(c["text_generation_end_time"]) for c in recs}

    # Completion events (what feeds the online predictor), in time order.
    completions = sorted(
        ((float(c["text_generation_end_time"]), int(c["num_token"]))
         for c in recs), key=lambda t: t[0])

    # Evaluation entries: (time, inclusive, kind, c, actual).
    # inclusive ⇒ history includes events with end_time == time (the chunk
    # boundary updates the predictor before the next unit starts). For
    # fraction f the eval time is start + f·(end-start): the moment token
    # c = floor(f·L) is emitted, so only earlier completions are in history.
    evals = []
    for c in recs:
        rid, ui = c["request_id"], c["unit_index"]
        own_end = float(c["text_generation_end_time"])
        length = int(c["num_token"])
        prev_end = end_time.get((rid, ui - 1)) if ui > 0 else None
        if prev_end is not None:
            start, incl = prev_end, True
        else:  # unit 0: start unknown → approximate with own end (strict <)
            start, incl = own_end, False
        for frac in PROGRESS_FRACTIONS:
            c_q = int(length * frac)
            if frac > 0.0 and c_q == 0:
                continue  # too short for a distinct progress point
            t_at = start + frac * (own_end - start) if incl else own_end
            evals.append((t_at, incl, f"c{int(frac * 100)}", c_q, length))
    evals.sort(key=lambda e: (e[0], e[1]))

    online = OnlinePosterior(predictor_cls)
    oracle = OracleTail([int(c["num_token"]) for c in recs])

    # Calibration cutoffs from the oracle distribution.
    all_len = [int(c["num_token"]) for c in recs]
    cutoffs = sorted({int(percentile(all_len, p))
                      for p in (25, 50, 75, 90, 99)})

    rows = []  # per-eval: kind, n_seen, warm, pred, oracle_pred, actual_rem
    calib = []  # (n_seen, predicted_p, outcome) at c=0
    curve = []  # (n_seen, q50, q90, q99) online unconditional, gate-aware
    j = 0
    for t, incl, kind, c_q, length in evals:
        while j < len(completions) and (
                completions[j][0] < t
                or (incl and completions[j][0] == t)):
            online.observe(completions[j][1])
            n = online.pred.sample_count
            if n % CONV_SAMPLE_EVERY == 0 or n <= 8:
                curve.append((n,
                              quantile_from_tail(online.tail, 0, 0.50),
                              quantile_from_tail(online.tail, 0, 0.90),
                              quantile_from_tail(online.tail, 0, 0.99)))
            j += 1
        pred = quantile_from_tail(online.tail, c_q, 0.50) - c_q
        opred = quantile_from_tail(oracle.tail, c_q, 0.50) - c_q
        actual_rem = length - c_q
        n_seen = online.pred.sample_count
        rows.append({"kind": kind, "n_seen": n_seen,
                     "warm": online.is_warm(c_q),
                     "pred": pred, "oracle_pred": opred,
                     "actual": actual_rem})
        if kind == "c0":
            for x in cutoffs:
                calib.append((n_seen, online.tail(x, 0),
                              1.0 if length > x else 0.0))
    return {"rows": rows, "calib": calib, "curve": curve,
            "oracle_q": {q: quantile_from_tail(oracle.tail, 0, q)
                         for q in (0.50, 0.90, 0.99)},
            "n_chunks": len(recs)}


def _errors(rows: list[dict], kind: str, warm_only: bool) -> dict | None:
    sel = [r for r in rows if r["kind"] == kind
           and (r["warm"] or not warm_only)]
    if not sel:
        return None
    ae = [abs(r["pred"] - r["actual"]) for r in sel]
    oae = [abs(r["oracle_pred"] - r["actual"]) for r in sel]
    rel = [abs(r["pred"] - r["actual"]) / r["actual"]
           for r in sel if r["actual"] > 0]
    orel = [abs(r["oracle_pred"] - r["actual"]) / r["actual"]
            for r in sel if r["actual"] > 0]
    return {"n": len(sel),
            "mae_online": statistics.fmean(ae),
            "mae_oracle": statistics.fmean(oae),
            "mape_online": statistics.fmean(rel) if rel else None,
            "mape_oracle": statistics.fmean(orel) if orel else None}


def _ece(samples: list[tuple[float, float]], bins: int = 10) -> float | None:
    """Expected calibration error over (predicted_p, outcome) pairs."""
    if not samples:
        return None
    binned: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, o in samples:
        binned[min(bins - 1, int(p * bins))].append((p, o))
    n = len(samples)
    ece = 0.0
    for b in binned:
        if not b:
            continue
        mp = statistics.fmean(p for p, _ in b)
        fo = statistics.fmean(o for _, o in b)
        ece += len(b) / n * abs(mp - fo)
    return ece


def summarize(res: dict) -> dict:
    rows = res["rows"]
    # Per-progress-fraction error: predict remaining length L-c after
    # f-fraction of the chunk emitted, online (live history) vs oracle (full
    # dist). warm = the gated/empirical regime; all = includes cold fallback.
    kinds = [f"c{int(f_ * 100)}" for f_ in PROGRESS_FRACTIONS]
    sharpening = {}
    for k in kinds:
        sharpening[k] = {"all": _errors(rows, k, warm_only=False),
                         "warm": _errors(rows, k, warm_only=True)}
    n_c0 = sum(1 for r in rows if r["kind"] == "c0")
    cold_c0 = sum(1 for r in rows if r["kind"] == "c0" and not r["warm"])

    early = [(p, o) for n, p, o in res["calib"] if n <= EARLY_EVENTS]
    late = [(p, o) for n, p, o in res["calib"] if n > EARLY_EVENTS]

    # Convergence: first event count where online q is within 10% of oracle.
    conv = {}
    for qi, q in ((1, "p50"), (2, "p90")):
        target = res["oracle_q"][0.50 if q == "p50" else 0.90]
        conv[q] = next((n for n, *qs in res["curve"]
                        if target and abs(qs[qi - 1] - target) / target
                        <= 0.10), None)
    return {"n_chunks": res["n_chunks"],
            "cold_frac_c0": cold_c0 / n_c0 if n_c0 else None,
            "sharpening": sharpening,
            "ece_early": _ece(early), "ece_late": _ece(late),
            "events_to_oracle_10pct": conv,
            "oracle_q": {f"p{int(q*100)}": v
                         for q, v in res["oracle_q"].items()}}


def f(x, d=3):
    return "" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "output"
    ap.add_argument("--runs-dir", type=Path, default=base / "runs")
    ap.add_argument("--out-dir", type=Path, default=base / "stats")
    args = ap.parse_args()

    predictor_cls = _load_chunk_len_predictor()

    results, summaries = {}, {}
    for cat_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        run1 = cat_dir / "baseline" / "run_1"
        rate_dirs = sorted(run1.glob("rate_*")) if run1.is_dir() else []
        chunks_path = next((rd / "chunks.jsonl" for rd in rate_dirs
                            if (rd / "chunks.jsonl").exists()), None)
        if chunks_path is None:
            continue
        chunks = read_jsonl(chunks_path)
        res = replay_category(chunks, predictor_cls)
        results[cat_dir.name] = res
        summaries[cat_dir.name] = summarize(res)
        print(f"replayed {cat_dir.name}: {res['n_chunks']} chunks")
    if not results:
        print(f"no category runs found under {args.runs_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "oracle_vs_posterior.json").write_text(
        json.dumps(summaries, indent=2))

    # Long format: one row per (category, progress fraction) — the sharpening
    # curve. mae/mape are for the conditional-median remaining-length
    # prediction; n_warm is the warm (non-cold-fallback) eval count.
    kinds = [f"c{int(f_ * 100)}" for f_ in PROGRESS_FRACTIONS]
    csv_path = args.out_dir / "oracle_vs_posterior.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "progress", "n_warm",
                    "mae_online", "mae_oracle", "mae_gap",
                    "mape_online", "mape_oracle"])
        for cat, s in summaries.items():
            for k in kinds:
                e = s["sharpening"][k]["warm"]
                if e is None:
                    w.writerow([cat, k, 0, "", "", "", "", ""])
                    continue
                gap = e["mae_online"] - e["mae_oracle"]
                w.writerow([cat, k, e["n"],
                            f(e["mae_online"], 2), f(e["mae_oracle"], 2),
                            f(gap, 3), f(e["mape_online"]),
                            f(e["mape_oracle"])])

    print("\n=== sharpening: MAE(online/oracle) of remaining length "
          "by progress (warm; tokens) ===")
    hdr = "".join(f"{k+' on/or':>14}" for k in kinds)
    print(f"{'category':<14} {'n_chunks':>9}{hdr}  {'ECE e/l':>11}")
    for cat, s in summaries.items():
        cells = ""
        for k in kinds:
            e = s["sharpening"][k]["warm"]
            cells += (f"{f(e['mae_online'],1)+'/'+f(e['mae_oracle'],1):>14}"
                      if e else f"{'-':>14}")
        print(f"{cat:<14} {s['n_chunks']:>9}{cells}  "
              f"{f(s['ece_early'],2)+'/'+f(s['ece_late'],2):>11}")

    _plots(results, summaries, args.out_dir)
    print(f"\nwrote stats + plots to {args.out_dir}")


def _plots(results: dict, summaries: dict, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cats = list(results)
    n = len(cats)

    # (a) convergence: online q50/q90/q99 trajectory vs oracle dashes.
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    for i, cat in enumerate(cats):
        ax = axes[0][i]
        curve = results[cat]["curve"]
        xs = [c[0] for c in curve]
        for qi, (q, color) in enumerate(
                ((0.50, "tab:blue"), (0.90, "tab:orange"),
                 (0.99, "tab:red")), start=1):
            ax.plot(xs, [c[qi] for c in curve], color=color, lw=1.2,
                    label=f"online p{int(q*100)}")
            ax.axhline(results[cat]["oracle_q"][q], color=color, ls="--",
                       lw=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("chunks observed")
        if i == 0:
            ax.set_ylabel("predicted length (tokens)")
            ax.legend(fontsize=7)
    fig.suptitle("Online posterior convergence to oracle quantiles "
                 "(dashed = oracle)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots / "posterior_convergence.png", dpi=140)
    plt.close(fig)

    # (b) sharpening curve: remaining-length MAE vs progress fraction,
    # online vs oracle, one panel per category. As progress grows the
    # conditional tail tightens (MAE falls); the online↔oracle gap shows
    # whether the live posterior keeps up under deeper conditioning.
    kinds = [f"c{int(f_ * 100)}" for f_ in PROGRESS_FRACTIONS]
    xs = [f_ * 100 for f_ in PROGRESS_FRACTIONS]
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False,
                             sharey=True)
    for i, cat in enumerate(cats):
        ax = axes[0][i]
        sh = summaries[cat]["sharpening"]
        on = [(sh[k]["warm"] or {}).get("mae_online") for k in kinds]
        orc = [(sh[k]["warm"] or {}).get("mae_oracle") for k in kinds]
        ax.plot(xs, on, "o-", color="tab:blue", lw=1.4, ms=4,
                label="online posterior")
        ax.plot(xs, orc, "s--", color="tab:green", lw=1.2, ms=4,
                label="oracle (full dist.)")
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("progress c = f·L  (%)")
        ax.set_xticks(xs)
        if i == 0:
            ax.set_ylabel("remaining-length MAE (tokens)")
            ax.legend(fontsize=7)
    fig.suptitle("Tail sharpening with progress: remaining-length error, "
                 "online vs oracle (warm)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots / "posterior_sharpening.png", dpi=140)
    plt.close(fig)

    # (c) tail calibration: early vs late reliability, per category.
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for i, cat in enumerate(cats):
        ax = axes[0][i]
        for sel, color, label in (
                (lambda ns: ns <= EARLY_EVENTS, "tab:red",
                 f"early (≤{EARLY_EVENTS})"),
                (lambda ns: ns > EARLY_EVENTS, "tab:blue", "late")):
            pts = [(p, o) for ns, p, o in results[cat]["calib"] if sel(ns)]
            if not pts:
                continue
            bx, by = [], []
            for b in range(10):
                bb = [(p, o) for p, o in pts
                      if min(9, int(p * 10)) == b]
                if bb:
                    bx.append(statistics.fmean(p for p, _ in bb))
                    by.append(statistics.fmean(o for _, o in bb))
            ax.plot(bx, by, "o-", color=color, ms=3, lw=1.0, label=label)
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1.0)
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("predicted P(L > x)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("observed frequency")
            ax.legend(fontsize=7)
    fig.suptitle("Tail-probability calibration (c = 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots / "posterior_calibration.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
