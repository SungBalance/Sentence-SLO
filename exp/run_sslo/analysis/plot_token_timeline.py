#!/usr/bin/env python3
"""Plot per-request cumulative-token timelines (generated vs consumed).

Two curves on the same axes, x = time since decoding_start_ts, y = cumulative
tokens:

* generated — within each chunk window [start_time_ts, end_time_ts] the
  cumulative count rises linearly with slope num_token / chunk_gen_duration
  (i.e. chunk-TPS, the inverse of chunk-TPOT).
* consumed  — reader / TTS consumption window starts at max(end_time_ts,
  prev_consumption_end) and lasts num_words * seconds_per_word; within it
  the count rises linearly with slope num_token / (num_words * spw).

`seconds_per_word` is read from `<run_dir>/sslo_config.json` (per-mode);
falls back to 0.28.

Usage:
  python3 plot_token_timeline.py --run-dir <run_<i>> --mode sslo \\
      [--request-idx N | default = average across all requests] \\
      [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsonl_utils import read_jsonl


def load_run(run_dir: Path, mode: str):
    chunks = [r for r in read_jsonl(run_dir / "chunks.jsonl")
              if r.get("mode") == mode]
    requests = [r for r in read_jsonl(run_dir / "requests.jsonl")
                if r.get("mode") == mode]
    spw = 0.28
    cfg_path = run_dir / "sslo_config.json"
    if cfg_path.exists():
        cfg = (json.loads(cfg_path.read_text()) or {}).get(mode, {})
        if cfg.get("seconds_per_word") is not None:
            spw = float(cfg["seconds_per_word"])
    return chunks, requests, spw


def build_curves(chunks_for_req: list[dict], decoding_start_ts: float,
                 spw: float) -> tuple[list[tuple[float, float]],
                                      list[tuple[float, float]]]:
    """Return (gen_breakpoints, cons_breakpoints), each (t_rel, cum_token).

    Both start at (0, 0). Generated curve is piecewise-linear over
    [start_rel, end_rel] per chunk. Consumed curve is piecewise-linear
    over [cons_start, cons_end] per chunk; if cons_start > prev cons_end
    a flat (idle) segment appears.
    """
    chunks = sorted(chunks_for_req, key=lambda c: c.get("chunk_idx", 0))
    gen: list[tuple[float, float]] = [(0.0, 0.0)]
    cons: list[tuple[float, float]] = [(0.0, 0.0)]
    cum_gen = 0
    cum_cons = 0
    cons_end = 0.0
    for c in chunks:
        n_tok = int(c.get("num_token") or 0)
        n_words = int(c.get("num_words") or 0)
        end_ts = c.get("end_time_ts")
        if end_ts is None:
            continue
        end_rel = float(end_ts) - decoding_start_ts
        cum_gen += n_tok
        gen.append((end_rel, cum_gen))

        cons_start = max(end_rel, cons_end)
        if cons_start > cons[-1][0]:
            # Reader was idle between cons[-1][0] and cons_start.
            cons.append((cons_start, cum_cons))
        cons_end = cons_start + n_words * spw
        cum_cons += n_tok
        cons.append((cons_end, cum_cons))
    return gen, cons


def resample_curve(breakpoints: list[tuple[float, float]],
                   t_grid: np.ndarray) -> np.ndarray:
    if not breakpoints:
        return np.full_like(t_grid, np.nan, dtype=float)
    xs = np.array([t for t, _ in breakpoints])
    ys = np.array([y for _, y in breakpoints])
    out = np.interp(t_grid, xs, ys).astype(float)
    out[t_grid < xs[0]] = np.nan
    out[t_grid > xs[-1]] = np.nan
    return out


def plot_single(gen, cons, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    if gen:
        gx, gy = zip(*gen)
        plt.plot(gx, gy, label="generated", color="tab:blue", linewidth=2)
    if cons:
        cx, cy = zip(*cons)
        plt.plot(cx, cy, label="consumed", color="tab:orange",
                 linewidth=2, linestyle="--")
    plt.xlabel("time since decoding start (s)")
    plt.ylabel("cumulative tokens")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_average(per_req_curves, title: str, output_path: Path) -> None:
    if not per_req_curves:
        print("no requests to plot")
        return
    max_t = 0.0
    for g, c in per_req_curves:
        if g:
            max_t = max(max_t, g[-1][0])
        if c:
            max_t = max(max_t, c[-1][0])
    if max_t <= 0:
        print("no positive time range")
        return
    t_grid = np.linspace(0, max_t, 500)
    gen_stack = np.stack([resample_curve(g, t_grid) for g, _ in per_req_curves])
    cons_stack = np.stack([resample_curve(c, t_grid) for _, c in per_req_curves])
    # Beyond the longest curve's end, all rows are NaN — nanmean would warn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gen_mean = np.nanmean(gen_stack, axis=0)
        cons_mean = np.nanmean(cons_stack, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, gen_mean, label="generated (mean)", color="tab:blue",
             linewidth=2)
    plt.plot(t_grid, cons_mean, label="consumed (mean)", color="tab:orange",
             linewidth=2, linestyle="--")
    plt.xlabel("time since decoding start (s)")
    plt.ylabel("cumulative tokens")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="run_<i> directory holding chunks.jsonl + requests.jsonl")
    p.add_argument("--mode", default="sslo")
    p.add_argument("--request-idx", type=int, default=None,
                   help="single request index; default = average all")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chunks, requests, spw = load_run(args.run_dir, args.mode)
    if not chunks or not requests:
        print(f"No chunks/requests for mode={args.mode} under {args.run_dir}")
        return

    chunks_by_req: dict[str, list[dict]] = {}
    for c in chunks:
        rid = str(c.get("request_id"))
        chunks_by_req.setdefault(rid, []).append(c)

    if args.request_idx is not None:
        target = next(
            (r for r in requests if r.get("request_idx") == args.request_idx),
            None)
        if target is None:
            print(f"request_idx={args.request_idx} not found in mode={args.mode}")
            return
        ds = target.get("decoding_start_ts")
        if ds is None:
            print(f"request_idx={args.request_idx} has no decoding_start_ts")
            return
        rid = str(target.get("request_id"))
        gen, cons = build_curves(chunks_by_req.get(rid, []), float(ds), spw)
        out = args.output or (
            args.run_dir
            / f"token_timeline_{args.mode}_req{args.request_idx}.png")
        plot_single(
            gen, cons,
            f"{args.mode}  request_idx={args.request_idx}  spw={spw}",
            out)
        print(f"wrote {out}")
        return

    per_req_curves = []
    for r in requests:
        ds = r.get("decoding_start_ts")
        if ds is None:
            continue
        rid = str(r.get("request_id"))
        gen, cons = build_curves(chunks_by_req.get(rid, []), float(ds), spw)
        if len(gen) > 1 and len(cons) > 1:
            per_req_curves.append((gen, cons))
    out = args.output or (
        args.run_dir / f"token_timeline_{args.mode}_avg.png")
    plot_average(
        per_req_curves,
        f"{args.mode}  avg of {len(per_req_curves)} requests  spw={spw}",
        out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
