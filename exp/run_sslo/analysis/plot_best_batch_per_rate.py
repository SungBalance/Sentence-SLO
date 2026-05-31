"""
For each (model, consume_model, policy, rate), pick the cap (batch size)
that satisfies unit_deadline_miss_rate < 0.5% and maximizes mean_handling_users.

Plot subplots (rows=model, cols=consume_model):
  X = request rate
  Y (left)  = max handling_users (solid)
  Y (right) = KV cache usage ratio (used_mean / total) (dashed)
Lines: baseline vs sslo_mlp, with cap annotated at each marker.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIOLATION_THRESHOLD_PCT = 0.5  # unit_deadline_miss_rate < 0.5%

MODELS = ["Qwen3.5-9B", "Qwen3.5-35B-A3B"]
CONSUME_LABELS = [
    ("read", "none", "read"),
    ("tts", "hexgrad__Kokoro-82M", "Kokoro"),
    ("tts", "Supertone__supertonic-3", "Supertone"),
]
POLICIES = ["baseline", "sslo_mlp"]
POLICY_COLOR = {"baseline": "tab:gray", "sslo_mlp": "tab:blue"}

RATE_RE = re.compile(r"rate_([0-9.]+)$")


def walk(root: Path):
    rows = []
    for s in root.rglob("summary.json"):
        parts = s.parts
        try:
            sentence_idx = parts.index("sentence")
        except ValueError:
            continue
        try:
            model_slug = parts[sentence_idx + 1]
            consume_mode = parts[sentence_idx + 2]
            tts_slug = parts[sentence_idx + 3]
            cap_dir = parts[sentence_idx + 4]
            policy = parts[sentence_idx + 5]
            run_dir = parts[sentence_idx + 6]
            rate_dir = parts[sentence_idx + 7]
        except IndexError:
            continue
        if not cap_dir.startswith("cap"):
            continue
        m = RATE_RE.match(rate_dir)
        if not m:
            continue
        cap = int(cap_dir[3:])
        rate = float(m.group(1))
        try:
            data = json.loads(s.read_text())
        except Exception:
            continue
        mode_metrics = data.get("metrics", {}).get(policy, {})
        wl = data.get("metrics", {}).get("workload", {}).get(policy, {})
        sched = data.get("metrics", {}).get("scheduler", {}).get(policy, {})
        udmr = wl.get("unit_deadline_miss_rate")  # ratio (0..1)
        if udmr is None:
            udmr = mode_metrics.get("unit_deadline_miss_rate")
        handling = mode_metrics.get("mean_handling_users")
        kv_used = sched.get("kv_blocks_used_mean")
        kv_total = sched.get("kv_blocks_total")
        rows.append({
            "model": model_slug,
            "consume_mode": consume_mode,
            "tts_slug": tts_slug,
            "policy": policy,
            "cap": cap,
            "rate": rate,
            "udmr_pct": (udmr * 100) if udmr is not None else None,
            "handling": handling,
            "kv_used": kv_used,
            "kv_total": kv_total,
        })
    return rows


def best_per_rate(rows, model, consume_mode, tts_slug, policy):
    """For given group, return [(rate, cap, handling, kv_ratio, udmr)] sorted by rate."""
    group = [
        r for r in rows
        if r["model"] == model
        and r["consume_mode"] == consume_mode
        and r["tts_slug"] == tts_slug
        and r["policy"] == policy
        and r["udmr_pct"] is not None
        and r["handling"] is not None
    ]
    by_rate = {}
    for r in group:
        if r["udmr_pct"] >= VIOLATION_THRESHOLD_PCT:
            continue
        cur = by_rate.get(r["rate"])
        if cur is None or r["handling"] > cur["handling"]:
            by_rate[r["rate"]] = r
    out = []
    for rate in sorted(by_rate):
        r = by_rate[rate]
        kv_ratio = (r["kv_used"] / r["kv_total"]) if (
            r["kv_used"] is not None and r["kv_total"]) else None
        out.append({
            "rate": rate, "cap": r["cap"], "handling": r["handling"],
            "kv_ratio": kv_ratio, "udmr_pct": r["udmr_pct"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="output_sweep_noabatch directory")
    ap.add_argument("--out", required=True, type=Path,
                    help="output PNG path")
    args = ap.parse_args()

    rows = walk(args.root)
    print(f"loaded {len(rows)} summaries")

    n_rows, n_cols = len(MODELS), len(CONSUME_LABELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False, sharex=True)

    for ri, model in enumerate(MODELS):
        for ci, (consume_mode, tts_slug, label) in enumerate(CONSUME_LABELS):
            ax = axes[ri][ci]
            ax2 = ax.twinx()
            for policy in POLICIES:
                pts = best_per_rate(rows, model, consume_mode, tts_slug, policy)
                if not pts:
                    continue
                xs = [p["rate"] for p in pts]
                ys = [p["handling"] for p in pts]
                kvs = [p["kv_ratio"] * 100 if p["kv_ratio"] is not None else None
                       for p in pts]
                color = POLICY_COLOR[policy]
                ax.plot(xs, ys, "o-", color=color, label=f"{policy} HU", lw=2)
                # KV usage on right axis (dashed)
                ax2.plot(xs, kvs, "s--", color=color, alpha=0.6,
                         label=f"{policy} KV%", lw=1.2)
                # Annotate cap chosen at each point
                for p in pts:
                    ax.annotate(f"c{p['cap']}",
                                xy=(p["rate"], p["handling"]),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=8, color=color)
            ax.set_title(f"{model} | {label}")
            ax.set_xlabel("request rate (req/s)")
            ax.set_ylabel("max handling_users (HU)")
            ax2.set_ylabel("KV usage %", color="dimgray")
            ax2.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            if ri == 0 and ci == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=8, loc="upper left")

    fig.suptitle(
        f"Max handling_users at cap with unit_deadline_miss < "
        f"{VIOLATION_THRESHOLD_PCT}% (cap label = chosen batch size)",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # Also dump a TSV of the chosen points for inspection.
    tsv_path = args.out.with_suffix(".tsv")
    with tsv_path.open("w") as f:
        f.write("model\tconsume\tpolicy\trate\tcap\thandling\tkv_ratio\tudmr_pct\n")
        for model in MODELS:
            for consume_mode, tts_slug, label in CONSUME_LABELS:
                for policy in POLICIES:
                    for p in best_per_rate(rows, model, consume_mode,
                                           tts_slug, policy):
                        kv = f"{p['kv_ratio']:.3f}" if p["kv_ratio"] is not None else ""
                        f.write(
                            f"{model}\t{label}\t{policy}\t{p['rate']:g}\t"
                            f"{p['cap']}\t{p['handling']:.1f}\t{kv}\t"
                            f"{p['udmr_pct']:.3f}\n")
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
