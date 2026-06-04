"""Build per-(language, task) prompt pools for the chunk-length study.

Loads wildchat + lmsys with the code/English filters OFF, classifies each
prompt by task (code vs dialogue, via lm_datasets._is_code_request) and
language (via langdetect), picks the TOP-K languages by sample count, and
writes one cache per (lang, task) category of up to TARGET_PER_CATEGORY
prompts. Each cache is a `processed_dataset.jsonl` that run_test.sh can read
unchanged via DATASET_CACHE_DIR + DATASET_NAME=combine.

No existing scripts are modified — this only consumes lm_datasets helpers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the dataset loaders + code classifier (read-only import).
_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
from lm_datasets import (  # noqa: E402
    _is_code_request,
    _load_lmsys,
    _load_wildchat,
)

from langdetect import DetectorFactory, detect  # noqa: E402

DetectorFactory.seed = 0  # langdetect is non-deterministic without this


def classify_language(text: str) -> str:
    """ISO-639-1 code (e.g. en/zh/ru) or 'unknown' on failure/too-short."""
    if len(text.strip()) < 12:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--load-per-dataset", type=int, default=6000,
                    help="raw prompts to pull from EACH of wildchat/lmsys")
    ap.add_argument("--target-per-category", type=int, default=512)
    ap.add_argument("--top-langs", type=int, default=3)
    args = ap.parse_args()

    # Filters OFF: we want code + multilingual prompts to classify ourselves.
    # Each source is best-effort: lmsys is gated and may be unavailable
    # without HF auth, in which case we proceed with wildchat alone (it is
    # also multilingual and contains code, covering both axes).
    sources = [("wildchat", _load_wildchat), ("lmsys", _load_lmsys)]
    prompts: list[str] = []
    loaded_counts: dict[str, int] = {}
    for name, loader in sources:
        print(f"loading {name} (≤{args.load_per_dataset}) ...", flush=True)
        try:
            got = loader(
                split="train", num_prompts=args.load_per_dataset,
                exclude_code=False, conversation_only=True,
                english_only=False)
        except Exception as e:
            print(f"  ** {name} unavailable, skipping: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            loaded_counts[name] = 0
            continue
        loaded_counts[name] = len(got)
        prompts.extend(got)
    if not prompts:
        raise RuntimeError("no datasets could be loaded")
    print(f"loaded {len(prompts)} raw prompts {loaded_counts}", flush=True)

    # Classify each prompt → (lang, task). Dedup identical prompts.
    meta: list[dict] = []
    seen: set[str] = set()
    lang_counts: Counter[str] = Counter()
    for p in prompts:
        if p in seen:
            continue
        seen.add(p)
        lang = classify_language(p)
        if lang == "unknown":
            continue
        task = "code" if _is_code_request(p) else "dialogue"
        meta.append({"prompt": p, "lang": lang, "task": task})
        lang_counts[lang] += 1

    print("\nlanguage sample counts (top 10):")
    for lang, n in lang_counts.most_common(10):
        print(f"  {lang}: {n}")
    top_langs = [lang for lang, _ in lang_counts.most_common(args.top_langs)]
    print(f"\nselected top-{args.top_langs} languages: {top_langs}")

    # Bucket and write one cache per (lang, task).
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in meta:
        if m["lang"] in top_langs:
            buckets[(m["lang"], m["task"])].append(m["prompt"])

    pools_dir = args.out_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    print("\nwriting category pools:")
    for lang in top_langs:
        for task in ("code", "dialogue"):
            items = buckets.get((lang, task), [])
            n_avail = len(items)
            chosen = items[:args.target_per_category]
            cat = f"{lang}-{task}"
            cat_dir = pools_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            with (cat_dir / "processed_dataset.jsonl").open("w") as f:
                for p in chosen:
                    f.write(json.dumps({"prompt": p},
                                       ensure_ascii=False) + "\n")
            short = "" if n_avail >= args.target_per_category else (
                f"  ** SHORT: only {n_avail} < {args.target_per_category}")
            print(f"  {cat}: wrote {len(chosen)} (available {n_avail}){short}")
            summary.append({"category": cat, "lang": lang, "task": task,
                            "written": len(chosen), "available": n_avail})

    # Persist classification meta + a machine-readable summary.
    with (args.out_dir / "pool_meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (args.out_dir / "pool_summary.json").open("w") as f:
        json.dump({"top_langs": top_langs,
                   "target_per_category": args.target_per_category,
                   "source_counts": loaded_counts,
                   "lang_counts": dict(lang_counts.most_common()),
                   "categories": summary}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote pool_meta.jsonl ({len(meta)} rows) + pool_summary.json")


if __name__ == "__main__":
    main()
