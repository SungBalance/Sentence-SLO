"""Remove image-generation prompts from the cached dataset.

Image-gen prompts (Midjourney / Stable Diffusion / DALL-E / SDXL etc.)
ask the model to produce long, dense comma-separated visual specs that
inflate chunk token-to-word ratios, causing deadline misses.
"""
import json
import re
import shutil
import sys
from pathlib import Path

CACHE = Path("/workspace/mlsys/exp/tools/dataset_cache/processed_dataset.jsonl")

# Case-insensitive substring match against the raw user prompt (chat
# template stripped before matching).
KEYWORDS = (
    "midjourney",
    "stable diffusion",
    "stablediffusion",
    "dall-e",
    "dalle",
    "dall·e",
    "sdxl",
    "image prompt",
    "img prompt",
    "ai image",
    "image generation",
    "text-to-image",
    "txt2img",
    "img2img",
    "imagine prompt",
    "/imagine",
    "prompt for generative ai",
    "prompt generator",
    "art prompt",
    "image with the following",
    "describe an image",
    "prompt for the ai to visualize",
    "for a generative ai",
)


_CHAT_RE = re.compile(
    r"<\|im_start\|>\s*user\s*(.*?)\s*<\|im_end\|>", re.DOTALL)


def strip_chat(p: str) -> str:
    m = _CHAT_RE.search(p)
    return (m.group(1) if m else p).strip().lower()


def is_image_prompt(p: str) -> str | None:
    raw = strip_chat(p)
    for kw in KEYWORDS:
        if kw in raw:
            return kw
    return None


def main():
    apply = "--apply" in sys.argv
    with CACHE.open() as f:
        prompts = [json.loads(L)["prompt"] for L in f if L.strip()]
    N = len(prompts)
    print(f"cache size: {N}")

    bad = []
    for i, p in enumerate(prompts):
        match = is_image_prompt(p)
        if match:
            bad.append((i, match, p))

    print(f"matched: {len(bad)}")
    print()
    # Sample
    for i, kw, p in bad[:10]:
        snippet = strip_chat(p)[:180]
        print(f"  [{kw}] {snippet!r}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")

    if not apply:
        print(f"\n(dry-run; pass --apply to remove {len(bad)} prompts)")
        return

    backup = CACHE.with_suffix(CACHE.suffix + ".bak_pre_image_filter")
    if not backup.exists():
        shutil.copy2(CACHE, backup)
        print(f"backup: {backup}")
    bad_set = {p for _, _, p in bad}
    kept = [p for p in prompts if p not in bad_set]
    tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
    with tmp.open("w") as f:
        for p in kept:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    tmp.replace(CACHE)
    print(f"REWROTE: {len(kept)} kept ({N - len(kept)} removed)")


if __name__ == "__main__":
    main()
