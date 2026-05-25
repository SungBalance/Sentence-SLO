#!/usr/bin/env python3
"""Prepare dialogue text chunks for TTS duration measurement."""

from __future__ import annotations

import argparse
import random as _random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    CHUNK_COLUMNS,
    chunk_output_paths,
    chunk_text,
    has_code_like_content,
    slugify,
    word_count,
    write_csv,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class DialogueTurn:
    dataset_item_id: str
    turn_idx: int
    role: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a dialogue dataset and write sentence/paragraph text chunks."
    )
    parser.add_argument("--dataset-name", default="combine",
                        choices=["combine", "wildchat", "lmsys"])
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Override default split; None = dataset default.",
    )
    parser.add_argument("--max-dialogues", type=int, default=2048)
    # SSLO
    parser.add_argument("--conversation-only", action="store_true", default=True)
    # SSLO
    parser.add_argument(
        "--no-conversation-only",
        dest="conversation_only",
        action="store_false",
    )
    # SSLO
    parser.add_argument("--english-only", action="store_true", default=True)
    # SSLO
    parser.add_argument("--no-english-only", dest="english_only", action="store_false")
    # SSLO
    parser.add_argument("--exclude-code", action="store_true", default=True)
    # SSLO
    parser.add_argument("--no-exclude-code", dest="exclude_code", action="store_false")
    # SSLO
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="combine shuffle seed (matches run_sslo).",
    )
    # SSLO: chunk-level random subsample fraction. Applied AFTER chunk
    # construction so TTS profile sample count = round(total_chunks * frac).
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.5,
        help="Random fraction of chunks to keep per chunk_unit (default 0.5).",
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def load_dialogue_turns(
    *,
    dataset_name: str,
    split: str | None,
    max_dialogues: int,
    conversation_only: bool,
    english_only: bool,
    exclude_code: bool,
    seed: int,
) -> list[DialogueTurn]:
    if max_dialogues <= 0:
        raise ValueError("--max-dialogues must be positive.")

    # SSLO
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    # SSLO
    from lm_datasets import load_dialogues

    # SSLO
    dialogues = load_dialogues(
        dataset_name,
        split=split,
        max_dialogues=max_dialogues,
        conversation_only=conversation_only,
        english_only=english_only,
        exclude_code=exclude_code,
        seed=seed,
    )
    turns: list[DialogueTurn] = []
    for idx, dialogue in enumerate(dialogues):
        turns.extend(
            extract_turns(
                dataset_item_id=f"{dataset_name}_{idx}",
                messages=dialogue,
            )
        )

    if not turns:
        raise RuntimeError("No usable dialogue turns after filtering.")
    return turns


def extract_turns(
    *,
    dataset_item_id: str,
    messages: list[dict[str, Any]],
) -> list[DialogueTurn]:
    turns: list[DialogueTurn] = []
    for turn_idx, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "system":
            continue
        text = clean_text(message.get("content", ""))
        if not text or has_code_like_content(text):
            continue
        turns.append(
            DialogueTurn(
                dataset_item_id=dataset_item_id,
                turn_idx=turn_idx,
                role=role or "unknown",
                text=text,
            )
        )
    return turns


def clean_text(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").strip()
    text = "\n\n".join(part.strip() for part in text.split("\n\n") if part.strip())
    return text


def build_chunk_rows(
    *,
    turns: list[DialogueTurn],
    dataset_name: str,
    chunk_unit: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        pieces = [piece for piece in chunk_text(turn.text, chunk_unit) if piece.strip()]
        for chunk_idx, piece in enumerate(pieces):
            wc = word_count(piece)
            rows.append(
                {
                    "chunk_id": (
                        f"{turn.dataset_item_id}::{turn.turn_idx}::{chunk_unit}::{chunk_idx}"
                    ),
                    "dataset_name": dataset_name,
                    "dataset_item_id": turn.dataset_item_id,
                    # SSLO: load_sslo_chunks.transform_row 가 request_id /
                    # num_words / num_token 키를 요구. dataset_item_id 를
                    # request_id 로 alias 하고, num_words 는 word_count 와
                    # 같은 값, num_token 은 알 수 없으므로 0.
                    "request_id": str(turn.dataset_item_id),
                    "turn_idx": turn.turn_idx,
                    "role": turn.role,
                    "chunk_unit": chunk_unit,
                    "chunk_idx": chunk_idx,
                    "text": piece,
                    "word_count": wc,
                    "num_words": wc,
                    "num_token": 0,
                    "char_count": len(piece),
                }
            )
    return rows


def write_chunk_unit_output(
    *,
    output_root: Path,
    chunk_unit: str,
    rows: list[dict[str, Any]],
    dataset_name: str,
    num_turns: int,
) -> None:
    paths = chunk_output_paths(output_root / chunk_unit / "text_chunks")
    write_jsonl(paths.chunks_jsonl, rows)
    write_csv(paths.chunks_csv, rows, columns=CHUNK_COLUMNS)
    write_json(
        paths.summary_json,
        {
            "dataset_name": dataset_name,
            "dataset_slug": slugify(dataset_name),
            "chunk_unit": chunk_unit,
            "num_turns": num_turns,
            "num_chunks": len(rows),
            "mean_words_per_chunk": (
                sum(int(row["word_count"]) for row in rows) / len(rows) if rows else 0.0
            ),
        },
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    turns = load_dialogue_turns(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        max_dialogues=args.max_dialogues,
        conversation_only=args.conversation_only,
        english_only=args.english_only,
        exclude_code=args.exclude_code,
        seed=args.seed,
    )

    for chunk_unit in ("sentence", "paragraph"):
        rows = build_chunk_rows(
            turns=turns,
            dataset_name=args.dataset_name,
            chunk_unit=chunk_unit,
        )
        # SSLO: chunk-level random subsample. Reduces TTS profile sample
        # count without changing the source distribution (dialogues are
        # shuffled upstream; rows here are in turn order, so sample()
        # gives uniform coverage across dialogues).
        if 0 < args.sample_frac < 1.0 and rows:
            rng = _random.Random(args.seed)
            n_keep = max(1, int(round(len(rows) * args.sample_frac)))
            rows = rng.sample(rows, n_keep)
        write_chunk_unit_output(
            output_root=output_root,
            chunk_unit=chunk_unit,
            rows=rows,
            dataset_name=args.dataset_name,
            num_turns=len(turns),
        )
        print(
            f"[{chunk_unit}] wrote {len(rows)} chunk rows under "
            f"{output_root / chunk_unit / 'text_chunks'}"
        )


if __name__ == "__main__":
    main()
