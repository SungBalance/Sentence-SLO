#!/usr/bin/env python3
"""Prepare dialogue text chunks for TTS duration measurement."""

from __future__ import annotations

import argparse
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


ULTRACHAT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SUPPORTED_DATASETS = {
    "ultrachat": ULTRACHAT_DATASET_ID,
    ULTRACHAT_DATASET_ID: ULTRACHAT_DATASET_ID,
}


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
    parser.add_argument("--dataset-name", default=ULTRACHAT_DATASET_ID)
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--max-dialogues", type=int, default=256)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def normalize_dataset_name(dataset_name: str) -> str:
    try:
        return SUPPORTED_DATASETS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dialogue dataset {dataset_name!r}. Supported: {supported}."
        ) from exc


def load_dialogue_turns(
    *,
    dataset_name: str,
    split: str,
    max_dialogues: int,
) -> list[DialogueTurn]:
    dataset_id = normalize_dataset_name(dataset_name)
    if max_dialogues <= 0:
        raise ValueError("--max-dialogues must be positive.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required in the execution container. "
            "Install it before running this experiment."
        ) from exc

    dataset = load_dataset(dataset_id, split=split)
    turns: list[DialogueTurn] = []
    kept_dialogues = 0
    for row_idx, row in enumerate(dataset):
        messages = row.get("messages") or []
        dialogue_turns = extract_turns(
            dataset_item_id=str(row.get("prompt_id") or row.get("id") or f"row_{row_idx}"),
            messages=messages,
        )
        if not dialogue_turns:
            continue
        turns.extend(dialogue_turns)
        kept_dialogues += 1
        if kept_dialogues >= min(max_dialogues, len(dataset)):
            break

    if not turns:
        raise RuntimeError("No usable dialogue turns remained after filtering.")
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
            rows.append(
                {
                    "chunk_id": (
                        f"{turn.dataset_item_id}::{turn.turn_idx}::{chunk_unit}::{chunk_idx}"
                    ),
                    "dataset_name": dataset_name,
                    "dataset_item_id": turn.dataset_item_id,
                    "turn_idx": turn.turn_idx,
                    "role": turn.role,
                    "chunk_unit": chunk_unit,
                    "chunk_idx": chunk_idx,
                    "text": piece,
                    "word_count": word_count(piece),
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
    )

    for chunk_unit in ("sentence", "paragraph"):
        rows = build_chunk_rows(
            turns=turns,
            dataset_name=args.dataset_name,
            chunk_unit=chunk_unit,
        )
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
