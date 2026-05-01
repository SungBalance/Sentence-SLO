"""Compute per-token boundary labels from a sequence of generated token IDs.

For each token position i in the generated sequence, the label is:
    remaining_tokens = (next_boundary_position - i)

where next_boundary_position is the index of the token AT which a
sentence/paragraph boundary is first detected (i.e. the final token of that chunk).
If no boundary exists after position i, remaining = seq_len - i.

Boundary detection reuses SentenceChunkDetector and ParagraphChunkDetector from
vllm.sslo.slo_state, which match the same logic used in benchmark.py streaming.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing vllm from the local checkout when running inside the container.
_VLLM_SRC = Path(__file__).resolve().parents[1] / "vllm"
if str(_VLLM_SRC) not in sys.path:
    sys.path.insert(0, str(_VLLM_SRC))

from vllm.sslo.slo_state import ParagraphChunkDetector, SentenceChunkDetector


def compute_boundary_labels(
    generated_token_ids: list[int],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Return (sentence_labels, paragraph_labels) for each generated token position.

    Each label[i] = number of remaining tokens until the next boundary (inclusive
    of the boundary token itself), so label=0 means the current token IS the
    boundary.  If no boundary exists after position i, label = seq_len - i - 1.

    Args:
        generated_token_ids: Token IDs of the *generated* portion only (no prompt).
        tokenizer: HuggingFace tokenizer used to decode tokens individually.

    Returns:
        Pair of integer lists, each of length len(generated_token_ids).
    """
    seq_len = len(generated_token_ids)
    sent_detector = SentenceChunkDetector()
    para_detector = ParagraphChunkDetector()

    # Find boundary positions by simulating the streaming accumulator.
    sent_boundaries: list[int] = []
    para_boundaries: list[int] = []
    sent_acc = ""
    para_acc = ""

    for i, tok_id in enumerate(generated_token_ids):
        token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        sent_acc += token_text
        para_acc += token_text

        if sent_detector.find_boundary(sent_acc) is not None:
            sent_boundaries.append(i)
            sent_acc = ""

        if para_detector.find_boundary(para_acc) is not None:
            para_boundaries.append(i)
            para_acc = ""

    sent_labels = _labels_from_boundaries(sent_boundaries, seq_len)
    para_labels = _labels_from_boundaries(para_boundaries, seq_len)
    return sent_labels, para_labels


def _labels_from_boundaries(boundaries: list[int], seq_len: int) -> list[int]:
    """For each position i, compute distance to the next boundary >= i.

    Default (no future boundary): distance to the last token (seq_len - 1 - i),
    treating end-of-sequence as an implicit boundary.
    """
    # Default: distance to end-of-sequence for each position.
    labels = [seq_len - 1 - i for i in range(seq_len)]
    # Walk boundaries in reverse so each position gets its nearest future boundary.
    for b in reversed(boundaries):
        for i in range(b + 1):
            labels[i] = b - i
    return labels
