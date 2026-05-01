#!/usr/bin/env python3
"""Stage 1: generate text with Qwen3.5 and extract last-layer hidden states + labels.

For each prompt:
  1. Generate text (greedy, up to --max-new-tokens)
  2. Run teacher-forced forward pass on (prompt + generated) to get last hidden states
  3. Compute sentence/paragraph boundary labels for each generated token
  4. Save a shard: {hidden_states, sentence_labels, paragraph_labels}

Run inside sk-sslo container:
    HF_HOME=/cache HF_HUB_CACHE=/cache/hub python3 length_predictor/prepare_features.py \\
        --model Qwen/Qwen3.5-7B \\
        --dataset koala --num-prompts 200 \\
        --output-dir length_predictor/features/koala
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HF_HUB_CACHE", "/cache/hub")

# Allow importing local modules without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "vllm"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp.tools.lm_datasets import load_prompts
from length_predictor.label_utils import compute_boundary_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract hidden states and boundary labels.")
    p.add_argument("--model", default="Qwen/Qwen3.5-7B",
                   help="HuggingFace model ID or local path.")
    p.add_argument("--dataset", default="koala",
                   help="Dataset alias (koala, wildchat, lmsys, or HF ID).")
    p.add_argument("--split", default=None,
                   help="Dataset split (defaults to dataset's natural split).")
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Number of prompts to process. None = all available.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate per prompt.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per forward-pass batch.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write shard_NNN.pt files.")
    return p.parse_args()


def load_model_and_tokenizer(model_id: str):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {model_id}  (bfloat16, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def process_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[dict]:
    """
    For each prompt:
    1. Tokenize with left-padding (needed for batched generation).
    2. Greedy-generate up to max_new_tokens.
    3. Teacher-forced forward pass on full sequence → last hidden states.
    4. Compute boundary labels for generated portion.

    Returns list of dicts with keys: hidden_states, sentence_labels, paragraph_labels.
    """
    # Tokenize with left-padding so all sequences in the batch align on the right.
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    # Move to the first device of the model (device_map="auto" shards the model).
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    attention_mask = attention_mask.to(first_device)

    # Step 1: Generate text (greedy).
    gen_out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # gen_out: [B, prompt_len + gen_len] (right-padded with generated tokens)

    # Step 2: Teacher-forced forward pass on full sequence.
    full_attention_mask = torch.ones(
        gen_out.shape, dtype=torch.long, device=first_device
    )
    tf_out = model(
        gen_out,
        attention_mask=full_attention_mask,
        output_hidden_states=True,
    )
    # last hidden state: [B, total_seq_len, hidden_size]
    last_hidden = tf_out.hidden_states[-1].float().cpu()

    results = []
    for b_idx, prompt_len in enumerate(prompt_lengths):
        full_ids = gen_out[b_idx].tolist()          # full sequence (prompt + generated)
        generated_ids = full_ids[prompt_len:]       # generated portion only
        if not generated_ids:
            continue

        # Hidden states for the generated portion (positions prompt_len .. end).
        gen_hidden = last_hidden[b_idx, prompt_len:prompt_len + len(generated_ids)]  # [G, H]

        # Boundary labels.
        sent_labels, para_labels = compute_boundary_labels(generated_ids, tokenizer)

        results.append({
            "hidden_states": gen_hidden,
            "sentence_labels": torch.tensor(sent_labels, dtype=torch.long),
            "paragraph_labels": torch.tensor(para_labels, dtype=torch.long),
        })

    return results


def save_shard(records: list[dict], path: Path) -> None:
    hidden = torch.cat([r["hidden_states"] for r in records], dim=0)
    sent = torch.cat([r["sentence_labels"] for r in records], dim=0)
    para = torch.cat([r["paragraph_labels"] for r in records], dim=0)
    torch.save(
        {"hidden_states": hidden, "sentence_labels": sent, "paragraph_labels": para},
        path,
    )
    print(
        f"  Saved {path.name}: {hidden.shape[0]} tokens, "
        f"hidden={list(hidden.shape)}, "
        f"sent_bdry={int((sent == 0).sum())}, "
        f"para_bdry={int((para == 0).sum())}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_raw = load_prompts(
        args.dataset, split=args.split, num_prompts=args.num_prompts
    )
    print(f"Loaded {len(prompts_raw)} prompts from '{args.dataset}'")

    model, tokenizer = load_model_and_tokenizer(args.model)

    # Apply chat template to each prompt.
    prompts = [apply_chat_template(tokenizer, p) for p in prompts_raw]

    shard_idx = 0
    batch_records: list[dict] = []

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        print(f"Processing prompts {i}..{i + len(batch) - 1} / {len(prompts)}")
        try:
            records = process_batch(model, tokenizer, batch, args.max_new_tokens)
        except Exception as exc:
            print(f"  WARNING: batch failed ({exc}), skipping.")
            continue

        batch_records.extend(records)

        # Write a shard after every 16 batches to limit peak memory.
        if len(batch_records) >= args.batch_size * 16:
            shard_path = output_dir / f"shard_{shard_idx:03d}.pt"
            save_shard(batch_records, shard_path)
            shard_idx += 1
            batch_records = []

    if batch_records:
        shard_path = output_dir / f"shard_{shard_idx:03d}.pt"
        save_shard(batch_records, shard_path)

    print(f"\nDone. {shard_idx + (1 if batch_records else 0)} shard(s) in {output_dir}")


if __name__ == "__main__":
    main()
