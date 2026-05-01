#!/usr/bin/env python3
"""Stage 2: train LengthPredictorHead on pre-extracted hidden states.

Uses HuggingFace Accelerate for multi-GPU DDP so only the small head (not
the 35B backbone) needs to be distributed.

Run:
    # Single GPU:
    python3 length_predictor/train.py --features-dir length_predictor/features/koala

    # Multi-GPU (e.g. 4 GPUs):
    accelerate launch --num_processes 4 length_predictor/train.py \\
        --features-dir length_predictor/features/koala \\
        --output-dir length_predictor/checkpoints/koala
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from length_predictor.dataset import FeatureDataset
from length_predictor.model import LengthPredictorHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LengthPredictorHead.")
    p.add_argument("--features-dir", required=True,
                   help="Directory containing shard_*.pt files from prepare_features.py.")
    p.add_argument("--output-dir", default="length_predictor/checkpoints",
                   help="Directory to save checkpoints.")
    p.add_argument("--hidden-size", type=int, default=4096,
                   help="Hidden size of the backbone (4096 for Qwen3.5 default).")
    p.add_argument("--inner-size", type=int, default=256,
                   help="Inner dimension of each MLP head.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Tokens per training step (across all GPUs).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of data used for validation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Delta for Huber loss (applied on log1p-scaled targets).")
    return p.parse_args()


def huber_loss_log1p(
    pred: torch.Tensor, labels: torch.Tensor, delta: float
) -> torch.Tensor:
    """Huber loss with log1p-scaled targets."""
    target = torch.log1p(labels.float())
    return F.huber_loss(pred, target, delta=delta, reduction="mean")


def evaluate(
    model: LengthPredictorHead,
    loader: DataLoader,
    accelerator: Accelerator,
    huber_delta: float,
) -> dict[str, float]:
    model.eval()
    total_loss = total_sent = total_para = 0.0
    n_batches = 0

    with torch.no_grad():
        for hidden, sent_lbl, para_lbl in loader:
            sent_pred, para_pred = model(hidden)
            sent_loss = huber_loss_log1p(sent_pred, sent_lbl, huber_delta)
            para_loss = huber_loss_log1p(para_pred, para_lbl, huber_delta)
            total_loss += (sent_loss + para_loss).item()
            total_sent += sent_loss.item()
            total_para += para_loss.item()
            n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_sent_loss": total_sent / max(n_batches, 1),
        "val_para_loss": total_para / max(n_batches, 1),
    }


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = FeatureDataset(args.features_dir)
    if accelerator.is_main_process:
        print(dataset)

    val_size = max(1, math.floor(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # --- Model + Optimizer ---
    model = LengthPredictorHead(
        hidden_size=args.hidden_size, inner_size=args.inner_size
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for hidden, sent_lbl, para_lbl in train_loader:
            sent_pred, para_pred = model(hidden)
            sent_loss = huber_loss_log1p(sent_pred, sent_lbl, args.huber_delta)
            para_loss = huber_loss_log1p(para_pred, para_lbl, args.huber_delta)
            loss = sent_loss + para_loss

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_train_loss = epoch_loss / max(n_steps, 1)

        # Validation (run on unwrapped model to avoid DDP issues with no_grad).
        unwrapped = accelerator.unwrap_model(model)
        val_metrics = evaluate(unwrapped, val_loader, accelerator, args.huber_delta)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train={avg_train_loss:.4f} | "
                f"val={val_metrics['val_loss']:.4f} "
                f"(sent={val_metrics['val_sent_loss']:.4f}, "
                f"para={val_metrics['val_para_loss']:.4f})"
            )

            # Save epoch checkpoint.
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pt"
            unwrapped.save(ckpt_path)

            # Track best.
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                unwrapped.save(output_dir / "best_model.pt")
                print(f"  → New best val_loss={best_val_loss:.4f}, saved best_model.pt")

    if accelerator.is_main_process:
        print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
        print(f"Checkpoints in: {output_dir}")


if __name__ == "__main__":
    main()
