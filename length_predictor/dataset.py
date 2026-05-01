"""FeatureDataset: loads pre-extracted hidden-state shards produced by prepare_features.py.

Each shard file (shard_NNN.pt) contains:
    {
        "hidden_states":     FloatTensor [N, hidden_size],
        "sentence_labels":   LongTensor  [N],
        "paragraph_labels":  LongTensor  [N],
    }

All shards in the directory are concatenated into a single flat dataset.
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, features_dir: str | Path) -> None:
        features_dir = Path(features_dir)
        shard_paths = sorted(glob.glob(str(features_dir / "shard_*.pt")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.pt files found in {features_dir}. "
                "Run prepare_features.py first."
            )

        hidden_list, sent_list, para_list = [], [], []
        for p in shard_paths:
            shard = torch.load(p, map_location="cpu", weights_only=True)
            hidden_list.append(shard["hidden_states"].float())
            sent_list.append(shard["sentence_labels"].long())
            para_list.append(shard["paragraph_labels"].long())

        self.hidden_states = torch.cat(hidden_list, dim=0)      # [N, H]
        self.sentence_labels = torch.cat(sent_list, dim=0)       # [N]
        self.paragraph_labels = torch.cat(para_list, dim=0)      # [N]

        assert self.hidden_states.shape[0] == self.sentence_labels.shape[0]
        assert self.hidden_states.shape[0] == self.paragraph_labels.shape[0]

    def __len__(self) -> int:
        return self.hidden_states.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.hidden_states[idx],
            self.sentence_labels[idx],
            self.paragraph_labels[idx],
        )

    @property
    def hidden_size(self) -> int:
        return self.hidden_states.shape[1]

    def __repr__(self) -> str:
        return (
            f"FeatureDataset(n={len(self)}, "
            f"hidden_size={self.hidden_size}, "
            f"sent_boundary_rate={float((self.sentence_labels == 0).float().mean()):.3f}, "
            f"para_boundary_rate={float((self.paragraph_labels == 0).float().mean()):.3f})"
        )
