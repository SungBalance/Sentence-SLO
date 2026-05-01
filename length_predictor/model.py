"""LengthPredictorHead: small MLP predicting remaining tokens to the next boundary.

Attaches to the last hidden layer of Qwen3.5 (hidden_size=4096 by default).
Two independent heads share the same input hidden state:
  - sentence_head: remaining tokens until next sentence boundary
  - paragraph_head: remaining tokens until next paragraph boundary

Both heads predict log1p(remaining_tokens) during training.
At inference, expm1(output) gives the raw token count estimate.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LengthPredictorHead(nn.Module):
    def __init__(self, hidden_size: int = 4096, inner_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.inner_size = inner_size

        self.sentence_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, 1),
        )
        self.paragraph_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [..., hidden_size]
        Returns:
            (sentence_pred, paragraph_pred) each shape [...], log1p-scaled.
        """
        s = self.sentence_head(hidden_states).squeeze(-1)
        p = self.paragraph_head(hidden_states).squeeze(-1)
        return s, p

    def predict(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: returns raw token count estimates (expm1 of output)."""
        with torch.no_grad():
            s, p = self.forward(hidden_states)
        return torch.expm1(s).clamp(min=0), torch.expm1(p).clamp(min=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden_size": self.hidden_size,
                "inner_size": self.inner_size,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "LengthPredictorHead":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(hidden_size=ckpt["hidden_size"], inner_size=ckpt["inner_size"])
        model.load_state_dict(ckpt["state_dict"])
        return model
