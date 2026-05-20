from __future__ import annotations

from torch import nn


class ConditioningMLP(nn.Module):
    """Map adapter-width bottleneck tokens into DiT conditioning width."""

    def __init__(
        self,
        *,
        adapter_dim: int,
        dit_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = max(adapter_dim, dit_hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dit_hidden_dim),
        )

    def forward(self, tokens):
        return self.net(tokens)

