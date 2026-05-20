from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~memory_mask.bool()
        attended, _ = self.attn(
            query=self.q_norm(queries),
            key=self.memory_norm(memory),
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        queries = queries + self.ffn(queries)
        return queries


class QueryBottleneck(nn.Module):
    """Fixed-size learnable query bank with cross-attention over Q-CAGE memory."""

    def __init__(
        self,
        *,
        num_queries: int,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        self.query_bank = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        queries = self.query_bank.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, memory_mask)
        return self.out_norm(queries)


class MeanPoolBottleneck(nn.Module):
    """Global pooled ablation that keeps output shape compatible with the generator."""

    def __init__(self, *, num_queries: int, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_offsets = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        weights = memory_mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (memory * weights.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.mlp(pooled)
        return pooled.unsqueeze(1) + self.query_offsets.unsqueeze(0)

