from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def _fit_mask(mask: torch.Tensor, token_count: int) -> torch.Tensor:
    if mask.shape[1] == token_count:
        return mask.bool()
    if mask.shape[1] > token_count:
        return mask[:, :token_count].bool()
    pad = torch.zeros(
        (mask.shape[0], token_count - mask.shape[1]),
        dtype=torch.bool,
        device=mask.device,
    )
    return torch.cat([mask.bool(), pad], dim=1)


def ensure_nonempty_memory(memory: torch.Tensor, memory_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Avoid all-masked cross-attention rows, which can create NaNs."""
    empty_rows = ~memory_mask.any(dim=1)
    if empty_rows.any():
        memory = memory.clone()
        memory_mask = memory_mask.clone()
        memory[empty_rows, 0] = 0
        memory_mask[empty_rows, 0] = True
    return memory, memory_mask


class SourceDepthProjector(nn.Module):
    """Normalize and project one source at one VLM depth into adapter width."""

    def __init__(self, input_dim: int, adapter_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, adapter_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MultiDepthTriSourceProjector(nn.Module):
    """Build Q-CAGE memory M from selected layer/source hidden states."""

    def __init__(
        self,
        *,
        vlm_hidden_dim: int,
        adapter_dim: int,
        selected_layers: Sequence[int],
        source_names: Sequence[str],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.selected_layers = [int(layer) for layer in selected_layers]
        self.source_names = list(source_names)
        self.projectors = nn.ModuleDict()
        for layer in self.selected_layers:
            for source in self.source_names:
                self.projectors[self._key(layer, source)] = SourceDepthProjector(
                    vlm_hidden_dim,
                    adapter_dim,
                    dropout=dropout,
                )

    @staticmethod
    def _key(layer: int, source: str) -> str:
        return f"layer_{int(layer)}__{source}"

    def forward(
        self,
        hidden_states: Mapping[int, torch.Tensor],
        source_masks: Mapping[str, torch.Tensor],
        valid_token_masks: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memories: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for layer in self.selected_layers:
            if layer not in hidden_states:
                available = ", ".join(str(key) for key in sorted(hidden_states.keys()))
                raise KeyError(f"Missing hidden state for layer {layer}; available: {available}")
            hidden = hidden_states[layer]
            token_count = hidden.shape[1]
            valid_mask = None
            if valid_token_masks is not None and layer in valid_token_masks:
                valid_mask = _fit_mask(valid_token_masks[layer], token_count)

            for source in self.source_names:
                if source not in source_masks:
                    continue
                projected = self.projectors[self._key(layer, source)](hidden)
                source_mask = _fit_mask(source_masks[source], token_count)
                if valid_mask is not None:
                    source_mask = source_mask & valid_mask
                memories.append(projected)
                masks.append(source_mask)

        if not memories:
            raise ValueError("No source memories were produced; check selected_layers and source_names")

        memory = torch.cat(memories, dim=1)
        memory_mask = torch.cat(masks, dim=1)
        return ensure_nonempty_memory(memory, memory_mask)

