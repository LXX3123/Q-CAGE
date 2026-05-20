from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from qcage.data.schema import DEFAULT_SOURCE_NAMES
from qcage.models.conditioning_mlp import ConditioningMLP
from qcage.models.projectors import MultiDepthTriSourceProjector
from qcage.models.query_bottleneck import MeanPoolBottleneck, QueryBottleneck


@dataclass
class QCAgeAdapterConfig:
    vlm_hidden_dim: int
    adapter_dim: int = 1024
    dit_hidden_dim: int = 4096
    selected_layers: Sequence[int] = (18, 24, 30)
    source_names: Sequence[str] = tuple(DEFAULT_SOURCE_NAMES)
    num_queries: int = 96
    num_heads: int = 16
    cross_attention_layers: int = 2
    dropout: float = 0.0
    bottleneck: str = "query"

    @classmethod
    def from_dict(cls, data: Mapping) -> "QCAgeAdapterConfig":
        return cls(
            vlm_hidden_dim=int(data["vlm_hidden_dim"]),
            adapter_dim=int(data.get("adapter_dim", 1024)),
            dit_hidden_dim=int(data.get("dit_hidden_dim", 4096)),
            selected_layers=tuple(int(layer) for layer in data.get("selected_layers", [18, 24, 30])),
            source_names=tuple(data.get("source_names", DEFAULT_SOURCE_NAMES)),
            num_queries=int(data.get("num_queries", 96)),
            num_heads=int(data.get("num_heads", 16)),
            cross_attention_layers=int(data.get("cross_attention_layers", 2)),
            dropout=float(data.get("dropout", 0.0)),
            bottleneck=str(data.get("bottleneck", "query")),
        )


@dataclass
class QCAgeAdapterOutput:
    condition_tokens: torch.Tensor
    bottleneck_tokens: torch.Tensor
    memory: torch.Tensor
    memory_mask: torch.Tensor


class QCAgeAdapter(nn.Module):
    """Trainable Q-CAGE interface between a frozen VLM and a frozen DiT generator."""

    def __init__(self, config: QCAgeAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.source_projectors = MultiDepthTriSourceProjector(
            vlm_hidden_dim=config.vlm_hidden_dim,
            adapter_dim=config.adapter_dim,
            selected_layers=config.selected_layers,
            source_names=config.source_names,
            dropout=config.dropout,
        )
        if config.bottleneck == "query":
            self.bottleneck = QueryBottleneck(
                num_queries=config.num_queries,
                dim=config.adapter_dim,
                num_heads=config.num_heads,
                num_layers=config.cross_attention_layers,
                dropout=config.dropout,
            )
        elif config.bottleneck == "mean_pool":
            self.bottleneck = MeanPoolBottleneck(
                num_queries=config.num_queries,
                dim=config.adapter_dim,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unsupported bottleneck: {config.bottleneck}")

        self.conditioning_mlp = ConditioningMLP(
            adapter_dim=config.adapter_dim,
            dit_hidden_dim=config.dit_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        hidden_states: Mapping[int, torch.Tensor],
        source_masks: Mapping[str, torch.Tensor],
        valid_token_masks: Mapping[int, torch.Tensor] | None = None,
    ) -> QCAgeAdapterOutput:
        memory, memory_mask = self.source_projectors(
            hidden_states=hidden_states,
            source_masks=source_masks,
            valid_token_masks=valid_token_masks,
        )
        bottleneck_tokens = self.bottleneck(memory, memory_mask)
        condition_tokens = self.conditioning_mlp(bottleneck_tokens)
        return QCAgeAdapterOutput(
            condition_tokens=condition_tokens,
            bottleneck_tokens=bottleneck_tokens,
            memory=memory,
            memory_mask=memory_mask,
        )

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]


def build_adapter_from_config(config: Mapping) -> QCAgeAdapter:
    adapter_cfg = QCAgeAdapterConfig.from_dict(config["model"]["adapter"])
    return QCAgeAdapter(adapter_cfg)

