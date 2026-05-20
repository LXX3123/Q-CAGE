from __future__ import annotations

from collections.abc import Iterable


def build_optimizer(parameters: Iterable, config: dict):
    import torch

    train_cfg = config["training"]
    return torch.optim.AdamW(
        parameters,
        lr=float(train_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        eps=float(train_cfg.get("eps", 1.0e-8)),
    )

