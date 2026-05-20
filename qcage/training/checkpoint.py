from __future__ import annotations

from pathlib import Path
from typing import Any


def unwrap_model(model):
    return getattr(model, "module", model)


def save_checkpoint(
    *,
    output_dir: str | Path,
    model,
    optimizer,
    step: int,
    epoch: int,
    config: dict[str, Any],
    name: str | None = None,
) -> Path:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = name or f"step_{step:08d}.pt"
    path = output_dir / ckpt_name
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        path,
    )
    latest = output_dir / "latest.pt"
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "epoch": epoch,
            "config": config,
        },
        latest,
    )
    return path


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

