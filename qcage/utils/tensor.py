from __future__ import annotations

from typing import Mapping


def move_to_device(batch, device: str):
    try:
        import torch
    except Exception:
        return batch

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)
    return batch

