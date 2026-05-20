from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qcage.data.schema import DEFAULT_SOURCE_NAMES


def _pad_token_tensors(tensors: Sequence[Any], pad_value: float = 0.0):
    import torch

    max_len = max(tensor.shape[0] for tensor in tensors)
    hidden_dim = tensors[0].shape[-1]
    batch = tensors[0].new_full((len(tensors), max_len, hidden_dim), pad_value)
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool)
    for row, tensor in enumerate(tensors):
        length = tensor.shape[0]
        batch[row, :length] = tensor
        mask[row, :length] = True
    return batch, mask


def _pad_bool_masks(masks: Sequence[Any], max_len: int):
    import torch

    batch = torch.zeros((len(masks), max_len), dtype=torch.bool)
    for row, mask in enumerate(masks):
        length = mask.shape[0]
        batch[row, :length] = mask.bool()
    return batch


def _stack_if_present(features: list[dict[str, Any]], key: str):
    import torch

    values = [feature.get(key) for feature in features]
    if any(value is None for value in values):
        return None
    try:
        return torch.stack(values)
    except RuntimeError:
        return values


def collate_qcage_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [item["sample"] for item in batch]
    if "features" not in batch[0]:
        return {"samples": samples}

    features = [item["features"] for item in batch]
    layer_ids = sorted(int(layer) for layer in features[0]["hidden_states"].keys())

    hidden_states = {}
    valid_token_masks = {}
    for layer in layer_ids:
        layer_key = layer if layer in features[0]["hidden_states"] else str(layer)
        tensors = [feature["hidden_states"][layer_key] for feature in features]
        padded, valid_mask = _pad_token_tensors(tensors)
        hidden_states[layer] = padded
        valid_token_masks[layer] = valid_mask

    max_len = max(tensor.shape[1] for tensor in hidden_states.values())
    source_names = list(features[0].get("source_masks", {}).keys()) or DEFAULT_SOURCE_NAMES
    source_masks = {}
    for source in source_names:
        masks = [feature["source_masks"].get(source) for feature in features]
        if any(mask is None for mask in masks):
            continue
        source_masks[source] = _pad_bool_masks(masks, max_len)

    result: dict[str, Any] = {
        "samples": samples,
        "hidden_states": hidden_states,
        "source_masks": source_masks,
        "valid_token_masks": valid_token_masks,
    }

    for key in [
        "prompt_embeds",
        "pooled_prompt_embeds",
        "target_latents",
        "noise_latents",
        "noisy_latents",
        "timesteps",
        "velocity_target",
        "txt_ids",
        "img_ids",
        "guidance",
    ]:
        value = _stack_if_present(features, key)
        if value is not None:
            result[key] = value

    return result
