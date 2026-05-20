from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    autocast_dtype: object | None


def get_precision(name: str) -> PrecisionConfig:
    normalized = name.lower()
    if normalized == "bf16":
        import torch

        return PrecisionConfig(name="bf16", autocast_dtype=torch.bfloat16)
    if normalized == "fp16":
        import torch

        return PrecisionConfig(name="fp16", autocast_dtype=torch.float16)
    if normalized in {"fp32", "float32"}:
        return PrecisionConfig(name="fp32", autocast_dtype=None)
    raise ValueError(f"Unsupported precision: {name}")


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

