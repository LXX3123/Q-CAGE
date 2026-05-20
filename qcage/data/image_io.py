from __future__ import annotations

from pathlib import Path


def load_rgb_image(path: str | Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

