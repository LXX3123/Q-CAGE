from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qcage.data.dataset import InterleavedJsonlDataset
from qcage.utils.config import load_config


QWEN_KEYS = ["hidden_states", "source_masks"]
FLUX_KEYS = ["prompt_embeds", "noisy_latents", "velocity_target", "timesteps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Q-CAGE cached feature files")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--require-flux", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _mask_counts(feature: dict) -> dict[str, int]:
    masks = feature.get("source_masks", {})
    return {name: int(mask.bool().sum().item()) for name, mask in masks.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_jsonl = args.input_jsonl or config["data"]["train_jsonl"]
    dataset = InterleavedJsonlDataset(
        input_jsonl,
        image_root=config["data"].get("image_root", "."),
        use_cached_features=False,
    )
    limit = min(args.max_samples or len(dataset), len(dataset))
    errors: list[str] = []
    reports: list[dict] = []

    for index in range(limit):
        sample = dataset[index]["sample"]
        if not sample.feature_path:
            errors.append(f"{sample.sample_id}: missing feature_path")
            continue
        path = Path(sample.feature_path)
        if not path.exists():
            errors.append(f"{sample.sample_id}: feature_path does not exist: {path}")
            continue
        feature = torch.load(path, map_location="cpu")
        missing = [key for key in QWEN_KEYS if key not in feature]
        if args.require_flux:
            missing.extend(key for key in FLUX_KEYS if key not in feature)
        if missing:
            errors.append(f"{sample.sample_id}: missing keys {missing}")
            continue
        reports.append(
            {
                "sample_id": sample.sample_id,
                "layers": sorted(int(layer) for layer in feature["hidden_states"].keys()),
                "mask_counts": _mask_counts(feature),
                "has_flux": all(key in feature for key in FLUX_KEYS),
            }
        )

    print(json.dumps({"checked": limit, "errors": errors, "examples": reports[:5]}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

