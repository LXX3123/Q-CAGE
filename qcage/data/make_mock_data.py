from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qcage.utils.config import load_config
from qcage.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny mock Q-CAGE dataset")
    parser.add_argument("--config", default="configs/mock_tiny.yaml")
    parser.add_argument("--output-root", default="data/mock")
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=12)
    return parser.parse_args()


def _feature(config: dict, num_tokens: int) -> dict:
    adapter_cfg = config["model"]["adapter"]
    hidden_dim = int(adapter_cfg["vlm_hidden_dim"])
    layers = [int(layer) for layer in adapter_cfg.get("selected_layers", [18, 24, 30])]

    hidden_states = {layer: torch.randn(num_tokens, hidden_dim) for layer in layers}
    masks = {
        "query": torch.zeros(num_tokens, dtype=torch.bool),
        "history_image": torch.zeros(num_tokens, dtype=torch.bool),
        "answer_text": torch.zeros(num_tokens, dtype=torch.bool),
    }
    split1 = max(1, num_tokens // 3)
    split2 = max(split1 + 1, 2 * num_tokens // 3)
    masks["query"][:split1] = True
    masks["history_image"][split1:split2] = True
    masks["answer_text"][split2:] = True

    return {
        "hidden_states": hidden_states,
        "source_masks": masks,
        "target_latents": torch.randn(4, 8, 8),
        "velocity_target": torch.randn(4, 8, 8),
    }


def _write_split(
    *,
    config: dict,
    root: Path,
    split: str,
    count: int,
    num_tokens: int,
) -> Path:
    feature_dir = root / "features" / split
    feature_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            sample_id = f"{split}_{index:04d}"
            feature_path = feature_dir / f"{sample_id}.pt"
            torch.save(_feature(config, num_tokens), feature_path)
            record = {
                "sample_id": sample_id,
                "history": [
                    {
                        "role": "user",
                        "text": "Create a recurring traveler in a rainy old town.",
                    },
                    {
                        "role": "assistant",
                        "text": "The traveler is near the station.",
                    },
                ],
                "query": {
                    "role": "user",
                    "text": "Move the same traveler to the open corner pub.",
                },
                "target_image": None,
                "answer_text": "The open corner pub should be the next destination.",
                "feature_path": str(feature_path),
            }
            handle.write(json.dumps(record) + "\n")
    return jsonl_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 7)))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    train_path = _write_split(
        config=config,
        root=root,
        split="train",
        count=args.train_size,
        num_tokens=args.num_tokens,
    )
    val_path = _write_split(
        config=config,
        root=root,
        split="val",
        count=args.val_size,
        num_tokens=args.num_tokens,
    )
    print(json.dumps({"train_jsonl": str(train_path), "val_jsonl": str(val_path)}, indent=2))


if __name__ == "__main__":
    main()

