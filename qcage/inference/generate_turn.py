from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qcage.data.collate import collate_qcage_batch
from qcage.data.dataset import InterleavedJsonlDataset
from qcage.models.flux_bridge import build_generator_bridge
from qcage.models.qcage_adapter import build_adapter_from_config
from qcage.training.checkpoint import load_checkpoint
from qcage.utils.config import load_config
from qcage.utils.device import default_device
from qcage.utils.logging import setup_logging
from qcage.utils.tensor import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Q-CAGE turn from a JSONL sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    device = default_device()

    dataset = InterleavedJsonlDataset(
        args.input_jsonl,
        image_root=config["data"].get("image_root", "."),
        use_cached_features=True,
    )
    batch = collate_qcage_batch([dataset[args.index]])
    batch = move_to_device(batch, device)

    adapter = build_adapter_from_config(config).to(device)
    checkpoint_path = config.get("inference", {}).get("checkpoint")
    if checkpoint_path:
        load_checkpoint(checkpoint_path, adapter, map_location=device)
        logger.info("Loaded adapter checkpoint: %s", checkpoint_path)
    adapter.eval()

    bridge = build_generator_bridge(config, device=device)
    output = adapter(
        hidden_states=batch["hidden_states"],
        source_masks=batch["source_masks"],
        valid_token_masks=batch.get("valid_token_masks"),
    )
    generated = bridge.generate(
        output.condition_tokens,
        batch=batch,
        samples=batch["samples"],
        num_steps=config.get("inference", {}).get("num_steps", 30),
        guidance_scale=config.get("inference", {}).get("guidance_scale", 3.5),
    )

    result = {
        "sample_id": batch["samples"][0].sample_id,
        "generation": generated,
    }
    output_json = args.output_json or config.get("inference", {}).get("output_json")
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", path)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
