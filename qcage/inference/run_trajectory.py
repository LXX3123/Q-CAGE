from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Run Q-CAGE over a JSONL trajectory/eval file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    device = default_device()

    input_jsonl = args.input_jsonl or config.get("inference", {}).get("input_jsonl")
    if input_jsonl is None:
        raise ValueError("Provide --input-jsonl or inference.input_jsonl in the config")

    output_dir = Path(args.output_dir or config.get("inference", {}).get("output_dir", "outputs/inference"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = InterleavedJsonlDataset(
        input_jsonl,
        image_root=config["data"].get("image_root", "."),
        use_cached_features=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_qcage_batch,
    )

    adapter = build_adapter_from_config(config).to(device)
    checkpoint_path = config.get("inference", {}).get("checkpoint")
    if checkpoint_path:
        load_checkpoint(checkpoint_path, adapter, map_location=device)
        logger.info("Loaded adapter checkpoint: %s", checkpoint_path)
    adapter.eval()
    bridge = build_generator_bridge(config, device=device)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for batch in tqdm(dataloader, desc="trajectory"):
            batch = move_to_device(batch, device)
            output = adapter(
                hidden_states=batch["hidden_states"],
                source_masks=batch["source_masks"],
                valid_token_masks=batch.get("valid_token_masks"),
            )
            generated = bridge.generate(
                output.condition_tokens,
                batch=batch,
                samples=batch["samples"],
                output_dir=str(output_dir),
                num_steps=config.get("inference", {}).get("num_steps", 30),
                guidance_scale=config.get("inference", {}).get("guidance_scale", 3.5),
            )
            record = {
                "sample_id": batch["samples"][0].sample_id,
                "generation": generated,
            }
            handle.write(json.dumps(record) + "\n")

    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
