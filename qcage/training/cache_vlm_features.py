from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from qcage.data.dataset import InterleavedJsonlDataset
from qcage.models.qwen_vlm import build_vlm_from_config
from qcage.utils.config import load_config
from qcage.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen VLM hidden states for Q-CAGE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-dir", default="cached_features")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--mock", action="store_true", help="Create synthetic feature caches for pipeline tests")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _mock_feature(config: dict) -> dict:
    adapter_cfg = config["model"]["adapter"]
    num_tokens = 128
    hidden_dim = int(adapter_cfg["vlm_hidden_dim"])
    layers = [int(layer) for layer in adapter_cfg.get("selected_layers", [18, 24, 30])]
    hidden_states = {layer: torch.randn(num_tokens, hidden_dim) for layer in layers}
    source_masks = {}
    thirds = num_tokens // 3
    source_masks["query"] = torch.zeros(num_tokens, dtype=torch.bool)
    source_masks["query"][:thirds] = True
    source_masks["history_image"] = torch.zeros(num_tokens, dtype=torch.bool)
    source_masks["history_image"][thirds : 2 * thirds] = True
    source_masks["answer_text"] = torch.zeros(num_tokens, dtype=torch.bool)
    source_masks["answer_text"][2 * thirds :] = True
    return {
        "hidden_states": hidden_states,
        "source_masks": source_masks,
        "target_latents": torch.randn(4, 32, 32),
        "velocity_target": torch.randn(4, 32, 32),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = args.input_jsonl or config["data"]["train_jsonl"]

    dataset = InterleavedJsonlDataset(
        input_jsonl,
        image_root=config["data"].get("image_root", "."),
        use_cached_features=False,
    )
    limit = args.num_samples or len(dataset)
    output_records: list[dict] = []

    if args.mock:
        for index in tqdm(range(min(limit, len(dataset))), desc="mock-cache"):
            sample = dataset[index]["sample"]
            path = output_dir / f"{sample.sample_id}.pt"
            if not (args.skip_existing and path.exists()):
                torch.save(_mock_feature(config), path)
            record = sample.to_dict()
            record["feature_path"] = str(path)
            output_records.append(record)
        if args.output_jsonl:
            out_jsonl = Path(args.output_jsonl)
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with out_jsonl.open("w", encoding="utf-8") as handle:
                for record in output_records:
                    handle.write(json.dumps(record) + "\n")
        logger.info("Wrote mock cached features to %s", output_dir)
        return

    vlm = build_vlm_from_config(config).load()
    for index in tqdm(range(min(limit, len(dataset))), desc="cache-vlm"):
        sample = dataset[index]["sample"]
        path = output_dir / f"{sample.sample_id}.pt"
        if not (args.skip_existing and path.exists()):
            output = vlm.extract([sample])
            torch.save(
                {
                    "hidden_states": output.hidden_states,
                    "source_masks": output.source_masks,
                    "answer_text": output.answer_text,
                    "sample_id": sample.sample_id,
                },
                path,
            )
        record = sample.to_dict()
        record["feature_path"] = str(path)
        output_records.append(record)
    if args.output_jsonl:
        out_jsonl = Path(args.output_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for record in output_records:
                handle.write(json.dumps(record) + "\n")
    logger.info("Wrote cached features to %s", output_dir)


if __name__ == "__main__":
    main()
