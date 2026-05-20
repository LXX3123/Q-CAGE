from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from qcage.data.collate import collate_qcage_batch
from qcage.data.dataset import InterleavedJsonlDataset
from qcage.models.flux_bridge import build_generator_bridge
from qcage.models.qcage_adapter import build_adapter_from_config
from qcage.evaluation.opening_official import write_official_output
from qcage.training.checkpoint import load_checkpoint
from qcage.training.distributed import cleanup_distributed, init_distributed
from qcage.utils.config import load_config
from qcage.utils.logging import setup_logging
from qcage.utils.tensor import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q-CAGE on an OpenING-style JSONL split")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    dist_state = init_distributed()
    device = f"cuda:{dist_state.local_rank}" if torch.cuda.is_available() else "cpu"

    benchmark_cfg = config["benchmark"]
    output_dir = Path(benchmark_cfg.get("output_dir", "outputs/opening"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = InterleavedJsonlDataset(
        benchmark_cfg["input_jsonl"],
        image_root=config["data"].get("image_root", "."),
        use_cached_features=True,
    )
    sampler = DistributedSampler(dataset, shuffle=False) if dist_state.enabled else None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_qcage_batch,
    )

    adapter = build_adapter_from_config(config).to(device)
    checkpoint_path = config.get("inference", {}).get("checkpoint")
    if checkpoint_path:
        load_checkpoint(checkpoint_path, adapter, map_location=device)
    adapter.eval()
    if dist_state.enabled:
        adapter = DistributedDataParallel(adapter, device_ids=[dist_state.local_rank])

    bridge = build_generator_bridge(config, device=device)
    shard_manifest = output_dir / f"manifest_rank{dist_state.rank:03d}.jsonl"
    official_dir = benchmark_cfg.get("official_model_output_dir")
    if official_dir is None and benchmark_cfg.get("opening_model_name"):
        official_dir = output_dir / f"{benchmark_cfg['opening_model_name']}_output"
    with shard_manifest.open("w", encoding="utf-8") as handle:
        for batch in tqdm(dataloader, disable=not dist_state.is_main_process, desc="benchmark"):
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
            official_json = None
            if official_dir is not None:
                official_json = write_official_output(
                    sample=batch["samples"][0],
                    generation=generated,
                    model_output_dir=official_dir,
                    answer_text=batch["samples"][0].answer_text,
                )
            handle.write(
                json.dumps(
                    {
                        "sample_id": batch["samples"][0].sample_id,
                        "candidate": benchmark_cfg.get("candidate_name", "qcage"),
                        "generation": generated,
                        "official_output": str(official_json) if official_json else None,
                    }
                )
                + "\n"
            )

    if dist_state.is_main_process:
        logger.info("Wrote benchmark shard(s) under %s", output_dir)
    cleanup_distributed(dist_state)


if __name__ == "__main__":
    main()
