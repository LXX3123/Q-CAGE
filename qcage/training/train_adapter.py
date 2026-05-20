from __future__ import annotations

import argparse
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
from qcage.training.checkpoint import load_checkpoint, save_checkpoint
from qcage.training.distributed import cleanup_distributed, init_distributed
from qcage.training.optimizer import build_optimizer
from qcage.utils.config import load_config
from qcage.utils.device import get_precision
from qcage.utils.logging import setup_logging
from qcage.utils.seed import seed_everything
from qcage.utils.tensor import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Q-CAGE adapter")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    dist_state = init_distributed()
    device = f"cuda:{dist_state.local_rank}" if torch.cuda.is_available() else "cpu"

    seed_everything(int(config.get("seed", 1234)) + dist_state.rank)
    precision = get_precision(config["training"].get("precision", "bf16"))

    if not config["data"].get("use_cached_features", False):
        raise RuntimeError(
            "Q-CAGE training expects cached feature_path tensors. Run "
            "qcage.training.cache_vlm_features and qcage.training.precompute_flux_features, "
            "then set data.use_cached_features=true."
        )

    dataset = InterleavedJsonlDataset(
        config["data"]["train_jsonl"],
        image_root=config["data"].get("image_root", "."),
        use_cached_features=True,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if dist_state.enabled else None
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size_per_gpu", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_qcage_batch,
        drop_last=True,
    )

    adapter = build_adapter_from_config(config).to(device)
    optimizer = build_optimizer(adapter.parameters(), config)
    resume_from = config["training"].get("resume_from")
    start_step = 0
    start_epoch = 0
    if resume_from:
        checkpoint = load_checkpoint(resume_from, adapter, optimizer=optimizer, map_location=device)
        start_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))

    if dist_state.enabled:
        adapter = DistributedDataParallel(adapter, device_ids=[dist_state.local_rank])

    generator_bridge = build_generator_bridge(config, device=device)
    generator_bridge.eval()

    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    max_steps_cfg = config["training"].get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    epochs = int(config["training"].get("epochs", 1))
    save_every = int(config["training"].get("save_every", 1000))
    log_every = int(config["training"].get("log_every", 10))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    output_dir = Path(config["training"]["output_dir"])

    if dist_state.is_main_process:
        trainable = sum(param.numel() for param in adapter.parameters() if param.requires_grad)
        logger.info("Training samples: %s", len(dataset))
        logger.info("Trainable Q-CAGE parameters: %.3f M", trainable / 1e6)
        logger.info("Device: %s | precision: %s | world_size: %s", device, precision.name, dist_state.world_size)

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(start_epoch, epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            progress = tqdm(
                dataloader,
                disable=not dist_state.is_main_process,
                desc=f"epoch {epoch}",
            )
            for micro_step, batch in enumerate(progress):
                batch = move_to_device(batch, device)
                autocast_enabled = precision.autocast_dtype is not None and device.startswith("cuda")
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    dtype=precision.autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    output = adapter(
                        hidden_states=batch["hidden_states"],
                        source_masks=batch["source_masks"],
                        valid_token_masks=batch.get("valid_token_masks"),
                    )
                    loss_out = generator_bridge.training_loss(output.condition_tokens, batch)
                    loss = loss_out.loss / grad_accum

                loss.backward()

                should_step = (micro_step + 1) % grad_accum == 0
                if should_step:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if dist_state.is_main_process and global_step % log_every == 0:
                        logs = " ".join(f"{key}={value:.5f}" for key, value in loss_out.logs.items())
                        progress.set_postfix(step=global_step, loss=f"{loss_out.loss.item():.5f}")
                        logger.info("step=%s loss=%.6f %s", global_step, loss_out.loss.item(), logs)

                    if dist_state.is_main_process and global_step % save_every == 0:
                        path = save_checkpoint(
                            output_dir=output_dir,
                            model=adapter,
                            optimizer=optimizer,
                            step=global_step,
                            epoch=epoch,
                            config=config,
                        )
                        logger.info("Saved checkpoint: %s", path)

                    if max_steps is not None and global_step >= max_steps:
                        break

            if max_steps is not None and global_step >= max_steps:
                break

        if dist_state.is_main_process:
            path = save_checkpoint(
                output_dir=output_dir,
                model=adapter,
                optimizer=optimizer,
                step=global_step,
                epoch=epochs,
                config=config,
                name="final.pt",
            )
            logger.info("Training complete. Final checkpoint: %s", path)
    finally:
        cleanup_distributed(dist_state)


if __name__ == "__main__":
    train()
