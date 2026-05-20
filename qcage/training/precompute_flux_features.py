from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from qcage.data.dataset import InterleavedJsonlDataset
from qcage.data.serialize import prompt_for_generator
from qcage.utils.config import load_config
from qcage.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute FLUX prompt and latent tensors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-dir", default="cached_features_flux")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-missing-qwen", action="store_true")
    return parser.parse_args()


def _dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _load_pipeline(config: dict, device: str):
    import diffusers

    generator_cfg = config["model"]["generator"]
    class_name = generator_cfg.get("pipeline_class", "FluxPipeline")
    pipeline_cls = getattr(diffusers, class_name)
    torch_dtype = _dtype(config.get("training", {}).get("precision", "bf16"))
    pipe = pipeline_cls.from_pretrained(generator_cfg["model_id"], torch_dtype=torch_dtype).to(device)
    pipe.vae.eval()
    for component_name in ["vae", "text_encoder", "text_encoder_2", "transformer"]:
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        component.eval()
        for param in component.parameters():
            param.requires_grad_(False)
    return pipe


def _preprocess_image(path: str | Path, resolution: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    image = ImageOps.fit(image, (resolution, resolution), method=Image.Resampling.LANCZOS)
    array = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    array = array.view(resolution, resolution, 3).permute(2, 0, 1).float() / 127.5 - 1.0
    return array.unsqueeze(0).to(device=device, dtype=dtype)


def _encode_prompt(pipe, prompt: str, device: str) -> dict[str, torch.Tensor]:
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("The selected FLUX pipeline does not expose encode_prompt().")
    with torch.no_grad():
        try:
            encoded = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
            )
        except TypeError:
            encoded = pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
            )

    if isinstance(encoded, dict):
        prompt_embeds = encoded.get("prompt_embeds")
        pooled = encoded.get("pooled_prompt_embeds")
        txt_ids = encoded.get("txt_ids")
        if txt_ids is None:
            txt_ids = encoded.get("text_ids")
    else:
        values = tuple(encoded)
        prompt_embeds = values[0]
        pooled = values[1] if len(values) > 1 else None
        txt_ids = values[2] if len(values) > 2 else None

    result = {"prompt_embeds": prompt_embeds.detach().cpu()}
    if pooled is not None:
        result["pooled_prompt_embeds"] = pooled.detach().cpu()
    if txt_ids is not None:
        result["txt_ids"] = txt_ids.detach().cpu()
    return result


def _vae_latents(pipe, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    scaling = float(getattr(pipe.vae.config, "scaling_factor", 1.0))
    shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
    return (latents - shift) * scaling


def _pack_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    if hasattr(pipe, "_pack_latents"):
        try:
            return pipe._pack_latents(latents, batch_size, channels, height, width)
        except TypeError:
            return pipe._pack_latents(latents, batch_size, channels, height, width, latents.dtype)
    return latents.flatten(2).transpose(1, 2)


def _image_ids(pipe, packed_latents: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, _ = packed_latents.shape
    _, _, height, width = latents.shape
    device = latents.device
    dtype = packed_latents.dtype
    if hasattr(pipe, "_prepare_latent_image_ids"):
        for h, w in [(height // 2, width // 2), (height, width)]:
            try:
                return pipe._prepare_latent_image_ids(batch_size, h, w, device, dtype).detach().cpu()
            except TypeError:
                try:
                    return pipe._prepare_latent_image_ids(h, w, device, dtype).detach().cpu()
                except TypeError:
                    continue
    return torch.zeros((packed_latents.shape[1], 3), dtype=dtype)


def _sample_noisy_latents(target_latents: torch.Tensor, timestep_scale: float = 1000.0):
    noise = torch.randn_like(target_latents)
    tau = torch.rand((target_latents.shape[0],), device=target_latents.device, dtype=target_latents.dtype)
    while tau.ndim < target_latents.ndim:
        tau = tau.unsqueeze(-1)
    noisy = (1.0 - tau) * noise + tau * target_latents
    velocity = target_latents - noise
    timesteps = tau.flatten()[: target_latents.shape[0]] * timestep_scale
    return noisy, velocity, timesteps


def _load_existing_feature(sample, allow_missing: bool) -> dict[str, Any]:
    if sample.feature_path and Path(sample.feature_path).exists():
        return torch.load(sample.feature_path, map_location="cpu")
    if allow_missing:
        return {}
    raise FileNotFoundError(
        f"Sample {sample.sample_id} does not have an existing Qwen feature_path. "
        "Run qcage.training.cache_vlm_features first, or pass --allow-missing-qwen."
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = _dtype(config.get("training", {}).get("precision", "bf16"))
    input_jsonl = args.input_jsonl or config["data"]["train_jsonl"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    dataset = InterleavedJsonlDataset(
        input_jsonl,
        image_root=config["data"].get("image_root", "."),
        use_cached_features=False,
    )
    pipe = _load_pipeline(config, device=device)
    limit = min(args.num_samples or len(dataset), len(dataset))
    records: list[dict[str, Any]] = []

    for index in tqdm(range(limit), desc="precompute-flux"):
        sample = dataset[index]["sample"]
        output_path = output_dir / f"{sample.sample_id}.pt"
        if args.skip_existing and output_path.exists():
            record = sample.to_dict()
            record["feature_path"] = str(output_path)
            records.append(record)
            continue
        if not sample.target_image:
            raise ValueError(f"Sample {sample.sample_id} is missing target_image")

        feature = _load_existing_feature(sample, allow_missing=args.allow_missing_qwen)
        feature.update(_encode_prompt(pipe, prompt_for_generator(sample), device=device))
        image_tensor = _preprocess_image(sample.target_image, args.resolution, device, precision)
        raw_latents = _vae_latents(pipe, image_tensor)
        target_latents = _pack_latents(pipe, raw_latents)
        noisy_latents, velocity_target, timesteps = _sample_noisy_latents(target_latents)
        feature["target_latents"] = target_latents.detach().cpu()
        feature["noisy_latents"] = noisy_latents.detach().cpu()
        feature["velocity_target"] = velocity_target.detach().cpu()
        feature["timesteps"] = timesteps.detach().cpu()
        feature["img_ids"] = _image_ids(pipe, target_latents, raw_latents)
        feature["sample_id"] = sample.sample_id

        torch.save(feature, output_path)
        record = sample.to_dict()
        record["feature_path"] = str(output_path)
        records.append(record)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    logger.info("Wrote FLUX-precomputed JSONL: %s", output_jsonl)


if __name__ == "__main__":
    main()
