from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class GeneratorLossOutput:
    loss: torch.Tensor
    logs: dict[str, float]


class MockFluxBridge(nn.Module):
    """Small differentiable bridge for shape tests and dry runs.

    This does not model FLUX quality. It only verifies that gradients can flow
    from a generator-side loss into Q-CAGE condition tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.zeros(()), persistent=False)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

    @staticmethod
    def compose_condition_sequence(condition_tokens: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            return condition_tokens
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.shape[0] != condition_tokens.shape[0]:
            raise ValueError("prompt_embeds batch size does not match condition_tokens")
        if prompt_embeds.shape[-1] != condition_tokens.shape[-1]:
            raise ValueError("prompt_embeds hidden dim does not match Q-CAGE condition dim")
        return torch.cat([prompt_embeds.to(condition_tokens.device), condition_tokens], dim=1)

    def training_loss(self, condition_tokens: torch.Tensor, batch: dict[str, Any]) -> GeneratorLossOutput:
        condition_sequence = self.compose_condition_sequence(condition_tokens, batch)
        target = batch.get("velocity_target")
        if target is None:
            target = batch.get("target_latents")
        if target is None:
            target = torch.zeros(
                (condition_tokens.shape[0],),
                dtype=condition_tokens.dtype,
                device=condition_tokens.device,
            )

        scalar_pred = condition_sequence.mean(dim=(1, 2))
        while scalar_pred.ndim < target.ndim:
            scalar_pred = scalar_pred.unsqueeze(-1)
        pred = scalar_pred.expand_as(target)
        loss = torch.mean((pred - target) ** 2)
        return GeneratorLossOutput(
            loss=loss,
            logs={
                "mock_flow_loss": float(loss.detach().cpu()),
                "condition_tokens": float(condition_tokens.shape[1]),
                "condition_sequence": float(condition_sequence.shape[1]),
            },
        )

    def generate(self, condition_tokens: torch.Tensor, **kwargs):
        samples = kwargs.get("samples") or []
        output_dir = kwargs.get("output_dir")
        image_paths: list[str] = []
        if output_dir is not None and samples:
            try:
                from PIL import Image, ImageDraw

                out_dir = Path(output_dir) / "images"
                out_dir.mkdir(parents=True, exist_ok=True)
                values = condition_tokens.detach().float().mean(dim=(1, 2)).cpu().tolist()
                for sample, value in zip(samples, values, strict=False):
                    normalized = int((abs(value) * 1000) % 255)
                    image = Image.new("RGB", (256, 256), (normalized, 96, 255 - normalized))
                    draw = ImageDraw.Draw(image)
                    draw.text((16, 116), getattr(sample, "sample_id", "mock"), fill=(255, 255, 255))
                    path = out_dir / f"{getattr(sample, 'sample_id', len(image_paths))}.png"
                    image.save(path)
                    image_paths.append(str(path))
            except Exception:
                image_paths = []
        return {
            "images": image_paths,
            "metadata": {
                "backend": "mock",
                "condition_shape": tuple(condition_tokens.shape),
            },
        }


class DiffusersFluxBridge(nn.Module):
    """Thin integration boundary for a frozen FLUX/DiT training stack."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bf16",
        pipeline_class: str = "FluxPipeline",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline_class = pipeline_class
        self.pipeline = None

    def load(self) -> "DiffusersFluxBridge":
        import torch
        import diffusers

        torch_dtype = torch.bfloat16 if self.dtype == "bf16" else torch.float16 if self.dtype == "fp16" else torch.float32
        pipeline_cls = getattr(diffusers, self.pipeline_class)
        self.pipeline = pipeline_cls.from_pretrained(self.model_id, torch_dtype=torch_dtype).to(self.device)
        self.freeze()
        return self

    def freeze(self) -> None:
        if self.pipeline is None:
            return
        for component_name in ["transformer", "vae", "text_encoder", "text_encoder_2"]:
            component = getattr(self.pipeline, component_name, None)
            if component is None:
                continue
            component.eval()
            for param in component.parameters():
                param.requires_grad_(False)

    @staticmethod
    def compose_condition_sequence(condition_tokens: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            return condition_tokens
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.shape[-1] != condition_tokens.shape[-1]:
            raise ValueError("prompt_embeds hidden dim does not match Q-CAGE condition dim")
        return torch.cat([prompt_embeds.to(condition_tokens.device), condition_tokens], dim=1)

    def _extend_txt_ids(self, txt_ids: torch.Tensor | None, extra_tokens: int, batch_size: int):
        if txt_ids is None:
            return None
        if txt_ids.ndim == 2:
            extra = torch.zeros(
                (extra_tokens, txt_ids.shape[-1]),
                dtype=txt_ids.dtype,
                device=txt_ids.device,
            )
            return torch.cat([txt_ids, extra], dim=0)
        if txt_ids.ndim == 3:
            extra = torch.zeros(
                (batch_size, extra_tokens, txt_ids.shape[-1]),
                dtype=txt_ids.dtype,
                device=txt_ids.device,
            )
            return torch.cat([txt_ids, extra], dim=1)
        raise ValueError(f"Unsupported txt_ids shape: {tuple(txt_ids.shape)}")

    @staticmethod
    def _model_sample(output):
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple):
            return output[0]
        return output

    def training_loss(self, condition_tokens: torch.Tensor, batch: dict[str, Any]) -> GeneratorLossOutput:
        if self.pipeline is None:
            raise RuntimeError("Call load() before training_loss().")

        noisy_latents = batch.get("noisy_latents")
        target_velocity = batch.get("velocity_target")
        if noisy_latents is None or target_velocity is None:
            raise KeyError(
                "FLUX training requires precomputed batch['noisy_latents'] and "
                "batch['velocity_target']. Run qcage.training.precompute_flux_features first."
            )

        condition_sequence = self.compose_condition_sequence(condition_tokens, batch)
        transformer = self.pipeline.transformer
        batch_size = condition_tokens.shape[0]
        txt_ids = self._extend_txt_ids(
            batch.get("txt_ids"),
            extra_tokens=condition_tokens.shape[1],
            batch_size=batch_size,
        )
        kwargs = {
            "hidden_states": noisy_latents,
            "timestep": batch.get("timesteps"),
            "encoder_hidden_states": condition_sequence,
            "pooled_projections": batch.get("pooled_prompt_embeds"),
            "txt_ids": txt_ids,
            "img_ids": batch.get("img_ids"),
            "return_dict": True,
        }
        guidance = batch.get("guidance")
        if guidance is not None:
            kwargs["guidance"] = guidance
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        pred_velocity = self._model_sample(transformer(**kwargs))
        loss = torch.mean((pred_velocity.float() - target_velocity.float()) ** 2)
        return GeneratorLossOutput(
            loss=loss,
            logs={
                "flow_loss": float(loss.detach().cpu()),
                "condition_tokens": float(condition_tokens.shape[1]),
                "condition_sequence": float(condition_sequence.shape[1]),
            },
        )

    def generate(self, condition_tokens: torch.Tensor, **kwargs):
        if self.pipeline is None:
            raise RuntimeError("Call load() before generate().")

        batch = kwargs.get("batch") or {}
        samples = kwargs.get("samples") or []
        output_dir = kwargs.get("output_dir")
        condition_sequence = self.compose_condition_sequence(condition_tokens, batch)

        call_kwargs = {
            "prompt_embeds": condition_sequence,
            "pooled_prompt_embeds": batch.get("pooled_prompt_embeds"),
            "num_inference_steps": kwargs.get("num_steps", 30),
            "guidance_scale": kwargs.get("guidance_scale", 3.5),
            "output_type": "pil",
        }
        if "height" in kwargs:
            call_kwargs["height"] = kwargs["height"]
        if "width" in kwargs:
            call_kwargs["width"] = kwargs["width"]
        call_kwargs = {key: value for key, value in call_kwargs.items() if value is not None}

        try:
            result = self.pipeline(**call_kwargs)
        except TypeError as exc:
            raise TypeError(
                "The selected diffusers pipeline did not accept prompt_embeds-style "
                "Q-CAGE inference. Check the FLUX/FLUX.2 pipeline signature and adapt "
                "DiffusersFluxBridge.generate for that checkpoint."
            ) from exc

        images = getattr(result, "images", result[0] if isinstance(result, tuple) else result)
        image_paths: list[str] = []
        if output_dir is not None:
            out_dir = Path(output_dir) / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images):
                sample_id = getattr(samples[index], "sample_id", f"sample_{index:04d}") if index < len(samples) else f"sample_{index:04d}"
                path = out_dir / f"{sample_id}.png"
                image.save(path)
                image_paths.append(str(path))

        return {
            "images": image_paths,
            "metadata": {
                "backend": "diffusers_flux",
                "condition_shape": tuple(condition_tokens.shape),
                "condition_sequence_shape": tuple(condition_sequence.shape),
            },
        }


class PrecomputedDiffusersFluxBridge(DiffusersFluxBridge):
    """Frozen FLUX transformer loss from precomputed latents and text embeddings.

    Expected batch fields:
    - noisy_latents or target/noise/timesteps from which noisy latents can be built upstream
    - velocity_target
    - prompt_embeds
    - pooled_prompt_embeds, txt_ids, img_ids when required by the FLUX transformer

    The bridge keeps all FLUX parameters frozen while preserving the computation
    graph from the transformer prediction back into Q-CAGE condition tokens.
    """

    def _extend_txt_ids(self, txt_ids: torch.Tensor | None, extra_tokens: int, batch_size: int):
        if txt_ids is None:
            return None
        if txt_ids.ndim == 2:
            extra = torch.zeros(
                (extra_tokens, txt_ids.shape[-1]),
                dtype=txt_ids.dtype,
                device=txt_ids.device,
            )
            return torch.cat([txt_ids, extra], dim=0)
        if txt_ids.ndim == 3:
            extra = torch.zeros(
                (batch_size, extra_tokens, txt_ids.shape[-1]),
                dtype=txt_ids.dtype,
                device=txt_ids.device,
            )
            return torch.cat([txt_ids, extra], dim=1)
        raise ValueError(f"Unsupported txt_ids shape: {tuple(txt_ids.shape)}")

    @staticmethod
    def _model_sample(output):
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, tuple):
            return output[0]
        return output

    def training_loss(self, condition_tokens: torch.Tensor, batch: dict[str, Any]) -> GeneratorLossOutput:
        if self.pipeline is None:
            raise RuntimeError("Call load() before training_loss().")

        noisy_latents = batch.get("noisy_latents")
        target_velocity = batch.get("velocity_target")
        if noisy_latents is None or target_velocity is None:
            raise KeyError(
                "diffusers_flux_precomputed requires batch['noisy_latents'] and "
                "batch['velocity_target']."
            )

        condition_sequence = self.compose_condition_sequence(condition_tokens, batch)
        transformer = self.pipeline.transformer
        batch_size = condition_tokens.shape[0]
        txt_ids = self._extend_txt_ids(
            batch.get("txt_ids"),
            extra_tokens=condition_tokens.shape[1],
            batch_size=batch_size,
        )

        kwargs = {
            "hidden_states": noisy_latents,
            "timestep": batch.get("timesteps"),
            "encoder_hidden_states": condition_sequence,
            "pooled_projections": batch.get("pooled_prompt_embeds"),
            "txt_ids": txt_ids,
            "img_ids": batch.get("img_ids"),
            "return_dict": True,
        }
        guidance = batch.get("guidance")
        if guidance is not None:
            kwargs["guidance"] = guidance
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        pred_velocity = self._model_sample(transformer(**kwargs))
        loss = torch.mean((pred_velocity.float() - target_velocity.float()) ** 2)
        return GeneratorLossOutput(
            loss=loss,
            logs={
                "flow_loss": float(loss.detach().cpu()),
                "condition_tokens": float(condition_tokens.shape[1]),
                "condition_sequence": float(condition_sequence.shape[1]),
            },
        )


def build_generator_bridge(config: dict, device: str = "cuda") -> nn.Module:
    generator_cfg = config["model"]["generator"]
    backend = generator_cfg.get("backend", "mock")
    if backend == "mock":
        bridge = MockFluxBridge().to(device)
        bridge.freeze()
        return bridge
    if backend == "diffusers_flux":
        return DiffusersFluxBridge(
            model_id=generator_cfg["model_id"],
            device=device,
            dtype=config.get("training", {}).get("precision", "bf16"),
            pipeline_class=generator_cfg.get("pipeline_class", "FluxPipeline"),
        ).load()
    if backend == "diffusers_flux_precomputed":
        return PrecomputedDiffusersFluxBridge(
            model_id=generator_cfg["model_id"],
            device=device,
            dtype=config.get("training", {}).get("precision", "bf16"),
            pipeline_class=generator_cfg.get("pipeline_class", "FluxPipeline"),
        ).load()
    raise ValueError(f"Unsupported generator backend: {backend}")
