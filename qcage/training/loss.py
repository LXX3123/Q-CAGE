from __future__ import annotations

import torch


def flow_matching_mse(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_velocity - target_velocity) ** 2)


def interpolate_flow_latent(noise_latents: torch.Tensor, target_latents: torch.Tensor, timesteps: torch.Tensor):
    """Compute z_tau = (1 - tau) z0 + tau z1 and target velocity z1 - z0."""
    tau = timesteps
    while tau.ndim < target_latents.ndim:
        tau = tau.unsqueeze(-1)
    noisy = (1.0 - tau) * noise_latents + tau * target_latents
    target_velocity = target_latents - noise_latents
    return noisy, target_velocity

