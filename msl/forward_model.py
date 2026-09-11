"""Differentiable post-stack seismic forward model.

The forward operator follows the standard acoustic-convolution workflow:
impedance -> normal-incidence reflection coefficient -> wavelet convolution.
It is used as a data-consistency loss in the example training loop.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def impedance_to_reflectivity(impedance: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert acoustic impedance [B, 1, T] to reflectivity [B, 1, T]."""

    if impedance.ndim != 3 or impedance.shape[1] != 1:
        raise ValueError("impedance must have shape [B, 1, T]")
    left = impedance[..., :-1]
    right = impedance[..., 1:]
    reflectivity = (right - left) / (right + left + eps)
    # The first sample has no preceding interface and is set to zero.
    return F.pad(reflectivity, (1, 0), mode="constant", value=0.0)


def ricker_wavelet(length: int = 31, dominant_frequency: float = 30.0, dt: float = 0.001) -> Tensor:
    """Create a normalized zero-phase Ricker wavelet."""

    if length % 2 == 0:
        raise ValueError("length must be odd for same-padding convolution")
    time = (torch.arange(length, dtype=torch.float32) - length // 2) * dt
    pi_f_t = math.pi * dominant_frequency * time
    wavelet = (1.0 - 2.0 * pi_f_t.square()) * torch.exp(-pi_f_t.square())
    return wavelet / wavelet.abs().sum().clamp_min(1e-8)


class ConvolutionalForwardModel(nn.Module):
    """Fixed-wavelet differentiable forward operator."""

    def __init__(self, wavelet: Tensor):
        super().__init__()
        if wavelet.ndim != 1 or wavelet.numel() % 2 == 0:
            raise ValueError("wavelet must be a one-dimensional odd-length tensor")
        self.register_buffer("wavelet", wavelet.detach().float().view(1, 1, -1))

    def forward(self, impedance: Tensor) -> Tensor:
        reflectivity = impedance_to_reflectivity(impedance)
        padding = self.wavelet.shape[-1] // 2
        return F.conv1d(reflectivity, self.wavelet, padding=padding)


def forward_consistency_loss(
    predicted_impedance: Tensor,
    observed_seismic: Tensor,
    forward_model: ConvolutionalForwardModel,
) -> Tensor:
    """Mean-squared synthetic/observed seismic reconstruction loss."""

    synthetic = forward_model(predicted_impedance)
    if synthetic.shape != observed_seismic.shape:
        raise ValueError(f"Shape mismatch: synthetic={synthetic.shape}, observed={observed_seismic.shape}")
    return F.mse_loss(synthetic, observed_seismic)
