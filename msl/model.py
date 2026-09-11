"""PyTorch implementation of the multiscale stepwise learning network.

Tensor convention throughout this module is ``[batch, channels, time]``.
The implementation mirrors the architecture specified in the manuscript:

* 80 CWT channels are split into four 20-channel frequency groups.
* Step 1 processes the groups progressively.  The first module receives 20
  channels; each subsequent module receives its 20-channel band concatenated
  with the preceding 20-channel output (40 channels).
* Each Step 1 module follows ``in-256-128-64-32-20``.
* The four Step 1 outputs and a 20-channel projection of the low-frequency
  initial model are concatenated into 100 channels.
* Step 2 uses an initial convolution, three residual blocks, and a final
  prediction convolution, following ``100-256-128-64-32-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ConvAct1D(nn.Module):
    """1-D convolution with ReLU and optional dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.conv(x)))


class Step1Module(nn.Module):
    """Frequency-band feature extractor used in Step 1."""

    def __init__(self, in_channels: int, kernel_size: int, dropout: float = 0.5):
        super().__init__()
        channel_sequence = [in_channels, 256, 128, 64, 32, 20]
        layers: list[nn.Module] = []
        for index, (cin, cout) in enumerate(zip(channel_sequence[:-1], channel_sequence[1:])):
            # Dropout is used only in Step 1.  The last 20-channel feature is
            # kept intact so that the branch output can be fused downstream.
            layer_dropout = dropout if index < len(channel_sequence) - 2 else 0.0
            layers.append(ConvAct1D(cin, cout, kernel_size, layer_dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ResidualBlock1D(nn.Module):
    """Residual block with a projection shortcut when channel counts differ."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        y = self.activation(self.conv1(x))
        y = self.conv2(y)
        return self.activation(y + residual)


class Step2Module(nn.Module):
    """Fusion and impedance prediction module."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.initial_conv = ConvAct1D(100, 256, kernel_size, dropout=0.0)
        self.residual_block_1 = ResidualBlock1D(256, 128, kernel_size)
        self.residual_block_2 = ResidualBlock1D(128, 64, kernel_size)
        self.residual_block_3 = ResidualBlock1D(64, 32, kernel_size)
        self.prediction = nn.Conv1d(32, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        return self.prediction(x)


@dataclass(frozen=True)
class MSLConfig:
    """Architecture and regularization settings."""

    cwt_channels: int = 80
    band_channels: int = 20
    low_frequency_channels: int = 20
    step1_kernels: tuple[int, int, int, int] = (9, 7, 5, 3)
    step2_kernel: int = 3
    dropout: float = 0.5


class MSLNetwork(nn.Module):
    """CWT-assisted multiscale stepwise learning network."""

    def __init__(self, config: MSLConfig | None = None):
        super().__init__()
        self.config = config or MSLConfig()
        cfg = self.config
        expected = 4 * cfg.band_channels
        if cfg.cwt_channels != expected:
            raise ValueError("cwt_channels must equal four times band_channels")

        self.step1 = nn.ModuleList(
            [
                Step1Module(cfg.band_channels, cfg.step1_kernels[0], cfg.dropout),
                Step1Module(2 * cfg.band_channels, cfg.step1_kernels[1], cfg.dropout),
                Step1Module(2 * cfg.band_channels, cfg.step1_kernels[2], cfg.dropout),
                Step1Module(2 * cfg.band_channels, cfg.step1_kernels[3], cfg.dropout),
            ]
        )
        self.low_frequency_projection = nn.Conv1d(
            1, cfg.low_frequency_channels, kernel_size=1, stride=1, padding=0
        )
        self.step2 = Step2Module(cfg.step2_kernel)

    @staticmethod
    def _ensure_trace_tensor(x: Tensor) -> Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3:
            return x
        raise ValueError(f"Expected [B,T] or [B,C,T], got {tuple(x.shape)}")

    def forward(self, cwt_data: Tensor, low_frequency_model: Tensor) -> Tensor:
        """Predict impedance from CWT data and a low-frequency model."""

        if cwt_data.ndim != 3:
            raise ValueError("cwt_data must have shape [B, 80, T]")
        if cwt_data.shape[1] != self.config.cwt_channels:
            raise ValueError(f"Expected {self.config.cwt_channels} CWT channels")

        low_frequency_model = self._ensure_trace_tensor(low_frequency_model)
        bands = torch.chunk(cwt_data, chunks=4, dim=1)
        outputs: list[Tensor] = []
        previous: Tensor | None = None
        for index, band in enumerate(bands):
            branch_input = band if index == 0 else torch.cat([band, previous], dim=1)
            previous = self.step1[index](branch_input)
            outputs.append(previous)

        low_features = self.low_frequency_projection(low_frequency_model)
        fused = torch.cat([*outputs, low_features], dim=1)
        return self.step2(fused)


def architecture_rows() -> list[dict[str, str]]:
    """Return a compact layer table for the README and manuscript supplement."""

    rows: list[dict[str, str]] = [
        {
            "stage": "Input seismic trace",
            "input": "B x 1 x T",
            "transformation": "identity",
            "kernel/stride/padding": "-",
            "output": "B x 1 x T",
        },
        {
            "stage": "CWT representation",
            "input": "B x 1 x T",
            "transformation": "morl; 2-80 Hz; 80 channels",
            "kernel/stride/padding": "-",
            "output": "B x 80 x T",
        },
        {
            "stage": "Step 1 (branch 1)",
            "input": "B x 20 x T",
            "transformation": "20-256-128-64-32-20",
            "kernel/stride/padding": "9 / 1 / 4",
            "output": "B x 20 x T",
        },
        {
            "stage": "Step 1 (branches 2-4)",
            "input": "B x 40 x T",
            "transformation": "40-256-128-64-32-20",
            "kernel/stride/padding": "7, 5, 3 / 1 / 3, 2, 1",
            "output": "B x 20 x T per branch",
        },
        {
            "stage": "Low-frequency projection",
            "input": "B x 1 x T",
            "transformation": "1-20 (1 x 1 convolution)",
            "kernel/stride/padding": "1 / 1 / 0",
            "output": "B x 20 x T",
        },
        {
            "stage": "Feature fusion",
            "input": "four Step 1 outputs + low-frequency features",
            "transformation": "20 x 5 = 100 channels",
            "kernel/stride/padding": "-",
            "output": "B x 100 x T",
        },
        {
            "stage": "Step 2 initial convolution",
            "input": "B x 100 x T",
            "transformation": "100-256",
            "kernel/stride/padding": "3 / 1 / 1",
            "output": "B x 256 x T",
        },
        {
            "stage": "Step 2 residual block 1",
            "input": "B x 256 x T",
            "transformation": "256-128",
            "kernel/stride/padding": "3 / 1 / 1; projection shortcut",
            "output": "B x 128 x T",
        },
        {
            "stage": "Step 2 residual block 2",
            "input": "B x 128 x T",
            "transformation": "128-64",
            "kernel/stride/padding": "3 / 1 / 1; projection shortcut",
            "output": "B x 64 x T",
        },
        {
            "stage": "Step 2 residual block 3",
            "input": "B x 64 x T",
            "transformation": "64-32",
            "kernel/stride/padding": "3 / 1 / 1; projection shortcut",
            "output": "B x 32 x T",
        },
        {
            "stage": "Prediction layer",
            "input": "B x 32 x T",
            "transformation": "32-1",
            "kernel/stride/padding": "3 / 1 / 1",
            "output": "B x 1 x T",
        },
    ]
    return rows
