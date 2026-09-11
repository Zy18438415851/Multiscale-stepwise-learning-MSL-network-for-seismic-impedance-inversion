"""Small synthetic dataset for testing the MSL implementation.

This module is deliberately independent of proprietary field data.  Replace
``build_demo_dataset`` with a loader for your seismic traces and well logs when
publishing the non-confidential repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .cwt import CWTConfig, cwt_transform
from .forward_model import ConvolutionalForwardModel, ricker_wavelet


@dataclass
class DatasetBundle:
    seismic: Tensor
    cwt: Tensor
    low_frequency_model: Tensor
    impedance: Tensor


def _smooth_trace(trace: np.ndarray, window: int = 25) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(trace, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _layered_impedance(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Generate a positive, piecewise-smooth impedance profile."""

    n_layers = int(rng.integers(5, 10))
    boundaries = np.sort(rng.choice(np.arange(8, n_samples - 8), size=n_layers - 1, replace=False))
    boundaries = np.concatenate(([0], boundaries, [n_samples]))
    values = rng.uniform(1.0, 3.0, size=n_layers)
    trace = np.empty(n_samples, dtype=np.float64)
    for start, stop, value in zip(boundaries[:-1], boundaries[1:], values):
        trace[start:stop] = value
    trace = _smooth_trace(trace, window=5)
    trace += 0.02 * rng.standard_normal(n_samples)
    return np.clip(trace, 0.5, None)


def build_demo_dataset(
    n_traces: int = 24,
    n_samples: int = 128,
    *,
    seed: int = 7,
    cwt_config: CWTConfig | None = None,
    noise_std: float = 0.01,
) -> DatasetBundle:
    """Create synthetic impedance, seismic, low-frequency and CWT arrays."""

    rng = np.random.default_rng(seed)
    impedance_np = np.stack([_layered_impedance(rng, n_samples) for _ in range(n_traces)])
    # The demonstration uses normalized impedance, as is common before neural
    # network training.  The field workflow should store the training-set
    # min/max (or mean/std) and invert this transform after prediction.
    z_min = float(impedance_np.min())
    z_max = float(impedance_np.max())
    impedance_np = (impedance_np - z_min) / max(z_max - z_min, 1e-8)
    impedance = torch.from_numpy(impedance_np.astype(np.float32)).unsqueeze(1)

    low_np = np.stack([_smooth_trace(trace, window=31) for trace in impedance_np])
    low_model = torch.from_numpy(low_np.astype(np.float32)).unsqueeze(1)

    wavelet = ricker_wavelet(length=31, dominant_frequency=30.0, dt=0.001)
    forward_model = ConvolutionalForwardModel(wavelet)
    with torch.no_grad():
        seismic = forward_model(impedance)
    noise_generator = torch.Generator().manual_seed(seed + 1)
    seismic = seismic + noise_std * torch.randn(seismic.shape, generator=noise_generator)

    cfg = cwt_config or CWTConfig()
    cwt_np = cwt_transform(seismic.squeeze(1).numpy(), cfg, normalize=True)
    cwt_data = torch.from_numpy(cwt_np)
    return DatasetBundle(
        seismic=seismic.float(),
        cwt=cwt_data.float(),
        low_frequency_model=low_model.float(),
        impedance=impedance.float(),
    )


def split_bundle(bundle: DatasetBundle, train_fraction: float = 0.75) -> tuple[DatasetBundle, DatasetBundle]:
    """Split traces without mixing the time samples of a trace."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    n = bundle.seismic.shape[0]
    n_train = max(1, min(n - 1, int(round(n * train_fraction))))

    def select(indices: slice) -> DatasetBundle:
        return DatasetBundle(
            seismic=bundle.seismic[indices],
            cwt=bundle.cwt[indices],
            low_frequency_model=bundle.low_frequency_model[indices],
            impedance=bundle.impedance[indices],
        )

    return select(slice(0, n_train)), select(slice(n_train, n))
