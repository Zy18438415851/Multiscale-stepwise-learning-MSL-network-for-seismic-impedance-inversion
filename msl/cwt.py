"""Continuous-wavelet preprocessing used by the MSL example.

The implementation follows the configuration described in the manuscript:
real Morlet wavelet (``morl``), 1-ms sampling interval, and a retained
frequency range of 2--80 Hz.  CWT is intentionally implemented as an offline
preprocessing step because PyWavelets' CWT is not differentiable in PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pywt


@dataclass(frozen=True)
class CWTConfig:
    """Configuration for converting seismic traces to CWT channels."""

    wavelet: str = "morl"
    sample_interval: float = 0.001
    f_min: float = 2.0
    f_max: float = 80.0
    n_channels: int = 80

    def frequencies(self) -> np.ndarray:
        """Return monotonically increasing pseudo-frequencies in Hz.

        ``n_channels=80`` means 80 uniformly sampled frequencies between
        2 and 80 Hz.  If the production implementation uses the integer
        frequencies 2, 3, ..., 80 Hz, set ``n_channels=79`` instead.
        """

        if self.n_channels < 1:
            raise ValueError("n_channels must be positive")
        return np.linspace(self.f_min, self.f_max, self.n_channels, dtype=np.float64)

    def scales(self) -> np.ndarray:
        """Convert the requested pseudo-frequencies to CWT scales."""

        central_frequency = pywt.central_frequency(self.wavelet)
        return central_frequency / (self.frequencies() * self.sample_interval)


def _as_batch_traces(data: np.ndarray | Iterable[float]) -> np.ndarray:
    traces = np.asarray(data, dtype=np.float64)
    if traces.ndim == 1:
        traces = traces[None, :]
    if traces.ndim != 2:
        raise ValueError(f"Expected [batch, time] traces, got shape {traces.shape}")
    return traces


def cwt_transform(
    data: np.ndarray | Iterable[float],
    config: CWTConfig | None = None,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Compute CWT coefficients with output shape ``[B, C, T]``.

    Parameters
    ----------
    data:
        Seismic traces with shape ``[T]`` or ``[B, T]``.
    config:
        CWT frequency and sampling configuration.
    normalize:
        If true, normalize each trace by its own maximum absolute coefficient.
        This changes absolute amplitude but preserves the within-trace relative
        energy pattern.  It is disabled by default.
    """

    cfg = config or CWTConfig()
    traces = _as_batch_traces(data)
    scales = cfg.scales()
    coefficients = []
    for trace in traces:
        coeffs, pseudo_frequencies = pywt.cwt(
            trace,
            scales,
            cfg.wavelet,
            sampling_period=cfg.sample_interval,
        )
        # Scales were constructed from the requested frequency grid; use the
        # explicit grid as the channel definition and keep low-to-high order.
        del pseudo_frequencies
        coeffs = np.asarray(coeffs, dtype=np.float32)
        if normalize:
            scale = np.max(np.abs(coeffs))
            if scale > 0:
                coeffs = coeffs / scale
        coefficients.append(coeffs)
    return np.stack(coefficients, axis=0)


def cwt_channel_summary(config: CWTConfig | None = None) -> dict[str, object]:
    """Return useful metadata for a README or experiment log."""

    cfg = config or CWTConfig()
    return {
        "wavelet": cfg.wavelet,
        "central_frequency_cycles_per_sample": float(pywt.central_frequency(cfg.wavelet)),
        "sample_interval_s": cfg.sample_interval,
        "frequency_range_hz": [cfg.f_min, cfg.f_max],
        "number_of_channels": cfg.n_channels,
        "frequencies_hz": cfg.frequencies().tolist(),
        "scales": cfg.scales().tolist(),
    }
