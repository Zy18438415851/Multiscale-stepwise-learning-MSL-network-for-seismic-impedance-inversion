"""Reproducible CWT-assisted MSL seismic impedance inversion example."""

from .cwt import CWTConfig, cwt_channel_summary, cwt_transform
from .data import DatasetBundle, build_demo_dataset, split_bundle
from .forward_model import ConvolutionalForwardModel, impedance_to_reflectivity, ricker_wavelet
from .model import MSLConfig, MSLNetwork, architecture_rows
from .training import TrainConfig, predict, regression_metrics, train_model

__all__ = [
    "CWTConfig",
    "cwt_channel_summary",
    "cwt_transform",
    "DatasetBundle",
    "build_demo_dataset",
    "split_bundle",
    "ConvolutionalForwardModel",
    "impedance_to_reflectivity",
    "ricker_wavelet",
    "MSLConfig",
    "MSLNetwork",
    "architecture_rows",
    "TrainConfig",
    "predict",
    "regression_metrics",
    "train_model",
]
