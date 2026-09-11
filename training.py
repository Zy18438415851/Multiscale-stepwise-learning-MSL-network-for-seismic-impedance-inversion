"""Training and evaluation utilities for the MSL demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle
from .forward_model import ConvolutionalForwardModel, forward_consistency_loss
from .model import MSLNetwork


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 5e-3
    weight_decay: float = 1e-6
    forward_loss_weight: float = 0.1
    device: str = "cpu"


def _loader(bundle: DatasetBundle, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(bundle.cwt, bundle.low_frequency_model, bundle.seismic, bundle.impedance)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: MSLNetwork,
    bundle: DatasetBundle,
    forward_model: ConvolutionalForwardModel,
    config: TrainConfig | None = None,
) -> list[dict[str, float]]:
    """Train with impedance supervision plus forward consistency."""

    cfg = config or TrainConfig()
    device = torch.device(cfg.device)
    model.to(device)
    forward_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(cfg.epochs):
        running_total = 0.0
        running_impedance = 0.0
        running_forward = 0.0
        n_batches = 0
        for cwt, low_model, seismic, target in _loader(bundle, cfg.batch_size, shuffle=True):
            cwt = cwt.to(device)
            low_model = low_model.to(device)
            seismic = seismic.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(cwt, low_model)
            impedance_loss = nn.functional.mse_loss(prediction, target)
            seismic_loss = forward_consistency_loss(prediction, seismic, forward_model)
            total_loss = impedance_loss + cfg.forward_loss_weight * seismic_loss
            total_loss.backward()
            optimizer.step()

            running_total += float(total_loss.detach())
            running_impedance += float(impedance_loss.detach())
            running_forward += float(seismic_loss.detach())
            n_batches += 1

        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": running_total / n_batches,
                "impedance_loss": running_impedance / n_batches,
                "forward_loss": running_forward / n_batches,
            }
        )
    return history


@torch.no_grad()
def predict(model: MSLNetwork, bundle: DatasetBundle, device: str = "cpu") -> Tensor:
    """Run inference and return [B, 1, T] impedance predictions."""

    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)
    return model(bundle.cwt.to(device_obj), bundle.low_frequency_model.to(device_obj)).cpu()


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Compute basic trace-wise aggregate metrics for the demo."""

    pred = prediction.detach().float().reshape(-1)
    true = target.detach().float().reshape(-1)
    mse = torch.mean((pred - true) ** 2)
    rmse = torch.sqrt(mse)
    covariance = torch.mean((pred - pred.mean()) * (true - true.mean()))
    correlation = covariance / (pred.std(unbiased=False) * true.std(unbiased=False) + 1e-8)
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "Correlation": float(correlation),
    }
