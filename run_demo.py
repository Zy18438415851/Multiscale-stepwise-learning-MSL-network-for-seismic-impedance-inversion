"""Run a small synthetic MSL experiment.

Example:
    python run_demo.py --epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from msl import (
    CWTConfig,
    ConvolutionalForwardModel,
    MSLNetwork,
    TrainConfig,
    build_demo_dataset,
    predict,
    regression_metrics,
    ricker_wavelet,
    split_bundle,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    torch.manual_seed(7)
    args.output.mkdir(parents=True, exist_ok=True)

    cwt_cfg = CWTConfig(wavelet="morl", sample_interval=0.001, f_min=2, f_max=80, n_channels=80)
    bundle = build_demo_dataset(n_traces=24, n_samples=128, cwt_config=cwt_cfg)
    train_bundle, test_bundle = split_bundle(bundle, train_fraction=0.75)

    model = MSLNetwork()
    forward_model = ConvolutionalForwardModel(ricker_wavelet(length=31, dominant_frequency=30, dt=0.001))
    history = train_model(
        model,
        train_bundle,
        forward_model,
        TrainConfig(epochs=args.epochs, batch_size=4, device="cpu"),
    )
    prediction = predict(model, test_bundle)
    print("CWT channels:", bundle.cwt.shape)
    print("Training history:", history)
    print("Test metrics:", regression_metrics(prediction, test_bundle.impedance))

    torch.save(model.state_dict(), args.output / "msl_demo_weights.pt")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    idx = 0
    axes[0].plot(test_bundle.impedance[idx, 0].numpy(), label="target")
    axes[0].plot(prediction[idx, 0].numpy(), label="prediction")
    axes[0].set_title("Impedance")
    axes[0].legend()
    axes[1].imshow(test_bundle.cwt[idx].numpy(), aspect="auto", origin="lower")
    axes[1].set_title("CWT channels")
    axes[1].set_xlabel("Time sample")
    axes[1].set_ylabel("Channel")
    axes[2].plot(test_bundle.seismic[idx, 0].numpy())
    axes[2].set_title("Observed seismic")
    axes[2].set_xlabel("Time sample")
    fig.savefig(args.output / "msl_demo_result.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
