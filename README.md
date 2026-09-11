# Reproducible CWT-assisted MSL seismic impedance inversion

This repository is a self-contained, non-confidential demonstration of the

continuous-wavelet-transform-assisted multiscale stepwise learning (MSL)

workflow described in the manuscript. It generates synthetic traces, applies

CWT, trains the MSL network with impedance and forward-model consistency

losses, and evaluates a held-out set.

The real 3-D seismic and well-log datasets are proprietary and are therefore

not included. To apply the workflow to the field data, replace

`msl.data.build_demo_dataset` with a loader that returns the same tensor

shapes.

## Contents

```text
msl_reproducibility/
├── msl/
│   ├── cwt.py             # Morlet CWT and frequency-channel construction
│   ├── data.py            # non-confidential synthetic demonstration data
│   ├── forward_model.py   # impedance -> reflectivity -> seismic convolution
│   ├── model.py           # Step 1, fusion, residual Step 2, layer table
│   └── training.py        # losses, training loop, prediction, metrics
├── notebooks/
│   └── 01_msl_demo.ipynb  # end-to-end executable example
├── run_demo.py            # command-line entry point
├── requirements.txt
└── README.md
```

## Installation and execution

```bash
python -m pip install -r requirements.txt
python run_demo.py --epochs 2
```

The command writes demonstration weights and a result figure to `outputs/`.

The notebook can be opened with Jupyter:

```bash
jupyter notebook notebooks/01_msl_demo.ipynb
```

The demonstration impedance profiles are min--max normalized before training.

For field data, save the training-set normalization parameters and invert the

normalization after inference.

## Tensor conventions

All network tensors use `[batch, channels, time]` ordering.

* Input seismic trace: `[B, 1, T]`.
  
* CWT representation: `[B, 80, T]`, using the real Morlet wavelet, a 1-ms
  
  sample interval, and 80 uniformly sampled pseudo-frequency channels between
  
  2 and 80 Hz.
  
* Four frequency groups: four `[B, 20, T]` tensors.
  
* Step 1: branch 1 receives 20 channels; branches 2--4 concatenate their
  
  20-channel band with the preceding branch output and therefore receive 40
  
  channels. Each branch follows `in-256-128-64-32-20`.
  
* Four Step 1 outputs and a 20-channel projection of the low-frequency initial
  
  model are fused into `[B, 100, T]`.
  
* Step 2 follows `100-256-128-64-32-1` using an initial convolution, three
  
  residual blocks, and a final prediction convolution.
  
All temporal convolutions use stride 1. Odd kernels use symmetric padding

`(kernel_size-1)/2`; thus the temporal dimension remains `T`. In particular,

the Step 2 kernel size 3 uses padding 1.

## Important frequency-channel note

The example uses 80 uniformly spaced channels from 2 to 80 Hz. If the

production implementation uses the integer frequencies 2, 3, ..., 80 Hz,

there are 79 channels; set `n_channels=79` and update the network channel

configuration accordingly. The manuscript and repository should use one

consistent convention.

## Forward-model constraint

The differentiable forward operator computes normal-incidence reflectivity

from the predicted impedance and convolves it with a fixed Ricker wavelet.

The training objective is the sum of impedance supervision and a weighted

synthetic-seismic reconstruction loss. Replace the demonstration wavelet with

the estimated field wavelet when reproducing the field workflow.

## Replacing the demonstration data

Prepare arrays with these shapes:

```text
seismic:              [N, 1, T]
low_frequency_model:  [N, 1, T]
impedance labels:     [N, 1, T]  # only where well labels are available
```

Run `cwt_transform(seismic[:, 0, :], CWTConfig(...))` to obtain `[N, 80, T]`.

For a real blind-well experiment, keep the blind well out of training,

normalization/statistics estimation, wavelet estimation, and hyperparameter

selection, as described in the manuscript.

