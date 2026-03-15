# Near-Field Beam Training for XL-MIMO Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

Official PyTorch implementation for:

> **Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning**
>
> J. Nie, Y. Cui et al.
>
> *IEEE Transactions on Mobile Computing (TMC)*, 2025
>
> 📄 [arXiv:2406.03249](https://arxiv.org/abs/2406.03249)

## 📋 Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Expected Results](#expected-results)
- [Citation](#citation)
- [License](#license)

## Overview

This repository implements a **deep learning-based near-field beam training** scheme for extremely large-scale MIMO (XL-MIMO) systems. In the near-field region, the spherical wavefront model requires beam training over both **distance and angle** dimensions, making conventional far-field DFT codebooks insufficient.

Our approach uses a **UNet-like CNN** to directly map estimated CSI to phase-only beamforming vectors, achieving near-optimal spectral efficiency with low computational overhead.

### Key Contributions

- **CNN-based beam training**: Maps estimated CSI → analog beamforming phases end-to-end
- **Rate-driven loss function**: Directly optimizes spectral efficiency (not proxy metrics)
- **Near-field aware**: Designed for spherical wave propagation in XL-MIMO
- **Low complexity**: Real-time inference suitable for practical deployment

## Mathematical Background

### Near-Field Channel Model

In the near-field region, the channel follows the **spherical wave model**:

$$h_n = \frac{\alpha}{r_n} \exp\left(-j \frac{2\pi}{\lambda} r_n\right)$$

where $r_n = \sqrt{r^2 + d_n^2 - 2rd_n\sin\theta}$ is the distance from antenna $n$ to the user, accounting for the spherical wavefront.

### Problem Formulation

Given estimated CSI $\hat{\mathbf{h}} \in \mathbb{C}^{N_t}$, find phase-only beamforming vector $\mathbf{v} = [e^{j\phi_1}, \ldots, e^{j\phi_{N_t}}]^T$ that maximizes:

$$R = \log_2\left(1 + \frac{\rho}{N_t} |\mathbf{h}^H \mathbf{v}|^2\right)$$

where $\rho$ is the SNR and $N_t$ is the number of antennas.

### CNN Architecture

The model processes the estimated CSI as input:
- **Input**: Real and imaginary parts concatenated → $(1, 2, N_t)$ tensor
- **Encoder**: 3 convolutional blocks with AvgPool downsampling
- **Decoder**: 2 transposed convolution blocks with skip-like structure
- **Output**: $N_t$ phase values via linear layer + tanh activation → mapped to unit-norm beamforming vector

```
Input (1, 2, 256)
    │
    ├──► [Conv-BN-ReLU]×2 ──► AvgPool ──► [Conv-BN-ReLU]×2 ──► AvgPool
    │                                                         ──► [Conv-BN-ReLU]×2
    │                                                              │
    │    [Conv-BN-ReLU]×2 ◄── ConvTranspose ◄─────────────────────┘
    │         │
    │    [Conv-BN-ReLU]×2 ◄── ConvTranspose
    │         │
    │      Flatten → Linear(512, 256) → Tanh
    │         │
    └──► Phase output (256,) → trans_vrf → Beamforming vector
```

## Project Structure

```
xl_mimo_beam_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── src/
│   ├── __init__.py
│   ├── model.py                 # CNN architecture (BeamTrainingNet)
│   ├── channel.py               # Near-field channel model (spherical wave)
│   ├── beamforming.py           # Beamforming codebook & precoding
│   ├── trainer.py               # Training pipeline
│   ├── evaluator.py             # Evaluation metrics & visualization
│   └── utils.py                 # Core algorithm (trans_vrf, rate_func)
├── tests/
│   ├── test_model.py            # Model architecture tests
│   ├── test_channel.py          # Channel model tests
│   ├── test_beamforming.py      # Beamforming tests
│   ├── test_trainer.py          # Training pipeline tests
│   └── test_end_to_end.py       # End-to-end integration tests
├── configs/
│   └── default.yaml             # Hyperparameters
├── examples/
│   ├── demo.ipynb               # Interactive Jupyter demo
│   └── reproduce_results.py     # Reproduce paper results
└── data/
    └── README.md                # Data preparation instructions
```

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy, SciPy, Matplotlib, scikit-learn

### Install from source

```bash
cd xl_mimo_beam_training
pip install -e ".[dev]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train with synthetic data

```python
from src.trainer import Trainer

config = {
    "num_antennas": 256,
    "batch_size": 100,
    "num_epochs": 200,
    "learning_rate": 0.001,
    "num_synthetic_samples": 5000,
    "checkpoint_dir": "checkpoints",
}

trainer = Trainer(config, device="cpu")
trainer.setup_model()
trainer.load_data()  # Uses synthetic data by default
history = trainer.train()
```

### 2. Evaluate the trained model

```python
from src.evaluator import Evaluator

evaluator = Evaluator.from_checkpoint("checkpoints/best_model.pth")
metrics = evaluator.evaluate_all_metrics(H_test, H_est_test)
evaluator.plot_rate_vs_snr(metrics["snr_dB"], metrics["spectral_efficiency"])
```

### 3. Run tests

```bash
cd tests
pytest -v
```

## Training

### With real data

Place `pcsi.mat` and `ecsi.mat` in the `data/` directory, then:

```bash
python examples/reproduce_results.py --data_path data --epochs 200 --device cpu
```

### With synthetic data

```bash
python examples/reproduce_results.py --samples 5000 --epochs 200 --device cuda
```

### Configuration

Edit `configs/default.yaml` or pass arguments via command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_antennas` | 256 | Number of transmit antennas |
| `batch_size` | 100 | Training batch size |
| `num_epochs` | 200 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `lr_patience` | 20 | LR scheduler patience |
| `num_synthetic_samples` | 5000 | Synthetic data samples |

## Evaluation Metrics

The evaluator computes:

1. **Spectral Efficiency**: Achievable rate $R = \log_2(1 + \frac{\rho}{N_t}|\mathbf{h}^H\mathbf{v}|^2)$
2. **Beamforming Gain**: $|\mathbf{h}^H\mathbf{v}|^2$ in dB
3. **Normalized MSE**: Between predicted and optimal (MRT) beamforming vectors
4. **Rate vs SNR curves**: Performance across SNR regimes [-20, 20] dB

## Expected Results

| SNR (dB) | Spectral Efficiency (bps/Hz) |
|----------|------------------------------|
| -20 | ~0.5 |
| -10 | ~1.8 |
| 0 | ~4.0 |
| 10 | ~6.5 |
| 20 | ~8.5 |

*Results on synthetic near-field channels with N_t = 256. Actual values may vary with channel conditions.*

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{nie2025near,
  title={Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning},
  author={Nie, Jingzhi and Cui, Yuanhao and others},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China. The authors thank the editor and anonymous reviewers for their constructive feedback.

---

<p align="center">
  Part of <a href="https://github.com/yuanhao-cui/awesome-integrated-sensing-and-communications">awesome-integrated-sensing-and-communications</a>
</p>
