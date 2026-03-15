# ISAC Capacity-Distortion Tradeoff Baseline

Implementation of the CRB-rate region analysis for Integrated Sensing and Communications (ISAC) under Gaussian channels.

## Reference Paper

**Title:** "On the Fundamental Tradeoff of Integrated Sensing and Communications Under Gaussian Channels"

**Authors:** Y. Xiong, F. Liu, Y. Cui, W. Yuan, et al.

**Venue:** IEEE Transactions on Information Theory, 2023

**arXiv:** https://arxiv.org/abs/2204.06938

## Overview

This baseline implements the fundamental tradeoff between sensing quality (measured by Bayesian Cramér-Rao Bound, BCRB) and communication rate (measured in nats/channel use) for point-to-point ISAC systems under Gaussian channels.

### Key Concepts

1. **CRB-Rate Region:** The set of all achievable pairs (e, R) where e is the BCRB and R is the communication rate.

2. **Two-Fold Tradeoff:**
   - **Subspace Tradeoff (ST):** Arises from resource allocation between sensing and communication subspaces.
   - **Deterministic-Random Tradeoff (DRT):** Arises from the choice between deterministic (sensing-optimal) and random (communication-optimal) waveform structures.

3. **Corner Points:**
   - **P_sc = (e_min, R_sc):** Sensing-constrained capacity point.
   - **P_cs = (e_cs, R_max):** Communication-constrained minimum CRB point.

## System Model

### Gaussian ISAC Channel (Eq. 2)

```
Communication: Y_c = H_c X + Z_c
Sensing:       Y_s = H_s X + Z_s
```

Where:
- **X** ∈ C^(M×T): Transmitted waveform
- **H**c ∈ C^(Nc×M): Communication channel matrix
- **H**s ∈ C^(Ns×M): Sensing channel matrix (depends on parameter η)
- **Z**c, **Z**s: i.i.d. circularly symmetric complex Gaussian noise

### Key Quantities

- **Ergodic Rate** (Eq. 4): `R = log|I + σc^{-2} Hc Rx Hc^H|`
- **Bayesian CRB** (Eq. 7): `e = tr{J^{-1}}` where J is the BFIM
- **BFIM** (Eq. 11): `J = (T/σs²) Φ(Rx)`

## Bounds

### Pentagon Inner Bound (Proposition 1)
The achievable region contains all points satisfying:
- e ≥ e_min
- R ≤ R_max
- e ≥ e_min + (e_cs - e_min)/(R_max - R_sc) × (R - R_sc)

### Gaussian Inner Bound
Assumes i.i.d. Gaussian signaling: X columns ~ CN(0, Rx_bar).

### Semi-Unitary Inner Bound
Uses uniform distribution over the Stiefel manifold for deterministic waveform structure.

### Outer Bound
Based on statistical covariance shaping optimization (Eq. 48).

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- numpy ≥ 1.24.0
- scipy ≥ 1.10.0
- cvxpy ≥ 1.3.0
- matplotlib ≥ 3.7.0

## Usage

### Basic Example

```python
import numpy as np
from src import (
    GaussianISACChannel,
    compute_rate,
    compute_crb,
    optimize_sensing_rx,
    optimize_comm_rx,
    covariance_shaping,
    gaussian_inner_bound,
    outer_bound,
)

# Setup channel
M, Nc, Ns, T = 10, 1, 10, 3
Hc = np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)
Hc /= np.linalg.norm(Hc)

# Compute corner points
Rx_sense = optimize_sensing_rx(P_T=1.0, M=M)
Rx_comm = optimize_comm_rx(P_T=1.0, M=M, Hc=Hc)

# Compute rate and CRB
R_sc = compute_rate(Rx_sense, Hc, sigma_c2=0.001)
R_max = compute_rate(Rx_comm, Hc, sigma_c2=0.001)

print(f"Rate at sensing-optimal: {R_sc:.4f}")
print(f"Maximum rate: {R_max:.4f}")
```

### Reproducing Paper Figures

```bash
cd examples
python reproduce_figures.py --output-dir results --figures 5,8,10
```

### Running Tests

```bash
cd tests
python -m pytest -v
```

## Project Structure

```
isac_capacity_distortion/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── __init__.py
│   ├── system_model.py     # Channel model, BFIM, CRB, rate
│   ├── bounds.py           # Inner and outer bounds
│   ├── optimization.py     # Optimization routines
│   └── case_study.py       # Paper figure generation
├── tests/                  # Test suite
│   ├── test_system_model.py
│   ├── test_bounds.py
│   ├── test_optimization.py
│   └── test_reproducibility.py
├── examples/               # Example scripts
│   └── reproduce_figures.py
└── results/                # Output directory for figures
```

## Parameters (Table I)

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 10 | Number of Tx antennas |
| Ns | 10 | Number of sensing Rx antennas |
| Nc | 1 | Number of communication Rx antennas |
| d | 0.5λ | Antenna spacing |
| Sensing SNR | 20 dB | Max sensing receiving SNR per antenna |
| Comm SNR | 33 dB | Max communication receiving SNR |

## Parameters (Table II)

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 4 | Number of Tx antennas |
| Ns | 4 | Number of sensing Rx antennas |
| Nc | 4 | Number of communication Rx antennas |
| σs² | 1 | Sensing noise variance |
| Sensing SNR | 24 dB | Sensing transmit SNR |
| Comm SNR | 24 dB | Communication transmit SNR |

## Key Equations

| Eq. | Description | Formula |
|-----|-------------|---------|
| (2) | System model | Y = HX + Z |
| (4) | Ergodic rate | R = log\|I + σc^{-2} Hc Rx Hc^H\| |
| (7) | Bayesian CRB | e = tr{J^{-1}} |
| (11) | BFIM | J = (T/σs²) Φ(Rx) |
| (14) | Sensing optimization | min tr{Φ(Rx)^{-1}} s.t. power |
| (48) | Covariance shaping | min (1-α)CRB - α·Rate |

## License

This implementation is provided for academic research purposes.
Please cite the original paper when using this code.

```bibtex
@article{xiong2023fundamental,
  title={On the fundamental tradeoff of integrated sensing and communications under {Gaussian} channels},
  author={Xiong, Yifeng and Liu, Fan and Cui, Yuanhao and Yuan, Wei and others},
  journal={IEEE Transactions on Information Theory},
  year={2023}
}
```
