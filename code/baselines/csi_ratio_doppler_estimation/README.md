# CSI-Ratio-based Doppler Frequency Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Reference implementation of **three algorithms** for Doppler frequency estimation using CSI-ratio in Integrated Sensing and Communications (ISAC).

> 📄 **Paper**: "CSI-Ratio-based Doppler Frequency Estimation in Integrated Sensing and Communications"  
> 👤 **Authors**: J. Andrew Zhang, Yuanhao Cui, et al.

---

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd csi_ratio_doppler_estimation

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scipy

# Run tests
pytest tests/

# Generate simulation figures
python examples/generate_figures.py
```

---

## Algorithm Overview

This repository implements three complementary algorithms for estimating Doppler frequency from CSI-ratio measurements:

### Algorithm 1: Möbius Transformation-based ⭐ (Primary)

**Estimates**: Signed Doppler frequency (can distinguish approaching vs receding)

**Steps**:
1. Compute CSI-ratio: `R(t) = H_m(t) / H_{m+1}(t)`
2. Fit circle to `R(t)` in complex plane via least-squares
3. Shift circle to origin: `R_s(t) = R(t) - C_0`
4. Calculate angle `θ_R(t) = arg(R_s(t))` and magnitude `a_R(t) = |R_s(t)|`
5. Weighted linear regression: `θ_R(t) = β_0 + β_1 * t` (weights = `a_R`)
6. `f_D = β_1 / (2π)`

**Key advantage**: The only method that preserves the sign of Doppler frequency.

### Algorithm 2: Periodicity-based

**Estimates**: `|f_D|` (magnitude only)

**Steps**:
1. Extract phase angle `γ(t) = arg(R(t))`
2. Search for zero-crossings relative to starting angle
3. `f_D = 1 / (S * T_s)` where `S` = cycle length in samples
4. Average all estimates

**Limitations**: Cannot determine sign; requires sufficient samples for at least one full cycle.

### Algorithm 3: Signal Difference-based

**Estimates**: `|f_D|` (magnitude only)

**Steps**:
1. For each lag `n`, compute average squared difference:  
   `Δ_Σ(n) = (1/(N-n)) * Σ |R(k+n) - R(k)|²`
2. Find `n* = argmin Δ_Σ(n)` (lag with minimum difference)
3. `f_D = 1 / (n* * T_s)`

**Limitations**: Cannot determine sign; resolution limited by sampling interval.

---

## Results

### Figure B1: CSI-Ratio Circle in Complex Plane

Synthetic CSI samples with known Doppler (50 Hz) form a perfect circle in the complex plane. The CSI-ratio cancels common phase terms (CFO, TMO, phase noise).

![B1 — CSI-Ratio Circle](./results/B1_csi_ratio_circle.png)

### Figure B2: Doppler Estimation Comparison

Real-time comparison of all three algorithms using a sliding window (100 ms). The Möbius-based method (Algorithm 1) tracks the true value most accurately and preserves the sign.

![B2 — Estimation Comparison](./results/B2_estimation_comparison.png)

### Figure B3: Estimation Error vs SNR

Median absolute error across 20 trials per SNR level. The Möbius-based estimator maintains sub-Hz accuracy even at low SNR.

![B3 — Error vs SNR](./results/B3_error_vs_snr.png)

### Figure B4: CSI-Ratio Trajectory & Circle Fitting

Visualization of the CSI-ratio trajectory (left) and the shifted circle used for Doppler extraction (right). The arrow indicates the rotation direction corresponding to positive Doppler.

![B4 — Trajectory & Circle Fitting](./results/B4_trajectory_circle_fit.png)

---

## API Reference

### Signal Model

```python
from signal_model import csi_with_doppler

# Generate synthetic CSI for two antennas with known Doppler
H1, H2 = csi_with_doppler(
    t,                # Time samples (np.ndarray)
    f_D=50.0,         # Doppler frequency (Hz)
    snr_db=30.0,      # Signal-to-noise ratio (dB)
    amplitude_ratio=1.2,
    phase_offset=np.pi/6,
    cfo_hz=50.0,      # Carrier frequency offset (Hz)
    tmo_hz=10.0       # Timing misalignment offset (Hz)
)
```

### CSI-Ratio Computation

```python
from csi_ratio import compute_csi_ratio, compute_csi_ratio_multi

# Single antenna pair
R = compute_csi_ratio(H_m, H_m1)  # H_m / H_{m+1}

# Multiple antenna pairs
R = compute_csi_ratio_multi(H, ref_antenna=0)  # Shape: (N, M-1)
```

### Doppler Estimators

#### Algorithm 1: Möbius-based (Signed)

```python
from mobius_estimator import mobius_doppler_estimate

result = mobius_doppler_estimate(
    R,                    # CSI-ratio samples
    T_s=0.0005,           # Sampling interval (s)
    circle_method="least_squares",  # "least_squares" | "kasa" | "pratt"
    unwrap_phases=True
)

# Returns:
#   result['f_D']           → Signed Doppler frequency (Hz)
#   result['f_D_magnitude'] → |f_D| (Hz)
#   result['direction']     → "approaching" or "receding"
#   result['center_A']      → Circle center (real)
#   result['center_B']      → Circle center (imag)
#   result['radius']        → Circle radius
#   result['r_squared']     → R² of linear fit
```

#### Algorithm 2: Periodicity-based (Magnitude only)

```python
from periodicity_estimator import periodicity_doppler_estimate

result = periodicity_doppler_estimate(
    R,
    T_s=0.0005,
    reference_method="start"  # "start" | "mean"
)

# Returns:
#   result['f_D']        → |f_D| (Hz), always non-negative
#   result['direction']  → "unknown"
#   result['num_crossings'] → Number of detected crossings
```

#### Algorithm 3: Difference-based (Magnitude only)

```python
from difference_estimator import difference_doppler_estimate

result = difference_doppler_estimate(
    R,
    T_s=0.0005,
    max_lag=None,        # Default: N // 2
    use_magnitude=False  # Use complex R or |R|
)

# Returns:
#   result['f_D']         → |f_D| (Hz)
#   result['n_star']      → Optimal lag (samples)
#   result['delta_sigma'] → Difference function array
```

### Circle Fitting

```python
from circle_fit import (
    least_squares_circle_fit,  # Eq. 11 from paper
    fit_circle_kasa,           # Algebraic fit
    fit_circle_pratt,          # Iterative refinement
    circle_fit_error
)

A, B, r = least_squares_circle_fit(R)  # Center (A, B), radius r
rms_error = circle_fit_error(R, A, B, r)
```

---

## Project Structure

```
csi_ratio_doppler_estimation/
├── src/                          # Core implementation
│   ├── __init__.py              # Package exports
│   ├── signal_model.py          # CSI signal generation (Eq. 2, 5)
│   ├── csi_ratio.py             # CSI-ratio computation (Eq. 6, 8)
│   ├── mobius_estimator.py      # Algorithm 1: Möbius-based (signed)
│   ├── periodicity_estimator.py # Algorithm 2: Periodicity-based
│   ├── difference_estimator.py  # Algorithm 3: Difference-based
│   ├── circle_fit.py            # Circle fitting (Eq. 11)
│   └── visualization.py         # Plotting utilities
├── tests/                        # Unit tests
│   ├── test_csi_ratio.py
│   ├── test_mobius.py
│   └── test_circle_fit.py
├── examples/                     # Usage examples
│   └── generate_figures.py      # Generate all simulation figures
├── results/                      # Generated figures
│   ├── B1_csi_ratio_circle.png
│   ├── B2_estimation_comparison.png
│   ├── B3_error_vs_snr.png
│   └── B4_trajectory_circle_fit.png
└── README.md                     # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2024csi,
  title={CSI-Ratio-based Doppler Frequency Estimation in Integrated Sensing and Communications},
  author={Zhang, J. Andrew and Cui, Yuanhao and others},
  journal={IEEE Transactions on Communications},
  year={2024},
  publisher={IEEE}
}
```

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

## Acknowledgments

This implementation is based on the theoretical framework established in the paper "CSI-Ratio-based Doppler Frequency Estimation in Integrated Sensing and Communications". The CSI-ratio approach elegantly cancels common impairments (CFO, TMO, phase noise) that plague conventional CSI-based sensing systems.
