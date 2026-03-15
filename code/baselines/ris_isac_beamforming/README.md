# RIS-ISAC Beamforming: SNR/CRB-Constrained Joint Beamforming and Reflection Designs

Implementation of joint beamforming and RIS phase shift optimization for RIS-assisted ISAC systems.

## Reference

**Rang Liu et al.**, "SNR/CRB-Constrained Joint Beamforming and Reflection Designs for RIS-ISAC Systems," *IEEE Transactions on Wireless Communications*, 2024.
[arXiv:2301.11134](https://arxiv.org/abs/2301.11134)

## System Model

- **Multi-antenna BS** (M=4 antennas) assisted by a **passive RIS** (L=30 elements)
- **MU-MISO communications**: K=2 single-antenna users
- **Radar sensing**: Target detection and angle estimation

## Problem Formulations

### Problem 1: SNR-Constrained (Target Detection)
```
max  Σ_k R_k                (sum rate)
s.t. SNR_sensing ≥ γ_min     (detection requirement)
     SINR_k ≥ γ_k            (communication QoS)
     Σ||w_k||² ≤ P_max      (power budget)
     |θ_l| = 1, ∀l           (RIS unit-modulus)
```

### Problem 2: CRB-Constrained (Parameter Estimation)
```
max  Σ_k R_k                (sum rate)
s.t. CRB(φ) ≤ ε_max         (estimation accuracy)
     SINR_k ≥ γ_k            (communication QoS)
     Σ||w_k||² ≤ P_max      (power budget)
     |θ_l| = 1, ∀l           (RIS unit-modulus)
```

## Algorithm

**Alternating Optimization (AO)** with:
- **SDR** (Semidefinite Relaxation) for beamforming optimization
- **Coordinate ascent** for RIS phase optimization
- **SCA** for non-convex constraint handling
- Solver fallback: MOSEK → SCS

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import RIS_ISAC_System, AlternatingOptimizationSolver

# Create system
system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)

# SNR-constrained solver
solver = AlternatingOptimizationSolver(system, problem_type='snr', snr_min_dB=5.0)
result = solver.solve()

print(f"Sum rate: {result['sum_rate']:.2f} bps/Hz")
print(f"Radar SNR: {10*np.log10(result['snr_sensing']):.1f} dB")
print(f"Converged: {result['converged']}")
```

## Parameters (Table I)

| Parameter | Value |
|-----------|-------|
| BS antennas M | 4 |
| UE antennas N | 2 |
| Data streams d | 2 |
| RIS elements L | 30 (vary 10-50) |
| P_max | 10 mW |
| SINR threshold | 10 dB |
| Bandwidth | 1 MHz |
| Rician factor K_R | 3 |
| Noise power | 3.98e-12 mW |

## Test Coverage (≥15 tests)

- `test_ris_unit_modulus`: |θ_l| = 1 for all elements
- `test_power_constraint`: Σ||w_k||² ≤ P_max
- `test_snr_constraint`: SNR_sensing ≥ γ_min when enforced
- `test_crb_constraint`: CRB ≤ ε_max when enforced
- `test_sum_rate_positive`: Sum rate > 0
- `test_ao_convergence`: AO converges
- `test_channel_dimensions`: Channel matrices have correct shapes

Run tests:
```bash
pytest tests/ -v
```
