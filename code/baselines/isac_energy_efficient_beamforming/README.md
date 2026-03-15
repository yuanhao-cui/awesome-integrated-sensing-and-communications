# Energy-Efficient Beamforming Design for ISAC

> Reproduction of: J. Zou, S. Sun, C. Masouros, **Y. Cui**, "Energy-Efficient Beamforming Design for Integrated Sensing and Communications Systems," IEEE Trans. Commun., 2024.
>
> [arXiv:2307.04002](https://arxiv.org/abs/2307.04002) | [IEEE](https://ieeexplore.ieee.org/document/10393498)

## Mathematical Background

### System Model
ISAC BS with M Tx antennas, K single-antenna users, N Rx antennas for sensing.

- **Communication EE**: EE_C = Σ_k log₂(1+SINR_k) / ((1/ε)Σ||w_k||² + P₀)
- **Sensing EE**: EE_S = CRB⁻¹ / (L·((1/ε)Σ||w_k||² + P₀))

### Algorithms Implemented
1. **Dinkelbach + Quadratic Transform** (Section III): Comm-EE maximization
2. **SCA** (Section IV-B): Sensing-EE maximization  
3. **Pareto Optimization** (Section V): Comm-EE ↔ Sensing-EE tradeoff

## Quick Start

```bash
cd code/baselines/isac_energy_efficient_beamforming
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Known Issues
- `test_sensing_ee_power_constraint`: SCA for sensing-centric EE may slightly violate power constraint in edge cases (numerical)
- `test_crb_reproducibility`: CRB computation has numerical instability for certain channel realizations

## Project Structure
- `src/system_model.py`: ISAC system model (SINR, channels)
- `src/ee_metrics.py`: EE_C, EE_S, CRB computation
- `src/dinkelbach_solver.py`: Dinkelbach method (Algorithm 1)
- `src/sca_solver.py`: SCA iterations (Algorithm 3)
- `src/pareto_optimizer.py`: Pareto boundary (Algorithm 4)
- `tests/`: 71 unit tests (69 passing)

## Test Status: 69/71 ✅
