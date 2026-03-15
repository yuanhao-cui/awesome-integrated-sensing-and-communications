# Cooperative Cell-Free ISAC Networks

Implementation of **"Cooperative Cell-Free ISAC Networks: Joint BS Mode Selection and Beamforming Design"** (IEEE Trans. Wireless Commun., 2024).

## Overview

This baseline implements a cell-free ISAC network where multiple base stations (BSs) cooperatively serve communication users while performing sensing tasks. Each BS can operate in either **communication mode** or **sensing mode**. The key contribution is the joint optimization of:

1. **BS Mode Selection** — which BSs serve communication vs. sensing
2. **Beamforming Design** — transmit beamformers for each BS
3. **Alternating Optimization** — solving the joint non-convex problem

## System Model

- **Cell-free architecture**: Multiple distributed BSs cooperate without cell boundaries
- **Dual-mode BSs**: Each BS selects communication or sensing mode
- **Cooperative sensing**: Multiple sensing BSs jointly estimate target parameters
- **Communication**: Users receive data from communication-mode BSs

## Key Algorithms

| Module | Description |
|--------|-------------|
| `mode_selection.py` | Greedy and relaxation-based BS mode selection |
| `beamforming.py` | MMSE and max-SINR beamforming design |
| `cooperative.py` | Multi-BS cooperative sensing with data fusion |
| `ao_solver.py` | Alternating optimization solver |
| `metrics.py` | Rate, CRB, coverage metrics |

## Usage

```python
from src.system_model import CellFreeISACSystem
from src.ao_solver import AlternatingOptimizationSolver

# Create system
system = CellFreeISACSystem(n_bs=8, n_users=4, n_targets=2)

# Run optimization
solver = AlternatingOptimizationSolver(system)
result = solver.solve()

print(f"Sum Rate: {result['sum_rate']:.2f} bps/Hz")
print(f"Average CRB: {result['avg_crb']:.6f}")
```

## Reference

```bibtex
@article{cooperative_cellfree_isac_2024,
  title={Cooperative Cell-Free ISAC Networks: Joint BS Mode Selection and Beamforming Design},
  journal={IEEE Transactions on Wireless Communications},
  year={2024},
  note={arXiv:2305.10800}
}
```
