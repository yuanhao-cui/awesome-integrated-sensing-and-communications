# ISAC Resource Allocation Framework

Implementation of **"Sensing as a Service in 6G Perceptive Networks: A Unified Framework for ISAC Resource Allocation"**

## Authors
- Fuwang Dong, Fan Liu, Yuanhao Cui, Wei Wang, Kaifeng Han, Zhiqin Wang
- IEEE Transactions on Wireless Communications, 2022
- arXiv: https://arxiv.org/abs/2202.09969

## Overview

This framework provides a unified optimization platform for ISAC resource allocation with three sensing QoS metrics:

1. **Detection QoS** (Eq. 18-21): Probability of target detection under Neyman-Pearson criterion
2. **Localization QoS** (Eq. 22-31): Cramér-Rao Bound based position/orientation estimation
3. **Tracking QoS** (Eq. 44-47): Posterior Cramér-Rao Bound for sequential tracking

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.system_model import ISACSystem
from src.ao_solver import AOSolver
import numpy as np

# Create system
system = ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                    P_total=40.0, B_total=100e6)

# Solve for detection QoS (max-min fairness)
solver = AOSolver(system, qos_type='detection', fairness='maxmin')
result = solver.solve(Gamma_c=1.0)  # 1 bps/Hz rate threshold

print(f"Power allocation: {result['p']}")
print(f"Bandwidth allocation: {result['b']}")
print(f"Detection probabilities: {result['detection_probs']}")
```

## File Structure

```
├── src/
│   ├── system_model.py       # ISAC system model (Eq. 1-9)
│   ├── detection_qos.py      # Detection probability (Eq. 18-21)
│   ├── localization_qos.py   # CRB-based localization (Eq. 22-31)
│   ├── tracking_qos.py       # PCRB tracking (Eq. 44-47)
│   ├── comm_rate.py          # Communication rate (Eq. 9)
│   ├── ao_solver.py          # Alternating Optimization (Alg 1)
│   └── fairness.py           # Max-min and proportional fairness
├── tests/
│   └── test_*.py             # Unit and integration tests
├── examples/
│   ├── demo.ipynb            # Interactive demo
│   └── reproduce_fig*.py     # Reproduce paper figures
└── configs/
    └── default.yaml          # Default configuration
```

## References

Key equations from the paper:
- **Eq. 9**: Communication rate formula
- **Eq. 10**: Unified optimization problem
- **Eq. 18**: Detection probability under Neyman-Pearson
- **Eq. 22**: CRB for range estimation
- **Eq. 44**: Posterior FIM recursion for tracking
