# 🏆 Leaderboard

> Standardized benchmark results for ISAC methods.

## Evaluation Protocol

All baselines are evaluated on the following metrics:

| Metric | Description | Unit |
|--------|-------------|------|
| BER | Bit Error Rate (communication performance) | - |
| CRB | Cramér-Rao Bound (sensing accuracy) | rad² / m² |
| Rate | Spectral Efficiency | bps/Hz |
| Pareto Score | Trade-off quality (Area under Pareto curve) | - |

## Results

> 🚧 No results yet. Baselines are being developed.

| Method | BER @ 20dB SNR | CRB (Delay) | Rate | Pareto | Date |
|--------|----------------|-------------|------|--------|------|
| - | - | - | - | - | - |

## How to Submit

1. Run `python benchmark/evaluate.py --method your_method`
2. Submit the generated `results.json` via PR
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details
