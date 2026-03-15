# 👋 Welcome to the ISAC Baselines

This directory contains rigorously-tested Python baselines for ISAC methods.

## Structure

Each baseline follows:
```
{method_name}/
├── README.md          # Math background + usage
├── requirements.txt   # Pinned dependencies
├── src/               # Source code
├── tests/             # Unit tests (≥80% coverage)
├── examples/          # Jupyter demos + reproduce scripts
└── configs/           # YAML configs
```

## Quick Start

```bash
# Run all tests
pytest code/baselines/ -v

# Run a specific baseline
cd code/baselines/ofdm_isac
pip install -r requirements.txt
pytest tests/ -v
python examples/reproduce_paper.py
```

## Adding a New Baseline

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the complete checklist.
