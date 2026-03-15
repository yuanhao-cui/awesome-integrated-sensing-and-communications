# Zak-OTFS for ISAC

> Reproduction of: S. K. Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O Relation and Data Communication," arXiv:2404.04182, 2024.

## Overview
OTFS modulation in delay-Doppler domain using Zak transform for predictable, non-fading I/O relation.

## Features
- OTFS modulation/demodulation (ISFFT + Heisenberg/Wigner)
- Zak transform implementation
- Point & spread pulsone generation
- Delay-Doppler ambiguity function
- PAPR analysis
- Channel estimation

## Quick Start
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Test Status: 18/21 ✅
- 3 pulsone spread tests have known issues (PAPR reduction not working as expected)
- Core OTFS, Zak, ambiguity, and channel modules all pass

## Known Issues
- Spread pulsone PAPR reduction needs debugging (shows 0 dB instead of ~6 dB)
