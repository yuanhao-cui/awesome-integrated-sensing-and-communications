# Contributing to Awesome-ISAC

Thank you for your interest in contributing! This is an academic infrastructure project, and we maintain rigorous standards.

## 📖 Adding a Paper

1. Find the appropriate category file in `paper/`
2. Add the paper in this format:
   ```markdown
   | [Paper Title](URL) | Venue | Year | [Code](URL) | Focus |
   ```
3. Ensure the link works (we run automated link checks)
4. Submit a Pull Request

## 💻 Contributing a Baseline

Baselines must meet our **strict quality standards**:

### Code Structure
```
code/baselines/{method_name}/
├── README.md               # Math background + usage
├── requirements.txt        # Pinned versions
├── src/
│   ├── model.py            # Core mathematical model
│   ├── solver.py           # Optimization/solver
│   └── metrics.py          # Performance metrics
├── tests/
│   ├── test_model.py       # Model correctness
│   ├── test_solver.py      # Solver correctness
│   ├── test_metrics.py     # Metrics correctness
│   └── test_reproducibility.py  # Paper result reproduction
├── examples/
│   ├── demo.ipynb          # Interactive demo
│   └── reproduce_paper.py  # Reproduce key results
└── configs/
    └── default.yaml        # Parameters aligned with paper
```

### Requirements
- [ ] Python ≥3.10, no hardware dependencies
- [ ] ≥80% test coverage (`pytest --cov`)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Results match published paper (within tolerance)
- [ ] README with mathematical background
- [ ] Jupyter notebook demo
- [ ] Code follows PEP 8

### Review Process
1. Submit PR with the complete baseline
2. Automated CI runs all tests
3. **Maintainer review** (mathematical correctness + code quality)
4. Feedback → revision → approval → merge

## 🐛 Reporting Issues

Use the issue templates:
- **New Paper**: Recommend a paper to add
- **Bug Report**: Report a bug in baseline code
- **Suggestion**: General improvements

## 📊 Submitting Benchmark Results

> Coming soon. Will define standardized evaluation protocols.

## Code of Conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
