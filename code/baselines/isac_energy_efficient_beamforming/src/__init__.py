"""
Energy-Efficient Beamforming Design for ISAC Systems
=====================================================

Implementation of algorithms from:
Zou, Sun, Masouros, Cui - IEEE Trans. Commun., 2024

Modules:
    system_model: ISAC system model (SINR, channels, steering vectors)
    ee_metrics: Energy efficiency metrics (EE_C, EE_S, CRB)
    dinkelbach_solver: Dinkelbach method for fractional programming
    quadratic_transform: Quadratic transform for log-SINR terms
    sdr_solver: Semidefinite relaxation with rank-1 recovery
    sca_solver: Successive convex approximation solver
    schur_complement: Schur complement for LMI constraints
    pareto_optimizer: Pareto boundary search (Algorithm 4)
    baselines: Baseline schemes for comparison
"""

from .system_model import ISACSystemModel
from .ee_metrics import compute_ee_c, compute_ee_s, compute_crb, compute_sinr
from .dinkelbach_solver import DinkelbachSolver
from .pareto_optimizer import ParetoOptimizer
from .baselines import EMaxBaseline, FixBeamBaseline, RandomBaseline

__all__ = [
    "ISACSystemModel",
    "compute_ee_c",
    "compute_ee_s",
    "compute_crb",
    "compute_sinr",
    "DinkelbachSolver",
    "ParetoOptimizer",
    "EMaxBaseline",
    "FixBeamBaseline",
    "RandomBaseline",
]

__version__ = "1.0.0"
