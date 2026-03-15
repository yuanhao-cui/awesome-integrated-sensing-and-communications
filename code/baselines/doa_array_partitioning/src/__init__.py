"""
DOA Estimation-Oriented Joint Array Partitioning and Beamforming for ISAC Systems.

Reference:
    Rang Liu, A. Lee Swindlehurst et al.,
    "DOA Estimation-Oriented Joint Array Partitioning and Beamforming Designs for ISAC Systems,"
    IEEE Transactions on Wireless Communications, 2024.
    arXiv: https://arxiv.org/abs/2410.12923
"""

from system_model import ISACSystem
from doa_crb import compute_crb, compute_fisher_info
from array_partition import ArrayPartitioner
from beamforming import BeamformingOptimizer
from admm_solver import ADMMSolver
from mm_solver import MMSolver
from heuristic import HeuristicStrategy

__all__ = [
    "ISACSystem",
    "compute_crb",
    "compute_fisher_info",
    "ArrayPartitioner",
    "BeamformingOptimizer",
    "ADMMSolver",
    "MMSolver",
    "HeuristicStrategy",
]
