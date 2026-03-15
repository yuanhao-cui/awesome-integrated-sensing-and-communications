"""RIS-ISAC Beamforming: SNR/CRB-Constrained Joint Beamforming and Reflection Designs.

Reference:
    Rang Liu et al., "SNR/CRB-Constrained Joint Beamforming and Reflection Designs
    for RIS-ISAC Systems," IEEE Trans. Wireless Commun., 2024.
    arXiv: https://arxiv.org/abs/2301.11134
"""

from .system_model import RIS_ISAC_System
from .channel_model import RISChannelModel
from .beamforming import BeamformingOptimizer
from .ris_phase import RISPhaseOptimizer
from .snr_constraint import SNRConstrainedSolver
from .crb_constraint import CRBConstrainedSolver
from .ao_solver import AlternatingOptimizationSolver

__all__ = [
    "RIS_ISAC_System",
    "RISChannelModel",
    "BeamformingOptimizer",
    "RISPhaseOptimizer",
    "SNRConstrainedSolver",
    "CRBConstrainedSolver",
    "AlternatingOptimizationSolver",
]
