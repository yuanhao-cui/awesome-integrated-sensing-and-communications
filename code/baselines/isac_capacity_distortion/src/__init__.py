"""
ISAC Capacity-Distortion Tradeoff Baseline

Implements the CRB-rate region analysis for integrated sensing and
communications (ISAC) under Gaussian channels, based on:

Y. Xiong, F. Liu, Y. Cui, W. Yuan, et al.,
"On the Fundamental Tradeoff of Integrated Sensing and Communications
Under Gaussian Channels,"
IEEE Transactions on Information Theory, 2023.
arXiv: https://arxiv.org/abs/2204.06938
"""

try:
    from .system_model import (
        GaussianISACChannel,
        compute_bfim,
        compute_crb,
        compute_rate,
    )
    from .bounds import (
        pentagon_inner_bound,
        gaussian_inner_bound,
        semi_unitary_inner_bound,
        outer_bound,
    )
    from .optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
        stiefel_sample,
    )
except ImportError:
    from system_model import (
        GaussianISACChannel,
        compute_bfim,
        compute_crb,
        compute_rate,
    )
    from bounds import (
        pentagon_inner_bound,
        gaussian_inner_bound,
        semi_unitary_inner_bound,
        outer_bound,
    )
    from optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
        stiefel_sample,
    )

__version__ = "0.1.0"
