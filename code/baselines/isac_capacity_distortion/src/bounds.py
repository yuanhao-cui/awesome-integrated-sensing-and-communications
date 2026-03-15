"""
Inner and Outer Bounds for the CRB-Rate Region.

Implements:
- Pentagon inner bound (Proposition 1)
- Gaussian signaling inner bound
- Semi-unitary (Stiefel manifold) inner bound
- Outer bound via covariance shaping

References:
    Xiong et al., "On the Fundamental Tradeoff of Integrated Sensing
    and Communications Under Gaussian Channels," IEEE TIT, 2023.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, List, Optional, Tuple

try:
    from .system_model import (
        compute_bfim,
        compute_crb,
        compute_rate,
        compute_phi_angle,
    )
    from .optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
        stiefel_sample,
        generate_isotropic_waveform,
    )
except ImportError:
    from system_model import (
        compute_bfim,
        compute_crb,
        compute_rate,
        compute_phi_angle,
    )
    from optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
        stiefel_sample,
        generate_isotropic_waveform,
    )


def pentagon_inner_bound(
    P_sc: Tuple[float, float],
    P_cs: Tuple[float, float],
    e_min: float,
    R_max: float,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the pentagon inner bound (Proposition 1).

    Any point (e, R) satisfying the following is achievable:
        (a) e >= e_min
        (b) R <= R_max
        (c) e >= e_min + (e_cs - e_min) / (R_max - R_sc) * (R - R_sc)

    This is achieved by time-sharing between the strategies at
    P_sc and P_cs.

    Args:
        P_sc (tuple): Sensing-constrained corner point (e_min, R_sc).
        P_cs (tuple): Communication-constrained corner point (e_cs, R_max).
        e_min (float): Minimum achievable CRB.
        R_max (float): Maximum achievable rate.
        n_points (int): Number of points for the boundary curve.

    Returns:
        Tuple of (e_boundary, R_boundary) arrays defining the
        pentagon boundary.

    References:
        Proposition 1 in the paper.
    """
    e_min_val, R_sc = P_sc
    e_cs, R_max_val = P_cs

    # Pentagon vertices
    # (e_min, 0), (e_min, R_sc), (e_cs, R_max), (∞, R_max)
    # The boundary consists of:
    # 1. Horizontal line: e = e_min, R in [0, R_sc]
    # 2. Diagonal line: from (e_min, R_sc) to (e_cs, R_max)
    # 3. Vertical asymptote: e -> inf at R = R_max

    e_boundary = []
    R_boundary = []

    # Segment 1: e = e_min, R in [0, R_sc]
    R_seg1 = np.linspace(0, R_sc, n_points // 3)
    e_seg1 = np.full_like(R_seg1, e_min_val)
    e_boundary.extend(e_seg1)
    R_boundary.extend(R_seg1)

    # Segment 2: Diagonal from (e_min, R_sc) to (e_cs, R_max)
    if R_max_val > R_sc and e_cs > e_min_val:
        R_seg2 = np.linspace(R_sc, R_max_val, n_points // 3)
        slope = (e_cs - e_min_val) / (R_max_val - R_sc)
        e_seg2 = e_min_val + slope * (R_seg2 - R_sc)
        e_boundary.extend(e_seg2)
        R_boundary.extend(R_seg2)

    # Segment 3: Vertical line at R = R_max, e in [e_cs, large]
    e_seg3 = np.linspace(e_cs, e_cs + 3 * (e_cs - e_min_val), n_points // 3)
    R_seg3 = np.full_like(e_seg3, R_max_val)
    e_boundary.extend(e_seg3)
    R_boundary.extend(R_seg3)

    return np.array(e_boundary), np.array(R_boundary)


def gaussian_inner_bound(
    alpha_values: np.ndarray,
    Hc: np.ndarray,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_c2: float = 1.0,
    sigma_s2: float = 1.0,
    P_T: float = 1.0,
    M: int = 4,
    Jp: Optional[np.ndarray] = None,
    Nc: Optional[int] = None,
    n_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Gaussian signaling inner bound.

    For each alpha in [0, 1], solves the covariance shaping problem
    and evaluates the achievable (e, R) with Gaussian signaling.

    The Gaussian inner bound assumes X has i.i.d. Gaussian columns
    ~ CN(0, Rx_bar), giving achievable rate equal to the ergodic
    capacity.

    Args:
        alpha_values (np.ndarray): Array of tradeoff parameters.
        Hc (np.ndarray): Communication channel, shape (Nc, M).
        Hs_func (callable, optional): Sensing channel function.
        phi_func (callable, optional): Custom Phi map for BFIM.
        T (int): Coherent processing interval.
        sigma_c2 (float): Communication noise variance.
        sigma_s2 (float): Sensing noise variance.
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        Jp (np.ndarray, optional): Prior information.
        Nc (int, optional): Number of comm Rx antennas.
        n_samples (int): Number of random samples for averaging.

    Returns:
        Tuple of (e_values, R_values, alpha_used) arrays.

    References:
        Section V-A, Gaussian signaling bound.
    """
    Hc = np.asarray(Hc, dtype=np.complex128)
    if Nc is None:
        Nc = Hc.shape[0]

    e_list = []
    R_list = []
    alpha_used = []

    for alpha in alpha_values:
        try:
            # Solve covariance shaping
            Rx_opt = covariance_shaping(
                alpha, P_T, M, Hc, Hs_func, phi_func,
                T, sigma_c2, sigma_s2, Jp, Nc
            )

            # Check if Rx is valid
            eigvals = np.linalg.eigvalsh(Rx_opt)
            if np.any(eigvals < -1e-6):
                continue

            # Compute rate (deterministic, no need for averaging)
            R = compute_rate(Rx_opt, Hc, sigma_c2)

            # Compute CRB (averaged over sensing parameters if applicable)
            if phi_func is not None:
                e = compute_crb(Rx_opt, T, sigma_s2, phi_func=phi_func, Jp=Jp)
            else:
                e = compute_crb(Rx_opt, T, sigma_s2, Jp=Jp)

            e_list.append(e)
            R_list.append(R)
            alpha_used.append(alpha)

        except Exception:
            continue

    return (
        np.array(e_list),
        np.array(R_list),
        np.array(alpha_used),
    )


def semi_unitary_inner_bound(
    alpha_values: np.ndarray,
    Hc: np.ndarray,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_c2: float = 1.0,
    sigma_s2: float = 1.0,
    P_T: float = 1.0,
    M: int = 4,
    M_sc: Optional[int] = None,
    Jp: Optional[np.ndarray] = None,
    Nc: Optional[int] = None,
    n_stiefel_samples: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the semi-unitary (Stiefel manifold) inner bound.

    For each alpha, solves the covariance shaping problem, then
    generates semi-unitary waveforms from the Stiefel manifold
    to evaluate achievable rate and CRB.

    The semi-unitary bound exploits the deterministic-random
    tradeoff (DRT): deterministic waveform structure improves
    sensing but may reduce communication rate.

    Args:
        alpha_values (np.ndarray): Array of tradeoff parameters.
        Hc (np.ndarray): Communication channel, shape (Nc, M).
        Hs_func (callable, optional): Sensing channel function.
        phi_func (callable, optional): Custom Phi map.
        T (int): Coherent processing interval.
        sigma_c2 (float): Communication noise variance.
        sigma_s2 (float): Sensing noise variance.
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        M_sc (int, optional): Sensing subspace dimension.
        Jp (np.ndarray, optional): Prior information.
        Nc (int, optional): Number of comm Rx antennas.
        n_stiefel_samples (int): Number of Stiefel samples.

    Returns:
        Tuple of (e_values, R_values, alpha_used) arrays.

    References:
        Section V-B, Semi-unitary inner bound.
    """
    Hc = np.asarray(Hc, dtype=np.complex128)
    if Nc is None:
        Nc = Hc.shape[0]
    if M_sc is None:
        M_sc = min(M, T)

    e_list = []
    R_list = []
    alpha_used = []

    for alpha in alpha_values:
        try:
            # Get the covariance shaping solution
            Rx_bar = covariance_shaping(
                alpha, P_T, M, Hc, Hs_func, phi_func,
                T, sigma_c2, sigma_s2, Jp, Nc
            )

            eigvals_Rx = np.linalg.eigvalsh(Rx_bar)
            if np.any(eigvals_Rx < -1e-6):
                continue

            # Eigendecomposition of Rx_bar
            eigvals, U = np.linalg.eigh(Rx_bar)
            eigvals = np.maximum(eigvals, 0)

            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            U = U[:, idx]

            # Use top M_sc eigenmodes
            M_use = min(M_sc, np.sum(eigvals > 1e-10))
            if M_use < 1:
                continue

            U_sc = U[:, :M_use]
            Lambda_sc = eigvals[:M_use]

            # Sample semi-unitary waveforms and average
            R_samples = []
            e_samples = []

            for _ in range(n_stiefel_samples):
                # Generate semi-unitary Q
                Q = stiefel_sample(M_use, T)

                # Construct waveform: X = U_sc * diag(sqrt(Lambda)) * Q
                X = U_sc @ np.diag(np.sqrt(np.maximum(Lambda_sc, 0))) @ Q

                # Evaluate rate
                Rx_sample = (X @ X.conj().T) / T
                R_sample = compute_rate(Rx_sample, Hc, sigma_c2)

                # Evaluate CRB
                if phi_func is not None:
                    e_sample = compute_crb(
                        Rx_sample, T, sigma_s2,
                        phi_func=phi_func, Jp=Jp
                    )
                else:
                    e_sample = compute_crb(Rx_sample, T, sigma_s2, Jp=Jp)

                R_samples.append(R_sample)
                e_samples.append(e_sample)

            e_list.append(np.mean(e_samples))
            R_list.append(np.mean(R_samples))
            alpha_used.append(alpha)

        except Exception:
            continue

    return (
        np.array(e_list),
        np.array(R_list),
        np.array(alpha_used),
    )


def outer_bound(
    alpha_values: np.ndarray,
    Hc: np.ndarray,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_c2: float = 1.0,
    sigma_s2: float = 1.0,
    P_T: float = 1.0,
    M: int = 4,
    Jp: Optional[np.ndarray] = None,
    Nc: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the outer bound for the CRB-rate region.

    The outer bound is derived from the covariance shaping
    optimization, treating the CRB and rate as functions of
    the statistical covariance Rx_bar only (without specifying
    the waveform distribution).

    For each alpha:
        e_out(alpha) = (sigma_s^2 / T) * tr{[Phi(Rx_bar(alpha))]^{-1}}
        R_out(alpha) = log|I + sigma_c^{-2} Hc Rx_bar(alpha) Hc^H|

    This gives an upper bound on the achievable region because
    the actual waveform must satisfy additional constraints.

    Args:
        alpha_values (np.ndarray): Tradeoff parameters.
        Hc (np.ndarray): Communication channel, shape (Nc, M).
        Hs_func (callable, optional): Sensing channel function.
        phi_func (callable, optional): Custom Phi map.
        T (int): Coherent processing interval.
        sigma_c2 (float): Communication noise variance.
        sigma_s2 (float): Sensing noise variance.
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        Jp (np.ndarray, optional): Prior information.
        Nc (int, optional): Number of comm Rx antennas.

    Returns:
        Tuple of (e_values, R_values, alpha_used) arrays.

    References:
        Eq. (50) in the paper.
    """
    Hc = np.asarray(Hc, dtype=np.complex128)
    if Nc is None:
        Nc = Hc.shape[0]

    e_list = []
    R_list = []
    alpha_used = []

    for alpha in alpha_values:
        try:
            Rx_bar = covariance_shaping(
                alpha, P_T, M, Hc, Hs_func, phi_func,
                T, sigma_c2, sigma_s2, Jp, Nc
            )

            eigvals = np.linalg.eigvalsh(Rx_bar)
            if np.any(eigvals < -1e-6):
                continue

            # Rate (Eq. 50a)
            R = compute_rate(Rx_bar, Hc, sigma_c2)

            # CRB (Eq. 50b)
            if phi_func is not None:
                bfim = compute_bfim(
                    Rx_bar, T, sigma_s2,
                    phi_func=phi_func, Jp=Jp
                )
            else:
                bfim = compute_bfim(Rx_bar, T, sigma_s2, Jp=Jp)

            e = (sigma_s2 / T) * np.real(np.trace(np.linalg.pinv(bfim)))

            e_list.append(e)
            R_list.append(R)
            alpha_used.append(alpha)

        except Exception:
            continue

    return (
        np.array(e_list),
        np.array(R_list),
        np.array(alpha_used),
    )


def compute_corner_points(
    Hc: np.ndarray,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_c2: float = 1.0,
    sigma_s2: float = 1.0,
    P_T: float = 1.0,
    M: int = 4,
    Jp: Optional[np.ndarray] = None,
    Nc: Optional[int] = None,
) -> dict:
    """Compute the two corner points P_sc and P_cs.

    P_sc = (e_min, R_sc): sensing-optimal point
    P_cs = (e_cs, R_max): communication-optimal point

    Returns:
        dict with keys:
            'e_min': Minimum CRB (sensing-optimal)
            'R_sc': Rate at sensing-optimal point
            'e_cs': CRB at comm-optimal point
            'R_max': Maximum rate
            'Rx_sc': Sensing-optimal covariance
            'Rx_cs': Comm-optimal covariance
    """
    Hc = np.asarray(Hc, dtype=np.complex128)

    # Sensing-optimal point
    Rx_sc = optimize_sensing_rx(P_T, M, phi_func=phi_func, T=T, sigma_s2=sigma_s2, Jp=Jp)

    if phi_func is not None:
        e_min = compute_crb(Rx_sc, T, sigma_s2, phi_func=phi_func, Jp=Jp)
    else:
        e_min = compute_crb(Rx_sc, T, sigma_s2, Jp=Jp)

    R_sc = compute_rate(Rx_sc, Hc, sigma_c2)

    # Communication-optimal point
    Rx_cs = optimize_comm_rx(P_T, M, Hc)

    R_max = compute_rate(Rx_cs, Hc, sigma_c2)

    if phi_func is not None:
        e_cs = compute_crb(Rx_cs, T, sigma_s2, phi_func=phi_func, Jp=Jp)
    else:
        e_cs = compute_crb(Rx_cs, T, sigma_s2, Jp=Jp)

    return {
        'e_min': e_min,
        'R_sc': R_sc,
        'e_cs': e_cs,
        'R_max': R_max,
        'Rx_sc': Rx_sc,
        'Rx_cs': Rx_cs,
    }
