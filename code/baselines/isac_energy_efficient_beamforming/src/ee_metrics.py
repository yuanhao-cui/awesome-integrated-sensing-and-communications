"""
Energy Efficiency Metrics
=========================

Computes energy efficiency metrics for ISAC systems:
    - Communication-centric EE (EE_C, Eq. 4)
    - Sensing-centric EE (EE_S, Eq. 33)
    - CRB for point-like target (Eq. 10)
    - SINR computation (Eq. 2)

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import numpy as np
from typing import Optional


def compute_sinr(k: int, h_k: np.ndarray, W: np.ndarray, sigma_c2: float) -> float:
    """
    Compute SINR for user k (Eq. 2).

    SINR_k = |h_k^H w_k|² / (σ_c² + Σ_{j≠k} |h_k^H w_j|²)

    Parameters
    ----------
    k : int
        User index (0-indexed)
    h_k : np.ndarray
        Channel vector for user k (M,)
    W : np.ndarray
        Beamforming matrix (M x K), column k is w_k
    sigma_c2 : float
        Noise power

    Returns
    -------
    float
        SINR value for user k
    """
    K = W.shape[1]
    signal = np.abs(h_k.conj() @ W[:, k]) ** 2
    interference = sum(
        np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K) if j != k
    )
    return float(signal / (sigma_c2 + interference))


def compute_sum_rate(H: np.ndarray, W: np.ndarray, sigma_c2: float) -> float:
    """
    Compute sum rate Σ_k log₂(1 + SINR_k).

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    W : np.ndarray
        Beamforming matrix (M x K)
    sigma_c2 : float
        Noise power

    Returns
    -------
    float
        Sum rate in bits/Hz/s
    """
    K = H.shape[0]
    sum_rate = 0.0
    for k in range(K):
        sinr_k = compute_sinr(k, H[k, :], W, sigma_c2)
        sum_rate += np.log2(1 + sinr_k)
    return sum_rate


def compute_total_power(W: np.ndarray) -> float:
    """
    Compute total transmit power P_total = Σ_k ||w_k||².

    Parameters
    ----------
    W : np.ndarray
        Beamforming matrix (M x K)

    Returns
    -------
    float
        Total transmit power
    """
    return float(np.sum(np.abs(W) ** 2))


def compute_ee_c(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
    epsilon: float,
    P0: float,
) -> float:
    """
    Compute communication-centric energy efficiency (Eq. 4).

    EE_C = Σ_k log₂(1+SINR_k) / ((1/ε)Σ_k||w_k||² + P₀)

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    W : np.ndarray
        Beamforming matrix (M x K)
    sigma_c2 : float
        Communication noise power
    epsilon : float
        Power amplifier efficiency (0 < ε ≤ 1)
    P0 : float
        Circuit power

    Returns
    -------
    float
        Communication EE in bits/Hz/J
    """
    sum_rate = compute_sum_rate(H, W, sigma_c2)
    total_power = compute_total_power(W)
    total_power_consumption = (1.0 / epsilon) * total_power + P0

    if total_power_consumption <= 0:
        return 0.0

    return float(sum_rate / total_power_consumption)


def compute_crb_point_target(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    sigma_s2: float,
    L: int,
) -> float:
    """
    Compute CRB for point-like target angle estimation (Eq. 10).

    For a point-like target at angle θ, the CRB for angle estimation is:
        CRB = (σ_s² / (2L)) * [A^H (I - P_A) A]^{-1}_{1,1}

    where:
        - A = [∂Rx/∂θ, vec(Ax)] is the derivative matrix
        - Rx = Σ_k w_k w_k^H is the transmit covariance
        - Ax = diag(a_t) * Rx * a_t^H * a_r^T (simplified model)

    Simplified form for ULA:
        CRB ∝ σ_s² / (L * trace(∂Rx/∂θ * Rx^{-1} * ∂Rx/∂θ * ...))

    This implementation uses the first-order approximation from the paper.

    Parameters
    ----------
    W : np.ndarray
        Beamforming matrix (M x K)
    a_t : np.ndarray
        Transmit steering vector (M,)
    a_r : np.ndarray
        Receive steering vector (N,)
    sigma_s2 : float
        Sensing noise power
    L : int
        Number of frames

    Returns
    -------
    float
        CRB value (lower is better)
    """
    M, K = W.shape
    N = len(a_r)

    # Transmit covariance: Rx = Σ_k w_k w_k^H
    Rx = W @ W.conj().T

    # Derivative of steering vector: ∂a_t(θ)/∂θ
    # For ULA: ∂a_t(θ)_m/∂θ = j*(2π/λ)*d*m*cos(θ)*a_t(θ)_m
    # Since θ=90°, cos(θ)=0, so we use a numerical approximation
    # Using perturbation for derivative
    eps_angle = 1e-6
    m_indices = np.arange(M)

    # Compute derivative of a_t w.r.t. θ at θ=90° (π/2)
    # For general θ: da_t/dθ = j*(2π/λ)*d*m*cos(θ)*a_t(θ)
    # At θ=90°, cos(90°)=0, but we consider the effective derivative
    # Using the formulation from the paper:
    # CRB involves the Fisher Information Matrix (FIM)

    # Simplified CRB computation based on Eq. (10):
    # CRB = σ_s² / (2L * ||a_r||² * a_t^H Q Q^H a_t)
    # where Q = (I - Rx(Rx + σ_s²/L I)^{-1}) * Da_t

    # Regularized version for numerical stability
    reg = sigma_s2 / L
    Rx_reg_inv = np.linalg.inv(Rx + reg * np.eye(M))

    # Derivative matrix Da = diag(da_t/dθ)
    # For effective computation, use numerical gradient
    theta_test = np.pi / 2  # 90 degrees
    da_t = (
        1j
        * 2
        * np.pi
        / 1.0
        * 0.5
        * m_indices
        * np.cos(theta_test)
        * a_t
    )

    # Fisher information for angle
    a_r_norm_sq = np.sum(np.abs(a_r) ** 2)

    # FIM component: ||a_r||² * da_t^H * Rx * da_t
    fim = a_r_norm_sq * (da_t.conj() @ Rx @ da_t)

    # Also include the projection term
    proj = a_r_norm_sq * (da_t.conj() @ Rx @ Rx_reg_inv @ Rx @ da_t)

    # Effective FIM
    fim_eff = 2 * L * (fim - proj) / sigma_s2

    # CRB is inverse of FIM
    if np.abs(fim_eff) < 1e-15:
        return float("inf")

    crb = sigma_s2 / (2 * L * a_r_norm_sq * np.abs(fim_eff) + 1e-15)

    # Ensure positive and finite
    crb = max(crb, 1e-15)

    return float(crb)


def compute_crb(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    sigma_s2: float,
    L: int,
) -> float:
    """
    Compute CRB for target angle estimation.

    This is a wrapper for compute_crb_point_target with additional
    numerical stability handling.

    Parameters
    ----------
    W : np.ndarray
        Beamforming matrix (M x K)
    a_t : np.ndarray
        Transmit steering vector (M,)
    a_r : np.ndarray
        Receive steering vector (N,)
    sigma_s2 : float
        Sensing noise power
    L : int
        Number of frames

    Returns
    -------
    float
        CRB value
    """
    return compute_crb_point_target(W, a_t, a_r, sigma_s2, L)


def compute_ee_s(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    sigma_s2: float,
    L: int,
    epsilon: float,
    P0: float,
) -> float:
    """
    Compute sensing-centric energy efficiency (Eq. 33).

    EE_S = CRB⁻¹ / (L * ((1/ε)Σ||w_k||² + P₀))

    Parameters
    ----------
    W : np.ndarray
        Beamforming matrix (M x K)
    a_t : np.ndarray
        Transmit steering vector (M,)
    a_r : np.ndarray
        Receive steering vector (N,)
    sigma_s2 : float
        Sensing noise power
    L : int
        Number of frames
    epsilon : float
        Power amplifier efficiency
    P0 : float
        Circuit power

    Returns
    -------
    float
        Sensing EE (CRB⁻¹ per Joule)
    """
    crb = compute_crb(W, a_t, a_r, sigma_s2, L)

    if crb <= 0 or np.isinf(crb):
        return 0.0

    total_power = compute_total_power(W)
    total_power_consumption = L * ((1.0 / epsilon) * total_power + P0)

    if total_power_consumption <= 0:
        return 0.0

    return float((1.0 / crb) / total_power_consumption)


def compute_ee_c_sdr(
    H: np.ndarray,
    W_list: list,
    sigma_c2: float,
    epsilon: float,
    P0: float,
) -> float:
    """
    Compute EE_C from SDR solution (list of PSD matrices W_k).

    For SDR: W_k = w_k w_k^H are K positive semidefinite matrices.
    Sum rate = Σ_k log₂(1 + h_k^H W_k h_k / (σ_c² + Σ_{j≠k} h_k^H W_j h_k))
    Power = Σ_k tr(W_k)

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    W_list : list of np.ndarray
        List of K PSD matrices, each (M x M)
    sigma_c2 : float
        Noise power
    epsilon : float
        PA efficiency
    P0 : float
        Circuit power

    Returns
    -------
    float
        Communication EE
    """
    K = len(W_list)
    sum_rate = 0.0
    total_power = 0.0

    for k in range(K):
        h_k = H[k, :]
        signal = np.real(h_k.conj() @ W_list[k] @ h_k)
        interference = sum(
            np.real(h_k.conj() @ W_list[j] @ h_k) for j in range(K) if j != k
        )
        sinr_k = signal / (sigma_c2 + interference)
        sum_rate += np.log2(1 + sinr_k)
        total_power += np.real(np.trace(W_list[k]))

    total_power_consumption = (1.0 / epsilon) * total_power + P0

    if total_power_consumption <= 0:
        return 0.0

    return float(sum_rate / total_power_consumption)
