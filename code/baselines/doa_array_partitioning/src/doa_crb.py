"""
DOA Estimation Cramér-Rao Bound (CRB) Computation.

For monostatic ISAC with partitionable array, the CRB for DOA estimation
depends on the Fisher Information Matrix derived from the radar received signal.
"""

import numpy as np
from typing import Optional, Tuple


def compute_fisher_info(
    a_theta: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma2_r: float,
    N: int = 1,
) -> np.ndarray:
    """
    Compute the Fisher Information Matrix for single-target DOA estimation.

    The radar received signal at RX antennas:
        y_r = sqrt(P_r) * (r ⊙ a(theta)) * (a(theta)^H * (t ⊙ w) * s)^H + noise

    FIM for scalar DOA theta:
        F = N * 2 * P_r * |a^H * (t ⊙ w)|^2 / sigma2_r * |r ⊙ a'(theta)|^2

    where a'(theta) = d(a(theta))/d(theta) is the derivative steering vector.

    Parameters
    ----------
    a_theta : np.ndarray, shape (M,)
        Steering vector a(theta) at target DOA.
    a : np.ndarray, shape (M,)
        Alias for a_theta (same value, kept for API consistency).
    w : np.ndarray, shape (M,) or (M, K)
        Transmit beamforming vector(s). If (M,K), summed.
    t : np.ndarray, shape (M,)
        TX partition binary vector.
    r : np.ndarray, shape (M,)
        RX partition binary vector.
    sigma2_r : float
        Radar noise power.
    N : int
        Number of samples.

    Returns
    -------
    F : float
        Scalar Fisher information for DOA theta.
    """
    if w.ndim == 2:
        w_total = np.sum(w, axis=1)
    else:
        w_total = w

    M = len(a_theta)
    antenna_positions = np.arange(M) * 0.5  # d=0.5 wavelengths
    # Derivative of steering vector: a'(theta)
    d_a_theta = 1j * 2 * np.pi * antenna_positions * np.cos(np.angle(a_theta[1:]) if M > 1 else 0)
    # Better: compute derivative properly
    phases = 2 * np.pi * antenna_positions * np.sin(np.arcsin(np.clip(
        np.angle(a_theta) / (2 * np.pi * antenna_positions + 1e-10), -1, 1
    )))
    # Actually, let's compute from the phase of the steering vector directly
    # a[m] = exp(j * 2*pi * m * d * sin(theta))
    # da[m]/dtheta = j * 2*pi * m * d * cos(theta) * exp(j * 2*pi * m * d * sin(theta))
    # We extract theta from a_theta
    if M > 1:
        # Extract theta from a_theta[1]/a_theta[0]
        phase_diff = np.angle(a_theta[1] / a_theta[0]) if np.abs(a_theta[0]) > 1e-10 else 0
        sin_theta = phase_diff / (2 * np.pi * 0.5)
        sin_theta = np.clip(sin_theta, -1, 1)
        cos_theta = np.sqrt(1 - sin_theta ** 2)
    else:
        cos_theta = 1.0

    a_deriv = 1j * 2 * np.pi * antenna_positions * cos_theta * a_theta

    # Transmit signal effective: t ⊙ w
    tx_signal = t * w_total
    # Receive response: r ⊙ a
    rx_response = r * a_theta

    # Transmit power factor
    tx_power = np.abs(np.conj(a_theta) @ tx_signal) ** 2
    # Receive gain
    rx_gain = np.linalg.norm(rx_response) ** 2

    # Derivative contribution
    deriv_gain = np.linalg.norm(r * a_deriv) ** 2

    F = 2 * N * tx_power * deriv_gain / (sigma2_r + 1e-10)
    return F


def compute_crb(
    a_theta: np.ndarray,
    w: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma2_r: float,
    N: int = 1,
) -> float:
    """
    Compute CRB for DOA estimation.

    CRB(theta) = F^{-1}

    Parameters
    ----------
    a_theta : np.ndarray, shape (M,)
        Steering vector at target DOA.
    w : np.ndarray, shape (M,) or (M, K)
        Transmit beamforming.
    t : np.ndarray, shape (M,)
        TX partition.
    r : np.ndarray, shape (M,)
        RX partition.
    sigma2_r : float
        Radar noise power.
    N : int
        Number of samples.

    Returns
    -------
    crb : float
        Cramér-Rao bound (lower bound on MSE of DOA estimator).
    """
    F = compute_fisher_info(a_theta, a_theta, w, t, r, sigma2_r, N)
    if F < 1e-15:
        return np.inf
    return 1.0 / F


def compute_crb_multi_target(
    target_angles: np.ndarray,
    antenna_positions: np.ndarray,
    w: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma2_r: float,
    N: int = 1,
) -> np.ndarray:
    """
    Compute CRB for multiple target DOAs.

    Parameters
    ----------
    target_angles : np.ndarray, shape (L,)
        L target DOAs in radians.
    antenna_positions : np.ndarray, shape (M,)
        Physical antenna positions.
    w : np.ndarray, shape (M,) or (M, K)
    t : np.ndarray, shape (M,)
    r : np.ndarray, shape (M,)
    sigma2_r : float
    N : int

    Returns
    -------
    crbs : np.ndarray, shape (L,)
        CRB per target.
    """
    crbs = np.zeros(len(target_angles))
    for l, theta in enumerate(target_angles):
        phases = 2 * np.pi * antenna_positions * np.sin(theta)
        a_theta = np.exp(1j * phases)
        crbs[l] = compute_crb(a_theta, w, t, r, sigma2_r, N)
    return crbs


def compute_crb_gradient(
    a_theta: np.ndarray,
    w: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma2_r: float,
    M: int,
    d: float = 0.5,
    N: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient of CRB w.r.t. TX partition t and beamforming w.

    Returns
    -------
    grad_t : np.ndarray, shape (M,)
    grad_w : np.ndarray, shape (M,)
    """
    antenna_positions = np.arange(M) * d

    if M > 1:
        phase_diff = np.angle(a_theta[1] / a_theta[0]) if np.abs(a_theta[0]) > 1e-10 else 0
        sin_theta = phase_diff / (2 * np.pi * d)
        sin_theta = np.clip(sin_theta, -1, 1)
        cos_theta = np.sqrt(1 - sin_theta ** 2)
    else:
        cos_theta = 1.0

    a_deriv = 1j * 2 * np.pi * antenna_positions * cos_theta * a_theta

    if w.ndim == 2:
        w_total = np.sum(w, axis=1)
    else:
        w_total = w

    tx_signal = t * w_total
    rx_response = r * a_theta

    tx_power = np.abs(np.conj(a_theta) @ tx_signal) ** 2
    deriv_gain = np.linalg.norm(r * a_deriv) ** 2

    F = 2 * N * tx_power * deriv_gain / (sigma2_r + 1e-10)
    if F < 1e-15:
        return np.zeros(M), np.zeros(M)

    # d(F)/d(t_m) = 2*N * deriv_gain / sigma2_r * d(|a^H diag(t) w|^2)/d(t_m)
    inner = np.conj(a_theta) @ tx_signal
    # d(|inner|^2)/d(t_m) = 2 * Re(inner* * conj(a_theta[m]) * w[m])
    d_tx_power_dt = 2 * np.real(np.conj(inner) * np.conj(a_theta) * w_total)
    grad_F_t = 2 * N * deriv_gain / (sigma2_r + 1e-10) * d_tx_power_dt

    # d(CRB)/dt = -F^{-2} * dF/dt
    grad_t = -(F ** -2) * grad_F_t

    # d(F)/d(w_m): similar
    d_tx_power_dw = 2 * np.real(np.conj(inner) * np.conj(a_theta) * t)
    grad_F_w = 2 * N * deriv_gain / (sigma2_r + 1e-10) * d_tx_power_dw
    grad_w = -(F ** -2) * grad_F_w

    return grad_t, grad_w
