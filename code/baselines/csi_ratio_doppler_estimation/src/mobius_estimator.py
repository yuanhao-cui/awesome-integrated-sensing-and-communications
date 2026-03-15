"""
Algorithm 1: Mobius Transformation-based Doppler Estimation.

This is the PRIMARY algorithm that can estimate the **signed** Doppler frequency.

Steps (from the paper):
1. Compute CSI-ratio R(t_k) = H_m(t_k) / H_{m+1}(t_k)
2. Fit circle to R(t_k) in complex plane via least-squares (Eq. 11)
3. Shift circle to origin: R_s(t_k) = R(t_k) - C_0
4. Calculate angle θ_R(t_k) = arg(R_s(t_k)) and magnitude a_R(t_k) = |R_s(t_k)|
5. Weighted linear regression: θ_R(t_k) = β_0 + β_1 * t_k (Eq. 14)
   Weights: w_k = a_R(t_k) (magnitudes)
6. f_D = β_1 / (2π * T_s)

The sign of f_D is preserved:
    f_D > 0: target approaching
    f_D < 0: target receding

Reference: Eq. (11)-(14) of the paper.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from circle_fit import least_squares_circle_fit, fit_circle_kasa, fit_circle_pratt


def mobius_doppler_estimate(
    R: np.ndarray,
    T_s: float,
    circle_method: str = "least_squares",
    unwrap_phases: bool = True,
) -> Dict[str, float]:
    """
    Estimate Doppler frequency using Mobius transformation (Algorithm 1).

    This is the ONLY method that can recover the sign of the Doppler frequency.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    T_s : float
        Sampling interval (seconds). For 2 kHz, T_s = 0.0005 s.
    circle_method : str
        Circle fitting method: 'least_squares', 'kasa', or 'pratt'.
        Default: 'least_squares' (Eq. 11).
    unwrap_phases : bool
        Whether to unwrap phase discontinuities before regression.
        Default: True.

    Returns
    -------
    result : dict
        'f_D': Estimated Doppler frequency (Hz), with sign.
        'f_D_magnitude': |f_D| (Hz).
        'direction': 'approaching' if f_D > 0, 'receding' if f_D < 0.
        'center_A': Real part of circle center.
        'center_B': Imaginary part of circle center.
        'radius': Circle radius.
        'beta_0': Intercept from weighted linear regression.
        'beta_1': Slope from weighted linear regression.
        'r_squared': R-squared of the linear fit.
        'rms_error': RMS error of circle fit.

    Notes
    -----
    Eq. (14): Weighted linear regression minimizes
        Σ w_k * (θ_R(t_k) - β_0 - β_1 * t_k)^2
    where w_k = |R_s(t_k)| are the magnitudes (more reliable samples
    near the circle edge get higher weight).

    The Doppler frequency is recovered as:
        f_D = β_1 / (2π)
    Note: the T_s in the paper's formulation is already in the time axis,
    so we compute f_D = β_1 / (2π) where β_1 is in rad/s.
    """
    N = len(R)
    t = np.arange(N) * T_s

    # Step 2: Circle fitting (Eq. 11)
    if circle_method == "least_squares":
        A, B, r = least_squares_circle_fit(R)
    elif circle_method == "kasa":
        A, B, r = fit_circle_kasa(R)
    elif circle_method == "pratt":
        A, B, r = fit_circle_pratt(R)
    else:
        raise ValueError(f"Unknown circle method: {circle_method}")

    # Step 3: Shift circle to origin
    C_0 = A + 1j * B
    R_s = R - C_0

    # Step 4: Calculate angle and magnitude
    theta_R = np.angle(R_s)  # angle in radians, range [-π, π]
    a_R = np.abs(R_s)  # magnitude

    # Unwrap phases to handle 2π discontinuities
    if unwrap_phases:
        theta_R = np.unwrap(theta_R)

    # Step 5: Weighted linear regression (Eq. 14)
    # θ_R(t_k) = β_0 + β_1 * t_k, weights = a_R
    beta_0, beta_1, r_squared = _weighted_linear_regression(t, theta_R, a_R)

    # Step 6: f_D = β_1 / (2π)
    f_D = beta_1 / (2 * np.pi)

    # Compute circle fit error
    distances = np.abs(R - C_0)
    rms_error = np.sqrt(np.mean((distances - r) ** 2))

    return {
        "f_D": f_D,
        "f_D_magnitude": abs(f_D),
        "direction": "approaching" if f_D > 0 else "receding",
        "center_A": A,
        "center_B": B,
        "radius": r,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "r_squared": r_squared,
        "rms_error": rms_error,
    }


def _weighted_linear_regression(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> Tuple[float, float, float]:
    """
    Weighted linear regression: y = β_0 + β_1 * x.

    Minimizes Σ w_i * (y_i - β_0 - β_1 * x_i)^2

    Parameters
    ----------
    x : np.ndarray
        Independent variable, shape (N,).
    y : np.ndarray
        Dependent variable, shape (N,).
    weights : np.ndarray
        Weights for each sample, shape (N,).

    Returns
    -------
    beta_0 : float
        Intercept.
    beta_1 : float
        Slope.
    r_squared : float
        Coefficient of determination.
    """
    w = weights / np.sum(weights)  # Normalize weights

    # Weighted means
    x_mean = np.sum(w * x)
    y_mean = np.sum(w * y)

    # Weighted covariance and variance
    S_xy = np.sum(w * (x - x_mean) * (y - y_mean))
    S_xx = np.sum(w * (x - x_mean) ** 2)

    # Slope and intercept
    if S_xx < 1e-20:
        beta_1 = 0.0
    else:
        beta_1 = S_xy / S_xx
    beta_0 = y_mean - beta_1 * x_mean

    # R-squared
    y_pred = beta_0 + beta_1 * x
    SS_res = np.sum(w * (y - y_pred) ** 2)
    SS_tot = np.sum(w * (y - y_mean) ** 2)

    if SS_tot < 1e-20:
        r_squared = 1.0 if SS_res < 1e-20 else 0.0
    else:
        r_squared = 1 - SS_res / SS_tot

    return beta_0, beta_1, r_squared
