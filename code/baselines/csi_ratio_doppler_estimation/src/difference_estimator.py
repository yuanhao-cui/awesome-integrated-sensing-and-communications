"""
Algorithm 3: Signal Difference-based Doppler Estimation.

Estimates |f_D| (magnitude only, no sign) by finding the periodicity
through signal self-similarity.

Steps:
1. For each lag n, compute average squared difference:
   Δ_Σ(n) = (1/(N-n)) * Σ_{k=1}^{N-n} |R(k+n) - R(k)|^2
2. Find n* = argmin Δ_Σ(n) (the lag with minimum difference)
3. f_D = 1 / (n* * T_s)

The minimum occurs when n* corresponds to one full period of the
Doppler-modulated CSI-ratio signal.

Limitations:
- Cannot determine sign of Doppler
- Resolution limited by sampling interval T_s
- Requires f_D to be well-separated from DC

Reference: Section III.C of the paper.
"""

import numpy as np
from typing import Dict, Optional


def difference_doppler_estimate(
    R: np.ndarray,
    T_s: float,
    max_lag: Optional[int] = None,
    use_magnitude: bool = False,
) -> Dict[str, float]:
    """
    Estimate |f_D| using signal difference method (Algorithm 3).

    Finds the lag n* that minimizes the average squared difference
    Δ_Σ(n) between the signal and its shifted version.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    T_s : float
        Sampling interval (seconds).
    max_lag : int, optional
        Maximum lag to search. Default: N // 2.
    use_magnitude : bool
        If True, use |R| instead of complex R for differences.
        Default: False (use full complex signal).

    Returns
    -------
    result : dict
        'f_D': Estimated |f_D| (Hz), always non-negative.
        'f_D_magnitude': Same as f_D.
        'direction': 'unknown' (Algorithm 3 cannot determine sign).
        'n_star': Optimal lag (samples) where minimum occurs.
        'delta_sigma': Array of Δ_Σ(n) values for all tested lags.
        'min_delta': Minimum value of Δ_Σ(n).

    Notes
    -----
    Δ_Σ(n) = (1/(N-n)) * Σ_{k=1}^{N-n} |R(k+n) - R(k)|^2

    This equals 0 when R is perfectly periodic with period n.
    For Doppler-modulated CSI-ratio:
        R(k) = A * exp(j*2π*f_D*k*T_s) * exp(j*φ_0)
    The minimum occurs at n* = 1/(f_D * T_s) = fs / f_D.

    The search range is limited to avoid spurious minima at large lags
    where fewer samples contribute to the average.
    """
    N = len(R)

    if max_lag is None:
        max_lag = N // 2

    # Ensure max_lag doesn't exceed signal length
    max_lag = min(max_lag, N - 1)

    # Use magnitude or complex signal
    if use_magnitude:
        signal = np.abs(R)
    else:
        signal = R

    # Step 1: Compute Δ_Σ(n) for each lag
    # Δ_Σ(n) = (1/(N-n)) * Σ_{k=0}^{N-n-1} |R(k+n) - R(k)|^2
    delta_sigma = np.zeros(max_lag)

    for n in range(1, max_lag + 1):
        # Difference between signal and its n-lagged version
        diff = signal[n:] - signal[: N - n]
        # Average squared difference
        if use_magnitude:
            delta_sigma[n - 1] = np.mean(diff ** 2)
        else:
            delta_sigma[n - 1] = np.mean(np.abs(diff) ** 2)

    # Step 2: Find n* = argmin Δ_Σ(n)
    # Exclude very small lags (they always have small difference)
    min_lag = max(2, max_lag // 20)  # At least 2, or 5% of max_lag
    search_range = delta_sigma[min_lag - 1:]
    n_star = min_lag + np.argmin(search_range)

    # Step 3: f_D = 1 / (n* * T_s)
    f_D = 1.0 / (n_star * T_s)

    return {
        "f_D": f_D,
        "f_D_magnitude": abs(f_D),
        "direction": "unknown",
        "n_star": n_star,
        "delta_sigma": delta_sigma,
        "min_delta": delta_sigma[n_star - 1],
    }


def difference_doppler_estimate_refined(
    R: np.ndarray,
    T_s: float,
    max_lag: Optional[int] = None,
) -> Dict[str, float]:
    """
    Refined difference-based estimation with parabolic interpolation.

    After finding the integer lag n* with minimum Δ_Σ, uses parabolic
    interpolation around the minimum to get sub-sample resolution.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    T_s : float
        Sampling interval (seconds).
    max_lag : int, optional
        Maximum lag to search. Default: N // 2.

    Returns
    -------
    result : dict
        Same as difference_doppler_estimate, plus:
        'n_star_refined': Sub-sample refined lag.
        'f_D_refined': Refined Doppler estimate.
    """
    # Get coarse estimate
    coarse = difference_doppler_estimate(R, T_s, max_lag)
    n_star = coarse["n_star"]
    delta_sigma = coarse["delta_sigma"]

    # Parabolic interpolation around minimum
    if n_star > 1 and n_star < len(delta_sigma):
        # Three points around minimum
        d0 = delta_sigma[n_star - 2]
        d1 = delta_sigma[n_star - 1]
        d2 = delta_sigma[n_star]

        # Parabolic fit: d(n) = a*n^2 + b*n + c
        # Minimum at n* = -b/(2a)
        denom = 2 * (d0 - 2 * d1 + d2)
        if abs(denom) > 1e-15:
            delta_n = (d0 - d2) / denom
            n_star_refined = n_star + delta_n
        else:
            n_star_refined = float(n_star)
    else:
        n_star_refined = float(n_star)

    # Refined f_D
    if n_star_refined > 0:
        f_D_refined = 1.0 / (n_star_refined * T_s)
    else:
        f_D_refined = coarse["f_D"]

    coarse["n_star_refined"] = n_star_refined
    coarse["f_D_refined"] = f_D_refined

    return coarse
