"""
Algorithm 2: Periodicity-based Doppler Estimation.

Estimates |f_D| (magnitude only, no sign) by detecting the periodicity
of the CSI-ratio phase.

Steps:
1. Compute angle γ(t_k) = arg(R(t_k)) of CSI-ratio samples
2. Search for zero-crossings relative to starting angle
3. f_D = 1 / (S * T_s) where S = cycle length in samples
4. Average all estimates

Limitations:
- Cannot determine sign of Doppler (approaching vs receding)
- Resolution limited by sampling rate
- Requires sufficient samples to observe at least one full cycle

Reference: Section III.B of the paper.
"""

import numpy as np
from typing import Dict, Optional


def periodicity_doppler_estimate(
    R: np.ndarray,
    T_s: float,
    reference_method: str = "start",
) -> Dict[str, float]:
    """
    Estimate |f_D| using periodicity of CSI-ratio phase (Algorithm 2).

    Detects zero-crossings of the phase relative to a reference angle,
    which occur when the phase has completed a full 2π cycle.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    T_s : float
        Sampling interval (seconds).
    reference_method : str
        How to determine the reference angle:
        - 'start': Use the angle of the first sample (default).
        - 'mean': Use the mean phase as reference.

    Returns
    -------
    result : dict
        'f_D': Estimated |f_D| (Hz), always non-negative.
        'f_D_magnitude': Same as f_D.
        'direction': 'unknown' (Algorithm 2 cannot determine sign).
        'num_crossings': Number of zero-crossings detected.
        'cycle_lengths': Array of detected cycle lengths in samples.
        'angles': Array of unwrapped phases used.

    Notes
    -----
    A zero-crossing occurs when the phase wraps past ±π, indicating
    completion of one full 2π cycle. The cycle length S in samples
    gives: f_D = 1 / (S * T_s).

    This method works best when:
    - f_D * T_s * N > 1 (at least one full cycle in the observation window)
    - SNR is sufficient for reliable phase measurements
    """
    N = len(R)

    # Step 1: Extract phase angle
    gamma = np.angle(R)

    # Determine reference angle
    if reference_method == "start":
        gamma_ref = gamma[0]
    elif reference_method == "mean":
        gamma_ref = np.angle(np.mean(R))
    else:
        raise ValueError(f"Unknown reference method: {reference_method}")

    # Compute relative angle (shifted by reference)
    gamma_rel = gamma - gamma_ref
    # Wrap to [-π, π]
    gamma_rel = np.angle(np.exp(1j * gamma_rel))

    # Unwrap for continuous phase
    gamma_unwrapped = np.unwrap(gamma_rel)

    # Step 2: Search for zero-crossings (2π crossings in unwrapped phase)
    # A crossing occurs when gamma_unwrapped crosses a multiple of ±2π
    crossings = []
    for i in range(1, N):
        # Check if phase crossed ±2π boundary
        prev_quotient = int(np.round(gamma_unwrapped[i - 1] / (2 * np.pi)))
        curr_quotient = int(np.round(gamma_unwrapped[i] / (2 * np.pi)))

        if curr_quotient != prev_quotient:
            # Linear interpolation to find exact crossing point
            delta = gamma_unwrapped[i] - gamma_unwrapped[i - 1]
            if abs(delta) > 1e-15:
                alpha = -(gamma_unwrapped[i - 1] - prev_quotient * 2 * np.pi) / delta
                crossing_idx = (i - 1) + alpha
                crossings.append(crossing_idx)

    # Alternative: find where unwrapped phase crosses multiples of 2π
    if len(crossings) == 0:
        # Direct detection of 2π crossings
        cycle_boundaries = np.where(np.diff(np.sign(gamma_unwrapped)))[0]
        # Filter for actual 2π crossings (not small oscillations)
        for i in cycle_boundaries:
            if abs(gamma_unwrapped[i + 1] - gamma_unwrapped[i]) > np.pi:
                crossings.append(i + 0.5)

    # Step 3: Compute cycle lengths and f_D
    cycle_lengths = []
    if len(crossings) >= 2:
        for i in range(1, len(crossings)):
            S = crossings[i] - crossings[i - 1]
            if S > 0:
                cycle_lengths.append(S)

    # Step 4: Average estimates
    if len(cycle_lengths) > 0:
        # Use median for robustness to outliers
        S_avg = np.median(cycle_lengths)
        f_D = 1.0 / (S_avg * T_s)
    else:
        # No complete cycle detected
        # Fallback: estimate from total phase change
        total_phase_change = abs(gamma_unwrapped[-1] - gamma_unwrapped[0])
        num_cycles = total_phase_change / (2 * np.pi)
        if num_cycles > 0.1:
            S_estimated = N / num_cycles
            f_D = 1.0 / (S_estimated * T_s)
        else:
            f_D = 0.0

    return {
        "f_D": f_D,
        "f_D_magnitude": abs(f_D),
        "direction": "unknown",
        "num_crossings": len(crossings),
        "cycle_lengths": np.array(cycle_lengths),
        "angles": gamma_unwrapped,
    }
