"""
CSI-Ratio Computation.

Implements Eq. (6) and (8) from the paper:
    R(t) = H_m(t) / H_{m+1}(t)

The CSI-ratio cancels:
- Timing Misalignment Offset (TMO)
- Carrier Frequency Offset (CFO)
- Phase noise (common to all antennas)

Properties:
- R(t) is a Mobius transform of z(t) = exp(j*2π*f_D*t)
- As z(t) traces the unit circle, R(t) traces a circle in complex plane
- The circle parameters encode the Doppler frequency
"""

import numpy as np
from typing import Optional, Tuple


def compute_csi_ratio(H_m: np.ndarray, H_m1: np.ndarray) -> np.ndarray:
    """
    Compute CSI-ratio between two adjacent receive antennas.

    Implements Eq. (6) of the paper:
        R(t_k) = H_m(t_k) / H_{m+1}(t_k)

    Parameters
    ----------
    H_m : np.ndarray
        CSI samples from antenna m, shape (N,) complex.
    H_m1 : np.ndarray
        CSI samples from antenna m+1, shape (N,) complex.

    Returns
    -------
    R : np.ndarray
        CSI-ratio samples, shape (N,) complex.

    Notes
    -----
    - If H_m1 has very small magnitude, the ratio will have large values.
      Consider adding a small regularization or clipping.
    - The ratio cancels all phase terms common to both antennas.
    """
    # Small regularization to avoid division by zero
    eps = 1e-15 * np.max(np.abs(H_m1))
    R = H_m / (H_m1 + eps)
    return R


def compute_csi_ratio_multi(H: np.ndarray, ref_antenna: int = 0) -> np.ndarray:
    """
    Compute CSI-ratios for all antenna pairs.

    Parameters
    ----------
    H : np.ndarray
        CSI matrix, shape (N, M) where N = time samples, M = antennas.
    ref_antenna : int
        Reference antenna index for forming ratios. Default: 0.

    Returns
    -------
    R : np.ndarray
        CSI-ratios for adjacent pairs, shape (N, M-1) complex.
        R[:, i] = H[:, i] / H[:, i+1] for i = 0, ..., M-2.
    """
    N, M = H.shape
    R = np.zeros((N, M - 1), dtype=complex)
    for i in range(M - 1):
        R[:, i] = compute_csi_ratio(H[:, i], H[:, i + 1])
    return R


def compute_csi_ratio_robust(
    H_m: np.ndarray,
    H_m1: np.ndarray,
    threshold_db: float = -30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CSI-ratio with robustness to low-SNR samples.

    Filters out samples where |H_m1| is below a threshold relative
    to the maximum magnitude.

    Parameters
    ----------
    H_m : np.ndarray
        CSI samples from antenna m, shape (N,) complex.
    H_m1 : np.ndarray
        CSI samples from antenna m+1, shape (N,) complex.
    threshold_db : float
        Threshold in dB below max magnitude. Samples with |H_m1|
        below this are excluded. Default: -30 dB.

    Returns
    -------
    R : np.ndarray
        CSI-ratio samples (filtered).
    mask : np.ndarray
        Boolean mask indicating which samples were kept, shape (N,).
    """
    abs_H_m1 = np.abs(H_m1)
    max_abs = np.max(abs_H_m1)
    threshold_linear = max_abs * 10 ** (threshold_db / 20)

    mask = abs_H_m1 > threshold_linear
    R = np.zeros_like(H_m)
    R[mask] = H_m[mask] / H_m1[mask]

    return R, mask
