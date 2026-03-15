"""
Circle Fitting for CSI-Ratio Samples.

Implements Eq. (11) from the paper - least-squares circle fit.

When the CSI-ratio R(t) = H_m(t)/H_{m+1}(t) is plotted in the complex
plane, the samples lie on a circle. This module finds the circle
parameters (center and radius).

The circle equation in the complex plane:
    |R - C_0|^2 = r^2
    where C_0 = A + jB is the center, r is the radius.

Expanding (Eq. 11):
    R_i = x_i + j*y_i
    x_i^2 + y_i^2 = 2*A*x_i + 2*B*y_i + C
    where C = r^2 - A^2 - B^2

This is a linear least-squares problem in (A, B, C).
"""

import numpy as np
from typing import Tuple


def least_squares_circle_fit(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a circle to complex samples using least-squares (Eq. 11).

    Solves the linear system:
        x_i^2 + y_i^2 = 2*A*x_i + 2*B*y_i + C

    in the least-squares sense, where:
        C_0 = A + jB  (circle center)
        r = sqrt(C + A^2 + B^2)  (circle radius)

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).

    Returns
    -------
    A : float
        Real part of circle center.
    B : float
        Imaginary part of circle center.
    r : float
        Circle radius.

    Notes
    -----
    This is the primary circle fitting method from Eq. (11) of the paper.
    The center C_0 = A + jB is used to shift the circle to the origin
    before computing the Doppler estimate.
    """
    x = np.real(R)
    y = np.imag(R)

    # Build the design matrix for: x^2 + y^2 = 2*A*x + 2*B*y + C
    # Stack as: [2x, 2y, 1] @ [A, B, C]^T = x^2 + y^2
    N = len(R)
    A_matrix = np.column_stack([2 * x, 2 * y, np.ones(N)])
    b = x ** 2 + y ** 2

    # Solve least-squares: min ||A_matrix @ theta - b||^2
    theta, residuals, rank, s = np.linalg.lstsq(A_matrix, b, rcond=None)

    A_center = theta[0]
    B_center = theta[1]
    C_const = theta[2]

    # Compute radius
    r = np.sqrt(max(C_const + A_center ** 2 + B_center ** 2, 0))

    return A_center, B_center, r


def fit_circle_kasa(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit circle using Kasa's method (algebraic fit).

    Simpler but less accurate than least-squares for noisy data.
    Included for comparison.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).

    Returns
    -------
    A : float
        Real part of circle center.
    B : float
        Imaginary part of circle center.
    r : float
        Circle radius.
    """
    x = np.real(R)
    y = np.imag(R)

    # Kasa: minimize sum of (x^2 + y^2 - 2Ax - 2By - C)^2
    M = np.column_stack([2 * x, 2 * y, np.ones(len(R))])
    b = x ** 2 + y ** 2

    theta = np.linalg.lstsq(M, b, rcond=None)[0]

    A, B, C = theta
    r = np.sqrt(max(C + A ** 2 + B ** 2, 0))
    return A, B, r


def fit_circle_pratt(R: np.ndarray, max_iter: int = 50) -> Tuple[float, float, float]:
    """
    Fit circle using Taubin-Pratt method (improved algebraic fit).

    Better accuracy than Kasa, iterative refinement.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    max_iter : int
        Maximum iterations. Default: 50.

    Returns
    -------
    A : float
        Real part of circle center.
    B : float
        Imaginary part of circle center.
    r : float
        Circle radius.
    """
    x = np.real(R)
    y = np.imag(R)
    N = len(R)

    # Start with Kasa estimate
    A, B, r = fit_circle_kasa(R)

    # Iterative refinement using geometric distances
    for _ in range(max_iter):
        # Compute distances from current circle
        dx = x - A
        dy = y - B
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist = np.maximum(dist, 1e-15)

        # Weights: closer points get higher weight
        w = 1.0 / dist
        w = w / np.sum(w)

        # Weighted least-squares
        W = np.diag(w)
        M = np.column_stack([2 * x, 2 * y, np.ones(N)])
        b = x ** 2 + y ** 2

        theta = np.linalg.lstsq(M.T @ W @ M, M.T @ W @ b, rcond=None)[0]
        A_new, B_new, C_new = theta
        r_new = np.sqrt(max(C_new + A_new ** 2 + B_new ** 2, 0))

        # Check convergence
        if abs(A_new - A) < 1e-10 and abs(B_new - B) < 1e-10:
            break

        A, B, r = A_new, B_new, r_new

    return A, B, r


def circle_fit_error(R: np.ndarray, A: float, B: float, r: float) -> float:
    """
    Compute RMS error of circle fit.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples.
    A, B : float
        Circle center coordinates.
    r : float
        Circle radius.

    Returns
    -------
    rms_error : float
        Root mean square error between sample distances and radius.
    """
    distances = np.abs(R - (A + 1j * B))
    return np.sqrt(np.mean((distances - r) ** 2))
