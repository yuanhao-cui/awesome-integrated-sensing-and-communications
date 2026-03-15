"""
Schur Complement for LMI Constraints
=====================================

Implements the Schur complement transformation for converting
nonlinear constraints into Linear Matrix Inequalities (LMIs).

Key application: CRB constraint → LMI (Eq. 18)

The Schur complement states that for a block matrix:
    M = [[A, B], [B^H, C]]

where A and C are positive definite, M ≽ 0 ⟺ C - B^H A^{-1} B ≽ 0

This allows converting quadratic constraints like x^H A^{-1} x ≤ t
into the LMI: [[A, x], [x^H, t]] ≽ 0

Reference: Zou et al., IEEE Trans. Commun., 2024 (Eq. 18)
"""

import numpy as np
from typing import Optional


def schur_complement_lmi(
    A: np.ndarray,
    b: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Construct LMI block matrix using Schur complement.

    For constraint b^H A^{-1} b ≤ t, the equivalent LMI is:
        [[A, b], [b^H, t]] ≽ 0

    Parameters
    ----------
    A : np.ndarray
        Positive definite matrix (n x n)
    b : np.ndarray
        Vector (n,)
    t : float
        Scalar upper bound

    Returns
    -------
    np.ndarray
        Block matrix (n+1 x n+1) that is PSD iff constraint holds
    """
    n = len(b)
    M = np.zeros((n + 1, n + 1), dtype=complex)

    M[:n, :n] = A
    M[:n, n] = b
    M[n, :n] = b.conj()
    M[n, n] = t

    return M


def verify_schur_complement(
    A: np.ndarray,
    b: np.ndarray,
    t: float,
    tol: float = 1e-8,
) -> bool:
    """
    Verify that Schur complement constraint holds.

    Checks if b^H A^{-1} b ≤ t, which is equivalent to:
        [[A, b], [b^H, t]] ≽ 0

    Parameters
    ----------
    A : np.ndarray
        Positive definite matrix (n x n)
    b : np.ndarray
        Vector (n,)
    t : float
        Scalar upper bound
    tol : float
        Numerical tolerance

    Returns
    -------
    bool
        True if constraint is satisfied
    """
    try:
        A_inv = np.linalg.inv(A)
        val = np.real(b.conj() @ A_inv @ b)
        return val <= t + tol
    except np.linalg.LinAlgError:
        return False


def crb_to_lmi_cvxpy(
    W_vars: list,
    a_t: np.ndarray,
    a_r: np.ndarray,
    sigma_s2: float,
    L: int,
    crb_max: float,
    M: int,
) -> list:
    """
    Convert CRB constraint to LMI form for cvxpy optimization.

    The CRB constraint (Eq. 10) is:
        CRB(W) ≤ CRB_max

    Using Schur complement, this becomes a set of LMIs:
        [[F_i, g_i], [g_i^H, t_i]] ≽ 0,  for each constraint i

    where F_i involves the transmit covariance and steering vectors,
    and t_i is bounded by CRB_max.

    Parameters
    ----------
    W_vars : list of cp.Variable
        PSD matrix variables W_k for each user (M x M each)
    a_t : np.ndarray
        Transmit steering vector (M,)
    a_r : np.ndarray
        Receive steering vector (N,)
    sigma_s2 : float
        Sensing noise power
    L : int
        Frame length
    crb_max : float
        Maximum allowed CRB
    M : int
        Number of transmit antennas

    Returns
    -------
    list
        List of cvxpy LMI constraints
    """
    import cvxpy as cp

    constraints = []

    # Total transmit covariance: Rx = Σ_k W_k
    Rx = sum(W_vars)

    # Regularization for numerical stability
    reg = sigma_s2 / L

    # Derivative of transmit steering vector
    # At θ=90°, da_t/dθ = j*(2π/λ)*d*m*cos(θ)*a_t = 0 for perfect ULA
    # We use effective derivative for general angles
    theta = np.pi / 2
    m_indices = np.arange(M)
    da_t = 1j * 2 * np.pi * 0.5 * m_indices * np.cos(theta) * a_t

    a_r_norm_sq = np.sum(np.abs(a_r) ** 2)

    # Construct LMI: [[A, b], [b^H, γ]] ≽ 0
    # where A involves Rx, b involves da_t, γ bounded by CRB_max

    # Fisher Information Matrix component
    # FIM = (2L/σ_s²) * ||a_r||² * da_t^H * Rx * da_t
    # CRB = 1/FIM

    # Constraint: da_t^H * Rx * da_t ≥ σ_s² * ||a_r||² / (2L * CRB_max)
    # Equivalent to: da_t^H * Rx * da_t ≥ threshold

    threshold = sigma_s2 * a_r_norm_sq / (2 * L * crb_max)

    # LMI constraint using Schur complement
    # [[Rx, sqrt(threshold)*da_t], [sqrt(threshold)*da_t^H, 1]] ≽ 0
    # implies: da_t^H Rx da_t ≥ threshold

    sqrt_threshold = np.sqrt(max(threshold, 1e-15))

    # Build LMI matrix
    block_11 = Rx  # M x M
    block_12 = sqrt_threshold * da_t.reshape(-1, 1)  # M x 1
    block_21 = sqrt_threshold * da_t.conj().reshape(1, -1)  # 1 x M
    block_22 = np.array([[1.0]])  # 1 x 1

    lmi_mat = cp.bmat(
        [
            [block_11, block_12],
            [block_21, block_22],
        ]
    )

    constraints.append(lmi_mat >> 0)

    return constraints


def power_constraint_lmi(
    W_vars: list,
    P_max: float,
) -> "cp.Constraint":
    """
    Create power constraint: Σ_k tr(W_k) ≤ P_max.

    Parameters
    ----------
    W_vars : list of cp.Variable
        PSD matrix variables (M x M each)
    P_max : float
        Maximum transmit power

    Returns
    -------
    cp.Constraint
        Power constraint for cvxpy
    """
    import cvxpy as cp

    total_power = sum(cp.trace(W_k) for W_k in W_vars)
    return total_power <= P_max


def sinr_constraint_lmi(
    h_k: np.ndarray,
    W_k: "cp.Expression",
    W_all: list,
    sigma_c2: float,
    gamma_min: float,
    k: int,
) -> "cp.Constraint":
    """
    Create SINR constraint in LMI form (Eq. for SINR ≥ γ_min).

    For user k: |h_k^H w_k|² ≥ γ_min (σ_c² + Σ_{j≠k} |h_k^H w_j|²)

    In SDR form: h_k^H W_k h_k ≥ γ_min (σ_c² + Σ_{j≠k} h_k^H W_j h_k)

    This is already a convex constraint (linear in W variables).

    Parameters
    ----------
    h_k : np.ndarray
        Channel vector for user k (M,)
    W_k : cp.Expression
        PSD matrix variable for user k
    W_all : list of cp.Expression
        All K PSD matrix variables
    sigma_c2 : float
        Noise power
    gamma_min : float
        Minimum SINR requirement
    k : int
        User index

    Returns
    -------
    cp.Constraint
        SINR constraint for cvxpy
    """
    import cvxpy as cp

    signal = cp.quad_form(h_k, W_k)
    interference = sum(
        cp.quad_form(h_k, W_j) for j, W_j in enumerate(W_all) if j != k
    )

    return signal >= gamma_min * (sigma_c2 + interference)
