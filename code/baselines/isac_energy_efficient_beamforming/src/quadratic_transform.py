"""
Quadratic Transform for Log-SINR
=================================

Implements the quadratic transform (Eq. 14) for converting non-convex
log(1+SINR) terms into more tractable forms.

The quadratic transform introduces auxiliary variables t_k to handle:
    log(1 + SINR_k) = log(1 + |h_k^H w_k|² / (σ_c² + Σ_{j≠k} |h_k^H w_j|²))

Using the quadratic transform, this becomes:
    2 Re{t_k^* h_k^H w_k} - |t_k|² (σ_c² + Σ_j |h_k^H w_j|²)

where t_k is the optimal auxiliary variable.

Reference: Zou et al., IEEE Trans. Commun., 2024 (Eq. 14)
"""

import numpy as np
from typing import Optional, Tuple


def quadratic_transform_objective(
    H: np.ndarray,
    W: np.ndarray,
    t: np.ndarray,
    sigma_c2: float,
) -> float:
    """
    Compute the quadratic transform objective for all users.

    Objective: Σ_k [2 Re{t_k^* h_k^H w_k} - |t_k|² (σ_c² + Σ_j |h_k^H w_j|²)]

    This is the reformulation of Σ_k log(1+SINR_k) using auxiliary
    variables t_k.

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    W : np.ndarray
        Beamforming matrix (M x K)
    t : np.ndarray
        Auxiliary variables (K,) complex
    sigma_c2 : float
        Noise power

    Returns
    -------
    float
        Quadratic transform objective value
    """
    K = H.shape[0]
    obj = 0.0

    for k in range(K):
        h_k = H[k, :]
        hw_k = h_k.conj() @ W[:, k]

        # Signal term: 2 Re{t_k^* h_k^H w_k}
        signal_term = 2 * np.real(np.conj(t[k]) * hw_k)

        # Power term: |t_k|² * (σ_c² + Σ_j |h_k^H w_j|²)
        total_power_k = sigma_c2 + sum(
            np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K)
        )
        power_term = np.abs(t[k]) ** 2 * total_power_k

        obj += signal_term - power_term

    return float(obj)


def optimize_t(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
) -> np.ndarray:
    """
    Compute optimal auxiliary variables t_k (Eq. 14, closed form).

    The optimal t_k is:
        t_k* = h_k^H w_k / (σ_c² + Σ_j |h_k^H w_j|²)

    This maximizes the quadratic transform for fixed W.

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
    np.ndarray
        Optimal auxiliary variables t (K,) complex
    """
    K = H.shape[0]
    t = np.zeros(K, dtype=complex)

    for k in range(K):
        h_k = H[k, :]
        hw_k = h_k.conj() @ W[:, k]

        total_power_k = sigma_c2 + sum(
            np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K)
        )

        if total_power_k > 1e-15:
            t[k] = hw_k / total_power_k
        else:
            t[k] = 0.0

    return t


def quadratic_transform_cvxpy_expr(
    h_k: np.ndarray,
    W_k: "cp.Expression",
    W_all: list,
    sigma_c2: float,
    t_k: complex,
) -> "cp.Expression":
    """
    Create cvxpy expression for quadratic transform of user k.

    Expression: 2 Re{t_k^* h_k^H W_k h_k} - |t_k|² (σ_c² + Σ_j h_k^H W_j h_k)

    For SDR: w_k w_k^H → W_k (PSD matrix)
    Then h_k^H w_k becomes h_k^H W_k h_k

    Parameters
    ----------
    h_k : np.ndarray
        Channel vector for user k (M,)
    W_k : cp.Expression
        PSD matrix variable for user k (M x M)
    W_all : list of cp.Expression
        All K PSD matrix variables
    sigma_c2 : float
        Noise power
    t_k : complex
        Auxiliary variable for user k

    Returns
    -------
    cp.Expression
        Quadratic transform expression for cvxpy
    """
    import cvxpy as cp

    # Signal term: 2 Re{t_k^* h_k^H W_k h_k}
    signal = cp.real(np.conj(t_k) * cp.quad_form(h_k, W_k))

    # Power term: |t_k|² * (σ_c² + Σ_j h_k^H W_j h_k)
    t_k_sq = np.abs(t_k) ** 2
    power_terms = [cp.quad_form(h_k, W_j) for W_j in W_all]
    power = t_k_sq * (sigma_c2 + sum(power_terms))

    return 2 * signal - power


def compute_sum_rate_quadratic(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> float:
    """
    Compute sum rate using iterative quadratic transform.

    Alternates between:
    1. Fix W, optimize t_k
    2. The objective value at optimal t_k equals log(1+SINR_k)

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    W : np.ndarray
        Beamforming matrix (M x K)
    sigma_c2 : float
        Noise power
    max_iter : int
        Maximum iterations (default: 10)
    tol : float
        Convergence tolerance

    Returns
    -------
    float
        Sum rate in bits/Hz
    """
    t = optimize_t(H, W, sigma_c2)

    # At optimal t_k, the quadratic transform equals log(1+SINR_k)
    # Direct computation is more accurate
    K = H.shape[0]
    sum_rate = 0.0

    for k in range(K):
        sinr_k = 0.0
        h_k = H[k, :]
        signal = np.abs(h_k.conj() @ W[:, k]) ** 2
        interference = sum(
            np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K) if j != k
        )
        sinr_k = signal / (sigma_c2 + interference)
        sum_rate += np.log2(1 + sinr_k)

    return float(sum_rate)


class QuadraticTransform:
    """
    Quadratic transform solver for log-SINR optimization.

    Implements the iterative procedure:
    1. Initialize auxiliary variables t_k
    2. Optimize W given t_k (convex subproblem)
    3. Optimize t_k given W (closed-form)
    4. Repeat until convergence

    Reference: Eq. (14) in Zou et al., 2024
    """

    def __init__(
        self,
        H: np.ndarray,
        sigma_c2: float,
    ):
        """
        Initialize quadratic transform solver.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix (K x M)
        sigma_c2 : float
            Communication noise power
        """
        self.H = H
        self.K, self.M = H.shape
        self.sigma_c2 = sigma_c2

    def solve(
        self,
        W_init: np.ndarray,
        max_iter: int = 20,
        tol: float = 1e-5,
    ) -> Tuple[np.ndarray, float]:
        """
        Run iterative quadratic transform optimization.

        Parameters
        ----------
        W_init : np.ndarray
            Initial beamforming matrix (M x K)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        W_opt : np.ndarray
            Optimized beamforming matrix
        obj_val : float
            Final objective value (sum rate approximation)
        """
        W = W_init.copy()
        prev_obj = -np.inf

        for iteration in range(max_iter):
            # Step 1: Optimize t_k given W
            t = optimize_t(self.H, W, self.sigma_c2)

            # Step 2: Compute objective
            obj = quadratic_transform_objective(self.H, W, t, self.sigma_c2)

            # Check convergence
            if abs(obj - prev_obj) < tol * (1 + abs(prev_obj)):
                break

            prev_obj = obj

            # Step 3: Update W (simplified gradient step for demonstration)
            # In full implementation, this would be a convex subproblem
            W = self._update_W(W, t)

        return W, obj

    def _update_W(self, W: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Simple gradient-based W update.

        In the full algorithm, this is replaced by a convex optimization
        subproblem solved via cvxpy.

        Parameters
        ----------
        W : np.ndarray
            Current beamforming matrix
        t : np.ndarray
            Auxiliary variables

        Returns
        -------
        np.ndarray
            Updated beamforming matrix
        """
        # Gradient of quadratic transform w.r.t. w_k
        W_new = np.zeros_like(W)

        for k in range(self.K):
            h_k = self.H[k, :]
            # Gradient direction: t_k^* h_k
            grad = np.conj(t[k]) * h_k

            # Normalized update
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-15:
                W_new[:, k] = W[:, k] + 0.1 * grad / grad_norm
            else:
                W_new[:, k] = W[:, k]

        return W_new
