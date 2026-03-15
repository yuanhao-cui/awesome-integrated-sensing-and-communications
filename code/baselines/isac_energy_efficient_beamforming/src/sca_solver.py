"""
Successive Convex Approximation (SCA) Solver
==============================================

Implements SCA for handling non-convex constraints in ISAC beamforming
optimization, specifically:
    - Rank-1 constraint: rank(W_k) = 1 (Eq. 22)
    - Non-convex constraints in sensing EE: ω ≤ ζ²/φ (Eq. 39)

SCA iteratively:
1. Linearizes non-convex constraints around current point
2. Solves the resulting convex subproblem
3. Updates the linearization point
4. Repeats until convergence

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
import cvxpy as cp


class SCASolver:
    """
    Successive Convex Approximation solver for non-convex ISAC problems.

    Handles:
    - Rank-1 relaxation (Eq. 22)
    - Non-convex quadratic constraints (Eq. 39)
    - Iterative refinement of SDR solutions
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-4,
        solver: str = "MOSEK",
        verbose: bool = False,
    ):
        """
        Initialize SCA solver.

        Parameters
        ----------
        max_iter : int
            Maximum SCA iterations
        tol : float
            Convergence tolerance
        solver : str
            CVXPY solver name
        verbose : bool
            Print iteration details
        """
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.verbose = verbose

    def solve_rank1_sca(
        self,
        H: np.ndarray,
        P_max: float,
        sigma_c2: float,
        a_t: Optional[np.ndarray] = None,
        a_r: Optional[np.ndarray] = None,
        sigma_s2: float = 1e-8,
        L: int = 30,
        crb_max: Optional[float] = None,
        gamma_min: Optional[float] = None,
        W_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Solve beamforming with SCA for rank-1 constraint (Eq. 22).

        The rank-1 constraint is handled by adding a penalty term:
            f(W) - μ * Σ_k (tr(W_k) - ||w_k||²)

        which encourages rank-1 solutions.

        At each iteration:
        1. Linearize the concave part: tr(W_k) - ||w_k||² around current W_k
        2. Solve the resulting convex problem
        3. Update linearization point

        Parameters
        ----------
        H : np.ndarray
            Channel matrix (K x M)
        P_max : float
            Maximum transmit power
        sigma_c2 : float
            Noise power
        a_t : np.ndarray, optional
            Transmit steering vector
        a_r : np.ndarray, optional
            Receive steering vector
        sigma_s2 : float
            Sensing noise power
        L : int
            Frame length
        crb_max : float, optional
            Maximum CRB
        gamma_min : float, optional
            Minimum SINR
        W_init : np.ndarray, optional
            Initial beamforming matrix (M x K)

        Returns
        -------
        W_opt : np.ndarray
            Optimized beamforming matrix (M x K)
        obj_history : list
            Objective value at each iteration
        """
        K, M = H.shape
        obj_history = []

        # Initialize with equal power allocation
        if W_init is None:
            W_init = np.zeros((M, K), dtype=complex)
            for k in range(K):
                h_k = H[k, :]
                w_k = h_k / np.linalg.norm(h_k)
                w_k *= np.sqrt(P_max / K)
                W_init[:, k] = w_k

        W_current = W_init.copy()
        mu = 1.0  # Penalty parameter

        for iteration in range(self.max_iter):
            # Solve convex subproblem with linearized rank penalty
            W_next, obj_val, status = self._solve_convex_subproblem(
                H, W_current, P_max, sigma_c2, mu,
                a_t, a_r, sigma_s2, L, crb_max, gamma_min,
            )

            if status not in ["optimal", "optimal_inaccurate"]:
                if self.verbose:
                    print(f"SCA iter {iteration}: solver status = {status}")
                break

            obj_history.append(obj_val)

            # Check convergence
            if iteration > 0:
                rel_change = abs(obj_history[-1] - obj_history[-2]) / (
                    abs(obj_history[-2]) + 1e-10
                )
                if rel_change < self.tol:
                    if self.verbose:
                        print(f"SCA converged at iteration {iteration}")
                    break

            # Update linearization point
            W_current = W_next

            # Increase penalty parameter
            mu *= 1.5

        return W_current, obj_history

    def _solve_convex_subproblem(
        self,
        H: np.ndarray,
        W_lin: np.ndarray,
        P_max: float,
        sigma_c2: float,
        mu: float,
        a_t: Optional[np.ndarray],
        a_r: Optional[np.ndarray],
        sigma_s2: float,
        L: int,
        crb_max: Optional[float],
        gamma_min: Optional[float],
    ) -> Tuple[np.ndarray, float, str]:
        """
        Solve convex subproblem with linearized constraints.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix
        W_lin : np.ndarray
            Linearization point (M x K)
        P_max : float
            Maximum power
        sigma_c2 : float
            Noise power
        mu : float
            Penalty parameter
        ... (other parameters)

        Returns
        -------
        W_opt : np.ndarray
            Solution (M x K)
        obj_val : float
            Objective value
        status : str
            Solver status
        """
        K, M = H.shape

        # Define variables as individual vectors
        W_vars = [cp.Variable((M, 1), complex=True) for _ in range(K)]

        # Construct W_k matrices
        W_mats = [W_vars[k] @ W_vars[k].H for k in range(K)]

        # Alternative: Use PSD matrix variables directly
        W_psd = [cp.Variable((M, M), hermitian=True) for _ in range(K)]

        constraints = []

        # PSD constraints
        for k in range(K):
            constraints.append(W_psd[k] >> 0)

        # Power constraint
        total_power = sum(cp.trace(W_psd[k]) for k in range(K))
        constraints.append(cp.real(total_power) <= P_max)

        # SINR constraints
        if gamma_min is not None:
            for k in range(K):
                h_k = H[k, :]
                signal = cp.quad_form(h_k, W_psd[k])
                interference = sum(
                    cp.quad_form(h_k, W_psd[j])
                    for j in range(K)
                    if j != k
                )
                constraints.append(
                    cp.real(signal) >= gamma_min * (sigma_c2 + cp.real(interference))
                )

        # CRB constraint (simplified LMI)
        if crb_max is not None and a_t is not None and a_r is not None:
            Rx = sum(W_psd)
            threshold = sigma_s2 * np.sum(np.abs(a_r) ** 2) / (2 * L * crb_max)
            constraints.append(cp.real(cp.quad_form(a_t, Rx)) >= threshold)

        # Linearized rank penalty
        # f(W_k) = tr(W_k) - ||w_k||² (concave)
        # Linearization: f(W_k) ≈ f(W_lin_k) + ⟨∇f(W_lin_k), W_k - W_lin_k⟩
        # ∇f(W_lin_k) = I - (w_lin_k w_lin_k^H) / ||w_lin_k||²

        rank_penalty = 0
        for k in range(K):
            w_lin_k = W_lin[:, k : k + 1]
            w_norm_sq = np.real(w_lin_k.conj().T @ w_lin_k)[0, 0]

            if w_norm_sq > 1e-15:
                # Gradient: I - w_lin w_lin^H / ||w_lin||²
                grad = np.eye(M) - (w_lin_k @ w_lin_k.conj().T) / w_norm_sq
            else:
                grad = np.eye(M)

            # Linear approximation
            rank_penalty += cp.real(
                cp.trace(grad @ (W_psd[k] - w_lin_k @ w_lin_k.conj().T))
            )

        # Objective: maximize sum rate approximation - μ * rank penalty
        obj_terms = []
        for k in range(K):
            h_k = H[k, :]
            # Use trace as proxy for signal power
            obj_terms.append(cp.real(cp.quad_form(h_k, W_psd[k])))

        objective = cp.Maximize(sum(obj_terms) - mu * rank_penalty)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            status = prob.status
        except (cp.error.SolverError, Exception):
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
                status = prob.status
            except Exception:
                return W_lin, -np.inf, "failed"

        if prob.status in ["optimal", "optimal_inaccurate"]:
            # Recover beamforming vectors from PSD solutions
            W_opt = np.zeros((M, K), dtype=complex)
            for k in range(K):
                W_k = W_psd[k].value
                if W_k is not None:
                    # Eigenvector corresponding to largest eigenvalue
                    eigenvalues, eigenvectors = np.linalg.eigh(W_k)
                    idx = np.argmax(eigenvalues)
                    w_k = eigenvectors[:, idx] * np.sqrt(max(eigenvalues[idx], 0))
                    W_opt[:, k] = w_k
            return W_opt, prob.value, prob.status
        else:
            return W_lin, -np.inf, prob.status

    def solve_sensing_ee_sca(
        self,
        H: np.ndarray,
        a_t: np.ndarray,
        a_r: np.ndarray,
        P_max: float,
        sigma_s2: float,
        L: int,
        epsilon: float,
        P0: float,
        W_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Solve sensing EE maximization using SCA (Algorithm 3).

        Handles the non-convex constraint ω ≤ ζ²/φ (Eq. 39) using SCA.

        The problem is:
            max EE_S = CRB⁻¹ / (L * ((1/ε)||W||² + P₀))
        subject to power constraint.

        SCA linearizes: ω ≤ ζ²/φ ≈ ω ≤ 2ζ₀ζ/φ₀ - ζ₀²φ/φ₀²

        Parameters
        ----------
        H : np.ndarray
            Channel matrix (for SINR constraints if needed)
        a_t : np.ndarray
            Transmit steering vector (M,)
        a_r : np.ndarray
            Receive steering vector (N,)
        P_max : float
            Maximum transmit power
        sigma_s2 : float
            Sensing noise power
        L : int
            Frame length
        epsilon : float
            PA efficiency
        P0 : float
            Circuit power
        W_init : np.ndarray, optional
            Initial beamforming

        Returns
        -------
        W_opt : np.ndarray
            Optimized beamforming matrix
        obj_history : list
            Objective values
        """
        K, M = H.shape
        obj_history = []

        if W_init is None:
            # Initialize with matched filter (beam toward target)
            W_init = np.zeros((M, K), dtype=complex)
            for k in range(K):
                W_init[:, k] = a_t / np.sqrt(K) * np.sqrt(P_max / K)

        W_current = W_init.copy()

        for iteration in range(self.max_iter):
            # Compute auxiliary variables at current point
            Rx = W_current @ W_current.conj().T

            # Compute CRB at current point
            from .ee_metrics import compute_crb, compute_total_power
            crb = compute_crb(W_current, a_t, a_r, sigma_s2, L)
            total_power = compute_total_power(W_current)
            total_consumption = (1 / epsilon) * total_power + P0

            if crb <= 0 or np.isinf(crb) or total_consumption <= 0:
                break

            # Current EE_S
            ee_s = (1 / crb) / (L * total_consumption)
            obj_history.append(ee_s)

            # Linearization variables for SCA
            # ω = CRB⁻¹, ζ = something related to signal, φ = power consumption
            # At current point: ω₀ = 1/crb, ζ₀ = ..., φ₀ = total_consumption

            # Solve convex subproblem
            W_next, status = self._solve_sensing_ee_subproblem(
                H, W_current, a_t, a_r, P_max, sigma_s2, L, epsilon, P0, crb, total_consumption,
            )

            if status not in ["optimal", "optimal_inaccurate"]:
                break

            # Check convergence
            if iteration > 0:
                rel_change = abs(obj_history[-1] - obj_history[-2]) / (
                    abs(obj_history[-2]) + 1e-10
                )
                if rel_change < self.tol:
                    break

            W_current = W_next

        return W_current, obj_history

    def _solve_sensing_ee_subproblem(
        self,
        H: np.ndarray,
        W_lin: np.ndarray,
        a_t: np.ndarray,
        a_r: np.ndarray,
        P_max: float,
        sigma_s2: float,
        L: int,
        epsilon: float,
        P0: float,
        crb_current: float,
        power_consumption_current: float,
    ) -> Tuple[np.ndarray, str]:
        """
        Solve convex subproblem for sensing EE maximization.

        Uses SCA linearization for non-convex constraints.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix
        W_lin : np.ndarray
            Linearization point
        a_t, a_r : np.ndarray
            Steering vectors
        P_max : float
            Power constraint
        sigma_s2 : float
            Sensing noise power
        L : int
            Frame length
        epsilon : float
            PA efficiency
        P0 : float
            Circuit power
        crb_current : float
            Current CRB value
        power_consumption_current : float
            Current total power consumption

        Returns
        -------
        W_opt : np.ndarray
            Solution
        status : str
            Solver status
        """
        K, M = H.shape

        # PSD matrix variables
        W_psd = [cp.Variable((M, M), hermitian=True) for _ in range(K)]

        constraints = []
        for k in range(K):
            constraints.append(W_psd[k] >> 0)

        # Power constraint
        total_power = sum(cp.trace(W_psd[k]) for k in range(K))
        constraints.append(cp.real(total_power) <= P_max)

        # Linearized CRB constraint (simplified)
        # CRB⁻¹ ∝ da_t^H Rx da_t, where Rx = Σ_k W_k
        Rx = sum(W_psd)

        # Derivative of steering vector
        theta = np.pi / 2
        m_indices = np.arange(M)
        da_t = 1j * 2 * np.pi * 0.5 * m_indices * np.cos(theta) * a_t

        a_r_norm_sq = np.sum(np.abs(a_r) ** 2)

        # Fisher information: FIM ∝ da_t^H Rx da_t
        fim_expr = cp.real(cp.quad_form(da_t, Rx))

        # Linearized EE_S objective
        # EE_S = (1/CRB) / (L * P_consumption)
        # At current point: ee_s₀ = (1/crb₀) / (L * P₀_cons)
        # Linearize using first-order Taylor expansion

        ee_s_current = (1 / crb_current) / (L * power_consumption_current)
        crb_inv_current = 1 / crb_current

        # Linear approximation of sensing EE
        # Objective: maximize linearized EE_S
        power_consumption_expr = (1 / epsilon) * cp.real(total_power) + P0

        # Use Dinkelbach-like formulation
        # max (FIM / (σ_s² * L * P_consumption))
        # ≈ max (FIM - λ * P_consumption) with λ updated iteratively

        sensing_rate = fim_expr / (sigma_s2 * L)

        objective = cp.Maximize(
            sensing_rate - ee_s_current * power_consumption_expr
        )

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            status = prob.status
        except (cp.error.SolverError, Exception):
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
                status = prob.status
            except Exception:
                return W_lin, "failed"

        if prob.status in ["optimal", "optimal_inaccurate"]:
            W_opt = np.zeros((M, K), dtype=complex)
            for k in range(K):
                W_k = W_psd[k].value
                if W_k is not None:
                    eigenvalues, eigenvectors = np.linalg.eigh(W_k)
                    idx = np.argmax(eigenvalues)
                    w_k = eigenvectors[:, idx] * np.sqrt(max(eigenvalues[idx], 0))
                    W_opt[:, k] = w_k
            return W_opt, status
        else:
            return W_lin, status
