"""
SDR Solver with Rank-1 Recovery
================================

Implements Semidefinite Relaxation (SDR) for beamforming optimization
with rank-1 recovery via Gaussian randomization.

Key concepts:
    - Replace w_k w_k^H with W_k ⪰ 0 (PSD matrix)
    - Solve relaxed SDP
    - Recover rank-1 solution via randomization

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import numpy as np
from typing import Optional, Tuple, List
import cvxpy as cp


class SDRSolver:
    """
    Semidefinite Relaxation solver for ISAC beamforming.

    Solves:
        max  f(W_1, ..., W_K)
        s.t. W_k ⪰ 0, rank(W_k) = 1, ∀k
             Σ_k tr(W_k) ≤ P_max
             SINR_k ≥ γ_min, ∀k
             CRB ≤ CRB_max

    Relaxation: Drop rank constraints → SDP
    Recovery: Gaussian randomization → rank-1 approximation
    """

    def __init__(
        self,
        M: int,
        K: int,
        P_max: float,
        sigma_c2: float,
        solver: str = "MOSEK",
    ):
        """
        Initialize SDR solver.

        Parameters
        ----------
        M : int
            Number of transmit antennas
        K : int
            Number of users
        P_max : float
            Maximum transmit power
        sigma_c2 : float
            Noise power
        solver : str
            CVXPY solver (default: MOSEK, fallback: SCS)
        """
        self.M = M
        self.K = K
        self.P_max = P_max
        self.sigma_c2 = sigma_c2
        self.solver = solver

    def solve_sdr(
        self,
        H: np.ndarray,
        a_t: Optional[np.ndarray] = None,
        a_r: Optional[np.ndarray] = None,
        sigma_s2: float = 1e-8,
        L: int = 30,
        crb_max: Optional[float] = None,
        gamma_min: Optional[float] = None,
        objective: str = "sum_rate",
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Solve the SDR problem.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix (K x M)
        a_t : np.ndarray, optional
            Transmit steering vector (M,)
        a_r : np.ndarray, optional
            Receive steering vector (N,)
        sigma_s2 : float
            Sensing noise power
        L : int
            Frame length
        crb_max : float, optional
            Maximum CRB constraint
        gamma_min : float, optional
            Minimum SINR per user
        objective : str
            Objective type: 'sum_rate', 'min_power'

        Returns
        -------
        W_list : list of np.ndarray
            PSD matrix solutions W_k (M x M each)
        info : dict
            Solver information (status, objective, etc.)
        """
        # Define PSD matrix variables
        W_vars = [cp.Variable((self.M, self.M), hermitian=True) for _ in range(self.K)]

        constraints = []

        # PSD constraints
        for k in range(self.K):
            constraints.append(W_vars[k] >> 0)

        # Power constraint: Σ_k tr(W_k) ≤ P_max
        total_power = sum(cp.trace(W_k) for W_k in W_vars)
        constraints.append(cp.real(total_power) <= self.P_max)

        # SINR constraints (if gamma_min specified)
        if gamma_min is not None:
            for k in range(self.K):
                h_k = H[k, :]
                signal = cp.quad_form(h_k, W_vars[k])
                interference = sum(
                    cp.quad_form(h_k, W_vars[j])
                    for j in range(self.K)
                    if j != k
                )
                constraints.append(
                    cp.real(signal) >= gamma_min * (self.sigma_c2 + cp.real(interference))
                )

        # CRB constraint (if specified)
        if crb_max is not None and a_t is not None and a_r is not None:
            from .schur_complement import crb_to_lmi_cvxpy

            crb_constraints = crb_to_lmi_cvxpy(
                W_vars, a_t, a_r, sigma_s2, L, crb_max, self.M
            )
            constraints.extend(crb_constraints)

        # Objective
        if objective == "sum_rate":
            # Approximate sum rate using quadratic lower bound
            obj_terms = []
            for k in range(self.K):
                h_k = H[k, :]
                # Use log(1+x) ≥ log(1+x_0) + (x-x_0)/(1+x_0) as lower bound
                # For initial iteration, use direct trace objective
                obj_terms.append(cp.quad_form(h_k, W_vars[k]))
            objective_expr = cp.Maximize(cp.real(sum(obj_terms)))
        elif objective == "min_power":
            objective_expr = cp.Minimize(cp.real(total_power))
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Solve
        prob = cp.Problem(objective_expr, constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            solver_used = "MOSEK"
        except (cp.error.SolverError, Exception):
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
                solver_used = "SCS"
            except Exception as e:
                return [], {"status": "failed", "error": str(e)}

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return [], {"status": prob.status}

        # Extract solution
        W_list = [W_vars[k].value for k in range(self.K)]

        info = {
            "status": prob.status,
            "optimal_value": prob.value,
            "solver": solver_used,
        }

        return W_list, info

    def rank1_recovery(
        self,
        W_list: List[np.ndarray],
        n_random: int = 100,
    ) -> np.ndarray:
        """
        Recover rank-1 beamforming vectors via Gaussian randomization.

        For each W_k:
        1. Eigendecomposition: W_k = U Λ U^H
        2. Generate random vector: r ~ CN(0, I)
        3. Construct: w_k = U Λ^{1/2} r
        4. Scale to satisfy power constraint

        Parameters
        ----------
        W_list : list of np.ndarray
            PSD matrix solutions (M x M each)
        n_random : int
            Number of randomization trials

        Returns
        -------
        np.ndarray
            Recovered beamforming matrix (M x K)
        """
        K = len(W_list)
        M = W_list[0].shape[0]

        best_W = None
        best_obj = -np.inf

        for _ in range(n_random):
            W_trial = np.zeros((M, K), dtype=complex)

            for k in range(K):
                W_k = W_list[k]

                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(W_k)
                eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

                # Random Gaussian vector
                r = (
                    np.random.standard_normal(M)
                    + 1j * np.random.standard_normal(M)
                ) / np.sqrt(2)

                # Construct w_k = U Λ^{1/2} r
                sqrt_lambda = np.sqrt(eigenvalues)
                w_k = eigenvectors @ (sqrt_lambda * r)

                W_trial[:, k] = w_k

            # Scale to satisfy power constraint
            total_power = np.sum(np.abs(W_trial) ** 2)
            if total_power > self.P_max:
                W_trial *= np.sqrt(self.P_max / total_power)

            # Evaluate objective (sum of diagonal entries of W_k)
            obj = np.sum(
                [np.real(np.trace(W_trial[:, k : k + 1] @ W_trial[:, k : k + 1].conj().T)) for k in range(K)]
            )

            if obj > best_obj:
                best_obj = obj
                best_W = W_trial.copy()

        return best_W if best_W is not None else np.zeros((M, K), dtype=complex)


def sdr_sum_rate_optimization(
    H: np.ndarray,
    P_max: float,
    sigma_c2: float,
    gamma_min: Optional[float] = None,
    n_random: int = 200,
) -> Tuple[np.ndarray, float]:
    """
    Convenience function for SDR-based sum rate maximization.

    Parameters
    ----------
    H : np.ndarray
        Channel matrix (K x M)
    P_max : float
        Maximum transmit power
    sigma_c2 : float
        Noise power
    gamma_min : float, optional
        Minimum SINR
    n_random : int
        Randomization trials

    Returns
    -------
    W_opt : np.ndarray
        Optimal beamforming matrix (M x K)
    sum_rate : float
        Achieved sum rate
    """
    K, M = H.shape
    solver = SDRSolver(M, K, P_max, sigma_c2)

    W_list, info = solver.solve_sdr(
        H, gamma_min=gamma_min, objective="sum_rate"
    )

    if not W_list:
        return np.zeros((M, K), dtype=complex), 0.0

    W_opt = solver.rank1_recovery(W_list, n_random)

    # Compute sum rate
    from .ee_metrics import compute_sum_rate
    sum_rate = compute_sum_rate(H, W_opt, sigma_c2)

    return W_opt, sum_rate
