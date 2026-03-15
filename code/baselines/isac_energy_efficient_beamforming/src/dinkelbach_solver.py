"""
Dinkelbach Solver for Fractional Programming
=============================================

Implements Dinkelbach's method for communication EE maximization (Algorithm 1).

The communication EE is a fractional program:
    max EE_C = f₁(W) / f₂(W)
    where f₁(W) = Σ_k log₂(1+SINR_k) (sum rate)
          f₂(W) = (1/ε)Σ_k||w_k||² + P₀ (power consumption)

Dinkelbach's method converts this to a parametric subtractive form:
    max f₁(W) - λ f₂(W)
    with λ updated as λ_{n+1} = f₁(W_n) / f₂(W_n)

The full Algorithm 1 (Section III) combines:
1. Dinkelbach method for fractional programming
2. Quadratic transform for log-SINR
3. SDR for non-convex beamforming
4. SCA for rank-1 recovery

Reference: Zou et al., IEEE Trans. Commun., 2024 (Algorithm 1)
"""

import numpy as np
from typing import Optional, Tuple, List, NamedTuple
import cvxpy as cp


class DinkelbachResult(NamedTuple):
    """Result from Dinkelbach solver."""
    W: np.ndarray           # Optimal beamforming matrix (M x K)
    ee_c: float             # Communication EE (bits/Hz/J)
    sum_rate: float         # Sum rate (bits/Hz)
    total_power: float      # Total transmit power
    n_iterations: int       # Number of Dinkelbach iterations
    converged: bool         # Whether converged
    obj_history: List[float]  # EE_C at each iteration


class DinkelbachSolver:
    """
    Dinkelbach solver for communication EE maximization (Algorithm 1).

    Combines:
    - Dinkelbach method (fractional → subtractive)
    - Quadratic transform (log-SINR handling)
    - SDR (beamforming relaxation)
    - SCA (rank-1 recovery)
    """

    def __init__(
        self,
        model: "ISACSystemModel",
        max_dinkelbach_iter: int = 30,
        max_inner_iter: int = 20,
        tol_outer: float = 1e-4,
        tol_inner: float = 1e-5,
        solver: str = "MOSEK",
        verbose: bool = False,
    ):
        """
        Initialize Dinkelbach solver.

        Parameters
        ----------
        model : ISACSystemModel
            System model with channels, parameters
        max_dinkelbach_iter : int
            Maximum Dinkelbach iterations (default: 30)
        max_inner_iter : int
            Maximum inner iterations for SCA (default: 20)
        tol_outer : float
            Outer loop convergence tolerance
        tol_inner : float
            Inner loop convergence tolerance
        solver : str
            CVXPY solver name
        verbose : bool
            Print iteration details
        """
        self.model = model
        self.M = model.M
        self.K = model.K
        self.max_dinkelbach_iter = max_dinkelbach_iter
        self.max_inner_iter = max_inner_iter
        self.tol_outer = tol_outer
        self.tol_inner = tol_inner
        self.solver = solver
        self.verbose = verbose

    def solve(
        self,
        target_angle_deg: float = 90.0,
        crb_max: Optional[float] = None,
        gamma_min: Optional[float] = None,
        W_init: Optional[np.ndarray] = None,
    ) -> DinkelbachResult:
        """
        Solve communication EE maximization (Algorithm 1).

        Steps:
        1. Initialize λ = 0, W
        2. Repeat:
           a. Solve: max f₁(W) - λ f₂(W) via SDR + SCA
           b. Update: λ = f₁(W) / f₂(W)
           c. Check: |f₁(W) - λ f₂(W)| < ε
        3. Return optimal W

        Parameters
        ----------
        target_angle_deg : float
            Target angle in degrees (default: 90°)
        crb_max : float, optional
            Maximum CRB constraint
        gamma_min : float, optional
            Minimum SINR per user
        W_init : np.ndarray, optional
            Initial beamforming matrix (M x K)

        Returns
        -------
        DinkelbachResult
            Optimization result
        """
        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        H = self.model.get_csi()
        sigma_c2 = self.model.sigma_c2
        sigma_s2 = self.model.sigma_s2
        epsilon = self.model.epsilon
        P0 = self.model.P0
        P_max = self.model.P_max
        L = self.model.L

        # Initialize beamforming
        if W_init is None:
            W_init = self._initialize_beamforming(H, a_t, P_max)

        W_current = W_init.copy()
        lambda_param = 0.0
        obj_history = []
        converged = False

        for dinkelbach_iter in range(self.max_dinkelbach_iter):
            # Compute current metrics
            sum_rate = self._compute_sum_rate(H, W_current, sigma_c2)
            total_power = self._compute_total_power(W_current)
            total_consumption = (1 / epsilon) * total_power + P0
            ee_c = sum_rate / total_consumption if total_consumption > 0 else 0

            obj_history.append(ee_c)

            if self.verbose:
                print(
                    f"Dinkelbach iter {dinkelbach_iter}: "
                    f"EE_C = {ee_c:.6f}, λ = {lambda_param:.6f}"
                )

            # Check convergence
            subtractive_obj = sum_rate - lambda_param * total_consumption
            if abs(subtractive_obj) < self.tol_outer * (1 + abs(sum_rate)):
                converged = True
                if self.verbose:
                    print(f"Dinkelbach converged at iteration {dinkelbach_iter}")
                break

            # Solve inner optimization: max f₁(W) - λ f₂(W)
            W_next, inner_status = self._solve_inner(
                H, W_current, a_t, a_r, sigma_c2, sigma_s2,
                epsilon, P0, P_max, L, lambda_param,
                crb_max, gamma_min,
            )

            if inner_status not in ["optimal", "optimal_inaccurate"]:
                if self.verbose:
                    print(f"Inner solver failed: {inner_status}")
                break

            W_current = W_next

            # Update λ
            sum_rate_new = self._compute_sum_rate(H, W_current, sigma_c2)
            total_power_new = self._compute_total_power(W_current)
            total_consumption_new = (1 / epsilon) * total_power_new + P0

            if total_consumption_new > 1e-15:
                lambda_param = sum_rate_new / total_consumption_new
            else:
                lambda_param = 0.0

        # Final metrics
        final_sum_rate = self._compute_sum_rate(H, W_current, sigma_c2)
        final_total_power = self._compute_total_power(W_current)
        final_consumption = (1 / epsilon) * final_total_power + P0
        final_ee_c = final_sum_rate / final_consumption if final_consumption > 0 else 0

        return DinkelbachResult(
            W=W_current,
            ee_c=final_ee_c,
            sum_rate=final_sum_rate,
            total_power=final_total_power,
            n_iterations=dinkelbach_iter + 1,
            converged=converged,
            obj_history=obj_history,
        )

    def _initialize_beamforming(
        self,
        H: np.ndarray,
        a_t: np.ndarray,
        P_max: float,
    ) -> np.ndarray:
        """
        Initialize beamforming matrix.

        Uses matched filter initialization: w_k ∝ h_k for communication,
        with a fraction of power allocated to sensing direction.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix (K x M)
        a_t : np.ndarray
            Transmit steering vector (M,)
        P_max : float
            Maximum power

        Returns
        -------
        np.ndarray
            Initial beamforming matrix (M x K)
        """
        M, K = self.M, self.K
        W = np.zeros((M, K), dtype=complex)

        # Allocate power: 50% communication, 50% sensing
        p_comm = 0.5 * P_max / K
        p_sense = 0.5 * P_max / K

        for k in range(K):
            h_k = H[k, :]
            # Communication component (matched filter)
            w_comm = h_k / (np.linalg.norm(h_k) + 1e-15) * np.sqrt(p_comm)
            # Sensing component (beam toward target)
            w_sense = a_t / (np.linalg.norm(a_t) + 1e-15) * np.sqrt(p_sense / K)
            W[:, k] = w_comm + w_sense

        return W

    def _solve_inner(
        self,
        H: np.ndarray,
        W_lin: np.ndarray,
        a_t: np.ndarray,
        a_r: np.ndarray,
        sigma_c2: float,
        sigma_s2: float,
        epsilon: float,
        P0: float,
        P_max: float,
        L: int,
        lambda_param: float,
        crb_max: Optional[float],
        gamma_min: Optional[float],
    ) -> Tuple[np.ndarray, str]:
        """
        Solve inner optimization: max f₁(W) - λ f₂(W).

        Uses SDR + quadratic transform + SCA.

        Parameters
        ----------
        (Various system parameters)
        lambda_param : float
            Dinkelbach parameter λ

        Returns
        -------
        W_opt : np.ndarray
            Optimized beamforming matrix
        status : str
            Solver status
        """
        K, M = H.shape

        # PSD matrix variables for SDR
        W_psd = [cp.Variable((M, M), hermitian=True) for _ in range(K)]

        constraints = []

        # PSD constraints
        for k in range(K):
            constraints.append(W_psd[k] >> 0)

        # Power constraint
        total_power_expr = sum(cp.real(cp.trace(W_psd[k])) for k in range(K))
        constraints.append(total_power_expr <= P_max)

        # SINR constraints (if specified)
        if gamma_min is not None:
            for k in range(K):
                h_k = H[k, :]
                signal = cp.real(cp.quad_form(h_k, W_psd[k]))
                interference = sum(
                    cp.real(cp.quad_form(h_k, W_psd[j]))
                    for j in range(K)
                    if j != k
                )
                constraints.append(signal >= gamma_min * (sigma_c2 + interference))

        # CRB constraint (if specified)
        if crb_max is not None:
            Rx = sum(W_psd)
            threshold = sigma_s2 * np.sum(np.abs(a_r) ** 2) / (2 * L * crb_max)

            # Linearized CRB constraint using SCA
            Rx_lin = W_lin @ W_lin.conj().T
            theta = np.pi / 2
            m_indices = np.arange(M)
            da_t = 1j * 2 * np.pi * 0.5 * m_indices * np.cos(theta) * a_t

            # First-order approximation
            fim_lin = np.real(da_t.conj() @ Rx_lin @ da_t)
            if fim_lin > 1e-15:
                grad_Rx = np.outer(da_t, da_t.conj())
                fim_approx = fim_lin + cp.real(
                    cp.trace(grad_Rx @ (Rx - Rx_lin))
                )
                constraints.append(fim_approx >= threshold)

        # Quadratic transform objective for sum rate
        # Use linearized approximation around W_lin
        sum_rate_approx = 0
        for k in range(K):
            h_k = H[k, :]
            hw_k_lin = h_k.conj() @ W_lin[:, k]
            total_power_k_lin = sigma_c2 + sum(
                np.abs(h_k.conj() @ W_lin[:, j]) ** 2 for j in range(K)
            )

            # Optimal t_k
            if total_power_k_lin > 1e-15:
                t_k = hw_k_lin / total_power_k_lin
            else:
                t_k = 0.0

            # Quadratic transform terms
            signal_term = 2 * cp.real(
                np.conj(t_k) * cp.quad_form(h_k, W_psd[k])
            )

            # Approximation of |t_k|² term (first-order)
            t_k_sq = np.abs(t_k) ** 2
            power_terms = sum(
                cp.real(cp.quad_form(h_k, W_psd[j])) for j in range(K)
            )
            power_term = t_k_sq * (sigma_c2 + power_terms)

            sum_rate_approx += signal_term - power_term

        # Power consumption term: (1/ε)tr(Σ_k W_k) + P₀
        power_consumption = (1 / epsilon) * total_power_expr + P0

        # Dinkelbach objective: f₁ - λ f₂
        objective = cp.Maximize(sum_rate_approx - lambda_param * power_consumption)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            status = prob.status
        except (cp.error.SolverError, Exception):
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
                status = prob.status
            except Exception:
                return W_lin, "failed"

        if prob.status in ["optimal", "optimal_inaccurate"]:
            # Rank-1 recovery via eigenvalue decomposition
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

    def _compute_sum_rate(
        self, H: np.ndarray, W: np.ndarray, sigma_c2: float
    ) -> float:
        """Compute sum rate Σ_k log₂(1 + SINR_k)."""
        K = H.shape[0]
        sum_rate = 0.0
        for k in range(K):
            h_k = H[k, :]
            signal = np.abs(h_k.conj() @ W[:, k]) ** 2
            interference = sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K) if j != k
            )
            sinr_k = signal / (sigma_c2 + interference)
            sum_rate += np.log2(1 + sinr_k)
        return sum_rate

    def _compute_total_power(self, W: np.ndarray) -> float:
        """Compute total transmit power."""
        return float(np.sum(np.abs(W) ** 2))
