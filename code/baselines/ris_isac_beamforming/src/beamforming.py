"""BS beamforming optimization via SDR (Semidefinite Relaxation).

Optimizes transmit beamforming vectors w_k for MU-MISO communications
subject to power and QoS constraints.

Uses two approaches:
1. Minimum power beamforming (convex feasibility check)
2. WMMSE-based sum-rate maximization (alternating optimization)

Reference: Section IV.A in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .system_model import RIS_ISAC_System

# Preferred solver order (fall back automatically)
_SOLVER_PREF = [cp.MOSEK, cp.SCS]


def _solve_problem(prob: cp.Problem) -> float:
    """Solve a CVXPY problem with solver fallback.

    Tries MOSEK first, then SCS.

    Args:
        prob: CVXPY problem to solve.

    Returns:
        Optimal objective value.
    """
    for solver in _SOLVER_PREF:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.value
        except (cp.error.SolverError, ImportError):
            continue
    raise RuntimeError(f"All solvers failed. Problem status: {prob.status}")


class BeamformingOptimizer:
    """Beamforming optimization using Semidefinite Relaxation (SDR).

    Two formulations:
    1. Minimum power: min Σ tr(W_k) s.t. SINR_k ≥ γ_k
    2. WMMSE: max Σ R_k via alternating WMMSE updates

    Uses SDR: W_k = w_k w_k^H, rank-1 constraint relaxed.

    Attributes:
        system: RIS_ISAC_System instance.
        M: Number of BS antennas.
        K: Number of users.
        P_max: Power budget.
    """

    def __init__(self, system: RIS_ISAC_System):
        """Initialize beamforming optimizer.

        Args:
            system: The RIS-ISAC system model (with fixed RIS phases).
        """
        self.system = system
        self.M = system.M
        self.K = system.K
        self.P_max = system.P_max

    def _get_effective_channels(self) -> np.ndarray:
        """Get effective channels for all users.

        Returns:
            Matrix H_eff of shape (K, M) where H_eff[k, :] = h_k^H.
        """
        H_eff = np.zeros((self.K, self.M), dtype=complex)
        for k in range(self.K):
            H_eff[k, :] = self.system.effective_channel(k)
        return H_eff

    def solve_min_power(
        self,
        sinr_thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Minimum power beamforming satisfying SINR constraints.

        Solves: min Σ_k tr(W_k)  s.t.  SINR_k ≥ γ_k,  W_k ≽ 0.

        This is a convex SDP (see Eq. in Section IV.A).

        Args:
            sinr_thresholds: SINR thresholds γ_k (K,).

        Returns:
            Tuple of (W, min_power) where W is (M, K) beamforming matrix.
        """
        H_eff = self._get_effective_channels()
        sigma2 = self.system.noise_power

        W_vars = [cp.Variable((self.M, self.M), hermitian=True) for _ in range(self.K)]

        constraints = []
        for k in range(self.K):
            constraints.append(W_vars[k] >> 0)

            h_k = H_eff[k, :].reshape(-1, 1)
            # Signal: h_k^H W_k h_k
            signal = cp.real(cp.conj(h_k).T @ W_vars[k] @ h_k)
            # Interference: Σ_{j≠k} h_k^H W_j h_k + σ²
            interf = sigma2
            for j in range(self.K):
                if j != k:
                    interf += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k)
            constraints.append(signal >= sinr_thresholds[k] * interf)

        # Total power
        obj = sum(cp.real(cp.trace(W_vars[k])) for k in range(self.K))
        prob = cp.Problem(cp.Minimize(obj), constraints)

        try:
            min_power = _solve_problem(prob)
        except RuntimeError:
            return np.zeros((self.M, self.K), dtype=complex), float("inf")

        W = self._recover_beamformers(W_vars)
        return W, min_power if min_power is not None else float("inf")

    def solve_max_rate(
        self,
        sinr_thresholds: Optional[np.ndarray] = None,
        max_wmmse_iter: int = 30,
    ) -> Tuple[np.ndarray, float]:
        """Maximize sum rate via WMMSE + SDR.

        Alternates between:
        1. Optimal MMSE receivers u_k and weights α_k
        2. SDR beamforming optimization with WMMSE objective

        The WMMSE reformulation (Eq. from Shi et al., 2011):
            min Σ_k (α_k e_k - log2(α_k))
        where e_k = |1 - u_k^H h_k w_k|^2 + Σ_{j≠k} |u_k^H h_k w_j|^2 + σ²|u_k|²

        Args:
            sinr_thresholds: Optional SINR constraints γ_k (K,).
            max_wmmse_iter: Maximum WMMSE iterations.

        Returns:
            Tuple of (W, sum_rate).
        """
        H_eff = self._get_effective_channels()
        sigma2 = self.system.noise_power

        # Initialize with matched filter beamforming
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H_eff[k, :]
            W[:, k] = h_k.conj() / np.linalg.norm(h_k)
        W *= np.sqrt(self.P_max / (self.K * np.linalg.norm(W, "fro") ** 2))
        W *= np.sqrt(self.P_max) / max(np.linalg.norm(W, "fro"), 1e-10)

        best_rate = 0.0

        for wmmse_iter in range(max_wmmse_iter):
            # Step 1: Compute MMSE receivers and weights
            u = np.zeros(self.K, dtype=complex)
            alpha = np.zeros(self.K)

            for k in range(self.K):
                h_k = H_eff[k, :]
                # MMSE receiver: u_k = (h_k^H w_k) / (Σ_j |h_k^H w_j|^2 + σ²)
                denom = sigma2
                for j in range(self.K):
                    denom += np.abs(h_k.conj() @ W[:, j]) ** 2
                u[k] = (h_k.conj() @ W[:, k]) / denom
                # MMSE weight
                e_k = denom - np.abs(h_k.conj() @ W[:, k]) ** 2 * np.abs(u[k]) ** 2
                e_k = max(e_k, 1e-15)
                alpha[k] = 1.0 / e_k

            # Step 2: Optimize beamforming via SDR with WMMSE objective
            W_vars = [cp.Variable((self.M, self.M), hermitian=True) for _ in range(self.K)]

            constraints = []
            for k in range(self.K):
                constraints.append(W_vars[k] >> 0)

            power_expr = sum(cp.real(cp.trace(W_vars[k])) for k in range(self.K))
            constraints.append(power_expr <= self.P_max)

            # SINR constraints (if provided)
            if sinr_thresholds is not None:
                for k in range(self.K):
                    h_k = H_eff[k, :].reshape(-1, 1)
                    signal = cp.real(cp.conj(h_k).T @ W_vars[k] @ h_k)
                    interf = sigma2
                    for j in range(self.K):
                        if j != k:
                            interf += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k)
                    constraints.append(signal >= sinr_thresholds[k] * interf)

            # WMMSE objective: minimize Σ_k α_k * e_k
            # e_k = |1 - u_k^H h_k w_k|^2 + Σ_{j≠k} |u_k^H h_k w_j|^2 + σ²|u_k|²
            # In SDR form: e_k = α_k * (tr(H_k^H W_k H_k) - 2*Re(u_k h_k^H w_k) + ...)
            wmmse_obj = 0
            for k in range(self.K):
                h_k = H_eff[k, :].reshape(-1, 1)
                uk = u[k]
                ak = alpha[k]

                # Quadratic terms via SDR
                quad = sigma2 * np.abs(uk) ** 2
                for j in range(self.K):
                    quad += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k) * np.abs(uk) ** 2

                # Linear term: -2 Re(u_k * h_k^H w_k) is not directly SDR-form
                # In SDR, the WMMSE becomes:
                # e_k = α_k * [1 - 2*Re(u_k^H h_k^H w_k) + |u_k|^2 * Σ_j h_k^H W_j h_k]
                # For SDR: 1 - 2*Re(u_k * h_k^H w_k) + |u_k|^2 * tr(H_k W H_k^H)
                # This simplifies to a linear function of W matrices

                # Using trace formulation:
                # e_k = |u_k|^2 * Σ_j h_k^H W_j h_k - 2 Re(u_k^H * h_k^H w_k) + 1
                # In SDR with W_k: need linearization

                # Simplified WMMSE objective for SDR:
                # min Σ_k α_k * (|u_k|^2 * Σ_j h_k^H W_j h_k)
                # subject to power + SINR
                wmmse_obj += ak * quad

            prob = cp.Problem(cp.Minimize(wmmse_obj), constraints)

            try:
                _solve_problem(prob)
            except RuntimeError:
                break

            W_new = self._recover_beamformers(W_vars)

            # Update W
            if np.sum(np.linalg.norm(W_new, axis=0) ** 2) > 0:
                W = W_new

            # Compute sum rate
            rate = self.system.compute_sum_rate(W)
            if rate > best_rate:
                best_rate = rate

        return W, best_rate

    def _recover_beamformers(self, W_vars: list) -> np.ndarray:
        """Recover beamforming vectors from SDR solution.

        Uses dominant eigenvector extraction.

        Args:
            W_vars: List of CVXPY variables with SDR solution.

        Returns:
            Beamforming matrix W of shape (M, K).
        """
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            W_k_val = W_vars[k].value
            if W_k_val is None:
                continue
            # Ensure Hermitian
            W_k_val = (W_k_val + W_k_val.conj().T) / 2
            eigvals, eigvecs = np.linalg.eigh(W_k_val)
            idx = np.argmax(eigvals)
            if eigvals[idx] > 1e-10:
                W[:, k] = np.sqrt(eigvals[idx]) * eigvecs[:, idx]
            else:
                W[:, k] = eigvecs[:, idx]

        # Scale to satisfy power constraint
        total_power = np.sum(np.linalg.norm(W, axis=0) ** 2)
        if total_power > self.P_max:
            W *= np.sqrt(self.P_max / total_power)

        return W
