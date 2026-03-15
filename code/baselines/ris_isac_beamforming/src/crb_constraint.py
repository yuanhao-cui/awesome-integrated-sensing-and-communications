"""CRB-constrained joint beamforming and reflection design.

Solves Problem 2 (parameter estimation formulation):
    max  Σ_k R_k  (sum rate)
    s.t. CRB(θ) ≤ ε_max  (estimation accuracy)
         SINR_k ≥ γ_k    (communication QoS)
         Σ||w_k||² ≤ P_max (power budget)
         |θ_l| = 1, ∀l   (RIS unit-modulus)

Reference: Section V, Problem P2 in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .system_model import RIS_ISAC_System
from .beamforming import BeamformingOptimizer, _solve_problem
from .ris_phase import RISPhaseOptimizer


class CRBConstrainedSolver:
    """Solver for CRB-constrained RIS-ISAC problem (Problem P2).

    The CRB for angle estimation:
        CRB(φ) = σ² / (2 * |d_s^H w_total|^2)

    where d_s is the derivative of the sensing channel w.r.t. φ.

    Attributes:
        system: RIS-ISAC system model.
        crb_max: Maximum allowed CRB (ε_max).
        sinr_thresh: SINR threshold (linear).
    """

    def __init__(
        self,
        system: RIS_ISAC_System,
        crb_max: float = 1e-2,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        """Initialize CRB-constrained solver.

        Args:
            system: RIS-ISAC system instance.
            crb_max: Maximum CRB (ε_max) for estimation accuracy.
            max_iter: Maximum AO iterations.
            tol: Convergence tolerance.
        """
        self.system = system
        self.M = system.M
        self.K = system.K
        self.L = system.L
        self.P_max = system.P_max
        self.noise_power = system.noise_power

        self.crb_max = crb_max
        self.sinr_thresh = system.sinr_thresh

        self.max_iter = max_iter
        self.tol = tol

        self.bf_optimizer = BeamformingOptimizer(system)
        self.ris_optimizer = RISPhaseOptimizer(system)

        # Target angle (for CRB computation)
        self.target_angle = 0.0  # rad

    def compute_crb(self, w_total: np.ndarray) -> float:
        """Compute the CRB for target angle estimation.

        CRB(φ) = σ² / (2 * |d_s^H w_total|^2)

        Args:
            w_total: Total beamforming vector (M,).

        Returns:
            CRB value.
        """
        d_s = self._compute_sensing_derivative()
        fisher_info = np.abs(d_s.conj() @ w_total) ** 2
        crb = self.noise_power / (2 * fisher_info + 1e-20)
        return crb

    def _compute_sensing_channel(self) -> np.ndarray:
        """Compute effective sensing channel h_s (M,)."""
        H_BR = self.system.channels["H_BR"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]
        Theta = self.system.ris_diagonal_matrix()
        return a_bs + a_ris.T @ Theta @ H_BR

    def _compute_sensing_derivative(self) -> np.ndarray:
        """Compute derivative of sensing channel w.r.t. target angle."""
        H_BR = self.system.channels["H_BR"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]
        Theta = self.system.ris_diagonal_matrix()
        phi = self.target_angle

        d_a_bs = 1j * np.pi * np.cos(phi) * np.arange(self.M) * a_bs
        d_a_ris = 1j * np.pi * np.cos(phi) * np.arange(self.L) * a_ris
        return d_a_bs + d_a_ris.T @ Theta @ H_BR

    def solve(self) -> dict:
        """Solve the CRB-constrained problem via alternating optimization.

        Returns:
            Dictionary with keys:
                'W': Optimal beamforming matrix (M, K).
                'theta': Optimal RIS phases (L,).
                'sum_rate': Achieved sum rate (bps/Hz).
                'crb': Achieved CRB.
                'converged': Whether AO converged.
                'iterations': Number of AO iterations.
                'history': List of sum_rate per iteration.
        """
        sinr_thresholds = np.full(self.K, self.sinr_thresh)

        # Initialize
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = self.system.effective_channel(k)
            W[:, k] = h_k.conj() / np.linalg.norm(h_k)
        W *= np.sqrt(self.P_max / self.K) / max(np.linalg.norm(W, "fro"), 1e-10)

        history = []
        prev_rate = -np.inf
        converged = False
        sum_rate = 0.0

        for it in range(self.max_iter):
            # Step 1: Optimize W for fixed θ (WMMSE + CRB constraint)
            W, _ = self._optimize_beamforming_wmmse_with_crb(sinr_thresholds)

            # Step 2: Optimize θ for fixed W (minimize CRB)
            self._optimize_ris_for_crb(W)

            # Evaluate
            sum_rate = self.system.compute_sum_rate(W)
            history.append(sum_rate)

            if abs(sum_rate - prev_rate) < self.tol * max(abs(sum_rate), 1e-6):
                converged = True
                break
            prev_rate = sum_rate

        w_total = np.sum(W, axis=1)
        crb_final = self.compute_crb(w_total)

        return {
            "W": W,
            "theta": self.system.theta.copy(),
            "sum_rate": sum_rate,
            "crb": crb_final,
            "converged": converged,
            "iterations": it + 1,
            "history": history,
        }

    def _optimize_beamforming_wmmse_with_crb(
        self,
        sinr_thresholds: np.ndarray,
        max_wmmse_iter: int = 15,
    ) -> Tuple[np.ndarray, float]:
        """Optimize beamforming via WMMSE with CRB constraint.

        Args:
            sinr_thresholds: SINR thresholds (K,).
            max_wmmse_iter: WMMSE inner iterations.

        Returns:
            Tuple of (W, objective_value).
        """
        H_eff = self.bf_optimizer._get_effective_channels()
        sigma2 = self.system.noise_power
        d_s = self._compute_sensing_derivative().reshape(-1, 1)  # (M, 1)

        # Initialize
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H_eff[k, :]
            W[:, k] = h_k.conj() / np.linalg.norm(h_k)
        W *= np.sqrt(self.P_max / self.K) / max(np.linalg.norm(W, "fro"), 1e-10)
        W *= np.sqrt(self.P_max) / max(np.linalg.norm(W, "fro"), 1e-10)

        best_rate = 0.0

        for wmmse_iter in range(max_wmmse_iter):
            # MMSE receivers
            u = np.zeros(self.K, dtype=complex)
            alpha = np.zeros(self.K)

            for k in range(self.K):
                h_k = H_eff[k, :]
                denom = sigma2
                for j in range(self.K):
                    denom += np.abs(h_k.conj() @ W[:, j]) ** 2
                u[k] = (h_k.conj() @ W[:, k]) / denom
                e_k = max(denom - np.abs(h_k.conj() @ W[:, k]) ** 2 * np.abs(u[k]) ** 2, 1e-15)
                alpha[k] = 1.0 / e_k

            # SDR optimization
            W_vars = [cp.Variable((self.M, self.M), hermitian=True) for _ in range(self.K)]

            constraints = []
            for k in range(self.K):
                constraints.append(W_vars[k] >> 0)

            power_expr = sum(cp.real(cp.trace(W_vars[k])) for k in range(self.K))
            constraints.append(power_expr <= self.P_max)

            # SINR constraints
            for k in range(self.K):
                h_k = H_eff[k, :].reshape(-1, 1)
                signal = cp.real(cp.conj(h_k).T @ W_vars[k] @ h_k)
                interf = sigma2
                for j in range(self.K):
                    if j != k:
                        interf += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k)
                constraints.append(signal >= sinr_thresholds[k] * interf)

            # CRB constraint: |d_s^H w_total|^2 ≥ σ² / (2 * ε_max)
            W_total = sum(W_vars[k] for k in range(self.K))
            crb_signal = cp.real(cp.conj(d_s).T @ W_total @ d_s)
            constraints.append(crb_signal >= sigma2 / (2 * self.crb_max))

            # WMMSE objective
            wmmse_obj = 0
            for k in range(self.K):
                h_k = H_eff[k, :].reshape(-1, 1)
                uk = u[k]
                ak = alpha[k]

                quad = 0
                for j in range(self.K):
                    quad += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k) * np.abs(uk) ** 2
                wmmse_obj += ak * quad

            prob = cp.Problem(cp.Minimize(wmmse_obj), constraints)

            try:
                _solve_problem(prob)
            except RuntimeError:
                # Fallback: try without CRB constraint
                constraints_no_crb = constraints[:-1]
                prob = cp.Problem(cp.Minimize(wmmse_obj), constraints_no_crb)
                try:
                    _solve_problem(prob)
                except RuntimeError:
                    break

            W_new = self.bf_optimizer._recover_beamformers(W_vars)
            if np.sum(np.linalg.norm(W_new, axis=0) ** 2) > 1e-10:
                W = W_new

            rate = self.system.compute_sum_rate(W)
            if rate > best_rate:
                best_rate = rate

        return W, best_rate

    def _optimize_ris_for_crb(self, W: np.ndarray):
        """Optimize RIS phases to minimize CRB for fixed beamforming.

        Maximizes |d_s^H w|^2 via grid search coordinate ascent.

        Args:
            W: Fixed beamforming matrix (M, K).
        """
        w_total = np.sum(W, axis=1)
        theta = self.system.theta.copy()

        def compute_fisher(th):
            self.system.theta = th
            d_s = self._compute_sensing_derivative()
            return np.abs(d_s.conj() @ w_total) ** 2

        current_fisher = compute_fisher(theta)

        for _ in range(30):
            improved = False
            for l in range(self.L):
                candidates = np.exp(
                    1j * np.linspace(0, 2 * np.pi, 16, endpoint=False)
                )
                best_fisher = current_fisher
                best_phase = theta[l]

                for phi_cand in candidates:
                    theta[l] = phi_cand
                    fisher = compute_fisher(theta)
                    if fisher > best_fisher:
                        best_fisher = fisher
                        best_phase = phi_cand
                        improved = True

                theta[l] = best_phase

            self.system.theta = theta
            current_fisher = best_fisher
            if not improved:
                break
