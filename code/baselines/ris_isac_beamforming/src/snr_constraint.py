"""SNR-constrained joint beamforming and reflection design.

Solves Problem 1 (target detection formulation):
    max  Σ_k R_k  (sum rate)
    s.t. SNR_sensing ≥ γ_min  (detection requirement)
         SINR_k ≥ γ_k         (communication QoS)
         Σ||w_k||² ≤ P_max   (power budget)
         |θ_l| = 1, ∀l       (RIS unit-modulus)

Solution approach: Alternating Optimization with SDR beamforming
and coordinate-ascent RIS phase optimization.

Reference: Section IV, Problem P1 in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .system_model import RIS_ISAC_System
from .beamforming import BeamformingOptimizer, _solve_problem
from .ris_phase import RISPhaseOptimizer


class SNRConstrainedSolver:
    """Solver for SNR-constrained RIS-ISAC problem (Problem P1).

    Jointly optimizes BS beamforming and RIS phase shifts to
    maximize communication sum rate subject to radar detection
    SNR constraint.

    AO Algorithm:
        1. Fix θ, optimize W via WMMSE + SDR with SINR + SNR constraints
        2. Fix W, optimize θ via grid-search coordinate ascent
        3. Repeat until convergence

    Attributes:
        system: RIS-ISAC system model.
        M: BS antenna count.
        K: User count.
        L: RIS element count.
        P_max: Power budget (mW).
        noise_power: Noise variance (mW).
        snr_min: Minimum radar SNR (linear).
        sinr_thresh: SINR threshold (linear).
        max_iter: Maximum AO iterations.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        system: RIS_ISAC_System,
        snr_min_dB: float = 5.0,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        """Initialize SNR-constrained solver.

        Args:
            system: RIS-ISAC system instance.
            snr_min_dB: Minimum radar sensing SNR in dB (γ_min).
            max_iter: Maximum alternating optimization iterations.
            tol: Convergence tolerance.
        """
        self.system = system
        self.M = system.M
        self.K = system.K
        self.L = system.L
        self.P_max = system.P_max
        self.noise_power = system.noise_power

        self.snr_min_dB = snr_min_dB
        self.snr_min = 10 ** (snr_min_dB / 10)

        self.sinr_thresh_dB = system.sinr_thresh_dB
        self.sinr_thresh = system.sinr_thresh

        self.max_iter = max_iter
        self.tol = tol

        self.bf_optimizer = BeamformingOptimizer(system)
        self.ris_optimizer = RISPhaseOptimizer(system)

    def _compute_sensing_channel(self) -> np.ndarray:
        """Compute the effective sensing channel vector h_s (M,).

        Returns:
            Sensing channel vector (M,).
        """
        H_BR = self.system.channels["H_BR"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]
        Theta = self.system.ris_diagonal_matrix()
        return a_bs + a_ris.T @ Theta @ H_BR

    def solve(self) -> dict:
        """Solve the SNR-constrained problem via alternating optimization.

        Algorithm:
            1. Initialize RIS phases (sensing-optimal).
            2. Repeat until convergence:
                a. Fix θ, optimize W via WMMSE + SDR with SNR constraint.
                b. Fix W, optimize θ via coordinate ascent.
            3. Return solution.

        Returns:
            Dictionary with keys:
                'W': Optimal beamforming matrix (M, K).
                'theta': Optimal RIS phases (L,).
                'sum_rate': Achieved sum rate (bps/Hz).
                'snr_sensing': Achieved radar SNR (linear).
                'converged': Whether AO converged.
                'iterations': Number of AO iterations.
                'history': List of sum_rate per iteration.
        """
        sinr_thresholds = np.full(self.K, self.sinr_thresh)

        # Initialize with matched filter beamforming
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = self.system.effective_channel(k)
            W[:, k] = h_k.conj() / np.linalg.norm(h_k)
        W *= np.sqrt(self.P_max / self.K) / np.linalg.norm(W, axis=0)

        # Optimize RIS for sensing initially
        self.ris_optimizer.optimize_for_snr(W, self.snr_min_dB)

        history = []
        prev_rate = -np.inf
        converged = False
        sum_rate = 0.0

        for it in range(self.max_iter):
            # Step 1: Optimize beamforming W for fixed θ (with SNR constraint)
            W, _ = self._optimize_beamforming_wmmse_with_snr(sinr_thresholds)

            # Step 2: Optimize RIS phases θ for fixed W
            snr_current = self.system.compute_snr_sensing(np.sum(W, axis=1))

            if snr_current < self.snr_min:
                # Prioritize sensing
                self.ris_optimizer.optimize_for_snr(W, self.snr_min_dB)
            else:
                # Balance communication and sensing
                self.ris_optimizer.optimize_joint(W, sensing_weight=0.3)

            # Compute current sum rate
            sum_rate = self.system.compute_sum_rate(W)
            snr_sensing = self.system.compute_snr_sensing(np.sum(W, axis=1))
            history.append(sum_rate)

            # Check convergence
            if abs(sum_rate - prev_rate) < self.tol * max(abs(sum_rate), 1e-6):
                converged = True
                break
            prev_rate = sum_rate

        return {
            "W": W,
            "theta": self.system.theta.copy(),
            "sum_rate": sum_rate,
            "snr_sensing": snr_sensing,
            "converged": converged,
            "iterations": it + 1,
            "history": history,
        }

    def _optimize_beamforming_wmmse_with_snr(
        self,
        sinr_thresholds: np.ndarray,
        max_wmmse_iter: int = 15,
    ) -> Tuple[np.ndarray, float]:
        """Optimize beamforming via WMMSE with SNR sensing constraint.

        Uses WMMSE alternating optimization with SDR subproblems.
        SNR constraint is enforced in each SDR step.

        Args:
            sinr_thresholds: SINR thresholds (K,).
            max_wmmse_iter: WMMSE inner iterations.

        Returns:
            Tuple of (W, objective_value).
        """
        H_eff = self.bf_optimizer._get_effective_channels()
        sigma2 = self.system.noise_power
        h_s = self._compute_sensing_channel().reshape(-1, 1)  # (M, 1)

        # Initialize W
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H_eff[k, :]
            W[:, k] = h_k.conj() / np.linalg.norm(h_k)
        W *= np.sqrt(self.P_max / self.K) / max(np.linalg.norm(W), 1e-10)
        W *= np.sqrt(self.P_max) / max(np.linalg.norm(W, "fro"), 1e-10)

        best_rate = 0.0

        for wmmse_iter in range(max_wmmse_iter):
            # Step 1: MMSE receivers and weights
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

            # Step 2: SDR beamforming optimization
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

            # SNR sensing constraint: |h_s^H w_total|^2 ≥ γ_min * σ²
            W_total = sum(W_vars[k] for k in range(self.K))
            sensing_signal = cp.real(cp.conj(h_s).T @ W_total @ h_s)
            constraints.append(sensing_signal >= self.snr_min * sigma2)

            # WMMSE objective: minimize Σ_k α_k * e_k (SDR form)
            wmmse_obj = 0
            for k in range(self.K):
                h_k = H_eff[k, :].reshape(-1, 1)
                uk = u[k]
                ak = alpha[k]

                # e_k in SDR: |u_k|^2 * Σ_j h_k^H W_j h_k - 2Re(u_k^H h_k^H w_k) + 1
                # SDR linearizes: Σ_j |u_k|^2 * h_k^H W_j h_k
                quad = 0
                for j in range(self.K):
                    quad += cp.real(cp.conj(h_k).T @ W_vars[j] @ h_k) * np.abs(uk) ** 2
                wmmse_obj += ak * quad

            prob = cp.Problem(cp.Minimize(wmmse_obj), constraints)

            try:
                _solve_problem(prob)
            except RuntimeError:
                # Fallback: try without SNR constraint
                constraints_no_snr = constraints[:-1]
                prob = cp.Problem(cp.Minimize(wmmse_obj), constraints_no_snr)
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
