"""RIS phase shift optimization.

Optimizes the passive RIS reflection coefficients θ to maximize
communication and sensing performance.

Reference: Section IV.B, AO for RIS in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .system_model import RIS_ISAC_System

_SOLVER_PREF = [cp.MOSEK, cp.SCS]


def _solve_problem(prob: cp.Problem) -> float:
    """Solve CVXPY problem with solver fallback."""
    for solver in _SOLVER_PREF:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.value
        except (cp.error.SolverError, ImportError):
            continue
    raise RuntimeError(f"All solvers failed. Status: {prob.status}")


class RISPhaseOptimizer:
    """RIS phase shift optimization under unit-modulus constraint.

    For fixed beamforming W, optimizes θ to maximize sum rate
    or satisfy sensing constraints.

    Unit-modulus constraint: |θ_l| = 1, ∀l ∈ {1,...,L}.
    """

    def __init__(self, system: RIS_ISAC_System):
        """Initialize RIS phase optimizer.

        Args:
            system: The RIS-ISAC system model.
        """
        self.system = system
        self.M = system.M
        self.K = system.K
        self.L = system.L

    def optimize_for_rate(self, W: np.ndarray) -> np.ndarray:
        """Optimize RIS phases to maximize sum rate for fixed beamforming.

        Uses SCA (Successive Convex Approximation) to handle the
        non-convex unit-modulus constraint. Each θ_l is written as
        θ_l = e^{jφ_l} and φ_l is optimized.

        Simplified approach: coordinate ascent over phase angles.

        Args:
            W: Fixed beamforming matrix (M, K).

        Returns:
            Optimized RIS phase vector (L,) with |θ_l| = 1.
        """
        H_BR = self.system.channels["H_BR"]  # (L, M)
        G = self.system.channels["G"]  # (K, L)
        h_d = self.system.channels["h_d"]  # (K, M)
        sigma2 = self.system.noise_power

        # Coordinate ascent over RIS phases
        theta = self.system.theta.copy()
        current_rate = self.system.compute_sum_rate(W)

        for _ in range(50):  # inner iterations
            improved = False
            for l in range(self.L):
                # Grid search over phase for element l
                best_phase = theta[l]
                best_rate = current_rate

                # Coarse grid
                candidates = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
                for phi_cand in candidates:
                    theta[l] = phi_cand
                    self.system.theta = theta.copy()
                    rate = self.system.compute_sum_rate(W)
                    if rate > best_rate:
                        best_rate = rate
                        best_phase = phi_cand
                        improved = True

                theta[l] = best_phase

            self.system.theta = theta
            current_rate = best_rate
            if not improved:
                break

        return theta

    def optimize_for_snr(
        self, W: np.ndarray, target_snr_dB: float
    ) -> Tuple[np.ndarray, float]:
        """Optimize RIS phases to maximize radar sensing SNR.

        For fixed communication beamforming W, optimize θ to
        satisfy SNR_sensing ≥ γ_min.

        SNR_s = |a_s^H θ|^2 / σ² where a_s is a composite sensing vector.

        Args:
            W: Fixed beamforming matrix (M, K).
            target_snr_dB: Target sensing SNR in dB.

        Returns:
            Tuple of (theta, achieved_snr_linear).
        """
        H_BR = self.system.channels["H_BR"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]

        w_total = np.sum(W, axis=1)  # total beamforming (M,)

        # The sensing SNR through RIS:
        # h_s = a_bs + diag(a_ris) H_BR w / ||w|| (normalized)
        # SNR = |h_s^H w|^2 / σ²
        # Effective: a_s[l] = a_ris[l] * (H_BR[l,:] @ w_total)
        a_s = a_ris * (H_BR @ w_total)  # (L,) composite sensing vector
        a_s = a_s / (np.linalg.norm(a_s) + 1e-15)

        # Maximize |a_s^H θ|² subject to |θ_l| = 1
        # Optimal: θ_l = e^{j * arg(a_s[l])}
        theta = np.exp(1j * np.angle(a_s.conj()))

        self.system.theta = theta
        achieved_snr = self.system.compute_snr_sensing(w_total)

        return theta, achieved_snr

    def optimize_joint(
        self,
        W: np.ndarray,
        sensing_weight: float = 0.5,
    ) -> np.ndarray:
        """Joint optimization of RIS phases for both communication and sensing.

        Weighted objective: α * rate + (1-α) * sensing_snr.

        Uses grid-based coordinate ascent.

        Args:
            W: Fixed beamforming matrix (M, K).
            sensing_weight: Weight for sensing vs communication (0 to 1).

        Returns:
            Optimized RIS phase vector (L,).
        """
        H_BR = self.system.channels["H_BR"]
        G = self.system.channels["G"]
        h_d = self.system.channels["h_d"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]
        sigma2 = self.system.noise_power

        w_total = np.sum(W, axis=1)
        a_s = a_ris * (H_BR @ w_total)

        theta = self.system.theta.copy()

        def compute_objective(th):
            self.system.theta = th
            rate = self.system.compute_sum_rate(W)
            snr = self.system.compute_snr_sensing(w_total)
            return (1 - sensing_weight) * rate + sensing_weight * snr

        current_obj = compute_objective(theta)

        for _ in range(30):
            improved = False
            for l in range(self.L):
                candidates = np.exp(1j * np.linspace(0, 2 * np.pi, 12, endpoint=False))
                best_obj = current_obj
                best_phase = theta[l]

                for phi_cand in candidates:
                    theta[l] = phi_cand
                    obj = compute_objective(theta)
                    if obj > best_obj:
                        best_obj = obj
                        best_phase = phi_cand
                        improved = True

                theta[l] = best_phase

            self.system.theta = theta
            current_obj = best_obj
            if not improved:
                break

        return theta
