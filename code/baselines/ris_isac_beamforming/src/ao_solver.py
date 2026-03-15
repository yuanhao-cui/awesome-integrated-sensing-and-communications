"""Alternating Optimization (AO) solver for RIS-ISAC.

Unified interface that dispatches to SNR-constrained or CRB-constrained
solvers based on the problem type. Coordinates the AO loop between
beamforming and RIS phase optimization.

Reference: Section IV-V, Algorithm 1 in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
from typing import Optional, Literal

from .system_model import RIS_ISAC_System
from .beamforming import BeamformingOptimizer
from .ris_phase import RISPhaseOptimizer
from .snr_constraint import SNRConstrainedSolver
from .crb_constraint import CRBConstrainedSolver


class AlternatingOptimizationSolver:
    """Unified alternating optimization solver for RIS-ISAC.

    Supports two problem formulations:
    1. SNR-constrained (target detection)
    2. CRB-constrained (parameter estimation)

    Attributes:
        system: RIS-ISAC system model.
        problem_type: 'snr' or 'crb'.
        max_iter: Maximum AO iterations.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        system: RIS_ISAC_System,
        problem_type: Literal["snr", "crb"] = "snr",
        snr_min_dB: float = 5.0,
        crb_max: float = 1e-2,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        """Initialize AO solver.

        Args:
            system: RIS-ISAC system instance.
            problem_type: 'snr' for detection, 'crb' for estimation.
            snr_min_dB: Minimum radar SNR (dB) for SNR-constrained.
            crb_max: Maximum CRB for CRB-constrained.
            max_iter: Maximum AO iterations.
            tol: Convergence tolerance.
        """
        self.system = system
        self.problem_type = problem_type
        self.max_iter = max_iter
        self.tol = tol

        if problem_type == "snr":
            self._solver = SNRConstrainedSolver(
                system, snr_min_dB=snr_min_dB, max_iter=max_iter, tol=tol
            )
        elif problem_type == "crb":
            self._solver = CRBConstrainedSolver(
                system, crb_max=crb_max, max_iter=max_iter, tol=tol
            )
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}. Use 'snr' or 'crb'.")

    def solve(self) -> dict:
        """Run the alternating optimization algorithm.

        Returns:
            Solution dictionary with optimized beamforming, RIS phases,
            and performance metrics.

        For SNR-constrained:
            'sum_rate', 'snr_sensing', 'W', 'theta', 'converged'

        For CRB-constrained:
            'sum_rate', 'crb', 'W', 'theta', 'converged'
        """
        return self._solver.solve()

    def evaluate(self, W: np.ndarray, theta: np.ndarray) -> dict:
        """Evaluate solution metrics for given W and θ.

        Args:
            W: Beamforming matrix (M, K).
            theta: RIS phase vector (L,).

        Returns:
            Dictionary of metrics: sum_rate, sinr_per_user, snr_sensing, crb.
        """
        self.system.set_ris_phases(theta)
        w_total = np.sum(W, axis=1)

        results = {
            "sum_rate": self.system.compute_sum_rate(W),
            "snr_sensing": self.system.compute_snr_sensing(w_total),
            "power_used": np.sum(np.linalg.norm(W, axis=0) ** 2),
            "sinr_per_user": [],
        }

        # Per-user SINR
        H_BR = self.system.channels["H_BR"]
        G = self.system.channels["G"]
        h_d = self.system.channels["h_d"]
        Theta = self.system.ris_diagonal_matrix()
        sigma2 = self.system.noise_power

        for k in range(self.system.K):
            h_k = G[k, :] @ Theta @ H_BR + h_d[k, :]
            signal = np.abs(h_k.conj() @ W[:, k]) ** 2
            interf = sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2
                for j in range(self.system.K)
                if j != k
            )
            sinr_k = signal / (interf + sigma2)
            results["sinr_per_user"].append(sinr_k)

        if self.problem_type == "crb":
            results["crb"] = self._solver.compute_crb(w_total)

        return results
