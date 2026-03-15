"""
Majorization-Minimization (MM) Solver for Beamforming Optimization.

Minimizes CRB subject to SINR and power constraints using MM framework.
Handles the non-convex beamforming optimization via convex upper bounds.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import cvxpy as cp


class MMSolver:
    """
    MM-based beamforming optimizer.

    Problem:
        min_{w}    CRB(theta; w, t, r)
        s.t.       SINR_k >= gamma_k,  for all k
                   sum_k ||w_k||^2 <= P_max

    MM approach:
        1. Construct upper bound g(w | w^i) at current iterate w^i
        2. Solve convex surrogate problem
        3. Update w^{i+1}
    """

    def __init__(
        self,
        M: int,
        K: int,
        P_max: float,
        sigma2_c: float,
        sigma2_r: float,
        max_iter: int = 50,
        tol: float = 1e-5,
        verbose: bool = False,
    ):
        self.M = M
        self.K = K
        self.P_max = P_max
        self.sigma2_c = sigma2_c
        self.sigma2_r = sigma2_r
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(
        self,
        H: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        a_theta: np.ndarray,
        sinr_thresholds: np.ndarray,
        w_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve beamforming optimization via MM.

        Parameters
        ----------
        H : np.ndarray, shape (M, K)
            Channel matrix.
        t : np.ndarray, shape (M,)
            TX partition vector.
        r : np.ndarray, shape (M,)
            RX partition vector.
        a_theta : np.ndarray, shape (M,)
            Steering vector at target DOA.
        sinr_thresholds : np.ndarray, shape (K,)
            Minimum SINR per user.
        w_init : np.ndarray, shape (M, K), optional
            Initial beamforming matrix.

        Returns
        -------
        w : np.ndarray, shape (M, K)
            Optimized beamforming matrix.
        info : dict
        """
        if w_init is None:
            w = self._initialize_beamforming(H, t, sinr_thresholds)
        else:
            w = w_init.copy()

        history = {"crb": [], "sinr": []}
        converged = False
        prev_crb = np.inf

        for it in range(self.max_iter):
            # Compute current CRB (as objective)
            crb_val = self._compute_crb_objective(w, t, r, a_theta)
            sinr_vals = self._compute_sinr(w, H, t)
            history["crb"].append(crb_val)
            history["sinr"].append(sinr_vals.copy())

            if self.verbose and it % 5 == 0:
                print(f"  MM iter {it}: CRB={crb_val:.4e}, SINR={sinr_vals}")

            # Check convergence
            if abs(crb_val - prev_crb) < self.tol * (1 + abs(prev_crb)):
                converged = True
                if self.verbose:
                    print(f"  MM converged at iteration {it}")
                break
            prev_crb_val = crb_val

            # MM surrogate: solve convex problem
            w_new = self._mm_step(w, H, t, r, a_theta, sinr_thresholds)
            if w_new is not None:
                w = w_new
            else:
                if self.verbose:
                    print(f"  MM step failed at iteration {it}")
                break

            prev_crb = prev_crb_val

        info = {
            "converged": converged,
            "iterations": it + 1,
            "crb_history": history["crb"],
            "sinr_history": history["sinr"],
            "final_crb": history["crb"][-1] if history["crb"] else np.inf,
        }
        return w, info

    def _initialize_beamforming(
        self, H: np.ndarray, t: np.ndarray, sinr_thresholds: np.ndarray
    ) -> np.ndarray:
        """Initialize beamforming with MRT-like approach."""
        w = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H[:, k]
            w[:, k] = np.sqrt(self.P_max / self.K) * h_k / (np.linalg.norm(h_k) + 1e-10)
        # Apply TX partition
        w = w * t[:, np.newaxis]
        return w

    def _compute_crb_objective(
        self, w: np.ndarray, t: np.ndarray, r: np.ndarray, a_theta: np.ndarray
    ) -> float:
        """Compute CRB for current beamforming."""
        from doa_crb import compute_crb

        crb = compute_crb(a_theta, w, t, r, self.sigma2_r)
        return crb

    def _compute_sinr(self, w: np.ndarray, H: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Compute SINR per user."""
        w_eff = w * t[:, np.newaxis]
        sinr = np.zeros(self.K)
        for k in range(self.K):
            h_k = H[:, k]
            signal = np.abs(h_k @ w_eff[:, k]) ** 2
            interference = sum(
                np.abs(h_k @ w_eff[:, j]) ** 2 for j in range(self.K) if j != k
            ) + self.sigma2_c
            sinr[k] = signal / (interference + 1e-15)
        return sinr

    def _mm_step(
        self,
        w: np.ndarray,
        H: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        a_theta: np.ndarray,
        sinr_thresholds: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Single MM iteration using convex surrogate.

        Constructs a weighted MMSE-type upper bound and solves via CVXPY.
        """
        try:
            w_var = cp.Variable((self.M, self.K), complex=True)
            constraints = []

            # Power constraint
            constraints.append(cp.sum(cp.norm(w_var, axis=0) ** 2) <= self.P_max)

            # TX partition: w_mk = 0 if t_m = 0
            for m in range(self.M):
                if t[m] < 0.5:
                    constraints.append(w_var[m, :] == 0)

            # SINR constraints (linearized)
            for k in range(self.K):
                h_k = H[:, k]
                w_k = w_var[:, k]
                # SINR_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma2) >= gamma_k
                signal = cp.real(cp.conj(h_k) @ w_k)
                signal_sq = signal ** 2  # Approximation for real part

                interference = self.sigma2_c
                for j in range(self.K):
                    if j != k:
                        interference += cp.abs(cp.conj(h_k) @ w_var[:, j]) ** 2

                constraints.append(
                    signal_sq >= sinr_thresholds[k] * interference
                )

            # Objective: minimize weighted sum of ||w||^2 (proxy for CRB)
            # MM surrogate: trace(W^H Q W) where Q captures CRB structure
            antenna_positions = np.arange(self.M) * 0.5
            if self.M > 1:
                phase_diff = np.angle(a_theta[1] / a_theta[0]) if np.abs(a_theta[0]) > 1e-10 else 0
                sin_theta = phase_diff / (2 * np.pi * 0.5)
                sin_theta = np.clip(sin_theta, -1, 1)
                cos_theta = np.sqrt(1 - sin_theta ** 2)
            else:
                cos_theta = 1.0

            a_deriv = 1j * 2 * np.pi * antenna_positions * cos_theta * a_theta

            # CRB-related objective: maximize Fisher info = minimize -F
            # F = 2 * |a^H (t ⊙ w_total)|^2 * ||r ⊙ a'||^2 / sigma2_r
            # We minimize the negative of a lower bound on F
            w_total = cp.sum(w_var, axis=1)
            tx_sig = cp.multiply(t, w_total)
            fisher_term = cp.abs(cp.conj(a_theta) @ tx_sig) ** 2
            objective = cp.Maximize(fisher_term)

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, warm_start=True, max_iters=5000, verbose=False)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                return w_var.value
            else:
                return None

        except Exception as e:
            if self.verbose:
                print(f"  MM step error: {e}")
            return None

    def solve_simplified(
        self,
        H: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
        a_theta: np.ndarray,
        sinr_thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Simplified beamforming: ZF + power allocation.

        Faster but suboptimal. Useful for warm-starting MM.
        """
        # Zero-forcing beamforming
        H_eff = H * t[:, np.newaxis]  # Effective channel (TX antennas only)
        W_zf = H_eff @ np.linalg.pinv(H_eff.conj().T @ H_eff)

        # Scale to meet power constraint
        total_power = np.sum(np.abs(W_zf) ** 2)
        if total_power > 1e-10:
            W_zf *= np.sqrt(self.P_max / total_power)

        return W_zf
