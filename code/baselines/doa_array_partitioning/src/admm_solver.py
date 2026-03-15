"""
ADMM Solver for Antenna Partitioning.

Solves the binary antenna partitioning problem (TX/RX assignment)
via ADMM relaxation with convergence guarantees.
"""

import numpy as np
from typing import Optional, Tuple, Dict


class ADMMSolver:
    """
    ADMM-based antenna partitioning solver.

    Solves:
        min_{c, z}   f(c, w, ...)     [CRB or related objective]
        s.t.         z = c
                     c ∈ [0, 1]^M     [continuous relaxation]
                     z ∈ {0, 1}^M     [binary]

    ADMM augmented Lagrangian:
        L_rho(c, z, u) = f(c) + (rho/2) * ||c - z + u||^2

    Alternating updates:
        c^{k+1} = argmin_c f(c) + (rho/2)||c - z^k + u^k||^2
        z^{k+1} = Π_{0,1}(c^{k+1} + u^k)
        u^{k+1} = u^k + c^{k+1} - z^{k+1}
    """

    def __init__(
        self,
        M: int,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        M : int
            Number of antennas.
        rho : float
            ADMM penalty parameter.
        max_iter : int
            Maximum ADMM iterations.
        tol : float
            Convergence tolerance on primal and dual residuals.
        verbose : bool
            Print iteration progress.
        """
        self.M = M
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(
        self,
        c_init: Optional[np.ndarray] = None,
        objective_fn: Optional[callable] = None,
        grad_fn: Optional[callable] = None,
        w: Optional[np.ndarray] = None,
        a_theta: Optional[np.ndarray] = None,
        sigma2_r: float = 1.0,
        N_samples: int = 1,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run ADMM to find optimal antenna partition.

        Parameters
        ----------
        c_init : np.ndarray, shape (M,)
            Initial continuous partition. Random if None.
        objective_fn : callable, optional
            f(c) -> float. If None, uses CRB-based objective.
        grad_fn : callable, optional
            grad_f(c) -> np.ndarray. If None, numerical gradient.
        w : np.ndarray, shape (M,) or (M, K)
            Beamforming vector(s), needed for default objective.
        a_theta : np.ndarray, shape (M,)
            Steering vector, needed for default objective.
        sigma2_r : float
        N_samples : int

        Returns
        -------
        z : np.ndarray, shape (M,)
            Binary partition (0=RX, 1=TX).
        info : dict
            Convergence information.
        """
        if c_init is None:
            c = np.random.default_rng().uniform(0.2, 0.8, self.M)
        else:
            c = c_init.copy()

        z = np.round(c).astype(float)
        u = np.zeros(self.M)

        history = {"primal_residual": [], "dual_residual": [], "objective": []}
        converged = False

        for it in range(self.max_iter):
            c_old = c.copy()
            z_old = z.copy()

            # c-update: gradient step on f(c) + (rho/2)||c - z + u||^2
            c = self._c_update(c, z, u, w, a_theta, sigma2_r, N_samples, objective_fn, grad_fn)

            # z-update: projection onto {0,1}
            z_new = c + u
            z = np.round(np.clip(z_new, 0, 1)).astype(float)

            # u-update: dual variable
            u = u + c - z

            # Compute residuals
            r_primal = np.linalg.norm(c - z)
            r_dual = self.rho * np.linalg.norm(z - z_old)

            history["primal_residual"].append(r_primal)
            history["dual_residual"].append(r_dual)

            if self.verbose and it % 10 == 0:
                print(f"  ADMM iter {it}: r_p={r_primal:.4e}, r_d={r_dual:.4e}")

            # Convergence check
            if r_primal < self.tol and r_dual < self.tol:
                converged = True
                if self.verbose:
                    print(f"  ADMM converged at iteration {it}")
                break

        info = {
            "converged": converged,
            "iterations": it + 1,
            "primal_residual": history["primal_residual"][-1],
            "dual_residual": history["dual_residual"][-1],
            "history": history,
        }
        return z, info

    def _c_update(
        self,
        c: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        w: Optional[np.ndarray],
        a_theta: Optional[np.ndarray],
        sigma2_r: float,
        N_samples: int,
        objective_fn: Optional[callable],
        grad_fn: Optional[callable],
    ) -> np.ndarray:
        """
        c-update via proximal gradient step.

        min_c f(c) + (rho/2)||c - z + u||^2
        → c = c - alpha * (grad_f(c) + rho*(c - z + u))
        """
        alpha = 1.0 / (self.rho + 1e-8)

        if grad_fn is not None:
            g = grad_fn(c)
        else:
            g = self._default_grad(c, w, a_theta, sigma2_r, N_samples)

        prox_grad = g + self.rho * (c - z + u)
        c_new = c - alpha * prox_grad

        # Project onto [0, 1]
        c_new = np.clip(c_new, 0.0, 1.0)
        return c_new

    def _default_grad(
        self,
        c: np.ndarray,
        w: Optional[np.ndarray],
        a_theta: Optional[np.ndarray],
        sigma2_r: float,
        N_samples: int,
    ) -> np.ndarray:
        """Numerical gradient of CRB w.r.t. c."""
        if w is None or a_theta is None:
            return np.zeros(self.M)

        eps = 1e-6
        grad = np.zeros(self.M)
        f0 = self._default_objective(c, w, a_theta, sigma2_r, N_samples)
        for m in range(self.M):
            c_plus = c.copy()
            c_plus[m] += eps
            f_plus = self._default_objective(c_plus, w, a_theta, sigma2_r, N_samples)
            grad[m] = (f_plus - f0) / eps
        return grad

    def _default_objective(
        self,
        c: np.ndarray,
        w: np.ndarray,
        a_theta: np.ndarray,
        sigma2_r: float,
        N_samples: int,
    ) -> float:
        """CRB-based objective for continuous partition c."""
        from doa_crb import compute_fisher_info

        t = c  # Continuous TX indicator
        r = 1 - c  # Continuous RX indicator

        F = compute_fisher_info(a_theta, a_theta, w, t, r, sigma2_r, N_samples)
        if F < 1e-15:
            return 1e10
        return 1.0 / F

    def run_admm_iteration(
        self,
        c: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        w: Optional[np.ndarray] = None,
        a_theta: Optional[np.ndarray] = None,
        sigma2_r: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single ADMM iteration for external use.

        Returns
        -------
        c_new, z_new, u_new
        """
        c_new = self._c_update(c, z, u, w, a_theta, sigma2_r, 1, None, None)
        z_new = np.round(np.clip(c_new + u, 0, 1)).astype(float)
        u_new = u + c_new - z_new
        return c_new, z_new, u_new
