"""
Array Partitioning Algorithm.

Joint optimization of TX/RX antenna partition and beamforming
via alternating optimization (Dinkelbach + ADMM + MM).
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from system_model import ISACSystem
from doa_crb import compute_crb, compute_fisher_info
from admm_solver import ADMMSolver
from mm_solver import MMSolver


class ArrayPartitioner:
    """
    Joint array partitioning and beamforming optimizer for ISAC.

    Alternating optimization framework:
        1. Fix partition → optimize beamforming (MM)
        2. Fix beamforming → optimize partition (ADMM)
        3. Repeat until convergence

    Also uses Dinkelbach's transform for fractional programming
    when the objective involves ratios.
    """

    def __init__(
        self,
        system: ISACSystem,
        max_outer_iter: int = 30,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        system : ISACSystem
            ISAC system model.
        max_outer_iter : int
            Maximum alternating optimization iterations.
        tol : float
            Convergence tolerance.
        verbose : bool
        """
        self.system = system
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.verbose = verbose

        self.admm = ADMMSolver(
            M=system.M,
            rho=1.0,
            max_iter=100,
            tol=1e-4,
            verbose=verbose,
        )
        self.mm = MMSolver(
            M=system.M,
            K=system.K,
            P_max=system.P_max,
            sigma2_c=system.sigma2_c,
            sigma2_r=system.sigma2_r,
            max_iter=30,
            tol=1e-5,
            verbose=verbose,
        )

    def optimize(
        self,
        H: np.ndarray,
        a_theta: np.ndarray,
        sinr_thresholds: np.ndarray,
        t_init: Optional[np.ndarray] = None,
        w_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Joint optimization of partition and beamforming.

        Parameters
        ----------
        H : np.ndarray, shape (M, K)
            Channel matrix.
        a_theta : np.ndarray, shape (M,)
            Steering vector at target DOA.
        sinr_thresholds : np.ndarray, shape (K,)
            Minimum SINR per user.
        t_init : np.ndarray, shape (M,), optional
            Initial TX partition.
        w_init : np.ndarray, shape (M, K), optional
            Initial beamforming.

        Returns
        -------
        t_opt : np.ndarray, shape (M,)
            Optimized TX partition.
        w_opt : np.ndarray, shape (M, K)
            Optimized beamforming.
        info : dict
        """
        M = self.system.M

        # Initialize
        if t_init is None:
            # Default: roughly half TX, half RX
            t = np.zeros(M)
            t[: M // 2] = 1.0
        else:
            t = t_init.copy()

        r = 1 - t

        if w_init is None:
            w = self.mm._initialize_beamforming(H, t, sinr_thresholds)
        else:
            w = w_init.copy()

        history = {"crb": [], "partition": [], "sinr_min": []}
        converged = False
        prev_crb = np.inf
        dinkelbach_lambda = 0.0

        for outer_it in range(self.max_outer_iter):
            # --- Step 1: Optimize beamforming given partition (MM) ---
            w, mm_info = self.mm.solve(H, t, r, a_theta, sinr_thresholds, w_init=w)

            # --- Step 2: Optimize partition given beamforming (ADMM) ---
            t, admm_info = self.admm.solve(
                c_init=t,
                w=w,
                a_theta=a_theta,
                sigma2_r=self.system.sigma2_r,
            )
            r = 1 - t

            # --- Step 3: Dinkelbach update (if using fractional form) ---
            crb = compute_crb(a_theta, w, t, r, self.system.sigma2_r)
            F = compute_fisher_info(a_theta, a_theta, w, t, r, self.system.sigma2_r)
            if F > 1e-10:
                dinkelbach_lambda = 1.0 / F

            # Record
            sinr = self.system.compute_sinr_with_partition(w, H, t)
            history["crb"].append(crb)
            history["partition"].append(t.copy())
            history["sinr_min"].append(np.min(sinr))

            if self.verbose:
                print(
                    f"Outer iter {outer_it}: CRB={crb:.4e}, "
                    f"min SINR={np.min(sinr):.4f}, "
                    f"TX antennas={int(np.sum(t))}"
                )

            # Check convergence
            if abs(crb - prev_crb) < self.tol * (1 + abs(prev_crb)):
                converged = True
                if self.verbose:
                    print(f"Converged at outer iteration {outer_it}")
                break
            prev_crb = crb

        info = {
            "converged": converged,
            "outer_iterations": outer_it + 1,
            "crb_history": history["crb"],
            "sinr_min_history": history["sinr_min"],
            "final_crb": history["crb"][-1] if history["crb"] else np.inf,
            "num_tx_antennas": int(np.sum(t)),
        }
        return t, w, info

    def optimize_dinkelbach(
        self,
        H: np.ndarray,
        a_theta: np.ndarray,
        sinr_thresholds: np.ndarray,
        max_dinkelbach_iter: int = 20,
        dinkelbach_tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Full optimization with Dinkelbach's transform for fractional programming.

        Solves: min_{t,w} CRB / P_sensing  (ratio form)
        via Dinkelbach: min_{t,w} CRB - lambda * P_sensing

        Parameters
        ----------
        H : np.ndarray
        a_theta : np.ndarray
        sinr_thresholds : np.ndarray
        max_dinkelbach_iter : int
        dinkelbach_tol : float

        Returns
        -------
        t_opt, w_opt, info
        """
        M = self.system.M
        t = np.ones(M) * 0.5  # Start balanced
        r = 1 - t
        w = self.mm._initialize_beamforming(H, t, sinr_thresholds)

        lambda_d = 0.0
        history = {"lambda": [], "crb": []}

        for d_it in range(max_dinkelbach_iter):
            # Solve inner problem with current lambda
            t, w, inner_info = self.optimize(H, a_theta, sinr_thresholds, t_init=t, w_init=w)

            # Update lambda via Dinkelbach
            crb = compute_crb(a_theta, w, t, r, self.system.sigma2_r)
            F = compute_fisher_info(a_theta, a_theta, w, t, r, self.system.sigma2_r)
            p_sensing = np.sum(np.abs(w) ** 2)

            if p_sensing > 1e-10:
                lambda_new = crb / p_sensing
            else:
                lambda_new = lambda_d

            history["lambda"].append(lambda_new)
            history["crb"].append(crb)

            if self.verbose:
                print(f"Dinkelbach iter {d_it}: lambda={lambda_new:.4e}, CRB={crb:.4e}")

            if abs(lambda_new - lambda_d) < dinkelbach_tol:
                break
            lambda_d = lambda_new

        info = {
            "dinkelbach_iterations": d_it + 1,
            "final_lambda": lambda_d,
            "history": history,
            **inner_info,
        }
        return t, w, info

    def get_partition_summary(self, t: np.ndarray) -> Dict:
        """Get human-readable partition summary."""
        tx_idx = np.where(t > 0.5)[0]
        rx_idx = np.where(t <= 0.5)[0]
        return {
            "tx_antennas": tx_idx.tolist(),
            "rx_antennas": rx_idx.tolist(),
            "num_tx": len(tx_idx),
            "num_rx": len(rx_idx),
            "tx_fraction": len(tx_idx) / len(t),
        }
