"""
Pareto Optimizer for ISAC Tradeoff (Algorithm 4)
=================================================

Implements Pareto boundary search for the communication-sensing
energy efficiency tradeoff.

Algorithm 4 (Section V):
    Maximize EE_C subject to EE_S ≥ ℰ (threshold)
    Vary ℰ to trace the Pareto boundary

The Pareto boundary shows the optimal tradeoff between:
    - Communication-centric EE (EE_C)
    - Sensing-centric EE (EE_S)

Reference: Zou et al., IEEE Trans. Commun., 2024 (Algorithm 4)
"""

import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from .dinkelbach_solver import DinkelbachSolver


class ParetoPoint(NamedTuple):
    """A point on the Pareto boundary."""
    ee_c: float         # Communication EE
    ee_s: float         # Sensing EE
    W: np.ndarray       # Beamforming matrix
    sum_rate: float     # Achieved sum rate
    total_power: float  # Total transmit power


class ParetoOptimizer:
    """
    Pareto optimizer for ISAC EE tradeoff (Algorithm 4).

    Traces the Pareto boundary by:
    1. Finding endpoints (pure comm-EE and pure sensing-EE)
    2. Varying the sensing EE threshold ℰ
    3. Solving constrained EE_C maximization at each ℰ
    """

    def __init__(
        self,
        model: "ISACSystemModel",
        n_pareto_points: int = 20,
        solver: str = "MOSEK",
        verbose: bool = False,
    ):
        """
        Initialize Pareto optimizer.

        Parameters
        ----------
        model : ISACSystemModel
            System model
        n_pareto_points : int
            Number of points on Pareto boundary
        solver : str
            CVXPY solver name
        verbose : bool
            Print progress
        """
        self.model = model
        self.n_pareto_points = n_pareto_points
        self.solver = solver
        self.verbose = verbose

    def trace_pareto_boundary(
        self,
        target_angle_deg: float = 90.0,
        gamma_min: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> List[ParetoPoint]:
        """
        Trace the Pareto boundary (Algorithm 4).

        Steps:
        1. Find EE_C_max (pure communication objective)
        2. Find EE_S_max (pure sensing objective)
        3. For ℰ ∈ [0, EE_S_max], solve:
            max EE_C s.t. EE_S ≥ ℰ
        4. Collect Pareto-optimal points

        Parameters
        ----------
        target_angle_deg : float
            Target angle in degrees
        gamma_min : float, optional
            Minimum SINR requirement
        n_points : int, optional
            Number of Pareto points (overrides constructor value)

        Returns
        -------
        list of ParetoPoint
            Points on Pareto boundary
        """
        if n_points is not None:
            self.n_pareto_points = n_points

        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        pareto_points = []

        # Step 1: Find EE_C maximum (endpoint 1)
        if self.verbose:
            print("Finding EE_C maximum...")

        ee_c_max_result = self._find_ee_c_max(gamma_min, target_angle_deg)
        if ee_c_max_result is not None:
            ee_c, ee_s, W = ee_c_max_result
            sum_rate = self._compute_sum_rate(W)
            total_power = self._compute_total_power(W)
            pareto_points.append(ParetoPoint(ee_c, ee_s, W, sum_rate, total_power))

            if self.verbose:
                print(f"  EE_C_max = {ee_c:.4f}, EE_S = {ee_s:.4f}")

        # Step 2: Find EE_S maximum (endpoint 2)
        if self.verbose:
            print("Finding EE_S maximum...")

        ee_s_max_result = self._find_ee_s_max(gamma_min, target_angle_deg)
        if ee_s_max_result is not None:
            ee_c, ee_s, W = ee_s_max_result
            sum_rate = self._compute_sum_rate(W)
            total_power = self._compute_total_power(W)
            pareto_points.append(ParetoPoint(ee_c, ee_s, W, sum_rate, total_power))

            if self.verbose:
                print(f"  EE_C = {ee_c:.4f}, EE_S_max = {ee_s:.4f}")

        # Step 3: Vary threshold and solve constrained problems
        if len(pareto_points) >= 2:
            ee_s_min = min(pt.ee_s for pt in pareto_points)
            ee_s_max = max(pt.ee_s for pt in pareto_points)

            # Generate threshold values
            thresholds = np.linspace(ee_s_min, ee_s_max, self.n_pareto_points)

            if self.verbose:
                print(f"Tracing Pareto boundary ({self.n_pareto_points} points)...")

            for i, threshold in enumerate(thresholds):
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"  Point {i + 1}/{self.n_pareto_points}")

                result = self._solve_constrained_ee_c(
                    threshold, gamma_min, target_angle_deg,
                )

                if result is not None:
                    ee_c, ee_s, W = result
                    sum_rate = self._compute_sum_rate(W)
                    total_power = self._compute_total_power(W)
                    pareto_points.append(
                        ParetoPoint(ee_c, ee_s, W, sum_rate, total_power)
                    )

        # Sort by EE_C for clean boundary
        pareto_points.sort(key=lambda p: p.ee_c)

        # Remove dominated points
        pareto_points = self._remove_dominated(pareto_points)

        return pareto_points

    def _find_ee_c_max(
        self,
        gamma_min: Optional[float],
        target_angle_deg: float,
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """
        Find maximum communication EE (pure comm objective).

        Parameters
        ----------
        gamma_min : float, optional
            Minimum SINR
        target_angle_deg : float
            Target angle

        Returns
        -------
        tuple or None
            (EE_C, EE_S, W) at EE_C maximum
        """
        solver = DinkelbachSolver(self.model, verbose=False)
        result = solver.solve(
            target_angle_deg=target_angle_deg,
            gamma_min=gamma_min,
        )

        if result.converged or result.ee_c > 0:
            from .ee_metrics import compute_ee_s
            ee_s = compute_ee_s(
                result.W,
                self.model.steering_vector_tx(np.radians(target_angle_deg)),
                self.model.steering_vector_rx(np.radians(target_angle_deg)),
                self.model.sigma_s2,
                self.model.L,
                self.model.epsilon,
                self.model.P0,
            )
            return (result.ee_c, ee_s, result.W)

        return None

    def _find_ee_s_max(
        self,
        gamma_min: Optional[float],
        target_angle_deg: float,
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """
        Find maximum sensing EE (pure sensing objective).

        Uses beamforming toward the target direction.

        Parameters
        ----------
        gamma_min : float, optional
            Minimum SINR
        target_angle_deg : float
            Target angle

        Returns
        -------
        tuple or None
            (EE_C, EE_S, W) at EE_S maximum
        """
        from .ee_metrics import compute_ee_c, compute_ee_s

        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        # Simple beamforming toward target
        M, K = self.M, self.K
        W = np.zeros((M, K), dtype=complex)
        P_per_user = self.model.P_max / K

        for k in range(K):
            W[:, k] = a_t / np.linalg.norm(a_t) * np.sqrt(P_per_user)

        H = self.model.get_csi()
        ee_c = compute_ee_c(H, W, self.model.sigma_c2, self.model.epsilon, self.model.P0)
        ee_s = compute_ee_s(
            W, a_t, a_r, self.model.sigma_s2,
            self.model.L, self.model.epsilon, self.model.P0,
        )

        return (ee_c, ee_s, W)

    @property
    def M(self):
        return self.model.M

    @property
    def K(self):
        return self.model.K

    def _solve_constrained_ee_c(
        self,
        ee_s_threshold: float,
        gamma_min: Optional[float],
        target_angle_deg: float,
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """
        Solve: max EE_C s.t. EE_S ≥ threshold.

        This is the core of Algorithm 4.

        Parameters
        ----------
        ee_s_threshold : float
            Minimum required sensing EE
        gamma_min : float, optional
            Minimum SINR
        target_angle_deg : float
            Target angle

        Returns
        -------
        tuple or None
            (EE_C, EE_S, W) for constrained optimum
        """
        import cvxpy as cp
        from .ee_metrics import compute_ee_c, compute_ee_s

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
        M, K = self.M, self.K

        # PSD matrix variables
        W_psd = [cp.Variable((M, M), hermitian=True) for _ in range(K)]

        constraints = []
        for k in range(K):
            constraints.append(W_psd[k] >> 0)

        # Power constraint
        total_power_expr = sum(cp.real(cp.trace(W_psd[k])) for k in range(K))
        constraints.append(total_power_expr <= P_max)

        # SINR constraints
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

        # Sensing EE constraint: EE_S ≥ threshold
        # EE_S = (1/CRB) / (L * P_consumption)
        # Linearized constraint using SCA
        Rx = sum(W_psd)
        m_indices = np.arange(M)
        da_t = 1j * 2 * np.pi * 0.5 * m_indices * np.cos(np.pi / 2) * a_t
        a_r_norm_sq = np.sum(np.abs(a_r) ** 2)

        # Simplified sensing constraint: signal power toward target
        fim_expr = cp.real(cp.quad_form(da_t, Rx))
        power_consumption_expr = (1 / epsilon) * total_power_expr + P0

        # EE_S ≥ threshold  →  FIM ≥ threshold * σ_s² * L * P_consumption
        sensing_threshold = ee_s_threshold * sigma_s2 * L
        constraints.append(fim_expr >= sensing_threshold * power_consumption_expr)

        # Objective: maximize sum rate (proxy for EE_C)
        sum_rate_obj = 0
        for k in range(K):
            h_k = H[k, :]
            sum_rate_obj += cp.real(cp.quad_form(h_k, W_psd[k]))

        objective = cp.Maximize(sum_rate_obj)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            status = prob.status
        except (cp.error.SolverError, Exception):
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
                status = prob.status
            except Exception:
                return None

        if status in ["optimal", "optimal_inaccurate"]:
            W_opt = np.zeros((M, K), dtype=complex)
            for k in range(K):
                W_k = W_psd[k].value
                if W_k is not None:
                    eigenvalues, eigenvectors = np.linalg.eigh(W_k)
                    idx = np.argmax(eigenvalues)
                    w_k = eigenvectors[:, idx] * np.sqrt(max(eigenvalues[idx], 0))
                    W_opt[:, k] = w_k

            ee_c = compute_ee_c(H, W_opt, sigma_c2, epsilon, P0)
            ee_s = compute_ee_s(W_opt, a_t, a_r, sigma_s2, L, epsilon, P0)

            return (ee_c, ee_s, W_opt)

        return None

    def _compute_sum_rate(self, W: np.ndarray) -> float:
        """Compute sum rate."""
        H = self.model.get_csi()
        sigma_c2 = self.model.sigma_c2
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

    def _remove_dominated(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Remove Pareto-dominated points.

        A point is dominated if there exists another point with
        both higher EE_C and higher EE_S.

        Parameters
        ----------
        points : list
            Candidate points sorted by EE_C

        Returns
        -------
        list
            Non-dominated points
        """
        if len(points) <= 1:
            return points

        # Build upper envelope (max EE_S for given EE_C)
        non_dominated = [points[0]]

        for pt in points[1:]:
            # Check if this point improves EE_S over last non-dominated
            if pt.ee_s > non_dominated[-1].ee_s:
                non_dominated.append(pt)

        return non_dominated
