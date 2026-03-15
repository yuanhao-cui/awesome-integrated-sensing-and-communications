"""
Alternating Optimization Solver for Joint BS Mode Selection and Beamforming.

Implements the alternating optimization (AO) approach to solve the joint
non-convex optimization problem: alternating between mode selection and
beamforming design until convergence.
"""

import numpy as np
from typing import Optional
from system_model import CellFreeISACSystem, BSMode
from mode_selection import ModeSelector
from beamforming import BeamformingDesigner
from cooperative import CooperativeSensing
from metrics import compute_rate, compute_crb, compute_coverage


class AlternatingOptimizationSolver:
    """
    Alternating optimization solver for cooperative cell-free ISAC.

    Algorithm:
    1. Initialize beamformers randomly
    2. Repeat until convergence:
       a. Fix beamformers -> optimize mode selection
       b. Fix modes -> optimize beamformers
       c. Evaluate objective
    3. Return optimal configuration

    The objective is a weighted sum of:
    - Total communication rate
    - Sensing accuracy (inverse CRB)
    - Coverage metric
    """

    def __init__(
        self,
        system: CellFreeISACSystem,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        verbose: bool = False,
    ):
        """
        Initialize the AO solver.

        Args:
            system: Cell-free ISAC system model.
            alpha: Weight for communication rate in objective.
            beta: Weight for sensing accuracy in objective.
            gamma: Weight for coverage in objective.
            max_iterations: Maximum AO iterations.
            tolerance: Convergence tolerance.
            verbose: Print iteration progress.
        """
        self.system = system
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        self.mode_selector = ModeSelector(system)
        self.bf_designer = BeamformingDesigner(system)
        self.coop_sensing = CooperativeSensing(system)

        # Convergence tracking
        self.history = {
            "objective": [],
            "sum_rate": [],
            "avg_crb": [],
            "coverage": [],
            "modes": [],
        }

    def compute_objective(self) -> float:
        """
        Compute the weighted objective function.

        Returns:
            Objective value (higher is better).
        """
        # Communication rate
        rate_result = compute_rate(self.system)
        sum_rate = rate_result["sum_rate"]

        # Sensing accuracy (inverse CRB)
        crb_result = compute_crb(self.system)
        avg_crb = crb_result["avg_crb"]
        sensing_metric = 1.0 / (avg_crb + 1e-10)

        # Coverage
        cov_result = compute_coverage(self.system)
        coverage = cov_result["coverage_fraction"]

        # Normalize and combine
        obj = (
            self.alpha * sum_rate +
            self.beta * sensing_metric * 1e-6 +  # Scale factor
            self.gamma * coverage
        )

        return obj

    def step_mode_selection(self) -> np.ndarray:
        """
        AO step: optimize mode selection with fixed beamformers.

        Tries multiple methods and selects the best.

        Returns:
            Optimized mode vector.
        """
        best_modes = None
        best_obj = -np.inf

        for method in ["greedy", "channel_norm", "distance", "equal_split"]:
            try:
                modes = self.mode_selector.optimize(method=method)
                obj = self.compute_objective()

                if obj > best_obj:
                    best_obj = obj
                    best_modes = modes.copy()
            except Exception:
                continue

        if best_modes is not None:
            self.system.set_mode_vector(best_modes)

        return best_modes if best_modes is not None else self.system.get_mode_vector()

    def step_beamforming(self) -> dict:
        """
        AO step: optimize beamformers with fixed modes.

        Tries multiple methods and selects the best.

        Returns:
            Beamforming result dictionary.
        """
        best_result = None
        best_obj = -np.inf

        for comm_method in ["mmse", "max_sinr"]:
            for sens_method in ["cooperative", "single_target"]:
                try:
                    self.bf_designer.initialize_beamformers()
                    result = self.bf_designer.design_all_beamformers(
                        comm_method=comm_method,
                        sens_method=sens_method,
                    )
                    obj = self.compute_objective()

                    if obj > best_obj:
                        best_obj = obj
                        best_result = result
                except Exception:
                    continue

        if best_result is None:
            # Fallback: equal power
            self.bf_designer.initialize_beamformers()
            best_result = self.bf_designer.equal_power_beamforming()

        return best_result

    def solve(self) -> dict:
        """
        Run the full alternating optimization algorithm.

        Returns:
            Dictionary with optimal configuration and metrics.
        """
        # Initialize
        self.bf_designer.initialize_beamformers()
        prev_obj = -np.inf

        for iteration in range(self.max_iterations):
            # Step 1: Optimize mode selection
            modes = self.step_mode_selection()

            # Step 2: Optimize beamforming
            bf_result = self.step_beamforming()

            # Evaluate
            obj = self.compute_objective()
            rate_result = compute_rate(self.system)
            crb_result = compute_crb(self.system)
            cov_result = compute_coverage(self.system)

            # Record history
            self.history["objective"].append(obj)
            self.history["sum_rate"].append(rate_result["sum_rate"])
            self.history["avg_crb"].append(crb_result["avg_crb"])
            self.history["coverage"].append(cov_result["coverage_fraction"])
            self.history["modes"].append(modes.copy())

            if self.verbose:
                print(
                    f"Iter {iteration+1:3d}: obj={obj:.6f}, "
                    f"rate={rate_result['sum_rate']:.4f}, "
                    f"crb={crb_result['avg_crb']:.6f}, "
                    f"cov={cov_result['coverage_fraction']:.4f}"
                )

            # Check convergence
            if abs(obj - prev_obj) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            prev_obj = obj

        # Final results
        final_modes = self.system.get_mode_vector()
        final_rate = compute_rate(self.system)
        final_crb = compute_crb(self.system)
        final_cov = compute_coverage(self.system)

        return {
            "modes": final_modes,
            "sum_rate": final_rate["sum_rate"],
            "per_user_rate": final_rate["per_user_rate"],
            "avg_crb": final_crb["avg_crb"],
            "per_target_crb": final_crb["per_target_crb"],
            "coverage_fraction": final_cov["coverage_fraction"],
            "iterations": len(self.history["objective"]),
            "converged": len(self.history["objective"]) < self.max_iterations,
            "history": self.history,
            "beamformers": {
                m: self.system.bs_configs[m].beamforming_vector
                for m in range(self.system.n_bs)
            },
        }

    def get_iteration_stats(self) -> list[dict]:
        """
        Get per-iteration statistics.

        Returns:
            List of dictionaries with iteration data.
        """
        n_iters = len(self.history["objective"])
        return [
            {
                "iteration": i + 1,
                "objective": self.history["objective"][i],
                "sum_rate": self.history["sum_rate"][i],
                "avg_crb": self.history["avg_crb"][i],
                "coverage": self.history["coverage"][i],
                "modes": self.history["modes"][i],
            }
            for i in range(n_iters)
        ]
