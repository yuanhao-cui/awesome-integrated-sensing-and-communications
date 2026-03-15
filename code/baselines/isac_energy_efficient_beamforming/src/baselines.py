"""
Baseline Schemes for ISAC Beamforming
======================================

Implements baseline schemes for comparison:
    1. EMaxBaseline: Energy maximization (maximize sum rate, ignore EE)
    2. FixBeamBaseline: Fixed beamforming (matched filter)
    3. RandomBaseline: Random beamforming with power normalization

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from .ee_metrics import compute_ee_c, compute_ee_s, compute_sum_rate, compute_total_power, compute_crb


class BaselineResult(NamedTuple):
    """Result from a baseline scheme."""
    W: np.ndarray          # Beamforming matrix (M x K)
    ee_c: float            # Communication EE
    ee_s: float            # Sensing EE
    sum_rate: float        # Sum rate
    total_power: float     # Total transmit power
    crb: float             # CRB value
    scheme_name: str       # Name of the baseline scheme


class EMaxBaseline:
    """
    Energy Maximization baseline.

    Maximizes sum rate subject to power constraint (ignoring EE).
    Uses matched filter beamforming scaled to maximum power.

    This is equivalent to maximizing throughput without considering
    energy efficiency, which typically results in using all available
    power.
    """

    def __init__(self, model: "ISACSystemModel"):
        """
        Initialize EMax baseline.

        Parameters
        ----------
        model : ISACSystemModel
            System model
        """
        self.model = model

    def solve(
        self,
        target_angle_deg: float = 90.0,
    ) -> BaselineResult:
        """
        Compute EMax beamforming.

        Uses matched filter: w_k ∝ h_k, scaled to P_max.

        Parameters
        ----------
        target_angle_deg : float
            Target angle for CRB computation

        Returns
        -------
        BaselineResult
            Beamforming result
        """
        H = self.model.get_csi()
        M, K = self.model.M, self.model.K
        P_max = self.model.P_max

        # Matched filter beamforming, scaled to maximum power
        W = np.zeros((M, K), dtype=complex)
        for k in range(K):
            h_k = H[k, :]
            w_k = h_k / (np.linalg.norm(h_k) + 1e-15)
            W[:, k] = w_k

        # Scale to use all available power
        W *= np.sqrt(P_max / K)

        # Compute metrics
        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        sum_rate = compute_sum_rate(H, W, self.model.sigma_c2)
        total_power = compute_total_power(W)
        ee_c = compute_ee_c(
            H, W, self.model.sigma_c2, self.model.epsilon, self.model.P0
        )
        ee_s = compute_ee_s(
            W, a_t, a_r, self.model.sigma_s2,
            self.model.L, self.model.epsilon, self.model.P0,
        )
        crb = compute_crb(W, a_t, a_r, self.model.sigma_s2, self.model.L)

        return BaselineResult(
            W=W,
            ee_c=ee_c,
            ee_s=ee_s,
            sum_rate=sum_rate,
            total_power=total_power,
            crb=crb,
            scheme_name="EMax (Matched Filter)",
        )


class FixBeamBaseline:
    """
    Fixed Beamforming baseline.

    Uses a fixed beamforming pattern:
    - Communication beams toward users (matched filter)
    - Sensing beam toward target
    - Equal power allocation between comm and sensing
    """

    def __init__(self, model: "ISACSystemModel"):
        """
        Initialize FixBeam baseline.

        Parameters
        ----------
        model : ISACSystemModel
            System model
        """
        self.model = model

    def solve(
        self,
        target_angle_deg: float = 90.0,
        sensing_fraction: float = 0.5,
    ) -> BaselineResult:
        """
        Compute fixed beamforming.

        Parameters
        ----------
        target_angle_deg : float
            Target angle
        sensing_fraction : float
            Fraction of power for sensing (0 to 1)

        Returns
        -------
        BaselineResult
            Beamforming result
        """
        H = self.model.get_csi()
        M, K = self.model.M, self.model.K
        P_max = self.model.P_max

        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        # Power allocation
        P_comm = (1 - sensing_fraction) * P_max
        P_sense = sensing_fraction * P_max

        W = np.zeros((M, K), dtype=complex)

        # Communication component (matched filter)
        for k in range(K):
            h_k = H[k, :]
            w_comm = h_k / (np.linalg.norm(h_k) + 1e-15)
            W[:, k] = w_comm * np.sqrt(P_comm / K)

        # Add sensing component (beam toward target)
        a_t_norm = a_t / (np.linalg.norm(a_t) + 1e-15)
        for k in range(K):
            W[:, k] += a_t_norm * np.sqrt(P_sense / K)

        # Compute metrics
        sum_rate = compute_sum_rate(H, W, self.model.sigma_c2)
        total_power = compute_total_power(W)
        ee_c = compute_ee_c(
            H, W, self.model.sigma_c2, self.model.epsilon, self.model.P0
        )
        ee_s = compute_ee_s(
            W, a_t, a_r, self.model.sigma_s2,
            self.model.L, self.model.epsilon, self.model.P0,
        )
        crb = compute_crb(W, a_t, a_r, self.model.sigma_s2, self.model.L)

        return BaselineResult(
            W=W,
            ee_c=ee_c,
            ee_s=ee_s,
            sum_rate=sum_rate,
            total_power=total_power,
            crb=crb,
            scheme_name=f"Fixed Beam (sense={sensing_fraction:.0%})",
        )


class RandomBaseline:
    """
    Random Beamforming baseline.

    Generates random beamforming vectors, normalized to satisfy
    power constraint. Useful as a lower bound reference.
    """

    def __init__(self, model: "ISACSystemModel", seed: Optional[int] = None):
        """
        Initialize Random baseline.

        Parameters
        ----------
        model : ISACSystemModel
            System model
        seed : int, optional
            Random seed
        """
        self.model = model
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        target_angle_deg: float = 90.0,
        n_trials: int = 100,
    ) -> BaselineResult:
        """
        Compute best random beamforming over multiple trials.

        Parameters
        ----------
        target_angle_deg : float
            Target angle
        n_trials : int
            Number of random trials

        Returns
        -------
        BaselineResult
            Best random beamforming result
        """
        H = self.model.get_csi()
        M, K = self.model.M, self.model.K
        P_max = self.model.P_max

        theta_rad = np.radians(target_angle_deg)
        a_t = self.model.steering_vector_tx(theta_rad)
        a_r = self.model.steering_vector_rx(theta_rad)

        best_ee_c = -np.inf
        best_W = None
        best_metrics = None

        for _ in range(n_trials):
            # Random beamforming
            W = (
                self.rng.standard_normal((M, K))
                + 1j * self.rng.standard_normal((M, K))
            ) / np.sqrt(2)

            # Normalize to power constraint
            total_power = np.sum(np.abs(W) ** 2)
            W *= np.sqrt(P_max / total_power)

            # Compute metrics
            ee_c = compute_ee_c(
                H, W, self.model.sigma_c2, self.model.epsilon, self.model.P0
            )

            if ee_c > best_ee_c:
                best_ee_c = ee_c
                best_W = W.copy()
                best_metrics = {
                    "sum_rate": compute_sum_rate(H, W, self.model.sigma_c2),
                    "total_power": compute_total_power(W),
                    "ee_s": compute_ee_s(
                        W, a_t, a_r, self.model.sigma_s2,
                        self.model.L, self.model.epsilon, self.model.P0,
                    ),
                    "crb": compute_crb(W, a_t, a_r, self.model.sigma_s2, self.model.L),
                }

        return BaselineResult(
            W=best_W,
            ee_c=best_ee_c,
            ee_s=best_metrics["ee_s"],
            sum_rate=best_metrics["sum_rate"],
            total_power=best_metrics["total_power"],
            crb=best_metrics["crb"],
            scheme_name=f"Random ({n_trials} trials)",
        )


def run_all_baselines(
    model: "ISACSystemModel",
    target_angle_deg: float = 90.0,
) -> dict:
    """
    Run all baseline schemes.

    Parameters
    ----------
    model : ISACSystemModel
        System model
    target_angle_deg : float
        Target angle

    Returns
    -------
    dict
        Results for each baseline scheme
    """
    results = {}

    # EMax baseline
    emax = EMaxBaseline(model)
    results["EMax"] = emax.solve(target_angle_deg)

    # Fixed beam baseline (multiple sensing fractions)
    fixbeam = FixBeamBaseline(model)
    for frac in [0.0, 0.3, 0.5, 0.7, 1.0]:
        key = f"FixBeam_{frac:.0%}"
        results[key] = fixbeam.solve(target_angle_deg, sensing_fraction=frac)

    # Random baseline
    random_bl = RandomBaseline(model)
    results["Random"] = random_bl.solve(target_angle_deg)

    return results
