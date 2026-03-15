"""
Joint Beamforming Design for Cell-Free ISAC Networks.

Implements beamforming algorithms for both communication and sensing modes.
Communication-mode BSs use MMSE/max-SINR beamforming to serve users.
Sensing-mode BSs use beampattern-focused beamforming for target illumination.
"""

import numpy as np
from system_model import CellFreeISACSystem, BSMode


class BeamformingDesigner:
    """
    Joint beamforming designer for cooperative cell-free ISAC.

    Optimizes beamforming vectors for all BSs considering their modes,
    power constraints, and multi-user/sensing interference.
    """

    def __init__(self, system: CellFreeISACSystem):
        self.system = system

    def initialize_beamformers(self):
        """Initialize beamforming vectors for all BSs."""
        for m in range(self.system.n_bs):
            bf = np.random.randn(self.system.n_antennas) + \
                 1j * np.random.randn(self.system.n_antennas)
            bf /= np.linalg.norm(bf)
            # Scale to satisfy power constraint
            bf *= np.sqrt(self.system.bs_configs[m].p_max)
            self.system.bs_configs[m].beamforming_vector = bf

    def mmse_beamforming(
        self,
        bs_idx: int,
        user_idx: int,
        regularization: float = None,
    ) -> np.ndarray:
        """
        MMSE beamforming for communication-mode BS.

        Computes the MMSE precoder for user k at BS m.

        Args:
            bs_idx: BS index.
            user_idx: Target user index.
            regularization: Regularization parameter (default: noise_power).

        Returns:
            Beamforming vector of shape (n_antennas,).
        """
        if regularization is None:
            regularization = self.system.noise_power

        H = self.system.get_communication_channels()
        h_mk = H[user_idx, bs_idx]  # Channel from BS m to user k

        # Compute covariance of interference + noise
        R = regularization * np.eye(self.system.n_antennas, dtype=complex)
        for k2 in range(self.system.n_users):
            if k2 != user_idx:
                h_mk2 = H[k2, bs_idx]
                R += np.outer(h_mk2, h_mk2.conj())

        # MMSE: w = R^{-1} h
        w = np.linalg.solve(R, h_mk)

        # Normalize to power constraint
        p_max = self.system.bs_configs[bs_idx].p_max
        w *= np.sqrt(p_max) / (np.linalg.norm(w) + 1e-12)

        return w

    def max_sinr_beamforming(
        self,
        bs_idx: int,
        user_idx: int,
    ) -> np.ndarray:
        """
        Max-SINR beamforming for communication-mode BS.

        Optimizes to maximize SINR of the target user.

        Args:
            bs_idx: BS index.
            user_idx: Target user index.

        Returns:
            Beamforming vector of shape (n_antennas,).
        """
        H = self.system.get_communication_channels()
        h_mk = H[user_idx, bs_idx]

        # Interference covariance
        R = self.system.noise_power * np.eye(self.system.n_antennas, dtype=complex)
        for k2 in range(self.system.n_users):
            if k2 != user_idx:
                h_mk2 = H[k2, bs_idx]
                R += np.outer(h_mk2, h_mk2.conj())

        # Max-SINR: w = R^{-1} h / ||R^{-1} h||
        w = np.linalg.solve(R, h_mk)
        p_max = self.system.bs_configs[bs_idx].p_max
        w *= np.sqrt(p_max) / (np.linalg.norm(w) + 1e-12)

        return w

    def sensing_beamforming(
        self,
        bs_idx: int,
        target_idx: int = 0,
    ) -> np.ndarray:
        """
        Sensing beamforming for sensing-mode BS.

        Steers beam toward the target using MRT (Matched Response Transmission).

        Args:
            bs_idx: BS index.
            target_idx: Target index to steer toward.

        Returns:
            Beamforming vector of shape (n_antennas,).
        """
        sens_channels = self.system.get_sensing_channels()
        a, pl = sens_channels[target_idx][bs_idx]

        # MRT: w = a* (conjugate of array response)
        w = a.conj()

        p_max = self.system.bs_configs[bs_idx].p_max
        w *= np.sqrt(p_max) / (np.linalg.norm(w) + 1e-12)

        return w

    def cooperative_sensing_beamforming(
        self,
        bs_idx: int,
    ) -> np.ndarray:
        """
        Cooperative sensing beamforming combining all targets.

        Weights beamforming toward multiple targets proportionally
        to their RCS.

        Args:
            bs_idx: BS index.

        Returns:
            Beamforming vector of shape (n_antennas,).
        """
        sens_channels = self.system.get_sensing_channels()
        w = np.zeros(self.system.n_antennas, dtype=complex)

        for t in range(self.system.n_targets):
            a, pl = sens_channels[t][bs_idx]
            rcs = self.system.target_configs[t].radar_cross_section
            weight = np.sqrt(rcs) * pl
            w += weight * a.conj()

        p_max = self.system.bs_configs[bs_idx].p_max
        if np.linalg.norm(w) > 1e-12:
            w *= np.sqrt(p_max) / np.linalg.norm(w)
        else:
            # Fallback: random beamformer
            w = np.random.randn(self.system.n_antennas) + \
                1j * np.random.randn(self.system.n_antennas)
            w *= np.sqrt(p_max) / np.linalg.norm(w)

        return w

    def design_all_beamformers(
        self,
        comm_method: str = "mmse",
        sens_method: str = "cooperative",
    ) -> dict:
        """
        Design beamformers for all BSs based on their current modes.

        Args:
            comm_method: Method for communication BSs ('mmse', 'max_sinr').
            sens_method: Method for sensing BSs ('single_target', 'cooperative').

        Returns:
            Dictionary with beamforming vectors and metadata.
        """
        beamformers = {}
        total_comm_power = 0.0
        total_sens_power = 0.0

        for m in range(self.system.n_bs):
            bs = self.system.bs_configs[m]

            if bs.mode == BSMode.COMMUNICATION:
                # Assign BS to serve user m % n_users (round-robin)
                user_idx = m % self.system.n_users

                if comm_method == "mmse":
                    w = self.mmse_beamforming(m, user_idx)
                elif comm_method == "max_sinr":
                    w = self.max_sinr_beamforming(m, user_idx)
                else:
                    raise ValueError(f"Unknown comm method: {comm_method}")

                total_comm_power += np.linalg.norm(w) ** 2

            else:  # SENSING
                if sens_method == "single_target":
                    w = self.sensing_beamforming(m, target_idx=m % max(1, self.system.n_targets))
                elif sens_method == "cooperative":
                    w = self.cooperative_sensing_beamforming(m)
                else:
                    raise ValueError(f"Unknown sens method: {sens_method}")

                total_sens_power += np.linalg.norm(w) ** 2

            bs.beamforming_vector = w
            beamformers[m] = w

        return {
            "beamformers": beamformers,
            "total_comm_power": total_comm_power,
            "total_sens_power": total_sens_power,
            "n_comm_bs": sum(1 for bs in self.system.bs_configs if bs.mode == BSMode.COMMUNICATION),
            "n_sens_bs": sum(1 for bs in self.system.bs_configs if bs.mode == BSMode.SENSING),
        }

    def verify_power_constraints(self) -> tuple[bool, list[float]]:
        """
        Verify all BSs satisfy power constraints.

        Returns:
            Tuple of (all_satisfied, list_of_power_ratios).
        """
        satisfied = True
        ratios = []

        for m in range(self.system.n_bs):
            bs = self.system.bs_configs[m]
            if bs.beamforming_vector is None:
                ratios.append(0.0)
                continue

            power = np.linalg.norm(bs.beamforming_vector) ** 2
            p_max = bs.p_max
            ratio = power / p_max
            ratios.append(ratio)

            if power > p_max * (1 + 1e-6):
                satisfied = False

        return satisfied, ratios

    def equal_power_beamforming(self) -> dict:
        """
        Simple equal-power random beamforming baseline.

        Returns:
            Dictionary with beamforming vectors.
        """
        beamformers = {}
        for m in range(self.system.n_bs):
            w = np.random.randn(self.system.n_antennas) + \
                1j * np.random.randn(self.system.n_antennas)
            w *= np.sqrt(self.system.bs_configs[m].p_max) / np.linalg.norm(w)
            self.system.bs_configs[m].beamforming_vector = w
            beamformers[m] = w

        return {"beamformers": beamformers}
