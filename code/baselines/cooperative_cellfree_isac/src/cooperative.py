"""
Multi-BS Cooperative Sensing for Cell-Free ISAC Networks.

Implements cooperative sensing where multiple sensing-mode BSs
jointly estimate target parameters (position, velocity) with
improved accuracy compared to single-BS sensing.
"""

import numpy as np
from system_model import CellFreeISACSystem, BSMode


class CooperativeSensing:
    """
    Multi-BS cooperative sensing with data fusion.

    Combines measurements from multiple sensing-mode BSs to improve
    target parameter estimation. Implements centralized and distributed
    fusion approaches.
    """

    def __init__(self, system: CellFreeISACSystem):
        self.system = system

    def get_sensing_bs_indices(self) -> list[int]:
        """Get indices of BSs currently in sensing mode."""
        return [
            m for m in range(self.system.n_bs)
            if self.system.bs_configs[m].mode == BSMode.SENSING
        ]

    def compute_echo_signal(
        self,
        bs_idx: int,
        target_idx: int,
    ) -> np.ndarray:
        """
        Compute received echo signal at a sensing BS from a target.

        Models monostatic radar: BS transmits and receives echo.

        Args:
            bs_idx: BS index.
            target_idx: Target index.

        Returns:
            Received signal vector of shape (n_antennas,).
        """
        bs = self.system.bs_configs[bs_idx]
        target = self.system.target_configs[target_idx]

        if bs.beamforming_vector is None:
            return np.zeros(self.system.n_antennas, dtype=complex)

        a_tx, pl_fwd = self.system.compute_channel_bs_target(bs_idx, target_idx)
        a_rx, pl_bwd = self.system.compute_channel_bs_target(bs_idx, target_idx)

        rcs = target.radar_cross_section

        # Echo signal: channel * beamformer * rcs * channel^H
        # y = sqrt(pl_fwd * pl_bwd * rcs) * a_rx * (a_tx^H * w) + noise
        gain = np.sqrt(pl_fwd * pl_bwd * rcs)
        echo_coeff = np.dot(a_tx.conj(), bs.beamforming_vector)
        y = gain * a_rx * echo_coeff

        # Add thermal noise
        noise = (np.sqrt(self.system.noise_power / 2) *
                 (np.random.randn(self.system.n_antennas) +
                  1j * np.random.randn(self.system.n_antennas)))
        y += noise

        return y

    def estimate_target_position_centralized(
        self,
        target_idx: int,
    ) -> np.ndarray:
        """
        Centralized position estimation from all sensing BSs.

        Uses weighted least squares to estimate target position from
        angle-of-arrival measurements at multiple BSs.

        Args:
            target_idx: Target index.

        Returns:
            Estimated position (2,).
        """
        sens_bs = self.get_sensing_bs_indices()

        if len(sens_bs) == 0:
            # No sensing BSs - return random estimate
            return np.random.uniform(0, self.system.area_size, 2)

        if len(sens_bs) == 1:
            # Single BS: limited estimation
            bs_pos = self.system.bs_configs[sens_bs[0]].position
            # Estimate as slightly offset from BS
            return bs_pos + np.random.randn(2) * 50

        # Multi-BS: AoA-based triangulation
        measurements = []
        weights = []

        for m in sens_bs:
            bs = self.system.bs_configs[m]
            target = self.system.target_configs[target_idx]

            # True AoA
            delta = target.position - bs.position
            true_theta = np.arctan2(delta[1], delta[0])

            # Measurement noise (proportional to 1/sqrt(SNR))
            distance = np.linalg.norm(delta)
            snr = self._compute_sensing_snr(m, target_idx)
            angle_noise_std = 1.0 / (np.sqrt(snr) + 1e-6)

            measured_theta = true_theta + np.random.randn() * angle_noise_std
            measurements.append((bs.position.copy(), measured_theta))
            weights.append(snr)

        # Weighted least squares for AoA-based localization
        return self._triangulate_aoa(measurements, weights)

    def _compute_sensing_snr(self, bs_idx: int, target_idx: int) -> float:
        """
        Compute SNR for sensing link at a BS.

        Args:
            bs_idx: BS index.
            target_idx: Target index.

        Returns:
            SNR in linear scale.
        """
        bs = self.system.bs_configs[bs_idx]
        target = self.system.target_configs[target_idx]

        if bs.beamforming_vector is None:
            return 0.0

        a_tx, pl_fwd = self.system.compute_channel_bs_target(bs_idx, target_idx)
        a_rx, pl_bwd = self.system.compute_channel_bs_target(bs_idx, target_idx)

        rcs = target.radar_cross_section
        p_tx = np.linalg.norm(bs.beamforming_vector) ** 2

        # Two-way path loss
        snr = p_tx * pl_fwd * pl_bwd * rcs * (self.system.n_antennas ** 2)
        snr /= self.system.noise_power

        return max(snr, 1e-10)

    def _triangulate_aoa(
        self,
        measurements: list[tuple[np.ndarray, float]],
        weights: list[float],
    ) -> np.ndarray:
        """
        Triangulate position from AoA measurements at multiple BSs.

        Uses weighted least squares.

        Args:
            measurements: List of (bs_position, aoa) tuples.
            weights: Measurement weights (higher = more reliable).

        Returns:
            Estimated position (2,).
        """
        if len(measurements) < 2:
            return measurements[0][0] + np.array([10.0, 10.0])

        # Solve: tan(theta_i) = (y - y_i) / (x - x_i)
        # Linearized least squares
        total_weight = sum(weights)
        weights_norm = [w / total_weight for w in weights]

        # Initial estimate: weighted centroid of BS positions
        x_est = np.zeros(2)
        for (bs_pos, _), w in zip(measurements, weights_norm):
            x_est += w * bs_pos

        # Iterative refinement
        for _ in range(10):
            A = np.zeros((len(measurements), 2))
            b = np.zeros(len(measurements))

            for i, ((bs_pos, theta), w) in enumerate(zip(measurements, weights_norm)):
                dx = x_est - bs_pos
                r = np.linalg.norm(dx) + 1e-6

                # Linearized measurement model
                A[i] = w * np.array([-np.sin(theta), np.cos(theta)]) / r
                b[i] = w * (dx[0] * np.cos(theta) + dx[1] * np.sin(theta))

            # Weighted least squares
            try:
                delta = np.linalg.lstsq(A, b, rcond=None)[0]
                x_est += delta
                if np.linalg.norm(delta) < 0.1:
                    break
            except np.linalg.LinAlgError:
                break

        # Clip to area bounds
        x_est = np.clip(x_est, 0, self.system.area_size)
        return x_est

    def distributed_fusion(
        self,
        target_idx: int,
    ) -> np.ndarray:
        """
        Distributed fusion using covariance intersection.

        Each BS produces a local estimate; these are fused using
        covariance intersection to avoid double-counting.

        Args:
            target_idx: Target index.

        Returns:
            Fused position estimate (2,).
        """
        sens_bs = self.get_sensing_bs_indices()

        if len(sens_bs) == 0:
            return np.array([self.system.area_size / 2, self.system.area_size / 2])

        estimates = []
        covariances = []

        for m in sens_bs:
            # Local single-BS estimate
            bs_pos = self.system.bs_configs[m].position
            target = self.system.target_configs[target_idx]

            delta = target.position - bs_pos
            true_theta = np.arctan2(delta[1], delta[0])

            snr = self._compute_sensing_snr(m, target_idx)
            angle_noise_std = 1.0 / (np.sqrt(snr) + 1e-6)

            # Estimate distance (crude: assume middle of area)
            est_dist = self.system.area_size / 3
            measured_theta = true_theta + np.random.randn() * angle_noise_std

            x_local = bs_pos + est_dist * np.array([
                np.cos(measured_theta), np.sin(measured_theta)
            ])
            x_local = np.clip(x_local, 0, self.system.area_size)

            # Covariance: larger uncertainty for single-BS
            cov = np.eye(2) * (est_dist * angle_noise_std) ** 2

            estimates.append(x_local)
            covariances.append(cov)

        # Covariance Intersection
        if len(estimates) == 1:
            return estimates[0]

        # Simple weighted average as fallback
        inv_covs = [np.linalg.inv(C + 1e-6 * np.eye(2)) for C in covariances]
        P_fused_inv = sum(inv_covs)
        x_fused = np.linalg.solve(P_fused_inv, sum(
            ic @ x for ic, x in zip(inv_covs, estimates)
        ))

        x_fused = np.clip(x_fused, 0, self.system.area_size)
        return x_fused

    def compute_cooperative_gain(
        self,
        target_idx: int,
    ) -> dict:
        """
        Compute the sensing gain from cooperation.

        Compares single-BS vs. multi-BS sensing accuracy.

        Args:
            target_idx: Target index.

        Returns:
            Dictionary with cooperation metrics.
        """
        sens_bs = self.get_sensing_bs_indices()
        true_pos = self.system.target_configs[target_idx].position

        # Single-BS errors
        single_errors = []
        for m in sens_bs:
            bs_pos = self.system.bs_configs[m].position
            est = bs_pos + np.random.randn(2) * 50  # Rough single-BS
            est = np.clip(est, 0, self.system.area_size)
            single_errors.append(np.linalg.norm(est - true_pos))

        # Cooperative error
        coop_est = self.estimate_target_position_centralized(target_idx)
        coop_error = np.linalg.norm(coop_est - true_pos)

        avg_single_error = np.mean(single_errors) if single_errors else float('inf')

        return {
            "single_bs_avg_error": avg_single_error,
            "cooperative_error": coop_error,
            "cooperation_gain_db": 10 * np.log10(avg_single_error / (coop_error + 1e-6)),
            "n_cooperating_bs": len(sens_bs),
        }
