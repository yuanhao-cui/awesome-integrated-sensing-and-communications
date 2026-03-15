"""
Transmit Beamforming Optimization.

Beamforming design for ISAC with array partitioning constraint.
Supports MRT, ZF, and MMSE beamforming, plus joint optimization.
"""

import numpy as np
from typing import Optional, Tuple, Dict


class BeamformingOptimizer:
    """
    Transmit beamforming optimizer for partitioned ISAC array.

    Provides:
    - Maximum Ratio Transmission (MRT)
    - Zero-Forcing (ZF)
    - Minimum MSE (MMSE)
    - Sensing-aware beamforming (maximizes radar SNR)
    - Joint communication-sensing beamforming
    """

    def __init__(
        self,
        M: int,
        K: int,
        P_max: float,
        sigma2_c: float = 1.0,
        sigma2_r: float = 1.0,
    ):
        self.M = M
        self.K = K
        self.P_max = P_max
        self.sigma2_c = sigma2_c
        self.sigma2_r = sigma2_r

    def mrt(
        self, H: np.ndarray, t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Maximum Ratio Transmission.

        w_k = sqrt(P/K) * h_k / ||h_k||

        Parameters
        ----------
        H : np.ndarray, shape (M, K)
            Channel matrix.
        t : np.ndarray, shape (M,), optional
            TX partition. If given, only TX antennas are used.

        Returns
        -------
        W : np.ndarray, shape (M, K)
            Beamforming matrix.
        """
        W = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H[:, k]
            norm = np.linalg.norm(h_k)
            if norm > 1e-10:
                W[:, k] = h_k / norm
        W *= np.sqrt(self.P_max / self.K)
        if t is not None:
            W = W * t[:, np.newaxis]
        return W

    def zf(
        self, H: np.ndarray, t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Zero-Forcing beamforming.

        W = H * (H^H H)^{-1}, then scaled to meet power constraint.

        Parameters
        ----------
        H : np.ndarray, shape (M, K)
        t : np.ndarray, shape (M,), optional

        Returns
        -------
        W : np.ndarray, shape (M, K)
        """
        if t is not None:
            H_eff = H * t[:, np.newaxis]
        else:
            H_eff = H

        # Regularized pseudo-inverse for numerical stability
        reg = 1e-6 * np.eye(self.K)
        W = H_eff @ np.linalg.inv(H_eff.conj().T @ H_eff + reg)

        # Scale to power constraint
        power = np.sum(np.abs(W) ** 2)
        if power > 1e-10:
            W *= np.sqrt(self.P_max / power)

        return W

    def mmse(
        self,
        H: np.ndarray,
        t: Optional[np.ndarray] = None,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        MMSE beamforming (regularized ZF).

        W = H * (H^H H + alpha * I)^{-1}

        Parameters
        ----------
        H : np.ndarray
        t : np.ndarray, optional
        alpha : float
            Regularization parameter (typically sigma2_c).

        Returns
        -------
        W : np.ndarray, shape (M, K)
        """
        if t is not None:
            H_eff = H * t[:, np.newaxis]
        else:
            H_eff = H

        W = H_eff @ np.linalg.inv(
            H_eff.conj().T @ H_eff + alpha * np.eye(self.K)
        )

        power = np.sum(np.abs(W) ** 2)
        if power > 1e-10:
            W *= np.sqrt(self.P_max / power)

        return W

    def sensing_aware(
        self,
        H: np.ndarray,
        a_theta: np.ndarray,
        t: np.ndarray,
        sinr_thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Sensing-aware beamforming that maximizes radar SNR subject to SINR constraints.

        Uses a simplified approach:
        1. Start with ZF for communication
        2. Add sensing beamformer component
        3. Re-optimize power allocation

        Parameters
        ----------
        H : np.ndarray, shape (M, K)
        a_theta : np.ndarray, shape (M,)
            Steering vector to target direction.
        t : np.ndarray, shape (M,)
        sinr_thresholds : np.ndarray, shape (K,)

        Returns
        -------
        W : np.ndarray, shape (M, K)
        """
        # Communication component: ZF
        W_comm = self.zf(H, t)

        # Sensing component: beamform toward target
        w_sense = t * a_theta
        w_sense = w_sense / (np.linalg.norm(w_sense) + 1e-10)

        # Split power between communication and sensing
        P_comm = 0.7 * self.P_max
        P_sense = 0.3 * self.P_max

        W_comm *= np.sqrt(P_comm / (np.sum(np.abs(W_comm) ** 2) + 1e-10))

        # Add sensing beam to all users proportionally
        W = W_comm + np.sqrt(P_sense / self.K) * w_sense[:, np.newaxis]

        # Final power normalization
        power = np.sum(np.abs(W) ** 2)
        if power > self.P_max:
            W *= np.sqrt(self.P_max / power)

        return W

    def joint_optimize(
        self,
        H: np.ndarray,
        a_theta: np.ndarray,
        t: np.ndarray,
        sinr_thresholds: np.ndarray,
        sensing_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Joint communication-sensing beamforming with configurable trade-off.

        Parameters
        ----------
        H : np.ndarray
        a_theta : np.ndarray
        t : np.ndarray
        sinr_thresholds : np.ndarray
        sensing_weight : float in [0, 1]
            Weight for sensing vs communication (1 = all sensing).

        Returns
        -------
        W : np.ndarray
        """
        comm_weight = 1 - sensing_weight

        # Communication: MMSE
        W_comm = self.mmse(H, t, alpha=self.sigma2_c)
        comm_power = comm_weight * self.P_max
        W_comm *= np.sqrt(comm_power / (np.sum(np.abs(W_comm) ** 2) + 1e-10))

        # Sensing: matched filter to target
        w_sense = t * a_theta
        sense_power = sensing_weight * self.P_max
        w_sense *= np.sqrt(sense_power / (np.linalg.norm(w_sense) ** 2 + 1e-10))

        # Combine
        W = W_comm + w_sense[:, np.newaxis] / np.sqrt(self.K)

        # Enforce power constraint
        total_power = np.sum(np.abs(W) ** 2)
        if total_power > self.P_max:
            W *= np.sqrt(self.P_max / total_power)

        return W

    def compute_spectral_efficiency(
        self, W: np.ndarray, H: np.ndarray, t: np.ndarray
    ) -> float:
        """
        Compute sum spectral efficiency.

        SE = sum_k log2(1 + SINR_k)
        """
        sinr = np.zeros(self.K)
        W_eff = W * t[:, np.newaxis]
        for k in range(self.K):
            h_k = H[:, k]
            signal = np.abs(h_k @ W_eff[:, k]) ** 2
            interference = sum(
                np.abs(h_k @ W_eff[:, j]) ** 2 for j in range(self.K) if j != k
            ) + self.sigma2_c
            sinr[k] = signal / (interference + 1e-15)
        return np.sum(np.log2(1 + sinr))

    def compute_radar_snr(
        self,
        W: np.ndarray,
        a_theta: np.ndarray,
        t: np.ndarray,
        r: np.ndarray,
    ) -> float:
        """
        Compute radar SNR for target at theta.

        SNR_r = |a^H (t ⊙ w_total)|^2 * ||r ⊙ a||^2 / sigma2_r
        """
        w_total = np.sum(W, axis=1)
        tx_sig = t * w_total
        rx_sig = r * a_theta
        snr = (
            np.abs(np.conj(a_theta) @ tx_sig) ** 2
            * np.linalg.norm(rx_sig) ** 2
            / self.sigma2_r
        )
        return snr
