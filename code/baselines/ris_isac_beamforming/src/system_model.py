"""RIS-ISAC system model.

Defines the physical layer model including:
- Received signal at users (communication)
- Reflected signal for radar sensing
- SINR, SNR, and rate computations

Reference: Section III.A, Eq. (4)-(7) in
    Rang Liu et al., IEEE TWC 2024, arXiv:2301.11134.
"""

import numpy as np
from typing import Optional
from .channel_model import RISChannelModel


class RIS_ISAC_System:
    """RIS-ISAC system model for joint communication and sensing.

    Models a multi-antenna BS assisted by a passive RIS serving K
    single-antenna users while performing radar sensing of a target.

    Attributes:
        M: BS antennas.
        K: Single-antenna users.
        L: RIS elements.
        P_max: Maximum transmit power (mW).
        noise_power: Noise variance (mW).
        sinr_thresh_dB: SINR threshold in dB.
        channels: Dictionary of channel matrices.
        theta: RIS phase shift vector (L,) with |theta_l| = 1.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 2,
        L: int = 30,
        P_max: float = 10e-3,
        noise_power: float = 3.98e-12,
        sinr_thresh_dB: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize RIS-ISAC system.

        Args:
            M: Number of BS antennas.
            K: Number of single-antenna users.
            L: Number of RIS elements.
            P_max: Maximum transmit power in mW (Table I: 10 mW).
            noise_power: Noise variance in mW (Table I: 3.98e-12 mW).
            sinr_thresh_dB: SINR threshold in dB (Table I: 10 dB).
            seed: Random seed for channel generation.
        """
        self.M = M
        self.K = K
        self.L = L
        self.P_max = P_max
        self.noise_power = noise_power
        self.sinr_thresh_dB = sinr_thresh_dB
        self.sinr_thresh = 10 ** (sinr_thresh_dB / 10)

        # Generate channels
        channel_model = RISChannelModel(M=M, K=K, L=L, seed=seed)
        self.channels = channel_model.generate_all_channels()

        # Initialize RIS phases randomly
        self.theta = np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, L))

    def ris_diagonal_matrix(self) -> np.ndarray:
        """Construct RIS diagonal matrix Θ = diag(θ).

        See Eq. (1): Θ = diag(θ_1, ..., θ_L) with |θ_l| = 1.

        Returns:
            Diagonal matrix Θ of shape (L, L).
        """
        return np.diag(self.theta)

    def effective_channel(self, user_idx: int) -> np.ndarray:
        """Compute effective channel for user k.

        The effective channel combines direct and RIS-reflected paths:
            h_k^eff = g_k^H Θ H_BR + h_{d,k}^H

        See Eq. (2): h_k^H = g_k^H Θ H_BR + h_{d,k}^H

        Args:
            user_idx: User index k (0-based).

        Returns:
            Effective channel vector of shape (M,).
        """
        H_BR = self.channels["H_BR"]  # (L, M)
        G = self.channels["G"]  # (K, L)
        h_d = self.channels["h_d"]  # (K, M)

        Theta = self.ris_diagonal_matrix()
        h_eff = G[user_idx, :] @ Theta @ H_BR + h_d[user_idx, :]
        return h_eff

    def compute_sinr(self, w_k: np.ndarray, W_interf: np.ndarray) -> float:
        """Compute SINR for a user given beamforming vectors.

        SINR_k = |h_k^H w_k|^2 / (Σ_{j≠k} |h_k^H w_j|^2 + σ^2)

        See Eq. (5).

        Args:
            w_k: Beamforming vector for user k (M,).
            W_interf: Stacked interference beamformers (M, K-1).

        Returns:
            SINR value (linear scale).
        """
        # This is a helper; the actual SINR computation uses effective channel
        # Called by the solver with the effective channel
        pass

    def compute_snr_sensing(self, w: np.ndarray) -> float:
        """Compute radar sensing SNR.

        SNR_s = |h_s^H w|^2 / σ^2

        where h_s is the sensing channel through RIS.

        See Eq. (6).

        Args:
            w: Total beamforming vector (M,) = Σ_k w_k.

        Returns:
            Sensing SNR (linear scale).
        """
        a_bs = self.channels["a_bs"]  # (M,)
        a_ris = self.channels["a_ris"]  # (L,)
        H_BR = self.channels["H_BR"]  # (L, M)
        Theta = self.ris_diagonal_matrix()

        # Round-trip sensing channel
        h_s = a_bs + (a_ris.T @ Theta @ H_BR)
        snr = np.abs(h_s.conj() @ w) ** 2 / self.noise_power
        return snr

    def compute_sum_rate(self, W: np.ndarray) -> float:
        """Compute sum rate over all users.

        R_k = log2(1 + SINR_k), Sum rate = Σ_k R_k

        See Eq. (4).

        Args:
            W: Beamforming matrix of shape (M, K), columns are w_k.

        Returns:
            Sum rate in bits/s/Hz.
        """
        H_BR = self.channels["H_BR"]
        G = self.channels["G"]
        h_d = self.channels["h_d"]
        Theta = self.ris_diagonal_matrix()

        sum_rate = 0.0
        for k in range(self.K):
            h_k = G[k, :] @ Theta @ H_BR + h_d[k, :]  # effective channel (M,)

            signal_power = np.abs(h_k.conj() @ W[:, k]) ** 2
            interference = 0.0
            for j in range(self.K):
                if j != k:
                    interference += np.abs(h_k.conj() @ W[:, j]) ** 2
            sinr_k = signal_power / (interference + self.noise_power)
            sum_rate += np.log2(1 + sinr_k)

        return sum_rate

    def set_ris_phases(self, theta: np.ndarray):
        """Set RIS phase shifts with unit-modulus enforcement.

        Args:
            theta: Complex phase vector (L,). Will be normalized to |θ_l| = 1.
        """
        self.theta = theta / np.abs(theta)

    def reset_channels(self, seed: Optional[int] = None):
        """Regenerate all channel matrices.

        Args:
            seed: New random seed (None for random).
        """
        cm = RISChannelModel(M=self.M, K=self.K, L=self.L, seed=seed)
        self.channels = cm.generate_all_channels()
