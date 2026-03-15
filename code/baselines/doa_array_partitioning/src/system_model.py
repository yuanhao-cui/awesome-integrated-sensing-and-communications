"""
ISAC System Model with Partitionable Array.

Monostatic ISAC base station where each antenna can be dynamically
partitioned as TX or RX. Supports multi-user MISO communications
and radar sensing for DOA estimation.
"""

import numpy as np
from typing import Optional, Tuple, List


class ISACSystem:
    """
    Monostatic ISAC system with M partitionable antennas.

    Parameters
    ----------
    M : int
        Total number of antennas.
    K : int
        Number of communication users.
    P_max : float
        Maximum transmit power (linear scale).
    sigma2_c : float
        Communication noise power.
    sigma2_r : float
        Radar noise power.
    d : float
        Antenna spacing (wavelengths).
    """

    def __init__(
        self,
        M: int = 16,
        K: int = 3,
        P_max: float = 10.0,
        sigma2_c: float = 1.0,
        sigma2_r: float = 1.0,
        d: float = 0.5,
    ):
        self.M = M
        self.K = K
        self.P_max = P_max
        self.sigma2_c = sigma2_c
        self.sigma2_r = sigma2_r
        self.d = d
        self.antenna_positions = np.arange(M) * d
        self._channel = None
        self._steering_vectors = {}

    def get_steering_vector(self, theta: float) -> np.ndarray:
        """
        Compute array steering vector for angle theta (radians).

        a(theta) = [1, exp(j*2*pi*d*sin(theta)), ..., exp(j*2*pi*(M-1)*d*sin(theta))]^T
        """
        if theta not in self._steering_vectors:
            phases = 2 * np.pi * self.antenna_positions * np.sin(theta)
            self._steering_vectors[theta] = np.exp(1j * phases)
        return self._steering_vectors[theta]

    def generate_channel(
        self,
        user_angles: Optional[np.ndarray] = None,
        fading_coeffs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate M x K communication channel matrix (LoS + NLoS).

        Parameters
        ----------
        user_angles : np.ndarray, shape (K,)
            User AoA in radians. Random if None.
        fading_coeffs : np.ndarray, shape (K,)
            Small-scale fading coefficients. Rayleigh if None.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        H : np.ndarray, shape (M, K)
            Channel matrix. Column k is the channel to user k.
        """
        rng = np.random.default_rng(seed)
        if user_angles is None:
            user_angles = rng.uniform(-np.pi / 2, np.pi / 2, self.K)
        if fading_coeffs is None:
            fading_coeffs = (
                rng.standard_normal(self.K) + 1j * rng.standard_normal(self.K)
            ) / np.sqrt(2)

        H = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            a = self.get_steering_vector(user_angles[k])
            H[:, k] = fading_coeffs[k] * a

        self._channel = H
        self._user_angles = user_angles
        return H

    @property
    def channel(self) -> np.ndarray:
        if self._channel is None:
            raise ValueError("Call generate_channel() first.")
        return self._channel

    @property
    def user_angles(self) -> np.ndarray:
        if not hasattr(self, "_user_angles") or self._user_angles is None:
            raise ValueError("Call generate_channel() first.")
        return self._user_angles

    def create_partition(
        self, tx_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binary TX/RX partition.

        Parameters
        ----------
        tx_indices : np.ndarray
            Indices of antennas assigned to TX.

        Returns
        -------
        t : np.ndarray, shape (M,)
            Binary vector: t[m] = 1 if antenna m is TX.
        r : np.ndarray, shape (M,)
            Binary vector: r[m] = 1 if antenna m is RX.
        """
        t = np.zeros(self.M)
        r = np.ones(self.M)
        t[tx_indices] = 1.0
        r[tx_indices] = 0.0
        return t, r

    def partition_from_continuous(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous relaxation [0,1] to binary partition.
        c > 0.5 → TX, c <= 0.5 → RX.
        """
        tx_indices = np.where(c > 0.5)[0]
        return self.create_partition(tx_indices)

    def compute_sinr(self, w: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Compute SINR for each user.

        Parameters
        ----------
        w : np.ndarray, shape (M, K)
            Beamforming matrix. Column k is beamformer for user k.
        H : np.ndarray, shape (M, K)
            Channel matrix.

        Returns
        -------
        sinr : np.ndarray, shape (K,)
            SINR per user.
        """
        sinr = np.zeros(self.K)
        for k in range(self.K):
            h_k = H[:, k]
            signal = np.abs(h_k @ w[:, k]) ** 2
            interference = 0.0
            for j in range(self.K):
                if j != k:
                    interference += np.abs(h_k @ w[:, j]) ** 2
            interference += self.sigma2_c
            sinr[k] = signal / interference
        return sinr

    def compute_sinr_with_partition(
        self, w: np.ndarray, H: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Compute SINR considering only TX antennas active.

        Parameters
        ----------
        w : np.ndarray, shape (M, K)
        H : np.ndarray, shape (M, K)
        t : np.ndarray, shape (M,)
            TX indicator vector.

        Returns
        -------
        sinr : np.ndarray, shape (K,)
        """
        # Apply partition: effective beamformer
        w_eff = w * t[:, np.newaxis]
        return self.compute_sinr(w_eff, H)

    def __repr__(self) -> str:
        return (
            f"ISACSystem(M={self.M}, K={self.K}, P_max={self.P_max:.1f}, "
            f"d={self.d})"
        )
