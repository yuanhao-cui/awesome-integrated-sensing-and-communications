"""
ISAC System Model
=================

Implements the system model for Integrated Sensing and Communications (ISAC)
with energy-efficient beamforming design.

System model (Section II):
    - ISAC BS with M transmit antennas
    - K single-antenna communication users
    - N receive antennas for sensing (point-like target at angle θ)
    - Beamforming matrix W = [w_1, ..., w_K]

Key equations:
    - SINR (Eq. 2): SINR_k = |h_k^H w_k|² / (σ_c² + Σ_{j≠k} |h_k^H w_j|²)
    - Communication EE (Eq. 4): EE_C = Σ_k log₂(1+SINR_k) / ((1/ε)Σ_k||w_k||² + P₀)
    - CRB (Eq. 10): Complex formula with steering vectors

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import numpy as np
from typing import Optional, Tuple


class ISACSystemModel:
    """
    ISAC system model for energy-efficient beamforming.

    Parameters
    ----------
    M : int
        Number of transmit antennas at BS (default: 16)
    K : int
        Number of single-antenna communication users (default: 4)
    N : int
        Number of receive antennas for sensing (default: 20)
    P_max_dbm : float
        Maximum transmit power in dBm (default: 30 dBm)
    P0_dbm : float
        Circuit power in dBm (default: 33 dBm)
    epsilon : float
        Power amplifier efficiency (default: 0.35)
    sigma_c_db : float
        Noise power for communication users in dB (default: -80 dB)
    sigma_s_db : float
        Noise power for sensing in dB (default: -80 dB)
    L : int
        Frame length (default: 30)
    wavelength : float
        Signal wavelength (default: 1.0, normalized)
    d : float
        Antenna spacing (default: 0.5 wavelength)

    Attributes
    ----------
    P_max : float
        Maximum transmit power (linear)
    P0 : float
        Circuit power (linear)
    sigma_c2 : float
        Communication noise power (linear)
    sigma_s2 : float
        Sensing noise power (linear)
    H : np.ndarray
        Channel matrix (K x M), Rayleigh fading
    """

    def __init__(
        self,
        M: int = 16,
        K: int = 4,
        N: int = 20,
        P_max_dbm: float = 30.0,
        P0_dbm: float = 33.0,
        epsilon: float = 0.35,
        sigma_c_db: float = -80.0,
        sigma_s_db: float = -80.0,
        L: int = 30,
        wavelength: float = 1.0,
        d: float = 0.5,
        seed: Optional[int] = None,
    ):
        """Initialize ISAC system model with parameters from Section VI."""
        self.M = M
        self.K = K
        self.N = N
        self.L = L
        self.epsilon = epsilon
        self.wavelength = wavelength
        self.d = d

        # Convert dB to linear
        self.P_max = 10 ** (P_max_dbm / 10) / 1000  # dBm to Watts
        self.P0 = 10 ** (P0_dbm / 10) / 1000
        self.sigma_c2 = 10 ** (sigma_c_db / 10)
        self.sigma_s2 = 10 ** (sigma_s_db / 10)

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Generate channel matrix (Rayleigh fading, Eq. model in Section VI)
        # h_k ~ CN(0, I_M), normalized
        self.H = (
            self.rng.standard_normal((K, M))
            + 1j * self.rng.standard_normal((K, M))
        ) / np.sqrt(2)

    def steering_vector_tx(self, theta_rad: float) -> np.ndarray:
        """
        Compute transmit steering vector a_t(θ).

        For ULA with M antennas, element m:
            a_t(θ)_m = exp(j * 2π/λ * d * m * sin(θ))

        Parameters
        ----------
        theta_rad : float
            Target angle in radians

        Returns
        -------
        np.ndarray
            Transmit steering vector (M,) complex
        """
        m_indices = np.arange(self.M)
        phase = 2 * np.pi / self.wavelength * self.d * m_indices * np.sin(theta_rad)
        return np.exp(1j * phase)

    def steering_vector_rx(self, theta_rad: float) -> np.ndarray:
        """
        Compute receive steering vector a_r(θ).

        For ULA with N antennas, element n:
            a_r(θ)_n = exp(j * 2π/λ * d * n * sin(θ))

        Parameters
        ----------
        theta_rad : float
            Target angle in radians

        Returns
        -------
        np.ndarray
            Receive steering vector (N,) complex
        """
        n_indices = np.arange(self.N)
        phase = 2 * np.pi / self.wavelength * self.d * n_indices * np.sin(theta_rad)
        return np.exp(1j * phase)

    def get_channel(self, k: int) -> np.ndarray:
        """
        Get channel vector for user k.

        Parameters
        ----------
        k : int
            User index (0-indexed)

        Returns
        -------
        np.ndarray
            Channel vector h_k (M,) complex
        """
        return self.H[k, :]

    def compute_sinr(self, k: int, W: np.ndarray) -> float:
        """
        Compute SINR for user k (Eq. 2).

        SINR_k = |h_k^H w_k|² / (σ_c² + Σ_{j≠k} |h_k^H w_j|²)

        Parameters
        ----------
        k : int
            User index (0-indexed)
        W : np.ndarray
            Beamforming matrix (M x K), column k is w_k

        Returns
        -------
        float
            SINR for user k
        """
        h_k = self.get_channel(k)
        signal = np.abs(h_k.conj() @ W[:, k]) ** 2
        interference = sum(
            np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(self.K) if j != k
        )
        return signal / (self.sigma_c2 + interference)

    def compute_sinr_vector(self, W: np.ndarray) -> np.ndarray:
        """
        Compute SINR for all users.

        Parameters
        ----------
        W : np.ndarray
            Beamforming matrix (M x K)

        Returns
        -------
        np.ndarray
            SINR values for each user (K,)
        """
        return np.array([self.compute_sinr(k, W) for k in range(self.K)])

    def compute_total_power(self, W: np.ndarray) -> float:
        """
        Compute total transmit power Σ_k ||w_k||².

        Parameters
        ----------
        W : np.ndarray
            Beamforming matrix (M x K)

        Returns
        -------
        float
            Total transmit power
        """
        return np.sum(np.abs(W) ** 2)

    def regenerate_channels(self, seed: Optional[int] = None):
        """
        Regenerate channel matrix with new random realization.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.H = (
            self.rng.standard_normal((self.K, self.M))
            + 1j * self.rng.standard_normal((self.K, self.M))
        ) / np.sqrt(2)

    def get_csi(self) -> np.ndarray:
        """
        Get full channel state information.

        Returns
        -------
        np.ndarray
            Channel matrix H (K x M) complex
        """
        return self.H.copy()
