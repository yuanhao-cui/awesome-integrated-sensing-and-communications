"""
Delay-Doppler Ambiguity Function
=================================
Computes the ambiguity function for Zak-OTFS waveforms.

The ambiguity function characterizes the sensing resolution in delay and
Doppler. For a waveform s(t), the ambiguity function is:

    A(tau, nu) = integral s(t) * s*(t - tau) * exp(j*2*pi*nu*t) dt

For a point pulsone at origin, the self-ambiguity is (Eq. 14 in [1]):
    A_b(tau, nu) = delta_Lambda(tau, nu)
    i.e., supported only on the period lattice Lambda = {(k*T_tau, l/T_nu)}

For the spread pulsone, the cross-ambiguity has peaks on the rotated
lattice Lambda*.

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import Tuple, Optional


class AmbiguityFunction:
    """
    Delay-Doppler ambiguity function calculator.

    Parameters
    ----------
    N : int
        Number of delay bins.
    M : int
        Number of Doppler bins.
    T_tau : float
        Delay period (seconds).
    T_nu : float
        Doppler period (Hz).
    """

    def __init__(self, N: int, M: int, T_tau: float = 1.0, T_nu: float = 1.0):
        self.N = N
        self.M = M
        self.T_tau = T_tau
        self.T_nu = T_nu
        self.NM = N * M

    def self_ambiguity(self, s: np.ndarray) -> np.ndarray:
        """
        Compute the self-ambiguity function of a time-domain signal.

        A(tau_k, nu_l) = sum_n s[n] * s*[(n - k) mod NM]
                         * exp(j*2*pi*l*n / M)

        where tau_k = k * T_tau / N and nu_l = l * T_nu / M.

        Parameters
        ----------
        s : np.ndarray, shape (N*M,)
            Time-domain signal.

        Returns
        -------
        A : np.ndarray, shape (N, M)
            Self-ambiguity function on the DD grid.
        """
        if len(s) != self.NM:
            raise ValueError(f"Signal length must be {self.NM}")

        A = np.zeros((self.N, self.M), dtype=complex)

        for k in range(self.N):
            for l in range(self.M):
                # Delay shift by k samples
                s_shifted = np.roll(s, k)
                # Doppler modulation
                n = np.arange(self.NM)
                doppler = np.exp(1j * 2 * np.pi * l * n / self.M)
                # Correlation
                A[k, l] = np.sum(s * np.conj(s_shifted) * doppler) / self.NM

        return A

    def cross_ambiguity(self, s_tx: np.ndarray, s_rx: np.ndarray) -> np.ndarray:
        """
        Compute the cross-ambiguity function between two signals.

        A_cross(tau_k, nu_l) = sum_n s_rx[n] * s_tx*[(n - k) mod NM]
                                * exp(j*2*pi*l*n / M)

        Used for matched filtering in sensing applications.

        Parameters
        ----------
        s_tx : np.ndarray, shape (N*M,)
            Transmitted signal.
        s_rx : np.ndarray, shape (N*M,)
            Received signal.

        Returns
        -------
        A : np.ndarray, shape (N, M)
            Cross-ambiguity function.
        """
        if len(s_tx) != self.NM or len(s_rx) != self.NM:
            raise ValueError(f"Signal lengths must be {self.NM}")

        A = np.zeros((self.N, self.M), dtype=complex)

        for k in range(self.N):
            for l in range(self.M):
                s_tx_shifted = np.roll(s_tx, k)
                n = np.arange(self.NM)
                doppler = np.exp(1j * 2 * np.pi * l * n / self.M)
                A[k, l] = np.sum(s_rx * np.conj(s_tx_shifted) * doppler) / self.NM

        return A

    def ambiguity_fast(self, s: np.ndarray) -> np.ndarray:
        """
        Fast computation of self-ambiguity function using FFT.

        More efficient than the direct computation for large N*M.

        A[k, l] = FFT_over_l { sum_n s[n] * s*[(n-k) mod NM] }

        Parameters
        ----------
        s : np.ndarray, shape (N*M,)
            Time-domain signal.

        Returns
        -------
        A : np.ndarray, shape (N, M)
            Self-ambiguity function.
        """
        if len(s) != self.NM:
            raise ValueError(f"Signal length must be {self.NM}")

        A = np.zeros((self.N, self.M), dtype=complex)

        for k in range(self.N):
            # Delay-shifted autocorrelation
            s_shifted = np.roll(s, k)
            r_k = s * np.conj(s_shifted)

            # Reshape and FFT for Doppler
            r_k_matrix = r_k.reshape(self.M, self.N)
            A[k, :] = np.fft.fft(r_k_matrix.sum(axis=1)) / self.M

        return A

    def range_doppler_map(self, s_tx: np.ndarray, s_rx: np.ndarray,
                           use_fft: bool = True) -> np.ndarray:
        """
        Compute the range-Doppler map (sensing output).

        The range-Doppler map is the magnitude of the cross-ambiguity
        function, showing target locations in delay-Doppler space.

        Parameters
        ----------
        s_tx : np.ndarray
            Transmitted signal.
        s_rx : np.ndarray
            Received signal (may contain reflections).
        use_fft : bool
            If True, use FFT-based fast computation.

        Returns
        -------
        rd_map : np.ndarray, shape (N, M)
            Range-Doppler map (magnitude).
        """
        if use_fft:
            A = self.cross_ambiguity_fft(s_tx, s_rx)
        else:
            A = self.cross_ambiguity(s_tx, s_rx)
        return np.abs(A)

    def cross_ambiguity_fft(self, s_tx: np.ndarray, s_rx: np.ndarray) -> np.ndarray:
        """
        Fast cross-ambiguity using FFT.

        Parameters
        ----------
        s_tx : np.ndarray
            Transmitted signal.
        s_rx : np.ndarray
            Received signal.

        Returns
        -------
        A : np.ndarray, shape (N, M)
            Cross-ambiguity function.
        """
        if len(s_tx) != self.NM or len(s_rx) != self.NM:
            raise ValueError(f"Signal lengths must be {self.NM}")

        A = np.zeros((self.N, self.M), dtype=complex)

        for k in range(self.N):
            s_tx_shifted = np.roll(s_tx, k)
            r_k = s_rx * np.conj(s_tx_shifted)
            r_k_matrix = r_k.reshape(self.M, self.N)
            A[k, :] = np.fft.fft(r_k_matrix.sum(axis=1)) / self.M

        return A

    def peak_to_sidelobe_ratio(self, A: np.ndarray) -> float:
        """
        Compute the peak-to-sidelobe ratio (PSR) of the ambiguity function.

        PSR = max(|A|) / max(|A| excluding main peak)

        Parameters
        ----------
        A : np.ndarray
            Ambiguity function.

        Returns
        -------
        psr : float
            Peak-to-sidelobe ratio in linear scale.
        """
        A_abs = np.abs(A)
        peak_val = np.max(A_abs)

        # Find the peak location
        peak_idx = np.unravel_index(np.argmax(A_abs), A_abs.shape)

        # Zero out a region around the peak (3x3)
        A_copy = A_abs.copy()
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni = (peak_idx[0] + di) % self.N
                nj = (peak_idx[1] + dj) % self.M
                A_copy[ni, nj] = 0

        sidelobe_max = np.max(A_copy)
        if sidelobe_max == 0:
            return np.inf

        return peak_val / sidelobe_max
