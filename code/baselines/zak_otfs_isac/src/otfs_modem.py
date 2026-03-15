"""
OTFS Modulation / Demodulation
===============================
Implements the OTFS modulator and demodulator using:
  1. ISFFT (Inverse Symplectic Finite Fourier Transform)
  2. Heisenberg transform (DD -> time domain)
  3. Wigner transform (time -> DD domain)

The OTFS modulation process (Section II-A in [1]):

    Transmitter:
        1. Start with DD domain symbols X[n, m], n=0..N-1, m=0..M-1
        2. ISFFT: X_tf[k, l] = (1/sqrt(NM)) * sum_{n,m} X[n,m]
                          * exp(j*2*pi*(nk/N - ml/M))
        3. Heisenberg transform: s(t) = Heisenberg{X_tf}

    Receiver:
        1. Wigner transform: Y_tf = Wigner{r(t)}
        2. SFFT (SFFT = ISFFT^{-1}): Y[n, m] = (1/sqrt(NM)) * sum_{k,l}
                           Y_tf[k, l] * exp(-j*2*pi*(nk/N - ml/M))

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import Optional


class OTFSModem:
    """
    OTFS modulator and demodulator.

    Implements the full OTFS modulation chain:
        DD domain -> ISFFT -> Time-Frequency -> Heisenberg -> Time domain

    And the demodulation chain:
        Time domain -> Wigner -> Time-Frequency -> SFFT -> DD domain

    Parameters
    ----------
    N : int
        Number of delay bins (time slots per frame).
    M : int
        Number of Doppler bins (subcarriers).
    """

    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.NM = N * M

    def isfft(self, X_dd: np.ndarray) -> np.ndarray:
        """
        Inverse Symplectic Finite Fourier Transform (ISFFT).

        Maps delay-Doppler domain symbols to time-frequency domain.

        X_tf[k, l] = (1/sqrt(N*M)) * sum_{n=0}^{N-1} sum_{m=0}^{M-1}
                      X_dd[n, m] * exp(j*2*pi*(n*k/N - m*l/M))

        This is equivalent to:
            1. FFT along columns (delay -> time)
            2. IFFT along rows (Doppler -> frequency)
            with normalization.

        Eq. (7) in [1].

        Parameters
        ----------
        X_dd : np.ndarray, shape (N, M)
            Delay-Doppler domain symbols.

        Returns
        -------
        X_tf : np.ndarray, shape (N, M)
            Time-frequency domain symbols.
        """
        if X_dd.shape != (self.N, self.M):
            raise ValueError(f"X_dd must have shape ({self.N}, {self.M})")

        # ISFFT = FFT along columns + IFFT along rows
        # This is equivalent to a 2D FFT with proper sign conventions
        X_tf = np.fft.fft(X_dd, axis=0) / np.sqrt(self.N)   # FFT along delay
        X_tf = np.fft.ifft(X_tf, axis=1) * np.sqrt(self.M)  # IFFT along Doppler

        return X_tf

    def sfft(self, Y_tf: np.ndarray) -> np.ndarray:
        """
        Symplectic Finite Fourier Transform (SFFT) - inverse of ISFFT.

        Maps time-frequency domain symbols to delay-Doppler domain.

        Y_dd[n, m] = (1/sqrt(N*M)) * sum_{k=0}^{N-1} sum_{l=0}^{M-1}
                      Y_tf[k, l] * exp(-j*2*pi*(n*k/N - m*l/M))

        Eq. (8) in [1].

        Parameters
        ----------
        Y_tf : np.ndarray, shape (N, M)
            Time-frequency domain symbols.

        Returns
        -------
        Y_dd : np.ndarray, shape (N, M)
            Delay-Doppler domain symbols.
        """
        if Y_tf.shape != (self.N, self.M):
            raise ValueError(f"Y_tf must have shape ({self.N}, {self.M})")

        # SFFT = IFFT along time + FFT along frequency (inverse of ISFFT)
        Y_dd = np.fft.ifft(Y_tf, axis=0) * np.sqrt(self.N)  # IFFT along time
        Y_dd = np.fft.fft(Y_dd, axis=1) / np.sqrt(self.M)   # FFT along frequency

        return Y_dd

    def heisenberg(self, X_tf: np.ndarray) -> np.ndarray:
        """
        Heisenberg transform: maps time-frequency domain to time-domain signal.

        For rectangular pulse shaping (OFDM-like):
            s(t) = sum_{k=0}^{N-1} sum_{l=0}^{M-1} X_tf[k, l]
                   * p(t - k*T/M) * exp(j*2*pi*l*delta_f*(t - k*T/M))

        For the discrete implementation with rectangular pulse shaping,
        this reduces to: s = vec(X_tf^T) (column-wise stacking of transpose).

        Eq. (9) in [1].

        Parameters
        ----------
        X_tf : np.ndarray, shape (N, M)
            Time-frequency domain symbols.

        Returns
        -------
        s : np.ndarray, shape (N*M,)
            Time-domain signal.
        """
        # For rectangular pulse shaping, the Heisenberg transform
        # is simply reshaping the TF matrix to a vector (column-major)
        s = X_tf.T.reshape(-1)
        return s

    def wigner(self, r: np.ndarray) -> np.ndarray:
        """
        Wigner transform: maps time-domain received signal to TF domain.

        This is the inverse of the Heisenberg transform.

        Eq. (10) in [1].

        Parameters
        ----------
        r : np.ndarray, shape (N*M,)
            Time-domain received signal.

        Returns
        -------
        Y_tf : np.ndarray, shape (N, M)
            Time-frequency domain received symbols.
        """
        if len(r) != self.NM:
            raise ValueError(f"r must have length {self.NM}")

        Y_tf = r.reshape(self.M, self.N).T
        return Y_tf

    def modulate(self, X_dd: np.ndarray) -> np.ndarray:
        """
        Full OTFS modulation: DD domain -> time domain.

        Steps:
            1. ISFFT: DD -> TF domain
            2. Heisenberg: TF -> time domain

        Parameters
        ----------
        X_dd : np.ndarray, shape (N, M)
            Delay-Doppler domain symbols.

        Returns
        -------
        s : np.ndarray, shape (N*M,)
            Time-domain transmitted signal.
        """
        X_tf = self.isfft(X_dd)
        s = self.heisenberg(X_tf)
        return s

    def demodulate(self, r: np.ndarray) -> np.ndarray:
        """
        Full OTFS demodulation: time domain -> DD domain.

        Steps:
            1. Wigner: time -> TF domain
            2. SFFT: TF -> DD domain

        Parameters
        ----------
        r : np.ndarray, shape (N*M,)
            Time-domain received signal.

        Returns
        -------
        Y_dd : np.ndarray, shape (N, M)
            Delay-Doppler domain received symbols.
        """
        Y_tf = self.wigner(r)
        Y_dd = self.sfft(Y_tf)
        return Y_dd
