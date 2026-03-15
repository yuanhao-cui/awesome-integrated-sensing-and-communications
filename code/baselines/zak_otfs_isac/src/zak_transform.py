"""
Zak Transform Implementation
============================
Implements the Zak transform for delay-Doppler domain signal processing.

The Zak transform of a signal s(t) is defined as (Eq. 2 in [1]):
    Z_s(t, f) = sum_{k=-inf}^{inf} s(t + k*T) * exp(j*2*pi*k*f*T)

where T is the delay period and f is the Doppler frequency.

For discrete implementation with N samples over period T:
    Z_s[n, m] = sum_{k=0}^{N-1} s[(n+k) mod N] * exp(j*2*pi*k*m/N)

The inverse Zak transform recovers the time-domain signal (Eq. 3):
    s(t) = (1/T) * integral_0^T Z_s(t, f) df

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import Tuple


class ZakTransform:
    """
    Zak transform for delay-Doppler domain processing.

    The Zak transform provides a unified representation in the delay-Doppler
    domain. Under the crystallization condition (delay period > max delay spread,
    Doppler period > max Doppler spread), the I/O relation becomes predictable
    and non-fading.

    Parameters
    ----------
    N_tau : int
        Number of delay samples (grid points along delay axis).
    N_nu : int
        Number of Doppler samples (grid points along Doppler axis).
    T_tau : float
        Delay period in seconds.
    T_nu : float
        Doppler period in Hz.
    """

    def __init__(self, N_tau: int, N_nu: int, T_tau: float = 1.0, T_nu: float = 1.0):
        self.N_tau = N_tau
        self.N_nu = N_nu
        self.T_tau = T_tau
        self.T_nu = T_nu

    def forward(self, s: np.ndarray) -> np.ndarray:
        """
        Compute the discrete Zak transform of a time-domain signal.

        Z_s[n, m] = sum_{k=0}^{N-1} s[(n+k) mod N] * exp(j*2*pi*k*m/N)

        This is equivalent to computing the FFT along columns of the
        folded signal matrix.

        Parameters
        ----------
        s : np.ndarray, shape (N_tau * N_nu,)
            Input time-domain signal of length N = N_tau * N_nu.

        Returns
        -------
        Z : np.ndarray, shape (N_tau, N_nu)
            Zak transform of the input signal (delay-Doppler representation).
        """
        N = self.N_tau * self.N_nu
        if len(s) != N:
            raise ValueError(f"Signal length must be {N} (= N_tau * N_nu)")

        # Reshape into N_tau x N_nu matrix (fold time into delay-Doppler grid)
        # Each column corresponds to a delay bin, each row to a Doppler bin
        S = s.reshape(self.N_tau, self.N_nu, order='F')

        # Apply FFT along Doppler axis (columns) to get Zak transform
        # This corresponds to the sum over k with the complex exponential
        Z = np.fft.fft(S, axis=1) / np.sqrt(self.N_nu)

        return Z

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the inverse Zak transform.

        s[(n+k) mod N] = (1/N_nu) * sum_{m=0}^{N_nu-1} Z[n, m] * exp(-j*2*pi*k*m/N_nu)

        Parameters
        ----------
        Z : np.ndarray, shape (N_tau, N_nu)
            Delay-Doppler domain signal.

        Returns
        -------
        s : np.ndarray, shape (N_tau * N_nu,)
            Reconstructed time-domain signal.
        """
        if Z.shape != (self.N_tau, self.N_nu):
            raise ValueError(f"Z must have shape ({self.N_tau}, {self.N_nu})")

        # Inverse FFT along Doppler axis
        S = np.fft.ifft(Z, axis=1) * np.sqrt(self.N_nu)

        # Reshape back to time domain
        s = S.reshape(-1, order='F')

        return s

    def delay_grid(self) -> np.ndarray:
        """Return the delay grid values (in seconds)."""
        return np.arange(self.N_tau) * (self.T_tau / self.N_tau)

    def doppler_grid(self) -> np.ndarray:
        """Return the Doppler grid values (in Hz)."""
        return np.arange(self.N_nu) * (self.T_nu / self.N_nu)

    def crystallization_satisfied(self, max_delay_spread: float,
                                   max_doppler_spread: float) -> bool:
        """
        Check if the crystallization condition is satisfied.

        Crystallization condition (Eq. 4 in [1]):
            T_tau >= tau_max  (delay period >= max delay spread)
            T_nu >= nu_max   (Doppler period >= max Doppler spread)

        When satisfied, the I/O relation in the DD domain is:
            y[tau, nu] = h[tau, nu] * x[tau, nu] + w[tau, nu]
        which is a circular convolution (predictable, non-fading).

        Parameters
        ----------
        max_delay_spread : float
            Maximum delay spread of the channel (seconds).
        max_doppler_spread : float
            Maximum Doppler spread of the channel (Hz).

        Returns
        -------
        bool
            True if crystallization condition is satisfied.
        """
        return (self.T_tau >= max_delay_spread) and (self.T_nu >= max_doppler_spread)
