"""
Delay-Doppler Domain Channel Model
===================================
Implements the DD domain channel for Zak-OTFS systems.

Under the crystallization condition, the DD domain I/O relation is
(Eq. 6 in [1]):

    y[tau, nu] = h[tau, nu] * x[tau, nu] + w[tau, nu]

where * denotes circular convolution, and h[tau, nu] is the DD domain
channel response.

The DD channel taps are related to the time-domain channel by:
    h[tau, nu] = sum_p h_p * delta[tau - tau_p] * delta[nu - nu_p]

where h_p, tau_p, nu_p are the complex gain, delay, and Doppler shift
of the p-th scatterer.

This I/O relation is:
- Predictable (deterministic given h)
- Non-fading (no random fluctuations within the DD frame)
- Enables model-free operation (estimate h directly)

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Scatterer:
    """Represents a single scatterer/path in the channel."""
    gain: complex         # Complex path gain (amplitude and phase)
    delay: float          # Delay in seconds
    doppler: float        # Doppler shift in Hz
    delay_bin: int = 0    # Nearest delay bin index
    doppler_bin: int = 0  # Nearest Doppler bin index


class DDChannel:
    """
    Delay-Doppler domain channel model for Zak-OTFS.

    The channel is characterized by a set of scatterers, each with a
    complex gain, delay, and Doppler shift. The DD domain channel response
    is constructed by placing these scatterers on the DD grid.

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
        self.delta_tau = T_tau / N
        self.delta_nu = T_nu / M
        self.scatterers: List[Scatterer] = []
        self.h_dd: Optional[np.ndarray] = None

    def add_scatterer(self, gain: complex, delay: float, doppler: float):
        """
        Add a scatterer (path) to the channel.

        Parameters
        ----------
        gain : complex
            Complex path gain.
        delay : float
            Path delay in seconds (should be < T_tau for crystallization).
        doppler : float
            Doppler shift in Hz (should be < T_nu for crystallization).
        """
        # Quantize to nearest DD grid point
        delay_bin = int(np.round(delay / self.delta_tau)) % self.N
        doppler_bin = int(np.round(doppler / self.delta_nu)) % self.M

        scatterer = Scatterer(
            gain=gain,
            delay=delay,
            doppler=doppler,
            delay_bin=delay_bin,
            doppler_bin=doppler_bin
        )
        self.scatterers.append(scatterer)
        self.h_dd = None  # Invalidate cached channel response

    def build_channel(self) -> np.ndarray:
        """
        Build the DD domain channel response from scatterers.

        h[tau, nu] = sum_p h_p * delta[tau - tau_p, nu - nu_p]

        Returns
        -------
        h_dd : np.ndarray, shape (N, M)
            DD domain channel response.
        """
        self.h_dd = np.zeros((self.N, self.M), dtype=complex)

        for scat in self.scatterers:
            self.h_dd[scat.delay_bin, scat.doppler_bin] += scat.gain

        return self.h_dd

    def get_channel(self) -> np.ndarray:
        """Get the DD channel response (builds if not cached)."""
        if self.h_dd is None:
            self.build_channel()
        return self.h_dd

    def apply_channel(self, x_dd: np.ndarray,
                      noise_power: float = 0.0) -> np.ndarray:
        """
        Apply the DD channel to input symbols via circular convolution.

        y[tau, nu] = h[tau, nu] * x[tau, nu] + w[tau, nu]

        The circular convolution is computed efficiently using FFT:
            Y = FFT2(H) .* FFT2(X) / (N*M)
            y = IFFT2(Y) * (N*M)

        Parameters
        ----------
        x_dd : np.ndarray, shape (N, M)
            Input symbols in DD domain.
        noise_power : float
            Noise power (variance of complex AWGN per sample).

        Returns
        -------
        y_dd : np.ndarray, shape (N, M)
            Output symbols after channel.
        """
        if x_dd.shape != (self.N, self.M):
            raise ValueError(f"x_dd must have shape ({self.N}, {self.M})")

        h_dd = self.get_channel()

        # Circular convolution via FFT
        H_fft = np.fft.fft2(h_dd)
        X_fft = np.fft.fft2(x_dd)
        Y_fft = H_fft * X_fft
        y_dd = np.fft.ifft2(Y_fft)

        # Add AWGN noise
        if noise_power > 0:
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(self.N, self.M) +
                1j * np.random.randn(self.N, self.M)
            )
            y_dd += noise

        return y_dd

    def apply_channel_time(self, s: np.ndarray, modem=None,
                            noise_power: float = 0.0) -> np.ndarray:
        """
        Apply channel in time domain (for verification).

        Parameters
        ----------
        s : np.ndarray
            Time-domain signal.
        modem : OTFSModem, optional
            OTFS modem for DD-time conversion.
        noise_power : float
            Noise power.

        Returns
        -------
        r : np.ndarray
            Time-domain received signal.
        """
        from otfs_modem import OTFSModem
        if modem is None:
            modem = OTFSModem(self.N, self.M)

        # Convert to DD domain, apply channel, convert back
        X_dd = modem.demodulate(s)  # Wigner + SFFT
        Y_dd = self.apply_channel(X_dd, noise_power)
        r = modem.modulate(Y_dd)    # ISFFT + Heisenberg

        return r

    def max_delay_spread(self) -> float:
        """Return the maximum delay among all scatterers."""
        if not self.scatterers:
            return 0.0
        return max(s.delay for s in self.scatterers)

    def max_doppler_spread(self) -> float:
        """Return the maximum Doppler among all scatterers."""
        if not self.scatterers:
            return 0.0
        return max(abs(s.doppler) for s in self.scatterers)

    def crystallization_check(self) -> bool:
        """
        Check if the crystallization condition is satisfied.

        Returns True if T_tau >= max_delay_spread AND T_nu >= max_doppler_spread.
        """
        return (self.T_tau >= self.max_delay_spread()) and \
               (self.T_nu >= self.max_doppler_spread())

    @classmethod
    def generate_random(cls, N: int, M: int, num_paths: int,
                        T_tau: float = 1.0, T_nu: float = 1.0,
                        max_delay: Optional[float] = None,
                        max_doppler: Optional[float] = None,
                        seed: Optional[int] = None) -> 'DDChannel':
        """
        Generate a random multi-path channel.

        Parameters
        ----------
        N, M : int
            Grid dimensions.
        num_paths : int
            Number of scatterers.
        T_tau, T_nu : float
            Delay and Doppler periods.
        max_delay : float, optional
            Maximum delay (default: 0.8 * T_tau).
        max_doppler : float, optional
            Maximum Doppler (default: 0.8 * T_nu).
        seed : int, optional
            Random seed.

        Returns
        -------
        channel : DDChannel
            Random channel instance.
        """
        rng = np.random.default_rng(seed)

        if max_delay is None:
            max_delay = 0.8 * T_tau
        if max_doppler is None:
            max_doppler = 0.8 * T_nu

        channel = cls(N, M, T_tau, T_nu)

        for _ in range(num_paths):
            gain = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            delay = rng.uniform(0, max_delay)
            doppler = rng.uniform(-max_doppler, max_doppler)
            channel.add_scatterer(gain, delay, doppler)

        return channel
