"""
Near-field channel model for XL-MIMO systems.

Implements the spherical-wave near-field channel model based on the
Fresnel/Kirchhoff approximation, appropriate for extremely large-scale
MIMO arrays where the far-field (planar-wave) assumption breaks down.

Reference:
    Section III-A of the paper describes the near-field channel model.
    The spherical wavefront model accounts for distance-dependent phase
    variations across the array aperture.

    For antenna n at position d_n and user at distance r, angle theta:
        h_n = (alpha / r_n) * exp(-j * 2*pi / lambda * r_n)

    where r_n = sqrt(r^2 + d_n^2 - 2*r*d_n*sin(theta)) is the distance
    from antenna n to the user (spherical wave model).
"""

import numpy as np
from typing import Optional, Tuple


class NearFieldChannel:
    """Near-field channel model with spherical wavefront propagation.

    Models the channel between a uniform linear array (ULA) and a single-antenna
    user in the radiative near-field region where the planar-wave assumption
    is invalid.

    Args:
        num_antennas: Number of antennas in the ULA (N_t).
        wavelength: Carrier wavelength lambda (meters).
        antenna_spacing: Inter-element spacing (meters). Default: lambda/2.
        bandwidth: System bandwidth (Hz). Default: 1e8 (100 MHz).
        noise_power_dBm: Noise power spectral density (dBm/Hz). Default: -174.
    """

    def __init__(
        self,
        num_antennas: int = 256,
        wavelength: float = 0.01,  # 30 GHz -> lambda = 1 cm
        antenna_spacing: Optional[float] = None,
        noise_power_dBm: float = -174.0,
    ):
        self.num_antennas = num_antennas
        self.wavelength = wavelength
        self.antenna_spacing = antenna_spacing or wavelength / 2.0
        self.noise_power_dBm = noise_power_dBm

        # Antenna positions (centered ULA)
        self.positions = (
            np.arange(num_antennas) - (num_antennas - 1) / 2.0
        ) * self.antenna_spacing

    def generate_channel(
        self,
        distance: float,
        angle: float,
        path_loss_dB: float = 0.0,
        num_paths: int = 1,
        angle_spread: float = 0.0,
    ) -> np.ndarray:
        """Generate a near-field channel vector.

        Args:
            distance: User distance from the array center (meters).
            angle: User angle of departure in radians (broadside = 0).
            path_loss_dB: Additional path loss in dB.
            num_paths: Number of multipath components.
            angle_spread: Angular spread for multipath (radians).

        Returns:
            Channel vector h of shape (num_antennas,) as complex numpy array.

        Note:
            For single-path (line-of-sight), uses exact spherical wave model.
            For multi-path, adds scattered components with random angles
            within the angular spread.
        """
        # Free-space path loss: alpha = lambda / (4*pi*r)
        alpha = self.wavelength / (4 * np.pi * distance)
        alpha *= 10 ** (path_loss_dB / 20.0)  # additional path loss

        h = np.zeros(self.num_antennas, dtype=np.complex128)

        # Line-of-sight (dominant path)
        h += self._spherical_wave_component(distance, angle, alpha)

        # Additional scattered paths
        for _ in range(1, num_paths):
            scattered_angle = angle + np.random.uniform(
                -angle_spread, angle_spread
            )
            scattered_distance = distance * np.random.uniform(0.95, 1.05)
            scattered_alpha = alpha * np.random.rayleigh(0.3)
            scattered_phase = np.random.uniform(0, 2 * np.pi)
            h += self._spherical_wave_component(
                scattered_distance, scattered_angle, scattered_alpha
            ) * np.exp(1j * scattered_phase)

        # Normalize to unit average power per antenna
        h = h / np.sqrt(np.mean(np.abs(h) ** 2))
        return h

    def _spherical_wave_component(
        self, distance: float, angle: float, amplitude: float
    ) -> np.ndarray:
        """Compute a single spherical wave channel component.

        Args:
            distance: Propagation distance (meters).
            angle: Departure angle (radians).
            amplitude: Path amplitude.

        Returns:
            Complex channel vector for this path component.
        """
        # Distance from each antenna to the user
        r_n = np.sqrt(
            distance**2
            + self.positions**2
            - 2 * distance * self.positions * np.sin(angle)
        )
        # Spherical wave model: amplitude decay + phase rotation
        h_n = (amplitude / r_n) * np.exp(-1j * 2 * np.pi / self.wavelength * r_n)
        return h_n

    def generate_channel_batch(
        self,
        num_samples: int,
        distance_range: Tuple[float, float] = (10.0, 100.0),
        angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
        num_paths: int = 3,
        angle_spread: float = 0.05,
    ) -> np.ndarray:
        """Generate a batch of near-field channel realizations.

        Args:
            num_samples: Number of channel samples to generate.
            distance_range: (min, max) user distances in meters.
            angle_range: (min, max) user angles in radians.
            num_paths: Number of multipath components per sample.
            angle_spread: Angular spread for scattered paths (radians).

        Returns:
            Channel matrix of shape (num_samples, num_antennas) as complex array.
        """
        channels = np.zeros((num_samples, self.num_antennas), dtype=np.complex128)
        for i in range(num_samples):
            distance = np.random.uniform(*distance_range)
            angle = np.random.uniform(*angle_range)
            channels[i] = self.generate_channel(
                distance, angle, num_paths=num_paths, angle_spread=angle_spread
            )
        return channels

    def estimate_channel(
        self,
        h_true: np.ndarray,
        snr_dB: float = 10.0,
        pilot_length: Optional[int] = None,
    ) -> np.ndarray:
        """Estimate the channel with pilot-based MMSE estimation.

        Uses a simple least-squares (LS) estimate with noise corruption,
        simulating practical channel estimation.

        Args:
            h_true: True channel vector of shape (num_antennas,).
            snr_dB: Signal-to-noise ratio in dB for estimation.
            pilot_length: Number of pilot symbols. If None, equals num_antennas.

        Returns:
            Estimated channel vector of shape (num_antennas,).
        """
        if pilot_length is None:
            pilot_length = self.num_antennas

        # Pilot matrix (orthogonal if pilot_length >= num_antennas)
        pilot_power = 1.0
        snr_linear = 10 ** (snr_dB / 10.0)
        noise_var = pilot_power / snr_linear

        # Simple LS estimate: h_est = h_true + noise
        noise = np.sqrt(noise_var / 2) * (
            np.random.randn(self.num_antennas) + 1j * np.random.randn(self.num_antennas)
        )
        # Effective noise reduces with more pilots
        noise /= np.sqrt(pilot_length)
        h_est = h_true + noise

        return h_est
