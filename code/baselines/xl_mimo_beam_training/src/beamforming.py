"""
Beamforming codebook and precoding for XL-MIMO systems.

Provides codebook generation for analog beamforming in near-field XL-MIMO,
including DFT-based far-field codebooks and polar-domain near-field codebooks.

Reference:
    Section III-B of the paper discusses the beamforming codebook design.
    Near-field beamforming requires joint consideration of distance and angle,
    unlike far-field codebooks which only depend on the angle.
"""

import numpy as np
from typing import Optional, Tuple


class BeamformingCodebook:
    """Beamforming codebook for analog beamforming in XL-MIMO.

    Supports:
    - DFT codebook (far-field baseline)
    - Polar-domain codebook (near-field, joint distance-angle)

    Args:
        num_antennas: Number of antennas (N_t).
        antenna_spacing: Inter-element spacing (meters).
        wavelength: Carrier wavelength (meters).
    """

    def __init__(
        self,
        num_antennas: int = 256,
        antenna_spacing: Optional[float] = None,
        wavelength: float = 0.01,
    ):
        self.num_antennas = num_antennas
        self.wavelength = wavelength
        self.antenna_spacing = antenna_spacing or wavelength / 2.0

    def generate_dft_codebook(self) -> np.ndarray:
        """Generate a DFT-based beamforming codebook (far-field).

        Creates N_t orthogonal beams covering the spatial domain.
        Each beam corresponds to a unique spatial direction.

        Returns:
            Codebook matrix of shape (N_t, N_t) where column k is the
            k-th beamforming vector.

        Note:
            This is the conventional far-field codebook. Each column v_k
            has unit norm: ||v_k|| = 1.
        """
        n = np.arange(self.num_antennas)
        k = np.arange(self.num_antennas)
        # DFT matrix: W[n,k] = exp(-j*2*pi*n*k/N) / sqrt(N)
        codebook = np.exp(-1j * 2 * np.pi * np.outer(n, k) / self.num_antennas)
        codebook /= np.sqrt(self.num_antennas)
        return codebook

    def generate_polar_codebook(
        self,
        num_beams: int,
        distance_grid: np.ndarray,
        angle_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a polar-domain near-field codebook.

        Creates beams spanning both distance and angle dimensions for
        near-field beam training. Each beam focuses at a specific
        (distance, angle) point using spherical wavefronts.

        Args:
            num_beams: Total number of beams (must equal len(distance_grid) * len(angle_grid)).
            distance_grid: Array of distance values (meters) to focus at.
            angle_grid: Array of angle values (radians) to focus at.

        Returns:
            Tuple of:
            - codebook: Complex matrix of shape (N_t, num_beams).
            - distances: 1D array of distance indices for each beam.
            - angles: 1D array of angle indices for each beam.

        Note:
            Each beamforming vector v(d, theta) compensates the spherical
            wavefront phase at distance d and angle theta:
                v_n(d, theta) = exp(j * 2*pi/lambda * r_n(d, theta)) / sqrt(N_t)
            where r_n is the distance from antenna n to the focal point.
        """
        positions = (
            np.arange(self.num_antennas) - (self.num_antennas - 1) / 2.0
        ) * self.antenna_spacing

        beams = []
        beam_distances = []
        beam_angles = []

        for d in distance_grid:
            for theta in angle_grid:
                r_n = np.sqrt(
                    d**2 + positions**2 - 2 * d * positions * np.sin(theta)
                )
                v = np.exp(1j * 2 * np.pi / self.wavelength * r_n)
                v /= np.sqrt(self.num_antennas)
                beams.append(v)
                beam_distances.append(d)
                beam_angles.append(theta)

        codebook = np.column_stack(beams)
        return codebook, np.array(beam_distances), np.array(beam_angles)

    @staticmethod
    def compute_beamforming_gain(h: np.ndarray, v: np.ndarray) -> float:
        """Compute the beamforming gain |h^H v|^2.

        Args:
            h: Channel vector of shape (N_t,).
            v: Beamforming vector of shape (N_t,).

        Returns:
            Beamforming gain (scalar, non-negative).
        """
        gain = np.abs(np.vdot(h, v)) ** 2
        return float(gain)

    @staticmethod
    def normalize_beamformer(v: np.ndarray) -> np.ndarray:
        """Normalize beamforming vector to unit norm.

        Args:
            v: Beamforming vector.

        Returns:
            Normalized beamforming vector with ||v|| = 1.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm
