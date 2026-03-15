"""
CSI Signal Model for ISAC Doppler Estimation.

Implements the channel model from Eq. (2) and (5) of the paper.

The received CSI at antenna m for a single subcarrier is:
    H_m(t) = a_m * exp(j*2π*f_c*τ_m(t)) * exp(j*φ_T(t)) * exp(j*φ_R(t))
             * exp(j*φ_CFO) * exp(j*φ_TMO) + noise

where:
    a_m: path amplitude (depends on antenna position and target range)
    f_c: carrier frequency
    τ_m(t): propagation delay = 2*d_m(t)/c (round-trip)
    φ_T(t), φ_R(t): transmitter/receiver phase noise
    φ_CFO: carrier frequency offset
    φ_TMO: timing misalignment offset

The CSI-ratio cancels shared phase terms:
    R(t) = H_m(t) / H_{m+1}(t)
         = (a_m / a_{m+1}) * exp(j*2π*f_D*t) * exp(j*φ_0)
         = A * z(t) * exp(j*φ_0)

where z(t) = exp(j*2π*f_D*t) traces the unit circle and f_D is the
Doppler frequency determined by the target's radial velocity.

Through Mobius transformation, R(t) traces a circle in the complex plane.
"""

import numpy as np
from typing import Tuple, Optional


def csi_signal_model(
    t: np.ndarray,
    f_c: float = 5.8e9,
    c: float = 3e8,
    d0: float = 5.0,
    v_r: float = 0.0,
    antenna_positions: Optional[np.ndarray] = None,
    snr_db: float = np.inf,
) -> np.ndarray:
    """
    Generate CSI samples for a multi-antenna receiver.

    Models the received CSI at each antenna as described in Eq. (2) of the paper.
    A target at range d0 with radial velocity v_r produces a time-varying
    round-trip delay at each antenna.

    Parameters
    ----------
    t : np.ndarray
        Time samples (seconds), shape (N,).
    f_c : float
        Carrier frequency (Hz). Default: 5.8 GHz (WiFi 5 GHz band).
    c : float
        Speed of light (m/s). Default: 3e8.
    d0 : float
        Initial range to target (meters). Default: 5.0 m.
    v_r : float
        Radial velocity of target (m/s). Positive = approaching.
    antenna_positions : np.ndarray, optional
        Antenna positions along one axis (meters), shape (M,).
        Default: [0, 0.0258, 0.0516] (Intel 5300, λ/2 at 5.8 GHz).
    snr_db : float
        Signal-to-noise ratio in dB. Default: infinite (no noise).

    Returns
    -------
    H : np.ndarray
        Complex CSI samples, shape (N, M) where M = number of antennas.
        H[k, m] is the CSI at time t[k] on antenna m.

    Notes
    -----
    - The phase includes path-dependent delay: τ_m(t) = 2*d_m(t)/c
    - d_m(t) = d0 - v_r*t + projection of antenna position
    - CFO and TMO are shared across antennas (cancelled by CSI-ratio)
    """
    if antenna_positions is None:
        # Intel 5300 default: half-wavelength spacing at 5.8 GHz
        wavelength = c / f_c  # ~5.17 cm
        antenna_positions = np.array([0, wavelength / 2, wavelength])

    M = len(antenna_positions)
    N = len(t)

    # Doppler frequency for each antenna (slightly different due to spacing)
    # f_D,m = 2 * f_c * v_r / c (Doppler shift for round-trip)
    f_D = 2 * f_c * v_r / c

    # Phase for each antenna at each time
    # Include antenna-dependent phase offset from spatial position
    H = np.zeros((N, M), dtype=complex)

    for m in range(M):
        # Antenna-dependent phase offset (from spatial separation)
        phi_antenna = 2 * np.pi * antenna_positions[m] / (c / f_c)

        # Doppler phase evolution
        # For a moving target, each antenna sees slightly different Doppler
        # due to geometry, but the dominant term is shared
        doppler_phase = 2 * np.pi * f_D * t

        # Combined phase: Doppler + antenna offset
        # Add shared CFO and TMO (will be cancelled by CSI-ratio)
        cfo_phase = 2 * np.pi * 50 * t  # 50 Hz CFO example
        tmo_phase = 2 * np.pi * 0.01 * t  # TMO example

        total_phase = doppler_phase + phi_antenna + cfo_phase + tmo_phase

        # Amplitude (depends on range and antenna)
        # For far-field, amplitudes are approximately equal
        range_m = d0 - v_r * t + antenna_positions[m] * 0.1  # small geometry effect
        amplitude = 1.0 / (range_m ** 2)  # Free-space path loss (simplified)

        H[:, m] = amplitude * np.exp(1j * total_phase)

    # Add noise
    if snr_db < np.inf:
        signal_power = np.mean(np.abs(H) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(N, M) + 1j * np.random.randn(N, M)
        )
        H = H + noise

    return H


def csi_with_doppler(
    t: np.ndarray,
    f_D: float,
    snr_db: float = np.inf,
    amplitude_ratio: float = 1.0,
    phase_offset: float = 0.0,
    cfo_hz: float = 0.0,
    tmo_hz: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CSI for two adjacent antennas with known Doppler.

    Directly generates CSI samples H1(t) and H2(t) such that their ratio
    R(t) = H1(t)/H2(t) exhibits the specified Doppler frequency f_D.

    This is useful for testing the three Doppler estimation algorithms
    with known ground truth.

    Parameters
    ----------
    t : np.ndarray
        Time samples (seconds), shape (N,).
    f_D : float
        Doppler frequency (Hz). Positive = approaching.
    snr_db : float
        Signal-to-noise ratio in dB. Default: infinite.
    amplitude_ratio : float
        |H1| / |H2| ratio. Default: 1.0.
    phase_offset : float
        Static phase difference between antennas (radians). Default: 0.
    cfo_hz : float
        Carrier frequency offset (Hz), shared by both antennas. Default: 0.
    tmo_hz : float
        Timing misalignment offset (Hz), shared by both antennas. Default: 0.

    Returns
    -------
    H1, H2 : np.ndarray
        Complex CSI samples for antenna 1 and antenna 2, each shape (N,).

    Notes
    -----
    The CSI-ratio is:
        R(t) = H1(t) / H2(t) = amplitude_ratio * exp(j*(2π*f_D*t + phase_offset))

    The shared CFO and TMO are cancelled:
        exp(j*2π*cfo_hz*t) appears in both H1 and H2, so R(t) is unaffected.

    This models Eq. (5-6) of the paper.
    """
    N = len(t)

    # Shared phase terms (cancelled by CSI-ratio)
    shared_phase = 2 * np.pi * cfo_hz * t + 2 * np.pi * tmo_hz * t

    # Antenna 1: base amplitude + shared phase + Doppler phase
    H1 = np.exp(1j * (shared_phase + 2 * np.pi * f_D * t + phase_offset))

    # Antenna 2: scaled amplitude + shared phase (no Doppler in ratio)
    H2 = (1.0 / amplitude_ratio) * np.exp(1j * shared_phase)

    # Add noise
    if snr_db < np.inf:
        for H in [H1, H2]:
            signal_power = np.mean(np.abs(H) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(N) + 1j * np.random.randn(N)
            )
            H += noise

    return H1, H2
