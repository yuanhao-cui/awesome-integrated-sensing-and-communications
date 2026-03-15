"""
Peak-to-Average Power Ratio (PAPR) Analysis
=============================================
Computes and analyzes PAPR for Zak-OTFS waveforms.

PAPR is defined as (Eq. 16 in [1]):
    PAPR = max|s(t)|^2 / E[|s(t)|^2]

For a point pulsone, the PAPR is approximately:
    PAPR_point ~ N * M (or ~10*log10(N*M) dB)

For a spread pulsone (using discrete chirp filter), the PAPR is
significantly reduced:
    PAPR_spread ~ 6 dB (approximately constant regardless of N, M)

The PAPR reduction is achieved by the chirp filter spreading energy
more uniformly across the signal.

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import Tuple


class PAPRAnalyzer:
    """
    PAPR computation and reduction analysis for OTFS waveforms.

    Parameters
    ----------
    N : int
        Number of delay bins.
    M : int
        Number of Doppler bins.
    """

    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M

    def compute_papr(self, s: np.ndarray) -> float:
        """
        Compute the PAPR of a time-domain signal.

        PAPR = max|s[n]|^2 / mean(|s[n]|^2)

        Parameters
        ----------
        s : np.ndarray
            Time-domain signal.

        Returns
        -------
        papr : float
            PAPR in linear scale.
        """
        power = np.abs(s) ** 2
        peak_power = np.max(power)
        avg_power = np.mean(power)

        if avg_power == 0:
            return np.inf

        return peak_power / avg_power

    def compute_papr_db(self, s: np.ndarray) -> float:
        """
        Compute PAPR in dB.

        Parameters
        ----------
        s : np.ndarray
            Time-domain signal.

        Returns
        -------
        papr_db : float
            PAPR in dB.
        """
        papr_lin = self.compute_papr(s)
        return 10 * np.log10(papr_lin) if papr_lin > 0 else -np.inf

    def complementary_cdf(self, signals: list,
                          num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the complementary CDF (CCDF) of PAPR.

        CCDF(x) = Pr(PAPR > x)

        Used to characterize the PAPR distribution across many signals.

        Parameters
        ----------
        signals : list of np.ndarray
            List of time-domain signals.
        num_points : int
            Number of points for the CCDF curve.

        Returns
        -------
        papr_values : np.ndarray
            PAPR values (dB) at which CCDF is evaluated.
        ccdf : np.ndarray
            Complementary CDF values.
        """
        paprs_db = np.array([self.compute_papr_db(s) for s in signals])

        papr_min = np.min(paprs_db)
        papr_max = np.max(paprs_db)
        papr_values = np.linspace(papr_min, papr_max, num_points)

        ccdf = np.array([np.mean(paprs_db > p) for p in papr_values])

        return papr_values, ccdf

    def analyze_waveform(self, s: np.ndarray) -> dict:
        """
        Comprehensive PAPR analysis of a waveform.

        Parameters
        ----------
        s : np.ndarray
            Time-domain signal.

        Returns
        -------
        stats : dict
            Dictionary with PAPR statistics:
            - papr_linear: PAPR in linear scale
            - papr_db: PAPR in dB
            - peak_power: Peak instantaneous power
            - avg_power: Average power
            - crest_factor: Peak-to-RMS ratio
        """
        power = np.abs(s) ** 2
        peak_power = np.max(power)
        avg_power = np.mean(power)
        rms = np.sqrt(avg_power)
        peak_amp = np.max(np.abs(s))

        return {
            'papr_linear': peak_power / avg_power if avg_power > 0 else np.inf,
            'papr_db': self.compute_papr_db(s),
            'peak_power': peak_power,
            'avg_power': avg_power,
            'crest_factor': peak_amp / rms if rms > 0 else np.inf,
        }

    def theoretical_papr_point(self) -> float:
        """
        Theoretical PAPR of a point pulsone.

        For a point pulsone, the time-domain signal after ISFFT + Heisenberg
        has all energy concentrated, giving:
            PAPR_point ~ N*M (linear) = 10*log10(N*M) dB

        Returns
        -------
        papr_db : float
            Theoretical PAPR in dB.
        """
        return 10 * np.log10(self.N * self.M)

    def theoretical_papr_spread(self) -> float:
        """
        Theoretical PAPR of a spread pulsone.

        The chirp filter distributes energy more uniformly, resulting in:
            PAPR_spread ~ 6 dB

        Returns
        -------
        papr_db : float
            Approximate PAPR in dB.
        """
        return 6.0  # Approximately constant

    def papr_reduction_gain(self, s_point: np.ndarray,
                             s_spread: np.ndarray) -> float:
        """
        Compute the PAPR reduction achieved by spreading.

        Parameters
        ----------
        s_point : np.ndarray
            Point pulsone time-domain signal.
        s_spread : np.ndarray
            Spread pulsone time-domain signal.

        Returns
        -------
        gain_db : float
            PAPR reduction in dB (positive means improvement).
        """
        papr_point = self.compute_papr_db(s_point)
        papr_spread = self.compute_papr_db(s_spread)
        return papr_point - papr_spread
