"""
Channel Estimation for Zak-OTFS
================================
Implements model-free and model-based channel estimation.

The Zak-OTFS framework enables model-free channel estimation by transmitting
a single point pulsone and reading the DD channel response directly:

    1. Transmit point pulsone b_{0,0}(t) at (tau=0, nu=0)
    2. Receive y[tau, nu] = h[tau, nu] * delta[tau, nu] + w[tau, nu]
                    = h[tau, nu] + w[tau, nu]
    3. The received DD symbols directly give the channel taps

This is possible because:
- The self-ambiguity of a point pulsone is a Kronecker delta on Lambda
- Under crystallization, the I/O relation is circular convolution
- Convolving with delta gives the channel directly

For model-based estimation, we can also use least-squares fitting
when the scatterer structure is known.

Reference:
    [1] Saif Khan Mohammed et al., "Zak-OTFS to Integrate Sensing the I/O
        Relation and Data Communication," arXiv:2404.04182, 2024.
"""

import numpy as np
from typing import Optional, List, Tuple


class ChannelEstimator:
    """
    Channel estimator for Zak-OTFS systems.

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

    def estimate_model_free(self, y_dd: np.ndarray,
                             noise_threshold: float = 0.0) -> np.ndarray:
        """
        Model-free channel estimation using received pulsone.

        When a point pulsone at (0,0) is transmitted, the received DD
        symbols are:
            y[tau, nu] = h[tau, nu] + w[tau, nu]

        So h_est = y directly (optionally thresholding noise).

        Parameters
        ----------
        y_dd : np.ndarray, shape (N, M)
            Received DD symbols after transmitting a point pulsone.
        noise_threshold : float
            Threshold to zero out small taps (noise rejection).

        Returns
        -------
        h_est : np.ndarray, shape (N, M)
            Estimated DD channel response.
        """
        h_est = y_dd.copy()

        if noise_threshold > 0:
            magnitude = np.abs(h_est)
            mask = magnitude < noise_threshold * np.max(magnitude)
            h_est[mask] = 0

        return h_est

    def estimate_ls(self, x_dd: np.ndarray, y_dd: np.ndarray) -> np.ndarray:
        """
        Least-squares channel estimation.

        Given known transmitted symbols x and received symbols y,
        estimate h such that y = h * x (circular convolution).

        In the frequency domain: H = Y ./ X
        where ./ is element-wise division.

        Parameters
        ----------
        x_dd : np.ndarray, shape (N, M)
            Known transmitted symbols.
        y_dd : np.ndarray, shape (N, M)
            Received symbols.

        Returns
        -------
        h_est : np.ndarray, shape (N, M)
            Estimated channel response.
        """
        X_fft = np.fft.fft2(x_dd)
        Y_fft = np.fft.fft2(y_dd)

        # Avoid division by zero
        H_fft = np.zeros_like(Y_fft)
        mask = np.abs(X_fft) > 1e-10
        H_fft[mask] = Y_fft[mask] / X_fft[mask]

        h_est = np.fft.ifft2(H_fft)
        return h_est

    def estimate_scatterers(self, h_dd: np.ndarray,
                            num_scatterers: int,
                            noise_floor: float = 0.1) -> List[Tuple[complex, int, int]]:
        """
        Estimate scatterer locations from channel response.

        Identifies the strongest taps in the DD channel as scatterers.

        Parameters
        ----------
        h_dd : np.ndarray, shape (N, M)
            DD channel response.
        num_scatterers : int
            Number of scatterers to estimate.
        noise_floor : float
            Fraction of max magnitude below which taps are ignored.

        Returns
        -------
        scatterers : list of (gain, delay_bin, doppler_bin)
            Estimated scatterer parameters.
        """
        magnitude = np.abs(h_dd)
        max_mag = np.max(magnitude)
        threshold = noise_floor * max_mag

        # Find all taps above threshold
        candidates = []
        for i in range(self.N):
            for j in range(self.M):
                if magnitude[i, j] > threshold:
                    candidates.append((magnitude[i, j], i, j))

        # Sort by magnitude (descending) and take top num_scatterers
        candidates.sort(reverse=True)
        top = candidates[:num_scatterers]

        scatterers = []
        for _, delay_bin, doppler_bin in top:
            gain = h_dd[delay_bin, doppler_bin]
            scatterers.append((gain, delay_bin, doppler_bin))

        return scatterers

    def nmse(self, h_true: np.ndarray, h_est: np.ndarray) -> float:
        """
        Compute the normalized mean square error (NMSE).

        NMSE = ||h_true - h_est||^2 / ||h_true||^2

        Parameters
        ----------
        h_true : np.ndarray
            True channel response.
        h_est : np.ndarray
            Estimated channel response.

        Returns
        -------
        nmse : float
            Normalized MSE in linear scale.
        """
        error = h_true - h_est
        mse = np.sum(np.abs(error) ** 2)
        signal_power = np.sum(np.abs(h_true) ** 2)

        if signal_power == 0:
            return np.inf

        return mse / signal_power

    def nmse_db(self, h_true: np.ndarray, h_est: np.ndarray) -> float:
        """
        Compute NMSE in dB.

        Parameters
        ----------
        h_true, h_est : np.ndarray
            True and estimated channel responses.

        Returns
        -------
        nmse_db : float
            NMSE in dB.
        """
        nmse_lin = self.nmse(h_true, h_est)
        return 10 * np.log10(nmse_lin) if nmse_lin > 0 else -np.inf

    def pilot_overhead(self, method: str = 'model_free') -> int:
        """
        Compute the pilot overhead (number of pilot symbols needed).

        Model-free: 1 pulsone (N*M symbols but only 1 DD bin is used)
        LS: depends on the number of unknowns

        Parameters
        ----------
        method : str
            'model_free' or 'ls'

        Returns
        -------
        overhead : int
            Number of pilot DD bins needed.
        """
        if method == 'model_free':
            return 1  # Single point pulsone
        elif method == 'ls':
            return self.N * self.M  # Full training sequence
        else:
            raise ValueError(f"Unknown method: {method}")

    def estimate_from_pulsone(self, received_dd: np.ndarray,
                               pulsone_loc: Tuple[int, int] = (0, 0),
                               threshold_db: float = -30.0) -> np.ndarray:
        """
        Estimate channel from received pulsone with thresholding.

        Parameters
        ----------
        received_dd : np.ndarray, shape (N, M)
            Received DD symbols after point pulsone transmission.
        pulsone_loc : tuple
            (delay_bin, doppler_bin) where pulsone was transmitted.
        threshold_db : float
            Threshold in dB below peak to zero out taps.

        Returns
        -------
        h_est : np.ndarray, shape (N, M)
            Estimated and thresholded channel response.
        """
        # The received signal is the channel shifted by pulsone location
        h_est = np.roll(received_dd, (-pulsone_loc[0], -pulsone_loc[1]), axis=(0, 1))

        # Apply threshold
        mag = np.abs(h_est)
        peak = np.max(mag)
        threshold = peak * 10 ** (threshold_db / 10)
        h_est[mag < threshold] = 0

        return h_est
