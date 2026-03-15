"""Tests for Mobius transformation-based Doppler estimator (Algorithm 1)."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_model import csi_with_doppler
from csi_ratio import compute_csi_ratio
from mobius_estimator import mobius_doppler_estimate


def generate_csi_ratio_samples(f_D, T_s, N, snr_db=np.inf):
    """Generate CSI-ratio samples with known Doppler frequency."""
    t = np.arange(N) * T_s
    H1, H2 = csi_with_doppler(t, f_D, snr_db=snr_db)
    R = compute_csi_ratio(H1, H2)
    return R, t


def test_mobius_doppler_estimate():
    """Algorithm 1 estimates f_D correctly on synthetic data."""
    T_s = 0.0005  # 2 kHz sampling
    N = 128
    f_D_true = 50.0  # Hz

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N, snr_db=np.inf)
    result = mobius_doppler_estimate(R, T_s)

    assert abs(result['f_D'] - f_D_true) / f_D_true < 0.01, \
        f"Estimated f_D={result['f_D']:.2f} != true {f_D_true:.2f}"

    assert result['direction'] == 'approaching'
    assert result['r_squared'] > 0.99, "Linear fit should be excellent for clean data"


def test_mobius_negative_doppler():
    """Algorithm 1 correctly identifies negative Doppler (receding target)."""
    T_s = 0.0005
    N = 128
    f_D_true = -30.0  # Receding target

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N)
    result = mobius_doppler_estimate(R, T_s)

    assert abs(result['f_D'] - f_D_true) / abs(f_D_true) < 0.05, \
        f"Estimated f_D={result['f_D']:.2f} != true {f_D_true:.2f}"

    assert result['direction'] == 'receding'
    assert result['f_D'] < 0, "f_D should be negative for receding target"


def test_mobius_with_noise():
    """Mobius estimator is robust to moderate noise."""
    T_s = 0.0005
    N = 128
    f_D_true = 40.0
    snr_db = 15.0  # 15 dB SNR

    np.random.seed(42)
    errors = []
    for _ in range(10):
        R, t = generate_csi_ratio_samples(f_D_true, T_s, N, snr_db=snr_db)
        result = mobius_doppler_estimate(R, T_s)
        errors.append(abs(result['f_D'] - f_D_true) / f_D_true)

    avg_error = np.mean(errors)
    assert avg_error < 0.15, f"Average error {avg_error:.2%} should be < 15% at 15 dB SNR"


def test_mobius_circle_fit_quality():
    """Circle fit quality metrics are reasonable."""
    T_s = 0.0005
    N = 128
    f_D_true = 60.0

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N)
    result = mobius_doppler_estimate(R, T_s)

    # Circle fit should be excellent for synthetic data
    assert result['rms_error'] < 0.01, \
        f"RMS circle fit error {result['rms_error']:.6f} should be very small"

    assert result['r_squared'] > 0.95


def test_mobius_different_frequencies():
    """Mobius estimator works for various Doppler frequencies."""
    T_s = 0.0005
    N = 128

    test_frequencies = [10, 25, 50, 75, 100, 200, 500]

    for f_D in test_frequencies:
        R, t = generate_csi_ratio_samples(f_D, T_s, N)
        result = mobius_doppler_estimate(R, T_s)
        rel_error = abs(result['f_D'] - f_D) / f_D

        assert rel_error < 0.05, \
            f"f_D={f_D}: estimated {result['f_D']:.2f}, error {rel_error:.2%}"


if __name__ == "__main__":
    test_mobius_doppler_estimate()
    test_mobius_negative_doppler()
    test_mobius_with_noise()
    test_mobius_circle_fit_quality()
    test_mobius_different_frequencies()
    print("All Mobius estimator tests passed!")
