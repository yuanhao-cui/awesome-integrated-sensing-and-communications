"""Tests for CSI-ratio computation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from csi_ratio import compute_csi_ratio, compute_csi_ratio_multi
from signal_model import csi_with_doppler


def test_csi_ratio_basic():
    """Test basic CSI-ratio computation."""
    H_m = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    H_m1 = np.array([1 + 0j, 2 + 0j, 3 + 0j])

    R = compute_csi_ratio(H_m, H_m1)

    expected = np.array([1 + 1j, 1 + 1j, 1 + 1j])
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_csi_ratio_cancels_offset():
    """Verify CSI-ratio cancels CFO/TMO."""
    N = 100
    fs = 2000  # 2 kHz
    T_s = 1.0 / fs
    t = np.arange(N) * T_s

    f_D = 50.0  # Hz
    cfo_hz = 100.0  # 100 Hz CFO
    tmo_hz = 20.0   # 20 Hz TMO

    H1, H2 = csi_with_doppler(t, f_D, snr_db=np.inf, cfo_hz=cfo_hz, tmo_hz=tmo_hz)
    R = compute_csi_ratio(H1, H2)

    # R should be constant magnitude (no CFO/TMO effect)
    magnitudes = np.abs(R)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-10,
                               err_msg="CSI-ratio magnitude varies (CFO/TMO not cancelled)")

    # Phase should increase linearly: angle(R) = 2π*f_D*t + const
    phase = np.unwrap(np.angle(R))
    # Slope should be 2π*f_D
    expected_slope = 2 * np.pi * f_D
    # Compute slope via linear regression
    slope, intercept = np.polyfit(t, phase, 1)
    assert abs(slope - expected_slope) < 1.0, \
        f"Phase slope {slope:.2f} != expected {expected_slope:.2f} (2π*{f_D})"


def test_csi_ratio_multi():
    """Test multi-antenna CSI-ratio computation."""
    N = 50
    M = 3  # 3 antennas
    H = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    R = compute_csi_ratio_multi(H)

    assert R.shape == (N, M - 1)
    for i in range(M - 1):
        expected = H[:, i] / H[:, i + 1]
        np.testing.assert_allclose(R[:, i], expected, atol=1e-10)


def test_csi_ratio_preserves_phase_difference():
    """CSI-ratio should preserve the phase difference between antennas."""
    N = 100
    t = np.arange(N) * 0.0005  # 2 kHz

    phase_diff = 0.5  # radians
    H1 = np.exp(1j * (2 * np.pi * 30 * t + phase_diff))
    H2 = np.exp(1j * 2 * np.pi * 30 * t)

    R = compute_csi_ratio(H1, H2)

    # All samples should have the same phase = phase_diff
    phases = np.angle(R)
    np.testing.assert_allclose(phases, phase_diff, atol=1e-10)


if __name__ == "__main__":
    test_csi_ratio_basic()
    test_csi_ratio_cancels_offset()
    test_csi_ratio_multi()
    test_csi_ratio_preserves_phase_difference()
    print("All CSI-ratio tests passed!")
