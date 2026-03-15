"""Tests for circle fitting algorithms."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from circle_fit import least_squares_circle_fit, fit_circle_kasa, fit_circle_pratt, circle_fit_error


def generate_circle_samples(center_A, center_B, radius, n_samples=100, noise_std=0.0):
    """Generate samples on a circle with optional noise."""
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x = center_A + radius * np.cos(angles) + noise_std * np.random.randn(n_samples)
    y = center_B + radius * np.sin(angles) + noise_std * np.random.randn(n_samples)
    return x + 1j * y


def test_circle_fit_accuracy():
    """Circle fitting recovers center within tolerance."""
    # Test case 1: Centered at origin
    R = generate_circle_samples(0, 0, 1.0, n_samples=200, noise_std=0.0)
    A, B, r = least_squares_circle_fit(R)
    assert abs(A) < 1e-10, f"Center A={A} should be ~0"
    assert abs(B) < 1e-10, f"Center B={B} should be ~0"
    assert abs(r - 1.0) < 1e-10, f"Radius r={r} should be ~1"

    # Test case 2: Off-center
    R = generate_circle_samples(2.5, -1.3, 3.7, n_samples=300, noise_std=0.0)
    A, B, r = least_squares_circle_fit(R)
    assert abs(A - 2.5) < 1e-8, f"Center A={A} should be ~2.5"
    assert abs(B - (-1.3)) < 1e-8, f"Center B={B} should be ~-1.3"
    assert abs(r - 3.7) < 1e-8, f"Radius r={r} should be ~3.7"


def test_circle_fit_with_noise():
    """Circle fitting works with noisy samples."""
    true_A, true_B, true_r = 1.5, -0.8, 2.0
    noise_std = 0.05  # 5% noise relative to radius

    np.random.seed(42)
    R = generate_circle_samples(true_A, true_B, true_r,
                                n_samples=200, noise_std=noise_std)

    A, B, r = least_squares_circle_fit(R)

    # Should be within 5% of true values
    assert abs(A - true_A) / true_r < 0.05
    assert abs(B - true_B) / true_r < 0.05
    assert abs(r - true_r) / true_r < 0.05


def test_circle_fit_methods_agree():
    """All circle fitting methods give similar results for clean data."""
    R = generate_circle_samples(1.0, -2.0, 1.5, n_samples=150)

    A1, B1, r1 = least_squares_circle_fit(R)
    A2, B2, r2 = fit_circle_kasa(R)
    A3, B3, r3 = fit_circle_pratt(R)

    # All methods should agree within 1%
    for (A, B, r) in [(A2, B2, r2), (A3, B3, r3)]:
        assert abs(A - A1) / r1 < 0.01
        assert abs(B - B1) / r1 < 0.01
        assert abs(r - r1) / r1 < 0.01


def test_circle_fit_error():
    """Circle fit error computation is correct."""
    R = generate_circle_samples(0, 0, 1.0, n_samples=100)
    error = circle_fit_error(R, 0, 0, 1.0)
    assert error < 1e-10, f"Error for perfect circle should be ~0, got {error}"

    # With wrong center
    error_wrong = circle_fit_error(R, 0.1, 0, 1.0)
    assert error_wrong > error, "Wrong center should give larger error"


if __name__ == "__main__":
    test_circle_fit_accuracy()
    test_circle_fit_with_noise()
    test_circle_fit_methods_agree()
    test_circle_fit_error()
    print("All circle fit tests passed!")
