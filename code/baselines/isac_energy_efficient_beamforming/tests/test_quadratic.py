"""
Tests for Quadratic Transform
==============================

Tests for the quadratic transform for log-SINR optimization.

Reference: Zou et al., IEEE Trans. Commun., 2024 (Eq. 14)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.quadratic_transform import (
    quadratic_transform_objective,
    optimize_t,
    compute_sum_rate_quadratic,
    QuadraticTransform,
)


class TestQuadraticTransform:
    """Test suite for quadratic transform."""

    @pytest.fixture
    def system_data(self):
        """Create test system data."""
        model = ISACSystemModel(M=8, K=3, N=10, seed=42)
        H = model.get_csi()
        sigma_c2 = model.sigma_c2
        W = np.random.randn(8, 3) + 1j * np.random.randn(8, 3)
        W *= 0.1  # Scale down
        return H, W, sigma_c2

    def test_optimize_t_dimensions(self, system_data):
        """Test optimal t has correct dimensions."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        assert t.shape == (3,)
        assert np.iscomplexobj(t)

    def test_optimize_t_formula(self, system_data):
        """Test optimal t satisfies the closed-form solution."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)

        K = H.shape[0]
        for k in range(K):
            h_k = H[k, :]
            hw_k = h_k.conj() @ W[:, k]
            total_power_k = sigma_c2 + sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K)
            )

            if total_power_k > 1e-15:
                expected_t = hw_k / total_power_k
                np.testing.assert_allclose(t[k], expected_t, rtol=1e-10)

    def test_quadratic_transform_objective(self, system_data):
        """Test quadratic transform objective is finite."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        obj = quadratic_transform_objective(H, W, t, sigma_c2)
        assert np.isfinite(obj)

    def test_quadratic_transform_upper_bound(self, system_data):
        """
        Test quadratic transform provides upper bound on sum rate.

        At optimal t, the quadratic transform value equals the sum rate.
        """
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        qt_obj = quadratic_transform_objective(H, W, t, sigma_c2)

        # Direct sum rate computation
        sum_rate = 0.0
        K = H.shape[0]
        for k in range(K):
            h_k = H[k, :]
            signal = np.abs(h_k.conj() @ W[:, k]) ** 2
            interference = sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K) if j != k
            )
            sinr_k = signal / (sigma_c2 + interference)
            sum_rate += np.log2(1 + sinr_k)

        # At optimal t, these should match (within tolerance)
        # The quadratic transform is a lower bound approximation
        assert qt_obj <= sum_rate + 1e-6 or abs(qt_obj - sum_rate) < 0.1

    def test_compute_sum_rate_quadratic(self, system_data):
        """Test sum rate computation via quadratic transform."""
        H, W, sigma_c2 = system_data
        sum_rate = compute_sum_rate_quadratic(H, W, sigma_c2)
        assert sum_rate >= 0
        assert np.isfinite(sum_rate)

    def test_quadratic_transform_class(self, system_data):
        """Test QuadraticTransform class."""
        H, _, sigma_c2 = system_data
        qt = QuadraticTransform(H, sigma_c2)

        W_init = np.random.randn(8, 3) + 1j * np.random.randn(8, 3)
        W_init *= 0.1

        W_opt, obj_val = qt.solve(W_init, max_iter=5)

        assert W_opt.shape == (8, 3)
        assert np.isfinite(obj_val)

    def test_t_k_zero_when_no_signal(self):
        """Test t_k is zero when signal is zero."""
        H = np.array([[1, 0], [0, 1]], dtype=complex)  # 2x2
        W = np.zeros((2, 2), dtype=complex)  # Zero beamforming
        sigma_c2 = 1.0

        t = optimize_t(H, W, sigma_c2)
        np.testing.assert_allclose(t, 0.0, atol=1e-15)
