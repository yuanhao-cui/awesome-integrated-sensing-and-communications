"""
Tests for Energy Efficiency Metrics
====================================

Tests for EE_C, EE_S, CRB, and SINR computations.

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.ee_metrics import (
    compute_sinr,
    compute_sum_rate,
    compute_total_power,
    compute_ee_c,
    compute_ee_s,
    compute_crb,
)


class TestEEMetrics:
    """Test suite for energy efficiency metrics."""

    @pytest.fixture
    def model(self):
        """Create test system model."""
        return ISACSystemModel(M=16, K=4, N=20, seed=42)

    @pytest.fixture
    def beamforming_matrix(self, model):
        """Create matched filter beamforming matrix."""
        H = model.get_csi()
        M, K = model.M, model.K
        W = np.zeros((M, K), dtype=complex)
        for k in range(K):
            h_k = H[k, :]
            W[:, k] = h_k / np.linalg.norm(h_k) * np.sqrt(model.P_max / K)
        return W

    def test_sinr_positive(self, model, beamforming_matrix):
        """Test SINR is always positive."""
        H = model.get_csi()
        for k in range(model.K):
            sinr_k = compute_sinr(k, H[k, :], beamforming_matrix, model.sigma_c2)
            assert sinr_k >= 0, f"SINR for user {k} is negative: {sinr_k}"

    def test_sinr_increases_with_power(self, model):
        """Test SINR increases with transmit power."""
        H = model.get_csi()
        h_0 = H[0, :]

        W_low = np.zeros((model.M, model.K), dtype=complex)
        W_low[:, 0] = h_0 * 0.1

        W_high = np.zeros((model.M, model.K), dtype=complex)
        W_high[:, 0] = h_0 * 1.0

        sinr_low = compute_sinr(0, h_0, W_low, model.sigma_c2)
        sinr_high = compute_sinr(0, h_0, W_high, model.sigma_c2)

        assert sinr_high > sinr_low

    def test_sum_rate_positive(self, model, beamforming_matrix):
        """Test sum rate is positive."""
        H = model.get_csi()
        sum_rate = compute_sum_rate(H, beamforming_matrix, model.sigma_c2)
        assert sum_rate >= 0

    def test_sum_rate_monotonic(self, model):
        """Test sum rate increases with total power."""
        H = model.get_csi()
        W_base = np.random.randn(model.M, model.K) + 1j * np.random.randn(model.M, model.K)

        sr_low = compute_sum_rate(H, W_base * 0.5, model.sigma_c2)
        sr_high = compute_sum_rate(H, W_base * 2.0, model.sigma_c2)

        assert sr_high > sr_low

    def test_total_power(self, model, beamforming_matrix):
        """Test total power computation."""
        power = compute_total_power(beamforming_matrix)
        assert power >= 0
        # Should be close to P_max (after normalization)
        assert power <= model.P_max + 1e-10

    def test_ee_fractional_structure(self, model, beamforming_matrix):
        """Test EE has correct fractional structure EE = Rate/Power."""
        H = model.get_csi()
        ee_c = compute_ee_c(
            H, beamforming_matrix, model.sigma_c2,
            model.epsilon, model.P0,
        )

        sum_rate = compute_sum_rate(H, beamforming_matrix, model.sigma_c2)
        total_power = compute_total_power(beamforming_matrix)
        total_consumption = (1 / model.epsilon) * total_power + model.P0

        ee_manual = sum_rate / total_consumption

        assert abs(ee_c - ee_manual) < 1e-10, f"EE mismatch: {ee_c} vs {ee_manual}"

    def test_ee_c_positive(self, model, beamforming_matrix):
        """Test communication EE is positive."""
        H = model.get_csi()
        ee_c = compute_ee_c(
            H, beamforming_matrix, model.sigma_c2,
            model.epsilon, model.P0,
        )
        assert ee_c >= 0

    def test_ee_c_zero_power(self, model):
        """Test EE_C is zero when power is zero."""
        H = model.get_csi()
        W_zero = np.zeros((model.M, model.K), dtype=complex)
        ee_c = compute_ee_c(H, W_zero, model.sigma_c2, model.epsilon, model.P0)
        assert ee_c == 0.0

    def test_crb_decreases_with_power(self, model):
        """Test CRB decreases as power increases."""
        theta_rad = np.pi / 2
        a_t = model.steering_vector_tx(theta_rad)
        a_r = model.steering_vector_rx(theta_rad)

        W_low = np.random.randn(model.M, model.K) + 1j * np.random.randn(model.M, model.K)
        W_low *= 0.1

        W_high = W_low * 10.0

        crb_low = compute_crb(W_low, a_t, a_r, model.sigma_s2, model.L)
        crb_high = compute_crb(W_high, a_t, a_r, model.sigma_s2, model.L)

        # CRB should decrease (better estimation) with more power
        # Note: CRB_high should be lower
        assert crb_high <= crb_low or crb_low == float('inf')

    def test_crb_positive(self, model, beamforming_matrix):
        """Test CRB is positive."""
        theta_rad = np.pi / 2
        a_t = model.steering_vector_tx(theta_rad)
        a_r = model.steering_vector_rx(theta_rad)

        crb = compute_crb(beamforming_matrix, a_t, a_r, model.sigma_s2, model.L)
        assert crb > 0 or np.isinf(crb)

    def test_ee_s_positive(self, model, beamforming_matrix):
        """Test sensing EE is positive or zero."""
        theta_rad = np.pi / 2
        a_t = model.steering_vector_tx(theta_rad)
        a_r = model.steering_vector_rx(theta_rad)

        ee_s = compute_ee_s(
            beamforming_matrix, a_t, a_r, model.sigma_s2,
            model.L, model.epsilon, model.P0,
        )
        assert ee_s >= 0

    def test_ee_s_zero_power(self, model):
        """Test EE_S is zero when power is zero."""
        theta_rad = np.pi / 2
        a_t = model.steering_vector_tx(theta_rad)
        a_r = model.steering_vector_rx(theta_rad)

        W_zero = np.zeros((model.M, model.K), dtype=complex)
        ee_s = compute_ee_s(
            W_zero, a_t, a_r, model.sigma_s2,
            model.L, model.epsilon, model.P0,
        )
        assert ee_s == 0.0

    def test_ee_c_increases_with_efficiency(self, model, beamforming_matrix):
        """Test EE_C increases with PA efficiency."""
        H = model.get_csi()

        ee_low = compute_ee_c(
            H, beamforming_matrix, model.sigma_c2, 0.2, model.P0,
        )
        ee_high = compute_ee_c(
            H, beamforming_matrix, model.sigma_c2, 0.8, model.P0,
        )

        assert ee_high >= ee_low
