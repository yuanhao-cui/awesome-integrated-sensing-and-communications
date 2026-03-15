"""
Tests for System Model
======================

Tests for the ISAC system model including:
- SINR computation
- Channel generation
- Steering vectors
- Power computation

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel


class TestSystemModel:
    """Test suite for ISACSystemModel."""

    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        assert model.M == 16
        assert model.K == 4
        assert model.N == 20
        assert model.H.shape == (4, 16)

    def test_power_conversion(self):
        """Test dBm to linear power conversion."""
        model = ISACSystemModel(P_max_dbm=30, P0_dbm=33)
        # 30 dBm = 1 Watt
        assert abs(model.P_max - 1.0) < 1e-6
        # 33 dBm ≈ 2 Watts
        assert abs(model.P0 - 2.0) < 0.1

    def test_channel_dimensions(self):
        """Test channel matrix has correct dimensions."""
        for M in [8, 16, 32]:
            for K in [2, 4, 8]:
                model = ISACSystemModel(M=M, K=K, N=20, seed=42)
                assert model.H.shape == (K, M)

    def test_channel_complex(self):
        """Test channel matrix is complex."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        assert np.iscomplexobj(model.H)

    def test_channel_normalization(self):
        """Test channel has correct power (approx unit variance)."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        # Each element should have variance ≈ 1 (CN(0,1))
        power_per_element = np.mean(np.abs(model.H) ** 2)
        assert 0.5 < power_per_element < 2.0

    def test_steering_vector_tx_dimensions(self):
        """Test transmit steering vector dimensions."""
        model = ISACSystemModel(M=16, K=4, N=20)
        a_t = model.steering_vector_tx(np.pi / 4)
        assert a_t.shape == (16,)
        assert np.iscomplexobj(a_t)

    def test_steering_vector_rx_dimensions(self):
        """Test receive steering vector dimensions."""
        model = ISACSystemModel(M=16, K=4, N=20)
        a_r = model.steering_vector_rx(np.pi / 4)
        assert a_r.shape == (20,)
        assert np.iscomplexobj(a_r)

    def test_steering_vector_unit_magnitude(self):
        """Test steering vectors have unit magnitude elements."""
        model = ISACSystemModel(M=16, K=4, N=20)
        a_t = model.steering_vector_tx(np.pi / 4)
        # Each element should have |a_t[m]| = 1
        np.testing.assert_allclose(np.abs(a_t), 1.0, atol=1e-10)

    def test_get_channel(self):
        """Test get_channel returns correct user channel."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        h_0 = model.get_channel(0)
        np.testing.assert_array_equal(h_0, model.H[0, :])

    def test_sinr_computation(self):
        """Test SINR is positive and increases with power."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        H = model.get_csi()

        # Create beamforming matrix (matched filter)
        W = np.zeros((16, 4), dtype=complex)
        for k in range(4):
            h_k = H[k, :]
            W[:, k] = h_k / np.linalg.norm(h_k)

        # Low power
        W_low = W * 0.1
        sinr_low = model.compute_sinr(0, W_low)

        # High power
        W_high = W * 1.0
        sinr_high = model.compute_sinr(0, W_high)

        # SINR should be positive
        assert sinr_low > 0
        assert sinr_high > 0

        # SINR should increase with power
        assert sinr_high > sinr_low

    def test_sinr_vector(self):
        """Test compute_sinr_vector returns correct dimensions."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        sinr_vec = model.compute_sinr_vector(W)
        assert sinr_vec.shape == (4,)
        assert np.all(sinr_vec >= 0)

    def test_total_power(self):
        """Test total power computation."""
        model = ISACSystemModel(M=16, K=4, N=20)
        W = np.ones((16, 4), dtype=complex)
        total_power = model.compute_total_power(W)
        assert total_power > 0
        # 16 * 4 * |1|² = 64
        assert abs(total_power - 64.0) < 1e-10

    def test_channel_regeneration(self):
        """Test channel regeneration produces different channels."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        H1 = model.get_csi()
        model.regenerate_channels(seed=123)
        H2 = model.get_csi()
        # Channels should be different
        assert not np.allclose(H1, H2)

    def test_power_constraint(self):
        """Test power constraint is enforced."""
        model = ISACSystemModel(M=16, K=4, N=20, P_max_dbm=30, seed=42)
        P_max = model.P_max

        # Random beamforming
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        W = W / np.sqrt(np.sum(np.abs(W) ** 2)) * np.sqrt(P_max)

        total_power = model.compute_total_power(W)
        assert total_power <= P_max + 1e-10

    def test_noise_power(self):
        """Test noise power conversion."""
        model = ISACSystemModel(sigma_c_db=-80)
        # -80 dB should be 1e-11
        assert abs(model.sigma_c2 - 1e-11) < 1e-8

    def test_csi_copy(self):
        """Test get_csi returns a copy."""
        model = ISACSystemModel(M=16, K=4, N=20, seed=42)
        H1 = model.get_csi()
        H1[0, 0] = 999  # Modify copy
        H2 = model.get_csi()
        assert H2[0, 0] != 999  # Original unchanged
