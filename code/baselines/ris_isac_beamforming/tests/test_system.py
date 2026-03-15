"""Tests for RIS-ISAC system model."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system_model import RIS_ISAC_System


class TestSystemModel:
    """System model unit tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)

    def test_ris_unit_modulus(self):
        """Test that all RIS phase elements have unit modulus |θ_l| = 1."""
        theta = self.system.theta
        magnitudes = np.abs(theta)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10,
                                   err_msg="RIS phases must have |θ_l| = 1")

    def test_ris_unit_modulus_after_set(self):
        """Test unit modulus after setting arbitrary phases."""
        theta_new = 3.0 * np.exp(1j * np.linspace(0, np.pi, self.system.L))
        self.system.set_ris_phases(theta_new)
        magnitudes = np.abs(self.system.theta)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)

    def test_channel_dimensions(self):
        """Test that channel matrices have correct shapes."""
        ch = self.system.channels
        M, K, L = self.system.M, self.system.K, self.system.L

        assert ch["H_BR"].shape == (L, M), f"H_BR shape: {ch['H_BR'].shape}"
        assert ch["G"].shape == (K, L), f"G shape: {ch['G'].shape}"
        assert ch["h_d"].shape == (K, M), f"h_d shape: {ch['h_d'].shape}"
        assert ch["a_bs"].shape == (M,), f"a_bs shape: {ch['a_bs'].shape}"
        assert ch["a_ris"].shape == (L,), f"a_ris shape: {ch['a_ris'].shape}"

    def test_ris_diagonal_matrix(self):
        """Test RIS diagonal matrix construction."""
        Theta = self.system.ris_diagonal_matrix()
        assert Theta.shape == (self.system.L, self.system.L)
        # Check it's diagonal
        off_diag = Theta - np.diag(np.diag(Theta))
        np.testing.assert_allclose(off_diag, 0, atol=1e-10)
        # Check diagonal entries are unit-modulus
        np.testing.assert_allclose(np.abs(np.diag(Theta)), 1.0, atol=1e-10)

    def test_effective_channel_shape(self):
        """Test effective channel has correct shape for each user."""
        for k in range(self.system.K):
            h_eff = self.system.effective_channel(k)
            assert h_eff.shape == (self.system.M,)

    def test_sum_rate_positive(self):
        """Test that sum rate is positive for valid beamforming."""
        M, K = self.system.M, self.system.K
        P_max = self.system.P_max
        # Simple beamforming: equal power allocation
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(P_max / K) / np.linalg.norm(W, axis=0)
        rate = self.system.compute_sum_rate(W)
        assert rate > 0, f"Sum rate should be positive, got {rate}"

    def test_power_constraint(self):
        """Test power constraint: Σ||w_k||² ≤ P_max."""
        M, K = self.system.M, self.system.K
        P_max = self.system.P_max
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(P_max / K) / np.linalg.norm(W, axis=0)
        total_power = np.sum(np.linalg.norm(W, axis=0) ** 2)
        assert total_power <= P_max * 1.01, f"Power {total_power} exceeds P_max {P_max}"

    def test_reset_channels(self):
        """Test channel regeneration with new seed."""
        H_BR_old = self.system.channels["H_BR"].copy()
        self.system.reset_channels(seed=123)
        H_BR_new = self.system.channels["H_BR"]
        # Should be different (with high probability)
        assert not np.allclose(H_BR_old, H_BR_new)
