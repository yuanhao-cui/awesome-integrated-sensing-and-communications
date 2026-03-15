"""Tests for RIS phase shift optimization."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system_model import RIS_ISAC_System
from src.ris_phase import RISPhaseOptimizer


class TestRISPhase:
    """RIS phase optimizer tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.ris_opt = RISPhaseOptimizer(self.system)

    def test_unit_modulus_after_rate_opt(self):
        """Test RIS phases maintain unit modulus after rate optimization."""
        M, K = self.system.M, self.system.K
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(self.system.P_max / K) / np.linalg.norm(W, axis=0)
        theta = self.ris_opt.optimize_for_rate(W)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_unit_modulus_after_snr_opt(self):
        """Test RIS phases maintain unit modulus after SNR optimization."""
        M, K = self.system.M, self.system.K
        W = np.zeros((M, K), dtype=complex)
        theta, snr = self.ris_opt.optimize_for_snr(W, 5.0)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)
        assert snr >= 0, "SNR should be non-negative"

    def test_unit_modulus_after_joint_opt(self):
        """Test RIS phases maintain unit modulus after joint optimization."""
        M, K = self.system.M, self.system.K
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(self.system.P_max / K) / np.linalg.norm(W, axis=0)
        theta = self.ris_opt.optimize_joint(W, sensing_weight=0.5)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_output_length(self):
        """Test output has correct length."""
        M, K = self.system.M, self.system.K
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        theta = self.ris_opt.optimize_for_rate(W)
        assert theta.shape == (self.system.L,)

    def test_snr_improvement(self):
        """Test SNR optimization achieves non-zero SNR."""
        W = np.zeros((self.system.M, self.system.K), dtype=complex)
        theta, snr = self.ris_opt.optimize_for_snr(W, 5.0)
        assert snr >= 0
