"""Tests for SNR-constrained solver."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system_model import RIS_ISAC_System
from src.snr_constraint import SNRConstrainedSolver


class TestSNRConstrained:
    """SNR-constrained solver tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.solver = SNRConstrainedSolver(self.system, snr_min_dB=5.0, max_iter=10)

    def test_snr_constraint_output_keys(self):
        """Test solver output contains expected keys."""
        result = self.solver.solve()
        expected_keys = {"W", "theta", "sum_rate", "snr_sensing", "converged", "iterations"}
        assert expected_keys.issubset(result.keys())

    def test_snr_constraint_beamforming_shape(self):
        """Test beamforming matrix has correct shape."""
        result = self.solver.solve()
        M, K = self.system.M, self.system.K
        assert result["W"].shape == (M, K)

    def test_snr_constraint_ris_unit_modulus(self):
        """Test RIS phases from SNR solver satisfy unit modulus."""
        result = self.solver.solve()
        theta = result["theta"]
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_snr_constraint_sensing_channel(self):
        """Test sensing channel computation."""
        h_s = self.solver._compute_sensing_channel()
        assert h_s.shape == (self.system.M,)
        assert h_s.dtype == complex

    def test_snr_constraint_positive_rate(self):
        """Test sum rate is positive."""
        result = self.solver.solve()
        assert result["sum_rate"] > 0, f"Sum rate: {result['sum_rate']}"

    def test_snr_constraint_history(self):
        """Test optimization history is recorded."""
        result = self.solver.solve()
        assert len(result["history"]) > 0
        assert len(result["history"]) <= self.solver.max_iter
