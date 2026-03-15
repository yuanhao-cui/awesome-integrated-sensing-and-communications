"""Tests for CRB-constrained solver."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system_model import RIS_ISAC_System
from src.crb_constraint import CRBConstrainedSolver


class TestCRBConstrained:
    """CRB-constrained solver tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.solver = CRBConstrainedSolver(self.system, crb_max=1e-2, max_iter=10)

    def test_crb_output_keys(self):
        """Test solver output contains expected keys."""
        result = self.solver.solve()
        expected_keys = {"W", "theta", "sum_rate", "crb", "converged", "iterations"}
        assert expected_keys.issubset(result.keys())

    def test_crb_beamforming_shape(self):
        """Test beamforming matrix has correct shape."""
        result = self.solver.solve()
        M, K = self.system.M, self.system.K
        assert result["W"].shape == (M, K)

    def test_crb_ris_unit_modulus(self):
        """Test RIS phases satisfy unit modulus."""
        result = self.solver.solve()
        theta = result["theta"]
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_crb_computation(self):
        """Test CRB computation returns positive value."""
        result = self.solver.solve()
        assert result["crb"] > 0, f"CRB should be positive, got {result['crb']}"

    def test_crb_positive_rate(self):
        """Test sum rate is positive."""
        result = self.solver.solve()
        assert result["sum_rate"] > 0

    def test_crb_constraint_positive(self):
        """Test CRB constraint function computes correctly."""
        w = np.random.randn(self.system.M) + 1j * np.random.randn(self.system.M)
        crb = self.solver.compute_crb(w)
        assert crb > 0
        assert np.isfinite(crb)

    def test_crb_history(self):
        """Test optimization history is recorded."""
        result = self.solver.solve()
        assert len(result["history"]) > 0
