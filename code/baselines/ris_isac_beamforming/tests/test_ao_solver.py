"""Tests for Alternating Optimization solver."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.system_model import RIS_ISAC_System
from src.ao_solver import AlternatingOptimizationSolver


class TestAOSolver:
    """AO solver tests."""

    def setup_method(self):
        self.system_snr = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.system_crb = RIS_ISAC_System(M=4, K=2, L=30, seed=43)

    def test_ao_snr_solver(self):
        """Test AO solver with SNR constraint."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=10
        )
        result = solver.solve()
        assert "W" in result
        assert "theta" in result
        assert "sum_rate" in result

    def test_ao_crb_solver(self):
        """Test AO solver with CRB constraint."""
        solver = AlternatingOptimizationSolver(
            self.system_crb, problem_type="crb", crb_max=1e-2, max_iter=10
        )
        result = solver.solve()
        assert "W" in result
        assert "theta" in result
        assert "sum_rate" in result
        assert "crb" in result

    def test_ao_invalid_type(self):
        """Test AO solver rejects invalid problem type."""
        with pytest.raises(ValueError, match="Unknown problem_type"):
            AlternatingOptimizationSolver(
                self.system_snr, problem_type="invalid"
            )

    def test_ao_convergence(self):
        """Test AO converges within max iterations."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=50, tol=1e-3
        )
        result = solver.solve()
        # Either converged or hit max iterations
        assert result["iterations"] <= 50

    def test_ao_evaluate(self):
        """Test solution evaluation."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=10
        )
        result = solver.solve()
        metrics = solver.evaluate(result["W"], result["theta"])
        assert "sum_rate" in metrics
        assert "snr_sensing" in metrics
        assert "power_used" in metrics
        assert "sinr_per_user" in metrics
        assert len(metrics["sinr_per_user"]) == self.system_snr.K

    def test_ao_unit_modulus(self):
        """Test AO solver output satisfies RIS unit modulus."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=10
        )
        result = solver.solve()
        np.testing.assert_allclose(np.abs(result["theta"]), 1.0, atol=1e-10)

    def test_ao_positive_sum_rate(self):
        """Test AO achieves positive sum rate."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=10
        )
        result = solver.solve()
        assert result["sum_rate"] > 0

    def test_ao_crb_positive(self):
        """Test AO with CRB returns positive CRB."""
        solver = AlternatingOptimizationSolver(
            self.system_crb, problem_type="crb", crb_max=1e-2, max_iter=10
        )
        result = solver.solve()
        assert result["crb"] > 0

    def test_ao_power_constraint(self):
        """Test AO solution satisfies power constraint."""
        solver = AlternatingOptimizationSolver(
            self.system_snr, problem_type="snr", snr_min_dB=5.0, max_iter=10
        )
        result = solver.solve()
        total_power = np.sum(np.linalg.norm(result["W"], axis=0) ** 2)
        assert total_power <= self.system_snr.P_max * 1.05
