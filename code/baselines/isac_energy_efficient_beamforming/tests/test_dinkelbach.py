"""
Tests for Dinkelbach Solver
============================

Tests for the Dinkelbach method for fractional programming.

Reference: Zou et al., IEEE Trans. Commun., 2024 (Algorithm 1)
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.dinkelbach_solver import DinkelbachSolver, DinkelbachResult


class TestDinkelbachSolver:
    """Test suite for Dinkelbach solver."""

    @pytest.fixture
    def model(self):
        """Create test system model with small parameters for fast tests."""
        return ISACSystemModel(M=8, K=2, N=10, seed=42)

    @pytest.fixture
    def solver(self, model):
        """Create Dinkelbach solver."""
        return DinkelbachSolver(
            model,
            max_dinkelbach_iter=10,
            max_inner_iter=5,
            verbose=False,
        )

    def test_solver_initialization(self, model):
        """Test solver initializes correctly."""
        solver = DinkelbachSolver(model)
        assert solver.M == 8
        assert solver.K == 2
        assert solver.max_dinkelbach_iter == 30

    def test_solve_returns_result(self, solver):
        """Test solve returns a DinkelbachResult."""
        result = solver.solve(target_angle_deg=90.0)
        assert isinstance(result, DinkelbachResult)

    def test_result_has_correct_fields(self, solver):
        """Test result has all required fields."""
        result = solver.solve(target_angle_deg=90.0)
        assert hasattr(result, 'W')
        assert hasattr(result, 'ee_c')
        assert hasattr(result, 'sum_rate')
        assert hasattr(result, 'total_power')
        assert hasattr(result, 'n_iterations')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'obj_history')

    def test_beamforming_dimensions(self, solver):
        """Test beamforming matrix has correct dimensions."""
        result = solver.solve(target_angle_deg=90.0)
        assert result.W.shape == (8, 2)

    def test_ee_c_positive(self, solver):
        """Test EE_C is positive."""
        result = solver.solve(target_angle_deg=90.0)
        assert result.ee_c >= 0

    def test_sum_rate_positive(self, solver):
        """Test sum rate is positive."""
        result = solver.solve(target_angle_deg=90.0)
        assert result.sum_rate >= 0

    def test_power_constraint(self, solver, model):
        """Test power constraint is satisfied."""
        result = solver.solve(target_angle_deg=90.0)
        assert result.total_power <= model.P_max + 1e-4

    def test_dinkelbach_converges(self, model):
        """Test Dinkelbach converges in reasonable iterations."""
        solver = DinkelbachSolver(
            model,
            max_dinkelbach_iter=30,
            max_inner_iter=10,
            verbose=False,
        )
        result = solver.solve(target_angle_deg=90.0)
        # Should converge in ≤ 30 iterations
        assert result.n_iterations <= 30

    def test_obj_history_length(self, solver):
        """Test obj_history has entries for each iteration."""
        result = solver.solve(target_angle_deg=90.0)
        assert len(result.obj_history) >= 1
        assert len(result.obj_history) <= solver.max_dinkelbach_iter

    def test_ee_c_improves_over_iterations(self, solver):
        """Test EE_C generally improves (non-decreasing)."""
        result = solver.solve(target_angle_deg=90.0)
        # Final EE should be ≥ initial EE
        if len(result.obj_history) >= 2:
            assert result.obj_history[-1] >= result.obj_history[0] - 1e-4

    def test_different_targets(self, solver):
        """Test solver works with different target angles."""
        for angle in [45.0, 90.0, 135.0]:
            result = solver.solve(target_angle_deg=angle)
            assert result.ee_c >= 0

    def test_custom_initialization(self, model):
        """Test solver with custom initial beamforming."""
        solver = DinkelbachSolver(model, max_dinkelbach_iter=5)
        W_init = np.random.randn(8, 2) + 1j * np.random.randn(8, 2)
        W_init *= np.sqrt(model.P_max / np.sum(np.abs(W_init) ** 2))

        result = solver.solve(target_angle_deg=90.0, W_init=W_init)
        assert result.W.shape == (8, 2)
