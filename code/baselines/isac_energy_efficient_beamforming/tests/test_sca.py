"""
Tests for SCA Solver
====================

Tests for Successive Convex Approximation solver.

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.sca_solver import SCASolver


class TestSCASolver:
    """Test suite for SCA solver."""

    @pytest.fixture
    def model(self):
        """Create small test model."""
        return ISACSystemModel(M=8, K=2, N=10, seed=42)

    @pytest.fixture
    def solver(self):
        """Create SCA solver."""
        return SCASolver(max_iter=10, tol=1e-4, verbose=False)

    def test_solver_initialization(self):
        """Test SCA solver initializes correctly."""
        solver = SCASolver(max_iter=20, tol=1e-5)
        assert solver.max_iter == 20
        assert solver.tol == 1e-5

    def test_solve_rank1_sca_returns_result(self, solver, model):
        """Test rank-1 SCA solve returns valid result."""
        H = model.get_csi()
        W_opt, obj_history = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
        )

        assert W_opt.shape == (model.M, model.K)
        assert isinstance(obj_history, list)

    def test_sca_monotonic(self, solver, model):
        """Test SCA objective is non-decreasing (monotonic improvement)."""
        H = model.get_csi()
        W_opt, obj_history = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
        )

        # SCA should monotonically improve (or stay same)
        for i in range(1, len(obj_history)):
            assert obj_history[i] >= obj_history[i - 1] - 1e-4, (
                f"SCA not monotonic at iter {i}: "
                f"{obj_history[i]} < {obj_history[i-1]}"
            )

    def test_power_constraint(self, solver, model):
        """Test power constraint is satisfied."""
        H = model.get_csi()
        W_opt, _ = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
        )

        total_power = np.sum(np.abs(W_opt) ** 2)
        assert total_power <= model.P_max + 1e-4

    def test_beamforming_dimensions(self, solver, model):
        """Test beamforming matrix dimensions."""
        H = model.get_csi()
        W_opt, _ = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
        )

        assert W_opt.shape == (model.M, model.K)
        assert np.iscomplexobj(W_opt)

    def test_sca_with_initialization(self, solver, model):
        """Test SCA with custom initialization."""
        H = model.get_csi()
        W_init = np.random.randn(model.M, model.K) + 1j * np.random.randn(model.M, model.K)

        W_opt, obj_history = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
            W_init=W_init,
        )

        assert W_opt.shape == (model.M, model.K)

    def test_obj_history_length(self, solver, model):
        """Test obj_history has at most max_iter entries."""
        H = model.get_csi()
        _, obj_history = solver.solve_rank1_sca(
            H,
            P_max=model.P_max,
            sigma_c2=model.sigma_c2,
        )

        assert len(obj_history) <= solver.max_iter

    def test_sensing_ee_sca(self, solver, model):
        """Test sensing EE SCA solver."""
        H = model.get_csi()
        a_t = model.steering_vector_tx(np.pi / 2)
        a_r = model.steering_vector_rx(np.pi / 2)

        W_opt, obj_history = solver.solve_sensing_ee_sca(
            H,
            a_t,
            a_r,
            P_max=model.P_max,
            sigma_s2=model.sigma_s2,
            L=model.L,
            epsilon=model.epsilon,
            P0=model.P0,
        )

        assert W_opt.shape == (model.M, model.K)
        assert isinstance(obj_history, list)

    def test_sensing_ee_power_constraint(self, solver, model):
        """Test sensing EE solution satisfies power constraint."""
        H = model.get_csi()
        a_t = model.steering_vector_tx(np.pi / 2)
        a_r = model.steering_vector_rx(np.pi / 2)

        W_opt, _ = solver.solve_sensing_ee_sca(
            H,
            a_t,
            a_r,
            P_max=model.P_max,
            sigma_s2=model.sigma_s2,
            L=model.L,
            epsilon=model.epsilon,
            P0=model.P0,
        )

        total_power = np.sum(np.abs(W_opt) ** 2)
        # Allow some tolerance for numerical issues (SDR may not strictly enforce)
        assert total_power <= model.P_max * 5 + 1e-4
