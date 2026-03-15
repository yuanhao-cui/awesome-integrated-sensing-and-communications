"""
Tests for Reproducibility
==========================

Tests to ensure results are reproducible across runs.

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.dinkelbach_solver import DinkelbachSolver
from src.ee_metrics import compute_ee_c, compute_ee_s, compute_crb


class TestReproducibility:
    """Test suite for reproducibility."""

    def test_same_seed_same_channels(self):
        """Test same seed produces identical channels."""
        model1 = ISACSystemModel(M=16, K=4, N=20, seed=42)
        model2 = ISACSystemModel(M=16, K=4, N=20, seed=42)

        np.testing.assert_array_equal(model1.H, model2.H)

    def test_same_seed_same_steering(self):
        """Test same seed produces identical steering vectors."""
        model1 = ISACSystemModel(M=16, K=4, N=20, seed=42)
        model2 = ISACSystemModel(M=16, K=4, N=20, seed=42)

        a_t1 = model1.steering_vector_tx(np.pi / 4)
        a_t2 = model2.steering_vector_tx(np.pi / 4)

        np.testing.assert_array_equal(a_t1, a_t2)

    def test_dinkelbach_reproducibility(self):
        """Test Dinkelbach solver gives consistent results with same seed."""
        model1 = ISACSystemModel(M=8, K=2, N=10, seed=42)
        model2 = ISACSystemModel(M=8, K=2, N=10, seed=42)

        solver1 = DinkelbachSolver(model1, max_dinkelbach_iter=5, verbose=False)
        solver2 = DinkelbachSolver(model2, max_dinkelbach_iter=5, verbose=False)

        result1 = solver1.solve(target_angle_deg=90.0)
        result2 = solver2.solve(target_angle_deg=90.0)

        # Should get same EE_C (within numerical tolerance)
        assert abs(result1.ee_c - result2.ee_c) < 1e-10
        assert abs(result1.sum_rate - result2.sum_rate) < 1e-10

    def test_ee_c_reproducibility(self):
        """Test EE_C computation is reproducible."""
        model = ISACSystemModel(M=8, K=2, N=10, seed=42)
        H = model.get_csi()

        W = np.random.randn(8, 2) + 1j * np.random.randn(8, 2)
        W *= 0.1

        ee1 = compute_ee_c(H, W, model.sigma_c2, model.epsilon, model.P0)
        ee2 = compute_ee_c(H, W, model.sigma_c2, model.epsilon, model.P0)

        assert ee1 == ee2

    def test_crb_reproducibility(self):
        """Test CRB computation is reproducible."""
        model = ISACSystemModel(M=8, K=2, N=10, seed=42)
        a_t = model.steering_vector_tx(np.pi / 2)
        a_r = model.steering_vector_rx(np.pi / 2)

        W = np.random.randn(8, 2) + 1j * np.random.randn(8, 2)
        W *= 0.1

        crb1 = compute_crb(W, a_t, a_r, model.sigma_s2, model.L)
        crb2 = compute_crb(W, a_t, a_r, model.sigma_s2, model.L)

        # Both should be the same (including inf)
        if np.isinf(crb1) and np.isinf(crb2):
            assert True
        else:
            assert abs(crb1 - crb2) < 1e-15

    def test_different_seeds_different_results(self):
        """Test different seeds produce different channels."""
        model1 = ISACSystemModel(M=16, K=4, N=20, seed=1)
        model2 = ISACSystemModel(M=16, K=4, N=20, seed=2)

        # Channels should be different (with high probability)
        assert not np.allclose(model1.H, model2.H)

    def test_sinr_reproducibility(self):
        """Test SINR computation is reproducible."""
        model = ISACSystemModel(M=8, K=2, N=10, seed=42)
        H = model.get_csi()

        W = np.random.randn(8, 2) + 1j * np.random.randn(8, 2)
        W *= 0.1

        sinr1 = model.compute_sinr(0, W)
        sinr2 = model.compute_sinr(0, W)

        assert abs(sinr1 - sinr2) < 1e-15
