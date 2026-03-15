"""Tests for bounds.py - Pentagon, Gaussian, Semi-unitary, and Outer bounds."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bounds import (
    pentagon_inner_bound,
    gaussian_inner_bound,
    semi_unitary_inner_bound,
    outer_bound,
    compute_corner_points,
)
from system_model import angle_to_hfunc, compute_phi_angle


def setup_test_params():
    """Setup common test parameters."""
    M, Ns, Nc = 4, 4, 1
    T = 5
    P_T = 1.0
    sigma_c2 = 0.1
    sigma_s2 = 0.1

    theta_c = np.deg2rad(42)
    theta_target = np.deg2rad(30)

    # LoS channel for communication
    positions = np.arange(M) * 0.5
    a_c = np.exp(1j * 2 * np.pi * positions * np.sin(theta_c))
    Hc = a_c.reshape(Nc, M)

    Hs_func = angle_to_hfunc(M, Ns)

    def phi_func(Rx):
        return compute_phi_angle(Rx, T, theta_target, M, Ns)

    Jp = np.array([[10.0]])  # Prior precision

    return {
        'M': M, 'Ns': Ns, 'Nc': Nc, 'T': T,
        'P_T': P_T, 'sigma_c2': sigma_c2, 'sigma_s2': sigma_s2,
        'Hc': Hc, 'Hs_func': Hs_func, 'phi_func': phi_func,
        'Jp': Jp,
    }


class TestPentagonInnerBound:
    """Tests for pentagon inner bound."""

    def test_pentagon_boundary_shape(self):
        """Test pentagon boundary has matching array lengths."""
        P_sc = (0.1, 1.0)
        P_cs = (0.5, 3.0)
        e_min = 0.1
        R_max = 3.0

        e_bound, R_bound = pentagon_inner_bound(P_sc, P_cs, e_min, R_max)
        assert len(e_bound) == len(R_bound)
        assert len(e_bound) > 0

    def test_pentagon_starts_at_emin(self):
        """Test pentagon boundary starts at e = e_min."""
        P_sc = (0.1, 1.0)
        P_cs = (0.5, 3.0)
        e_min = 0.1
        R_max = 3.0

        e_bound, R_bound = pentagon_inner_bound(P_sc, P_cs, e_min, R_max)
        # First point should be at e_min
        assert np.abs(e_bound[0] - e_min) < 1e-10

    def test_pentagon_satisfies_constraints(self):
        """Test all pentagon points satisfy the region constraints."""
        P_sc = (0.1, 1.0)
        P_cs = (0.5, 3.0)
        e_min = 0.1
        R_max = 3.0

        e_bound, R_bound = pentagon_inner_bound(P_sc, P_cs, e_min, R_max)

        # All e values should be >= e_min
        assert np.all(e_bound >= e_min - 1e-10)

        # All R values should be <= R_max
        assert np.all(R_bound <= R_max + 1e-10)

    def test_pentagon_corners(self):
        """Test pentagon includes the corner points."""
        e_min, R_sc = 0.1, 1.0
        e_cs, R_max = 0.5, 3.0

        e_bound, R_bound = pentagon_inner_bound(
            (e_min, R_sc), (e_cs, R_max), e_min, R_max
        )

        # Should pass through P_sc
        assert np.any(np.abs(R_bound - R_sc) < 0.1)
        assert np.any(np.abs(e_bound - e_min) < 0.1)


class TestCornerPoints:
    """Tests for corner point computation."""

    def test_corner_points_emin_le_ecs(self):
        """Test e_min <= e_cs (sensing-optimal CRB <= comm-optimal CRB)."""
        p = setup_test_params()

        corners = compute_corner_points(
            p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        assert corners['e_min'] <= corners['e_cs'] + 1e-6

    def test_corner_points_rsc_le_rmax(self):
        """Test R_sc <= R_max."""
        p = setup_test_params()

        corners = compute_corner_points(
            p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        assert corners['R_sc'] <= corners['R_max'] + 1e-6

    def test_corner_points_positive(self):
        """Test all corner point values are positive."""
        p = setup_test_params()

        corners = compute_corner_points(
            p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        assert corners['e_min'] > 0
        assert corners['e_cs'] > 0
        assert corners['R_sc'] >= 0
        assert corners['R_max'] >= 0


class TestGaussianInnerBound:
    """Tests for Gaussian signaling inner bound."""

    def test_gaussian_bound_output_shape(self):
        """Test Gaussian bound returns arrays."""
        p = setup_test_params()
        alpha_vals = np.linspace(0, 1, 5)

        e_vals, R_vals, alpha_used = gaussian_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        assert len(e_vals) == len(R_vals)
        assert len(e_vals) == len(alpha_used)

    def test_gaussian_bound_monotonic_rate(self):
        """Test rate increases with alpha (communication priority)."""
        p = setup_test_params()
        alpha_vals = np.linspace(0, 0.95, 20)

        e_vals, R_vals, alpha_used = gaussian_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        if len(R_vals) > 2:
            # Rate should generally increase with alpha
            # (allowing for numerical noise)
            assert R_vals[-1] >= R_vals[0] - 0.1

    def test_gaussian_bound_values_positive(self):
        """Test Gaussian bound produces positive values."""
        p = setup_test_params()
        alpha_vals = np.array([0.0, 0.5, 1.0])

        e_vals, R_vals, _ = gaussian_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        if len(e_vals) > 0:
            assert np.all(e_vals > 0)
            assert np.all(R_vals >= 0)


class TestSemiUnitaryInnerBound:
    """Tests for semi-unitary (Stiefel) inner bound."""

    def test_su_bound_output_shape(self):
        """Test SU bound returns arrays."""
        p = setup_test_params()
        alpha_vals = np.linspace(0, 1, 3)

        e_vals, R_vals, alpha_used = semi_unitary_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], M_sc=min(p['M'], p['T']),
            Jp=p['Jp'], Nc=p['Nc'], n_stiefel_samples=5
        )

        assert len(e_vals) == len(R_vals)

    def test_su_bound_values_positive(self):
        """Test SU bound produces positive values."""
        p = setup_test_params()
        alpha_vals = np.array([0.5])

        e_vals, R_vals, _ = semi_unitary_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], M_sc=min(p['M'], p['T']),
            Jp=p['Jp'], Nc=p['Nc'], n_stiefel_samples=5
        )

        if len(e_vals) > 0:
            assert np.all(e_vals > 0)


class TestOuterBound:
    """Tests for outer bound."""

    def test_outer_bound_output_shape(self):
        """Test outer bound returns arrays."""
        p = setup_test_params()
        alpha_vals = np.linspace(0, 1, 5)

        e_vals, R_vals, alpha_used = outer_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        assert len(e_vals) == len(R_vals)

    def test_outer_bound_tightness(self):
        """Test outer bound is >= Gaussian inner bound."""
        p = setup_test_params()
        alpha_vals = np.array([0.5])

        e_outer, R_outer, _ = outer_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            p['T'], p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        if len(e_outer) > 0 and len(e_gauss) > 0:
            # Outer bound rate should be >= inner bound rate
            assert R_outer[0] >= R_gauss[0] - 1e-6
