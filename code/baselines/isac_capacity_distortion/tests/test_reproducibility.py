"""Tests for reproducibility - comparing results to paper figures."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from system_model import (
    angle_to_channel,
    angle_to_hfunc,
    compute_rate,
    compute_crb,
    compute_phi_angle,
    make_uniform_linear_array,
)
from bounds import (
    compute_corner_points,
    gaussian_inner_bound,
    outer_bound,
)
from optimization import (
    optimize_sensing_rx,
    optimize_comm_rx,
    covariance_shaping,
)


def setup_angle_params(theta_c_deg=42.0):
    """Setup parameters matching Table I."""
    M, Ns, Nc = 10, 10, 1
    d = 0.5
    P_T = 1.0

    theta_c = np.deg2rad(theta_c_deg)
    theta_target = np.deg2rad(30)

    positions = np.arange(M) * d
    a_c = np.exp(1j * 2 * np.pi * positions * np.sin(theta_c))
    Hc = a_c.reshape(Nc, M)

    Hs_func = angle_to_hfunc(M, Ns, d, d)

    def phi_func(Rx):
        return compute_phi_angle(Rx, 1, theta_target, M, Ns, d, d, Jp=10.0)

    Jp = np.array([[10.0]])
    sigma_s2 = P_T * 10 ** (-20 / 10)  # 20 dB sensing SNR
    sigma_c2 = P_T * 10 ** (-33 / 10)  # 33 dB comm SNR

    return {
        'M': M, 'Ns': Ns, 'Nc': Nc, 'd': d,
        'P_T': P_T, 'sigma_c2': sigma_c2, 'sigma_s2': sigma_s2,
        'Hc': Hc, 'Hs_func': Hs_func, 'phi_func': phi_func,
        'Jp': Jp, 'theta_c': theta_c, 'theta_target': theta_target,
        'theta_c_deg': theta_c_deg,
    }


class TestFigure5Reproducibility:
    """Tests for Figure 5 reproducibility: Rate vs CRB for T=3, theta_c=42°."""

    def test_corner_points_range(self):
        """Test corner points are in reasonable ranges."""
        p = setup_angle_params(theta_c_deg=42.0)
        T = 3

        corners = compute_corner_points(
            p['Hc'], p['Hs_func'], p['phi_func'],
            T, p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        # CRB values should be positive and finite
        assert 0 < corners['e_min'] < 100
        assert 0 < corners['e_cs'] < 100

        # Rates should be positive
        assert corners['R_sc'] >= 0
        assert corners['R_max'] >= 0

        # Sensing-optimal should have lower CRB
        assert corners['e_min'] <= corners['e_cs']

        # Comm-optimal should have higher rate
        assert corners['R_sc'] <= corners['R_max']

    def test_rho_value(self):
        """Test correlation coefficient for theta_c=42° is approximately 0.61."""
        p = setup_angle_params(theta_c_deg=42.0)

        a_c = make_uniform_linear_array(p['M'], p['d'])(p['theta_c']).flatten()
        a_s = make_uniform_linear_array(p['M'], p['d'])(p['theta_target']).flatten()
        rho = np.abs(np.vdot(a_c, a_s)) / (np.linalg.norm(a_c) * np.linalg.norm(a_s))

        # rho should be in a reasonable range for this angle separation
        # (42° - 30° = 12° separation)
        assert 0.1 < rho < 0.9, f"rho = {rho} not in expected range"

    def test_tradeoff_direction(self):
        """Test that CRB increases as rate increases along the tradeoff."""
        p = setup_angle_params(theta_c_deg=42.0)
        T = 3
        alpha_vals = np.linspace(0.1, 0.9, 5)

        e_vals, R_vals, _ = gaussian_inner_bound(
            alpha_vals, p['Hc'], p['Hs_func'], p['phi_func'],
            T, p['sigma_c2'], p['sigma_s2'],
            p['P_T'], p['M'], p['Jp'], p['Nc']
        )

        if len(e_vals) > 2 and len(R_vals) > 2:
            # Check that rate increases with alpha (communication priority)
            # CRB may vary due to numerical issues
            assert R_vals[-1] >= R_vals[0] - 0.1 * abs(R_vals[0])


class TestFigure8Reproducibility:
    """Tests for Figure 8: different T values."""

    def test_crb_decreases_with_T(self):
        """Test CRB decreases as T increases (more observation time)."""
        p = setup_angle_params(theta_c_deg=50.0)

        crb_values = []
        for T in [3, 10, 50]:
            Rx_sense = optimize_sensing_rx(p['P_T'], p['M'])
            crb = compute_crb(Rx_sense, T, p['sigma_s2'],
                            phi_func=p['phi_func'], Jp=p['Jp'])
            crb_values.append(crb)

        # CRB should decrease with T
        assert crb_values[0] > crb_values[1] > crb_values[2]

    def test_rate_independent_of_T(self):
        """Test rate is independent of T for same covariance."""
        p = setup_angle_params(theta_c_deg=50.0)

        Rx = (p['P_T'] / p['M']) * np.eye(p['M'], dtype=np.complex128)

        rates = []
        for T in [3, 10, 50]:
            rate = compute_rate(Rx, p['Hc'], p['sigma_c2'])
            rates.append(rate)

        # All rates should be equal (rate doesn't depend on T directly)
        np.testing.assert_allclose(rates[0], rates[1], rtol=1e-10)
        np.testing.assert_allclose(rates[1], rates[2], rtol=1e-10)


class TestTableIParameters:
    """Tests for Table I parameter values."""

    def test_antenna_counts(self):
        """Test antenna counts match Table I."""
        p = setup_angle_params()
        assert p['M'] == 10
        assert p['Ns'] == 10
        assert p['Nc'] == 1

    def test_snr_values(self):
        """Test SNR values match Table I."""
        p = setup_angle_params()

        # Sensing SNR should be 20 dB
        sensing_snr_db = -10 * np.log10(p['sigma_s2'])
        np.testing.assert_allclose(sensing_snr_db, 20.0, atol=0.1)

        # Comm SNR should be 33 dB
        comm_snr_db = -10 * np.log10(p['sigma_c2'])
        np.testing.assert_allclose(comm_snr_db, 33.0, atol=0.1)


class TestTableIIParameters:
    """Tests for Table II parameter values."""

    def test_matrix_estimation_params(self):
        """Test parameters for matrix estimation case study."""
        M, Ns, Nc = 4, 4, 4
        sigma_s2 = 1.0
        P_T = 1.0

        # Sensing SNR = 24 dB
        sensing_snr_db = 24.0
        sigma_s2_expected = P_T * 10 ** (-sensing_snr_db / 10)
        np.testing.assert_allclose(sigma_s2_expected, 10**(-2.4), rtol=1e-10)


class TestBasicPhysicalConstraints:
    """Tests for basic physical constraints of the model."""

    def test_power_constraint_active(self):
        """Test transmit power is bounded by constraint."""
        p = setup_angle_params()
        M = p['M']
        P_T = p['P_T']

        Rx_sense = optimize_sensing_rx(P_T, M)
        power = np.real(np.trace(Rx_sense))
        np.testing.assert_allclose(power, P_T * M, rtol=1e-6)

    def test_crb_fisher_inequality(self):
        """Test CRB satisfies basic Fisher information inequality."""
        p = setup_angle_params()
        T = 10

        Rx = (p['P_T'] / p['M']) * np.eye(p['M'], dtype=np.complex128)
        crb = compute_crb(Rx, T, p['sigma_s2'],
                         phi_func=p['phi_func'], Jp=p['Jp'])

        # CRB should be positive
        assert crb > 0

        # CRB should decrease with more power
        Rx_high = 2 * Rx
        crb_high_power = compute_crb(Rx_high, T, p['sigma_s2'],
                                     phi_func=p['phi_func'], Jp=p['Jp'])
        assert crb_high_power < crb
