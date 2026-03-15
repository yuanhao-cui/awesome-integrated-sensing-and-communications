"""Tests for system_model.py - Channel model, BFIM, CRB, and rate computations."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from system_model import (
    GaussianISACChannel,
    compute_bfim,
    compute_crb,
    compute_rate,
    make_uniform_linear_array,
    angle_to_channel,
    angle_to_hfunc,
    compute_phi_angle,
    compute_rate_per_symbol,
)


class TestGaussianISACChannel:
    """Tests for the GaussianISACChannel class."""

    def test_channel_creation(self):
        """Test basic channel model creation."""
        M, Nc, Ns, T = 4, 2, 3, 10
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Hs_func = lambda eta: (np.random.randn(Ns, M) + 1j * np.random.randn(Ns, M)) / np.sqrt(2)

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        assert channel.M == M
        assert channel.Nc == Nc
        assert channel.Ns == Ns
        assert channel.T == T

    def test_comm_channel_shape(self):
        """Test comm channel matrix has correct shape."""
        M, Nc, Ns, T = 4, 2, 3, 10
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Hs_func = lambda eta: np.eye(Ns, M)

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        assert channel.comm_channel().shape == (Nc, M)

    def test_sensing_channel_shape(self):
        """Test sensing channel matrix has correct shape."""
        M, Nc, Ns, T = 4, 2, 3, 10
        Hc = np.eye(Nc, M)
        Hs_func = lambda eta: np.eye(Ns, M) * float(eta[0])

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        Hs = channel.sensing_channel(np.array([1.0]))
        assert Hs.shape == (Ns, M)

    def test_noise_generation_shape(self):
        """Test noise matrix has correct shape."""
        M, Nc, Ns, T = 4, 2, 3, 10
        Hc = np.eye(Nc, M)
        Hs_func = lambda eta: np.eye(Ns, M)

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        noise = channel.generate_noise(Nc)
        assert noise.shape == (Nc, T)

    def test_noise_statistics(self):
        """Test generated noise has approximately correct statistics."""
        np.random.seed(42)
        M, Nc, Ns, T = 4, 2, 3, 10000
        Hc = np.eye(Nc, M)
        Hs_func = lambda eta: np.eye(Ns, M)

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        noise = channel.generate_noise(1)

        # Mean should be near zero
        assert np.abs(np.mean(noise)) < 0.05
        # Variance should be near 0.5 per real/imag component
        assert np.abs(np.var(noise.real) - 0.5) < 0.05
        assert np.abs(np.var(noise.imag) - 0.5) < 0.05

    def test_comm_receive_shape(self):
        """Test communication receive signal shape."""
        M, Nc, Ns, T = 4, 2, 3, 10
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Hs_func = lambda eta: np.eye(Ns, M)

        channel = GaussianISACChannel(Hc, Hs_func, 1.0, 1.0, M, Nc, Ns, T)
        X = (np.random.randn(M, T) + 1j * np.random.randn(M, T)) / np.sqrt(2)
        Yc = channel.comm_receive(X)
        assert Yc.shape == (Nc, T)


class TestBFIM:
    """Tests for BFIM computation."""

    def test_bfim_shape(self):
        """Test BFIM has correct shape."""
        M = 4
        Rx = np.eye(M) * 0.5
        J = compute_bfim(Rx, T=10, sigma_s2=1.0)
        assert J.shape == (M, M)

    def test_bfim_scaling(self):
        """Test BFIM scales correctly with T and sigma_s2."""
        M = 4
        Rx = np.eye(M) * 0.5

        J1 = compute_bfim(Rx, T=10, sigma_s2=1.0)
        J2 = compute_bfim(Rx, T=20, sigma_s2=1.0)

        # Doubling T should quadruple the BFIM (since J ~ T^2)
        # Phi(Rx) = T * Rx, J = T/sigma_s2 * Phi(Rx) = T^2/sigma_s2 * Rx
        np.testing.assert_allclose(J2, 4 * J1, rtol=1e-10)

    def test_bfim_with_prior(self):
        """Test BFIM includes prior information."""
        M = 4
        Rx = np.eye(M) * 0.5
        Jp = np.eye(M) * 2.0

        J_no_prior = compute_bfim(Rx, T=10, sigma_s2=1.0, Jp=None)
        J_with_prior = compute_bfim(Rx, T=10, sigma_s2=1.0, Jp=Jp)

        # With prior should be larger
        assert np.trace(J_with_prior).real > np.trace(J_no_prior).real

    def test_bfim_psd(self):
        """Test BFIM is positive semi-definite."""
        M = 4
        Rx = np.eye(M) * 0.5
        J = compute_bfim(Rx, T=10, sigma_s2=1.0)

        eigvals = np.linalg.eigvalsh(J)
        assert np.all(eigvals >= -1e-10)


class TestCRB:
    """Tests for CRB computation."""

    def test_crb_positive(self):
        """Test CRB is positive."""
        M = 4
        Rx = np.eye(M) * 0.5
        crb = compute_crb(Rx, T=10, sigma_s2=1.0)
        assert crb > 0

    def test_crb_decreases_with_power(self):
        """Test CRB decreases as transmit power increases."""
        M = 4
        crb_low = compute_crb(np.eye(M) * 0.1, T=10, sigma_s2=1.0)
        crb_high = compute_crb(np.eye(M) * 1.0, T=10, sigma_s2=1.0)
        assert crb_high < crb_low

    def test_crb_decreases_with_T(self):
        """Test CRB decreases with longer coherent interval."""
        M = 4
        Rx = np.eye(M) * 0.5
        crb_short = compute_crb(Rx, T=5, sigma_s2=1.0)
        crb_long = compute_crb(Rx, T=20, sigma_s2=1.0)
        assert crb_long < crb_short

    def test_crb_matches_manual(self):
        """Test CRB matches manual computation."""
        M = 2
        Rx = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        T = 10
        sigma_s2 = 1.0

        crb = compute_crb(Rx, T, sigma_s2)

        # Manual: Phi(Rx) = T * Rx, J = (T/sigma_s2) * Phi(Rx) = T^2/sigma_s2 * Rx
        # For Rx = I, J = T^2/sigma_s2 * I
        # J_inv = sigma_s2/T^2 * I
        # tr(J_inv) = M * sigma_s2 / T^2
        J = (T / sigma_s2) * T * Rx
        J_inv = np.linalg.inv(J)
        crb_manual = np.real(np.trace(J_inv))

        np.testing.assert_allclose(crb, crb_manual, rtol=1e-10)


class TestRate:
    """Tests for communication rate computation."""

    def test_rate_positive(self):
        """Test rate is non-negative."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = np.eye(M) * 0.5
        rate = compute_rate(Rx, Hc, sigma_c2=1.0)
        assert rate >= 0

    def test_rate_zero_for_zero_input(self):
        """Test rate is zero when input power is zero."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = np.zeros((M, M), dtype=np.complex128)
        rate = compute_rate(Rx, Hc, sigma_c2=1.0)
        np.testing.assert_allclose(rate, 0, atol=1e-10)

    def test_rate_increases_with_power(self):
        """Test rate increases with transmit power."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        rate_low = compute_rate(np.eye(M) * 0.1, Hc, sigma_c2=1.0)
        rate_high = compute_rate(np.eye(M) * 1.0, Hc, sigma_c2=1.0)
        assert rate_high > rate_low

    def test_rate_awgn_capacity(self):
        """Test rate matches AWGN capacity for SISO."""
        Hc = np.array([[1.0]], dtype=np.complex128)
        P = 10.0
        sigma_c2 = 1.0
        Rx = np.array([[P]], dtype=np.complex128)

        rate = compute_rate(Rx, Hc, sigma_c2)
        # Capacity = log(1 + P/sigma_c2)
        capacity = np.log(1 + P / sigma_c2)

        np.testing.assert_allclose(rate, capacity, rtol=1e-10)

    def test_rate_mimo_capacity(self):
        """Test MIMO rate formula."""
        M, Nc = 2, 2
        Hc = np.eye(Nc, M, dtype=np.complex128)
        P = 5.0
        sigma_c2 = 1.0
        Rx = (P / M) * np.eye(M, dtype=np.complex128)

        rate = compute_rate(Rx, Hc, sigma_c2)
        # With Hc = I, rate = log det(I + (P/M) I / sigma_c2)
        # = Nc * log(1 + P/(M * sigma_c2))
        rate_expected = Nc * np.log(1 + P / (M * sigma_c2))

        np.testing.assert_allclose(rate, rate_expected, rtol=1e-10)


class TestULA:
    """Tests for uniform linear array functions."""

    def test_steering_vector_shape(self):
        """Test steering vector has correct shape."""
        M = 10
        a_func = make_uniform_linear_array(M)
        a = a_func(np.deg2rad(30))
        assert a.shape == (M, 1)

    def test_steering_vector_norm(self):
        """Test steering vector has unit norm."""
        M = 10
        a_func = make_uniform_linear_array(M)
        a = a_func(np.deg2rad(30))
        norm = np.linalg.norm(a)
        np.testing.assert_allclose(norm, np.sqrt(M), rtol=1e-10)

    def test_angle_to_channel_shape(self):
        """Test channel matrix has correct shape."""
        M, N = 8, 4
        H = angle_to_channel(np.deg2rad(45), M, N)
        assert H.shape == (N, M)

    def test_hfunc_output_shape(self):
        """Test Hs function output has correct shape."""
        M, Ns = 10, 10
        hfunc = angle_to_hfunc(M, Ns)
        Hs = hfunc(np.array([np.deg2rad(30)]))
        assert Hs.shape == (Ns, M)


class TestPhiAngle:
    """Tests for angle-specific Phi function."""

    def test_phi_angle_positive(self):
        """Test Phi is positive for valid inputs."""
        M, Ns = 10, 10
        Rx = np.eye(M) * 0.5
        phi = compute_phi_angle(Rx, T=1, theta=np.deg2rad(30), M=M, Ns=Ns)
        assert phi[0, 0] > 0

    def test_phi_angle_scales_with_power(self):
        """Test Phi scales with transmit power."""
        M, Ns = 10, 10
        phi1 = compute_phi_angle(np.eye(M) * 0.1, 1, np.deg2rad(30), M, Ns)
        phi2 = compute_phi_angle(np.eye(M) * 1.0, 1, np.deg2rad(30), M, Ns)
        assert phi2[0, 0] > phi1[0, 0]


class TestRatePerSymbol:
    """Tests for sample-based rate computation."""

    def test_rate_per_symbol_shape(self):
        """Test rate computation from waveform."""
        M, T, Nc = 4, 10, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        X = (np.random.randn(M, T) + 1j * np.random.randn(M, T)) / np.sqrt(2)

        rate = compute_rate_per_symbol(X, Hc, sigma_c2=1.0)
        assert isinstance(rate, float)
        assert rate >= 0
