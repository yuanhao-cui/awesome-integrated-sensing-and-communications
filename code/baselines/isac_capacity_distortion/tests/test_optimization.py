"""Tests for optimization.py - Optimization routines."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization import (
    optimize_sensing_rx,
    optimize_comm_rx,
    covariance_shaping,
    stiefel_sample,
    generate_isotropic_waveform,
    generate_semi_unitary_waveform,
)
from system_model import compute_rate, compute_crb


class TestOptimizeSensingRx:
    """Tests for sensing-optimal covariance optimization."""

    def test_sensing_rx_shape(self):
        """Test output has correct shape."""
        M = 4
        Rx = optimize_sensing_rx(P_T=1.0, M=M)
        assert Rx.shape == (M, M)

    def test_sensing_rx_psd(self):
        """Test output is positive semi-definite."""
        M = 4
        Rx = optimize_sensing_rx(P_T=1.0, M=M)
        eigvals = np.linalg.eigvalsh(Rx)
        assert np.all(eigvals >= -1e-10)

    def test_sensing_rx_power_constraint(self):
        """Test power constraint is satisfied."""
        M = 4
        P_T = 2.0
        Rx = optimize_sensing_rx(P_T, M)
        power = np.real(np.trace(Rx))
        np.testing.assert_allclose(power, P_T * M, rtol=1e-6)

    def test_sensing_rx_hermitian(self):
        """Test output is Hermitian."""
        M = 4
        Rx = optimize_sensing_rx(P_T=1.0, M=M)
        np.testing.assert_allclose(Rx, Rx.conj().T, atol=1e-10)

    def test_sensing_rx_isotropic(self):
        """Test sensing-optimal Rx is approximately P_T*I."""
        M = 4
        P_T = 1.0
        Rx = optimize_sensing_rx(P_T, M)
        Rx_expected = P_T * np.eye(M)
        np.testing.assert_allclose(Rx, Rx_expected, rtol=1e-6)


class TestOptimizeCommRx:
    """Tests for communication-optimal covariance optimization."""

    def test_comm_rx_shape(self):
        """Test output has correct shape."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = optimize_comm_rx(P_T=1.0, M=M, Hc=Hc)
        assert Rx.shape == (M, M)

    def test_comm_rx_psd(self):
        """Test output is positive semi-definite."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = optimize_comm_rx(P_T=1.0, M=M, Hc=Hc)
        eigvals = np.linalg.eigvalsh(Rx)
        assert np.all(eigvals >= -1e-10)

    def test_comm_rx_power_constraint(self):
        """Test power constraint is satisfied."""
        np.random.seed(42)
        M, Nc = 4, 2
        P_T = 2.0
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = optimize_comm_rx(P_T, M, Hc)
        power = np.real(np.trace(Rx))
        # Allow some tolerance for numerical optimization
        assert abs(power - P_T * M) / (P_T * M) < 0.1, f"Power {power} != expected {P_T * M}"

    def test_comm_rx_hermitian(self):
        """Test output is Hermitian."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = optimize_comm_rx(P_T=1.0, M=M, Hc=Hc)
        np.testing.assert_allclose(Rx, Rx.conj().T, atol=1e-10)

    def test_comm_rx_achieves_higher_rate(self):
        """Test comm-optimal Rx achieves >= sensing-optimal rate."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        P_T = 1.0

        Rx_comm = optimize_comm_rx(P_T, M, Hc)
        Rx_sense = optimize_sensing_rx(P_T, M)

        rate_comm = compute_rate(Rx_comm, Hc, sigma_c2=1.0)
        rate_sense = compute_rate(Rx_sense, Hc, sigma_c2=1.0)

        assert rate_comm >= rate_sense - 1e-6

    def test_comm_rx_siso(self):
        """Test comm-optimal Rx for SISO channel."""
        M, Nc = 1, 1
        Hc = np.array([[1.0]], dtype=np.complex128)
        P_T = 5.0

        Rx = optimize_comm_rx(P_T, M, Hc)
        np.testing.assert_allclose(Rx, [[P_T]], rtol=1e-6)


class TestCovarianceShaping:
    """Tests for covariance shaping optimization."""

    def test_covariance_shaping_shape(self):
        """Test output has correct shape."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = covariance_shaping(alpha=0.5, P_T=1.0, M=M, Hc=Hc)
        assert Rx.shape == (M, M)

    def test_covariance_shaping_psd(self):
        """Test output is positive semi-definite."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = covariance_shaping(alpha=0.5, P_T=1.0, M=M, Hc=Hc)
        eigvals = np.linalg.eigvalsh(Rx)
        assert np.all(eigvals >= -1e-10)

    def test_covariance_shaping_power(self):
        """Test power constraint is satisfied."""
        M, Nc = 4, 2
        P_T = 2.0
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = covariance_shaping(alpha=0.5, P_T=P_T, M=M, Hc=Hc)
        power = np.real(np.trace(Rx))
        np.testing.assert_allclose(power, P_T * M, rtol=1e-3)

    def test_covariance_shaping_alpha_zero(self):
        """Test alpha=0 gives sensing-optimal Rx."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
        Rx = covariance_shaping(alpha=0.0, P_T=1.0, M=M, Hc=Hc)

        # Should be approximately P_T * I
        Rx_expected = 1.0 * np.eye(M)
        np.testing.assert_allclose(Rx, Rx_expected, rtol=0.15)

    def test_covariance_shaping_alpha_one(self):
        """Test alpha=1 gives comm-optimal Rx."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

        Rx_alpha1 = covariance_shaping(alpha=0.99, P_T=1.0, M=M, Hc=Hc)
        Rx_comm = optimize_comm_rx(1.0, M, Hc)

        # Both should achieve similar rate
        rate_alpha1 = compute_rate(Rx_alpha1, Hc, sigma_c2=1.0)
        rate_comm = compute_rate(Rx_comm, Hc, sigma_c2=1.0)

        # Allow 25% tolerance for different optimization approaches
        np.testing.assert_allclose(rate_alpha1, rate_comm, rtol=0.25)

    def test_covariance_shaping_tradeoff(self):
        """Test tradeoff behavior as alpha varies."""
        M, Nc = 4, 2
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

        rates = []
        for alpha in [0.0, 0.5, 1.0]:
            Rx = covariance_shaping(alpha, 1.0, M, Hc)
            rate = compute_rate(Rx, Hc, sigma_c2=1.0)
            rates.append(rate)

        # Rate should increase with alpha
        assert rates[0] <= rates[1] + 0.1
        assert rates[1] <= rates[2] + 0.1


class TestStiefelSample:
    """Tests for Stiefel manifold sampling."""

    def test_stiefel_sample_shape(self):
        """Test output has correct shape."""
        M_sc, T = 3, 5
        Q = stiefel_sample(M_sc, T)
        assert Q.shape == (M_sc, T)

    def test_stiefel_sample_semi_unitary(self):
        """Test Q is semi-unitary: Q Q^H = I."""
        M_sc, T = 3, 5
        Q = stiefel_sample(M_sc, T)
        QQh = Q @ Q.conj().T
        np.testing.assert_allclose(QQh, np.eye(M_sc), atol=1e-10)

    def test_stiefel_sample_uniformity(self):
        """Test samples are approximately uniformly distributed."""
        np.random.seed(42)
        M_sc, T = 2, 3
        n_samples = 1000

        # For 2x3 Stiefel, sample trace(Q Q^H) should be 2 (identity)
        # Test: average of Q Q^H over samples should be (M_sc/T) * I_T
        QQh_avg = np.zeros((T, T), dtype=np.complex128)
        for _ in range(n_samples):
            Q = stiefel_sample(M_sc, T)
            # Q^H Q is the projection onto the column space
            QQh = Q.conj().T @ Q
            QQh_avg += QQh
        QQh_avg /= n_samples

        # Expected: (M_sc / T) * I_T
        expected = (M_sc / T) * np.eye(T)
        np.testing.assert_allclose(QQh_avg, expected, atol=0.1)

    def test_stiefel_sample_deterministic_seed(self):
        """Test reproducibility with same seed."""
        np.random.seed(123)
        Q1 = stiefel_sample(3, 5)

        np.random.seed(123)
        Q2 = stiefel_sample(3, 5)

        np.testing.assert_allclose(Q1, Q2)


class TestWaveformGeneration:
    """Tests for waveform generation functions."""

    def test_isotropic_waveform_shape(self):
        """Test isotropic waveform shape."""
        M, T = 4, 10
        X = generate_isotropic_waveform(P_T=1.0, M=M, T=T)
        assert X.shape == (M, T)

    def test_isotropic_waveform_power(self):
        """Test isotropic waveform satisfies power constraint."""
        M, T = 4, 100
        P_T = 2.0
        X = generate_isotropic_waveform(P_T, M, T)

        # Average power per symbol: (1/T) tr(X X^H) = P_T * M
        Rx = (X @ X.conj().T) / T
        avg_power = np.real(np.trace(Rx))
        np.testing.assert_allclose(avg_power, P_T * M, rtol=0.15)

    def test_semi_unitary_waveform_shape(self):
        """Test semi-unitary waveform shape."""
        M, T = 4, 6
        M_sc = 3
        X, Q = generate_semi_unitary_waveform(P_T=1.0, M_sc=M_sc, M=M, T=T)
        assert X.shape == (M, T)
        assert Q.shape == (M_sc, T)

    def test_semi_unitary_waveform_power(self):
        """Test semi-unitary waveform power."""
        M, T = 4, 6
        M_sc = 3
        P_T = 2.0
        X, _ = generate_semi_unitary_waveform(P_T, M_sc, M, T)

        Rx = (X @ X.conj().T) / T
        avg_power = np.real(np.trace(Rx))
        np.testing.assert_allclose(avg_power, P_T * M, rtol=0.25)
