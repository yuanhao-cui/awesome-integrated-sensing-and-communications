"""Tests for Zak transform."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zak_transform import ZakTransform


def test_zak_roundtrip():
    """Test that Zak transform + inverse = identity."""
    N_tau, N_nu = 8, 16
    zt = ZakTransform(N_tau, N_nu)

    rng = np.random.default_rng(42)
    s = rng.standard_normal(N_tau * N_nu) + 1j * rng.standard_normal(N_tau * N_nu)

    Z = zt.forward(s)
    s_recovered = zt.inverse(Z)

    error = np.max(np.abs(s - s_recovered))
    assert error < 1e-10, f"Zak roundtrip error: {error}"
    print(f"  Zak roundtrip error: {error:.2e} ✓")


def test_zak_shape():
    """Test output shapes."""
    N_tau, N_nu = 4, 8
    zt = ZakTransform(N_tau, N_nu)

    s = np.random.randn(N_tau * N_nu) + 1j * np.random.randn(N_tau * N_nu)
    Z = zt.forward(s)

    assert Z.shape == (N_tau, N_nu), f"Wrong shape: {Z.shape}"
    print(f"  Zak output shape: {Z.shape} ✓")


def test_zak_energy():
    """Test energy preservation."""
    N_tau, N_nu = 8, 8
    zt = ZakTransform(N_tau, N_nu)

    rng = np.random.default_rng(123)
    s = rng.standard_normal(N_tau * N_nu) + 1j * rng.standard_normal(N_tau * N_nu)

    energy_s = np.sum(np.abs(s) ** 2)
    Z = zt.forward(s)
    energy_Z = np.sum(np.abs(Z) ** 2)

    rel_error = abs(energy_s - energy_Z) / energy_s
    assert rel_error < 1e-10, f"Energy not preserved: {rel_error}"
    print(f"  Energy error: {rel_error:.2e} ✓")


def test_crystallization():
    """Test crystallization condition check."""
    N_tau, N_nu = 8, 8
    T_tau, T_nu = 1e-3, 400.0  # 1ms delay, 400Hz Doppler period
    zt = ZakTransform(N_tau, N_nu, T_tau, T_nu)

    # Channel within limits
    assert zt.crystallization_satisfied(0.5e-3, 200.0), "Should satisfy"

    # Channel exceeds delay limit
    assert not zt.crystallization_satisfied(2e-3, 200.0), "Should not satisfy"

    # Channel exceeds Doppler limit
    assert not zt.crystallization_satisfied(0.5e-3, 500.0), "Should not satisfy"

    print("  Crystallization check ✓")


def test_grids():
    """Test delay and Doppler grids."""
    N_tau, N_nu = 8, 16
    T_tau, T_nu = 1e-3, 100.0
    zt = ZakTransform(N_tau, N_nu, T_tau, T_nu)

    delay_grid = zt.delay_grid()
    doppler_grid = zt.doppler_grid()

    assert len(delay_grid) == N_tau
    assert len(doppler_grid) == N_nu
    assert np.isclose(delay_grid[1] - delay_grid[0], T_tau / N_tau)
    assert np.isclose(doppler_grid[1] - doppler_grid[0], T_nu / N_nu)
    print(f"  Delay resolution: {T_tau/N_tau*1e6:.2f} µs ✓")
    print(f"  Doppler resolution: {T_nu/N_nu:.2f} Hz ✓")


if __name__ == '__main__':
    print("Running Zak transform tests...")
    test_zak_roundtrip()
    test_zak_shape()
    test_zak_energy()
    test_crystallization()
    test_grids()
    print("All Zak transform tests passed! ✓")
