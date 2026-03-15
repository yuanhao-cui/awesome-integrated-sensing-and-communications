"""Tests for OTFS modulation/demodulation."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from otfs_modem import OTFSModem


def test_otfs_roundtrip():
    """Test that ISFFT + SFFT = identity (modulation roundtrip)."""
    N, M = 8, 16
    modem = OTFSModem(N, M)

    # Random DD symbols
    rng = np.random.default_rng(42)
    X_dd = rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))

    # ISFFT then SFFT should give back X_dd
    X_tf = modem.isfft(X_dd)
    X_recovered = modem.sfft(X_tf)

    error = np.max(np.abs(X_dd - X_recovered))
    assert error < 1e-10, f"Roundtrip error too large: {error}"
    print(f"  ISFFT+SFFT roundtrip error: {error:.2e} ✓")


def test_modem_roundtrip():
    """Test full modulation + demodulation roundtrip."""
    N, M = 8, 16
    modem = OTFSModem(N, M)

    rng = np.random.default_rng(123)
    X_dd = rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))

    # Modulate then demodulate (without channel)
    s = modem.modulate(X_dd)
    X_recovered = modem.demodulate(s)

    error = np.max(np.abs(X_dd - X_recovered))
    assert error < 1e-10, f"Full roundtrip error too large: {error}"
    print(f"  Full modem roundtrip error: {error:.2e} ✓")


def test_isfft_unitarity():
    """Test that ISFFT preserves energy (Parseval's theorem)."""
    N, M = 8, 16
    modem = OTFSModem(N, M)

    rng = np.random.default_rng(456)
    X_dd = rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))

    energy_dd = np.sum(np.abs(X_dd) ** 2)
    X_tf = modem.isfft(X_dd)
    energy_tf = np.sum(np.abs(X_tf) ** 2)

    rel_error = abs(energy_dd - energy_tf) / energy_dd
    assert rel_error < 1e-10, f"Energy not preserved: {rel_error}"
    print(f"  Energy preservation error: {rel_error:.2e} ✓")


def test_identity_matrices():
    """Test ISFFT matrix properties."""
    N, M = 4, 4
    modem = OTFSModem(N, M)

    # Test with delta function
    X_dd = np.zeros((N, M), dtype=complex)
    X_dd[0, 0] = 1.0

    X_tf = modem.isfft(X_dd)

    # ISFFT of delta should have constant magnitude
    mag = np.abs(X_tf)
    assert np.allclose(mag, mag[0, 0], atol=1e-10), "ISFFT of delta should have constant magnitude"
    print(f"  ISFFT delta magnitude constant: {mag[0, 0]:.4f} ✓")


if __name__ == '__main__':
    print("Running OTFS modem tests...")
    test_otfs_roundtrip()
    test_modem_roundtrip()
    test_isfft_unitarity()
    test_identity_matrices()
    print("All OTFS tests passed! ✓")
