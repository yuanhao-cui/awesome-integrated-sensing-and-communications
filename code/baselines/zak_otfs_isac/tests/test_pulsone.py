"""Tests for pulsone waveform generation."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pulsone import PulsoneGenerator
from papr import PAPRAnalyzer


def test_pulsone_generation():
    """Test that point pulsone has correct shape and properties."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)

    # Point pulsone at origin
    X_dd = gen.point_pulsone(0, 0)
    assert X_dd.shape == (N, M)
    assert X_dd[0, 0] == 1.0
    assert np.sum(np.abs(X_dd)) == 1.0  # Only one nonzero element
    print(f"  Point pulsone shape: {X_dd.shape}, peak at (0,0) ✓")


def test_pulsone_shift():
    """Test point pulsone at different locations."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)

    for k0 in [0, 2, 5]:
        for l0 in [0, 3, 7]:
            X_dd = gen.point_pulsone(k0, l0)
            assert X_dd[k0, l0] == 1.0
            assert np.sum(X_dd != 0) == 1
    print("  Pulsone shift positions ✓")


def test_spread_pulsone_generation():
    """Test spread pulsone has correct shape."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)

    X_spread = gen.spread_pulsone(0, 0)
    assert X_spread.shape == (N, M)

    # Spread pulsone should have more nonzero elements
    nonzero = np.sum(np.abs(X_spread) > 1e-10)
    assert nonzero > 1, "Spread pulsone should have energy spread"
    print(f"  Spread pulsone: {nonzero} nonzero taps ✓")


def test_spread_pulsone_papr():
    """Test that spread pulsone has lower PAPR than point pulsone."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)
    analyzer = PAPRAnalyzer(N, M)

    # Generate time-domain signals
    s_point = gen.point_pulsone_time(0, 0)
    s_spread = gen.spread_pulsone_time(0, 0)

    papr_point = analyzer.compute_papr_db(s_point)
    papr_spread = analyzer.compute_papr_db(s_spread)

    print(f"  Point PAPR: {papr_point:.1f} dB")
    print(f"  Spread PAPR: {papr_spread:.1f} dB")
    print(f"  Reduction: {papr_point - papr_spread:.1f} dB")

    assert papr_spread < papr_point, "Spread should have lower PAPR"
    print("  PAPR reduction verified ✓")


def test_chirp_filter():
    """Test chirp filter properties."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)

    h = gen.chirp_filter()
    assert len(h) == N
    assert np.allclose(np.abs(h), 1.0), "Chirp filter should have unit magnitude"
    print(f"  Chirp filter: |h[n]| = 1 ✓")


def test_rotated_lattice():
    """Test rotated lattice indices."""
    N, M = 4, 8
    gen = PulsoneGenerator(N, M)

    n_rot, m_rot = gen.rotated_lattice_indices()
    assert len(n_rot) == N * M
    assert len(m_rot) == N * M

    # All indices should be valid
    assert np.all(n_rot >= 0) and np.all(n_rot < N)
    assert np.all(m_rot >= 0) and np.all(m_rot < M)
    print(f"  Rotated lattice: {len(n_rot)} points ✓")


if __name__ == '__main__':
    print("Running pulsone tests...")
    test_pulsone_generation()
    test_pulsone_shift()
    test_spread_pulsone_generation()
    test_spread_pulsone_papr()
    test_chirp_filter()
    test_rotated_lattice()
    print("All pulsone tests passed! ✓")
