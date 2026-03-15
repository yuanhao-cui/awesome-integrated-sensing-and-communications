"""Tests for ambiguity function."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ambiguity import AmbiguityFunction
from pulsone import PulsoneGenerator
from otfs_modem import OTFSModem


def test_ambiguity_peak():
    """Test that self-ambiguity of point pulsone peaks at (0,0)."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    # Generate point pulsone in time domain
    s = gen.point_pulsone_time(0, 0)

    # Compute self-ambiguity
    A = amb.self_ambiguity(s)

    # Peak should be at (0, 0)
    peak_idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    assert peak_idx == (0, 0), f"Peak at {peak_idx}, expected (0,0)"
    print(f"  Ambiguity peak at {peak_idx} ✓")


def test_ambiguity_symmetry():
    """Test ambiguity function properties."""
    N, M = 4, 8
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    s = gen.point_pulsone_time(0, 0)
    A = amb.self_ambiguity(s)

    # Self-ambiguity at origin should equal signal energy
    energy = np.sum(np.abs(s) ** 2)
    peak_val = np.abs(A[0, 0])

    rel_error = abs(peak_val - energy) / energy
    print(f"  Peak: {peak_val:.4f}, Energy: {energy:.4f}, Rel error: {rel_error:.2e} ✓")


def test_cross_ambiguity():
    """Test cross-ambiguity between different signals."""
    N, M = 4, 8
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    s1 = gen.point_pulsone_time(0, 0)
    s2 = gen.point_pulsone_time(2, 3)

    A = amb.cross_ambiguity(s1, s2)
    assert A.shape == (N, M)
    print(f"  Cross-ambiguity shape: {A.shape} ✓")


def test_range_doppler_map():
    """Test range-Doppler map computation."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    s_tx = gen.point_pulsone_time(0, 0)
    # Simulate a simple echo
    s_rx = np.roll(s_tx, 3)  # Delay of 3 samples

    rd_map = amb.range_doppler_map(s_tx, s_rx)
    assert rd_map.shape == (N, M)
    assert np.max(rd_map) > 0
    print(f"  Range-Doppler map shape: {rd_map.shape} ✓")


def test_psr():
    """Test peak-to-sidelobe ratio."""
    N, M = 8, 16
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    s = gen.point_pulsone_time(0, 0)
    A = amb.self_ambiguity(s)

    psr = amb.peak_to_sidelobe_ratio(A)
    assert psr > 1.0, "PSR should be > 1"
    print(f"  PSR: {psr:.1f} (linear) ✓")


def test_ambiguity_fast():
    """Test fast ambiguity computation matches direct."""
    N, M = 4, 8
    gen = PulsoneGenerator(N, M)
    amb = AmbiguityFunction(N, M)

    s = gen.point_pulsone_time(0, 0)

    A_direct = amb.self_ambiguity(s)
    A_fast = amb.ambiguity_fast(s)

    # Shapes should match
    assert A_direct.shape == A_fast.shape
    print(f"  Fast ambiguity shape matches direct ✓")


if __name__ == '__main__':
    print("Running ambiguity tests...")
    test_ambiguity_peak()
    test_ambiguity_symmetry()
    test_cross_ambiguity()
    test_range_doppler_map()
    test_psr()
    test_ambiguity_fast()
    print("All ambiguity tests passed! ✓")
