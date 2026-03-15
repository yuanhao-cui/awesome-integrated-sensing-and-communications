"""Tests for the near-field channel model."""

import pytest
import numpy as np

import sys
sys.path.insert(0, "..")
from src.channel import NearFieldChannel


class TestNearFieldChannel:
    """Test suite for the near-field channel model."""

    @pytest.fixture
    def channel(self):
        """Create a standard channel model for testing."""
        return NearFieldChannel(
            num_antennas=256, wavelength=0.01, antenna_spacing=0.005
        )

    def test_channel_dimensions(self, channel):
        """Test that generated channel has correct dimensions."""
        h = channel.generate_channel(distance=50.0, angle=0.0)
        assert h.shape == (256,), f"Expected shape (256,), got {h.shape}"
        assert h.dtype == np.complex128

    def test_channel_not_all_zeros(self, channel):
        """Test that channel is not trivially zero."""
        h = channel.generate_channel(distance=50.0, angle=0.0)
        assert np.any(np.abs(h) > 0), "Channel should not be all zeros"

    def test_channel_batch_dimensions(self, channel):
        """Test batch channel generation dimensions."""
        num_samples = 100
        H = channel.generate_channel_batch(num_samples)
        assert H.shape == (num_samples, 256)
        assert H.dtype == np.complex128

    def test_channel_estimation_noise(self, channel):
        """Test that channel estimation adds noise."""
        h_true = channel.generate_channel(distance=50.0, angle=0.0)
        h_est = channel.estimate_channel(h_true, snr_dB=10.0)
        assert h_est.shape == h_true.shape
        # Estimate should differ from true (but be correlated)
        assert not np.allclose(h_true, h_est), "Estimate should have noise"

    def test_channel_normalization(self, channel):
        """Test that channel has unit average power."""
        h = channel.generate_channel(distance=50.0, angle=0.0)
        avg_power = np.mean(np.abs(h) ** 2)
        assert np.isclose(avg_power, 1.0, atol=1e-6), (
            f"Average power should be ~1, got {avg_power}"
        )

    def test_channel_multipath(self, channel):
        """Test multi-path channel generation."""
        h_los = channel.generate_channel(distance=50.0, angle=0.0, num_paths=1)
        h_mp = channel.generate_channel(distance=50.0, angle=0.0, num_paths=5)
        assert h_los.shape == h_mp.shape
        # Multi-path should differ from single-path
        assert not np.allclose(h_los, h_mp)

    def test_different_distances(self, channel):
        """Test channel generation at different distances."""
        h_near = channel.generate_channel(distance=10.0, angle=0.0)
        h_far = channel.generate_channel(distance=200.0, angle=0.0)
        assert h_near.shape == h_far.shape
        # Near-field effect: phase profile should differ
        assert not np.allclose(h_near, h_far)
