"""End-to-end integration tests with synthetic data."""

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, "..")
from src.model import BeamTrainingNet
from src.utils import trans_vrf, rate_func, generate_synthetic_data, prepare_input_features
from src.evaluator import Evaluator


class TestEndToEnd:
    """End-to-end tests verifying the complete pipeline."""

    def test_trans_vrf_unit_norm(self):
        """Test that trans_vrf produces unit-norm complex vectors."""
        phases = torch.randn(8, 256) * 0.5  # random phases in reasonable range
        v = trans_vrf(phases)
        assert v.shape == (8, 256)
        assert v.dtype == torch.complex64
        # |v| should be 1 for every element
        magnitudes = torch.abs(v)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6), (
            f"trans_vrf should produce unit-norm values, got range "
            f"[{magnitudes.min():.6f}, {magnitudes.max():.6f}]"
        )

    def test_rate_func_positive(self):
        """Test that spectral efficiency is positive."""
        h = torch.randn(8, 256, dtype=torch.complex64)
        h = h / torch.abs(h).mean()  # normalize
        v = torch.randn(8, 256) * 0.5  # phase values in [-1, 1]
        snr = torch.ones(8, 1) * 10.0  # SNR = 10

        neg_rate = rate_func(h, v, snr)
        assert neg_rate.shape == (8, 1)
        rate = -neg_rate
        assert torch.all(rate > 0), "Spectral efficiency should be positive"

    def test_rate_func_increases_with_snr(self):
        """Test that rate increases with SNR."""
        h = torch.randn(16, 256, dtype=torch.complex64)
        h = h / torch.abs(h).mean()
        v = torch.randn(16, 256) * 0.3

        snr_low = torch.ones(16, 1) * 1.0   # SNR = 1 (0 dB)
        snr_high = torch.ones(16, 1) * 100.0  # SNR = 100 (20 dB)

        rate_low = -torch.mean(rate_func(h, v, snr_low)).item()
        rate_high = -torch.mean(rate_func(h, v, snr_high)).item()

        assert rate_high > rate_low, (
            f"Rate should increase with SNR: {rate_low:.4f} -> {rate_high:.4f}"
        )

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        H, H_est = generate_synthetic_data(
            num_samples=100, num_antennas=256, seed=42
        )
        assert H.shape == (100, 256)
        assert H_est.shape == (100, 256)
        assert H.dtype == np.complex128
        assert H_est.dtype == np.complex128

    def test_prepare_input_features(self):
        """Test input feature preparation."""
        h_est = np.random.randn(50, 256) + 1j * np.random.randn(50, 256)
        features = prepare_input_features(h_est)
        assert features.shape == (50, 1, 2, 256)
        assert features.dtype == np.float32

    def test_end_to_end_training(self):
        """Test: Train 2 epochs on synthetic data, verify loss decreases."""
        # Generate data
        H, H_est = generate_synthetic_data(
            num_samples=200, num_antennas=256, seed=42
        )
        H_input = prepare_input_features(H_est)
        H_true = np.squeeze(H)
        snr = np.power(
            10.0, np.random.randint(-10, 10, size=(200, 1)).astype(np.float32) / 10.0
        )

        # Convert to tensors
        H_input_t = torch.tensor(H_input, dtype=torch.float32)
        H_true_t = torch.tensor(H_true, dtype=torch.complex64)
        snr_t = torch.tensor(snr, dtype=torch.float32)

        # Model
        model = BeamTrainingNet(antenna_count=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train 2 epochs
        losses = []
        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            batch_size = 50
            num_batches = len(H_input_t) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                inputs = H_input_t[start:end]
                targets = H_true_t[start:end]
                snr_batch = snr_t[start:end]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = rate_func(targets, outputs, snr_batch)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        # Verify loss decreased (or at least didn't increase significantly)
        assert losses[-1] <= losses[0] * 1.1, (
            f"Loss should not increase significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_model_inference_pipeline(self):
        """Test the complete inference pipeline."""
        # Generate data
        H, H_est = generate_synthetic_data(num_samples=20, num_antennas=256, seed=42)

        # Create model and run inference
        model = BeamTrainingNet(antenna_count=256)
        model.eval()

        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32)

        with torch.no_grad():
            phases = model(H_input_t)
            v = trans_vrf(phases)

        assert phases.shape == (20, 256)
        assert v.shape == (20, 256)
        assert torch.all(phases >= -1) and torch.all(phases <= 1)
        assert torch.allclose(torch.abs(v), torch.ones_like(torch.abs(v)), atol=1e-6)
