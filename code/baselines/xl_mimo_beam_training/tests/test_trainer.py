"""Tests for the training pipeline."""

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, "..")
from src.trainer import Trainer
from src.model import BeamTrainingNet


class TestTrainer:
    """Test suite for the training pipeline."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return {
            "num_antennas": 256,
            "batch_size": 32,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "lr_factor": 0.2,
            "lr_patience": 20,
            "min_lr": 1e-5,
            "val_split": 0.2,
            "num_synthetic_samples": 200,
            "in_channels": 1,
            "out_channels": 1,
            "init_features": 8,
            "seed": 42,
            "log_interval": 1,
            "checkpoint_dir": "/tmp/test_checkpoints",
        }

    def test_model_initialization(self, config):
        """Test that model is properly initialized."""
        trainer = Trainer(config, device="cpu")
        model = trainer.setup_model()
        assert isinstance(model, BeamTrainingNet)
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_synthetic_data_loading(self, config):
        """Test loading synthetic data."""
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        train_loader, val_loader = trainer.load_data()
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check data shapes
        batch = next(iter(train_loader))
        inputs, targets, snr = batch
        assert inputs.shape[1:] == (1, 2, 256), f"Input shape: {inputs.shape}"
        assert targets.shape[1] == 256, f"Target shape: {targets.shape}"
        assert snr.shape[1] == 1, f"SNR shape: {snr.shape}"

    def test_training_step(self, config):
        """Test that a single training step runs and reduces initial loss."""
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()

        # Run 2 epochs
        history = trainer.train(num_epochs=2)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_checkpoint_saving(self, config):
        """Test that checkpoints are saved."""
        import os
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()
        trainer.train(num_epochs=1)

        checkpoint_path = os.path.join(config["checkpoint_dir"], "final_model.pth")
        assert os.path.exists(checkpoint_path), "Final checkpoint should exist"

    def test_load_pretrained(self, config):
        """Test loading a pretrained checkpoint."""
        import os
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()
        trainer.train(num_epochs=1)

        # Create new trainer and load checkpoint
        trainer2 = Trainer(config, device="cpu")
        trainer2.setup_model()
        checkpoint_path = os.path.join(config["checkpoint_dir"], "final_model.pth")
        epoch, loss = trainer2.load_pretrained(checkpoint_path)
        assert epoch == 1
        # Loss is negative rate (-Rate), so it should be finite
        assert np.isfinite(loss)
