"""
Evaluation metrics and testing pipeline for beam training.

Provides comprehensive evaluation including spectral efficiency,
beamforming gain, normalized MSE, and visualization.

Reference:
    Evaluation setup from Section V of the paper.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import BeamTrainingNet
from .utils import (
    generate_synthetic_data,
    load_channel_data,
    load_checkpoint,
    prepare_input_features,
    rate_func,
    trans_vrf,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation pipeline for the beam training model.

    Computes multiple metrics across different SNR regimes:
    - Spectral efficiency (achievable rate in bps/Hz)
    - Beamforming gain |h^H v|^2
    - Normalized MSE between predicted and optimal beamforming vectors
    - Rate vs SNR curves

    Args:
        model: Trained BeamTrainingNet model.
        device: Device for inference.
        num_antennas: Number of transmit antennas.
    """

    def __init__(
        self,
        model: BeamTrainingNet,
        device: str = "cpu",
        num_antennas: int = 256,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_antennas = num_antennas

    def evaluate_rate_vs_snr(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
        snr_range: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Evaluate spectral efficiency across SNR values.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.
            snr_range: List of SNR values in dB. Default: [-20, -15, ..., 20].

        Returns:
            Tuple of (snr_dB_list, rate_list) for plotting.
        """
        if snr_range is None:
            snr_range = list(range(-20, 21, 5))

        H_input = prepare_input_features(H_est)
        H_true = np.squeeze(H)

        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)
        H_true_t = torch.tensor(H_true, dtype=torch.complex64).to(self.device)

        self.model.eval()
        rates = []

        with torch.no_grad():
            outputs = self.model(H_input_t)  # (N, Nt) phase values

            for snr_dB in snr_range:
                snr_linear = 10 ** (snr_dB / 10.0)
                snr_tensor = torch.full(
                    (H_true_t.shape[0], 1), snr_linear, dtype=torch.float32
                ).to(self.device)

                loss = rate_func(H_true_t, outputs, snr_tensor)
                avg_rate = -torch.mean(loss).item()
                rates.append(avg_rate)
                logger.info(f"SNR: {snr_dB} dB, Rate: {avg_rate:.4f} bps/Hz")

        return snr_range, rates

    def compute_beamforming_gain(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
    ) -> np.ndarray:
        """Compute beamforming gain |h^H v|^2 for each sample.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.

        Returns:
            Array of beamforming gains of shape (num_samples,).
        """
        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)
        H_true = np.squeeze(H)

        self.model.eval()
        with torch.no_grad():
            phases = self.model(H_input_t)  # (N, Nt)
            v = trans_vrf(phases).cpu().numpy()  # (N, Nt) complex

        gains = np.abs(np.sum(np.conj(H_true) * v, axis=1)) ** 2
        return gains

    def compute_normalized_mse(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
    ) -> float:
        """Compute normalized MSE between predicted and MRT beamforming vectors.

        Compares the predicted beamforming vector against the maximum ratio
        transmission (MRT) optimal vector v_mrt = h / ||h||.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.

        Returns:
            Average normalized MSE.
        """
        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)
        H_true = np.squeeze(H)

        # Optimal MRT beamforming
        norms = np.linalg.norm(H_true, axis=1, keepdims=True)
        v_opt = H_true / norms  # (N, Nt)

        self.model.eval()
        with torch.no_grad():
            phases = self.model(H_input_t)
            v_pred = trans_vrf(phases).cpu().numpy()  # (N, Nt)

        # Normalized MSE
        diff = np.abs(v_pred - v_opt) ** 2
        mse = np.mean(np.sum(diff, axis=1) / self.num_antennas)
        return float(mse)

    def evaluate_all_metrics(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
        snr_range: Optional[List[int]] = None,
    ) -> Dict:
        """Run all evaluation metrics.

        Args:
            H: Perfect CSI.
            H_est: Estimated CSI.
            snr_range: SNR values in dB.

        Returns:
            Dictionary with all computed metrics.
        """
        snr_list, rate_list = self.evaluate_rate_vs_snr(H, H_est, snr_range)
        gains = self.compute_beamforming_gain(H, H_est)
        nmse = self.compute_normalized_mse(H, H_est)

        metrics = {
            "snr_dB": snr_list,
            "spectral_efficiency": rate_list,
            "avg_beamforming_gain_dB": float(10 * np.log10(np.mean(gains))),
            "beamforming_gains": gains,
            "normalized_mse": nmse,
        }

        logger.info(
            f"Evaluation results: "
            f"avg_gain={metrics['avg_beamforming_gain_dB']:.2f} dB, "
            f"NMSE={nmse:.6f}"
        )
        return metrics

    @staticmethod
    def plot_rate_vs_snr(
        snr_range: List[float],
        rates: List[float],
        save_path: Optional[str] = None,
        title: str = "Near-Field Beam Training Performance",
    ) -> None:
        """Plot spectral efficiency vs SNR curve.

        Args:
            snr_range: SNR values in dB.
            rates: Achievable rates in bps/Hz.
            save_path: If provided, save figure to this path.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(snr_range, rates, marker="o", linewidth=2, markersize=6)
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("Spectral Efficiency (bps/Hz)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        num_antennas: int = 256,
    ) -> "Evaluator":
        """Create an Evaluator from a saved checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            device: Device for inference.
            num_antennas: Number of antennas.

        Returns:
            Initialized Evaluator with loaded model.
        """
        model = BeamTrainingNet(antenna_count=num_antennas)
        load_checkpoint(model, None, checkpoint_path, device)
        model.eval()
        return cls(model, device, num_antennas)
