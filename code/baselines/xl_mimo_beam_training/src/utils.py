"""
Utility functions for near-field beam training.

Core algorithm functions preserved from the original implementation:
- trans_vrf: Phase-to-complex conversion (Eq. in paper)
- rate_func: Spectral efficiency loss function (Eq. 14 in paper)
- load_channel_data: Data loading for .mat files

Reference:
    These functions implement the core algorithm from:
    J. Nie, Y. Cui et al., IEEE TMC, 2025.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

Nt = 256  # Default number of transmit antennas


def trans_vrf(temp: torch.Tensor) -> torch.Tensor:
    """Convert phase values to complex unit-norm beamforming vectors.

    Maps output values from the CNN (in [-1, 1]) to complex exponentials
    representing phase-only beamforming vectors for analog beamforming.

    Args:
        temp: Phase values of shape (batch, Nt) in [-1, 1]. These are
            multiplied by pi internally to get actual phases in [-pi, pi].

    Returns:
        Complex beamforming vectors of shape (batch, Nt) with |v| = 1.

    Note:
        This implements the phase mapping v_n = exp(j * pi * temp_n).
        For analog beamforming, each antenna applies a phase shift only,
        hence |v_n| = 1 for all n.

        PRESERVED EXACTLY from original implementation.
    """
    v_real = torch.cos(temp * math.pi)
    v_imag = torch.sin(temp * math.pi)
    vrf = torch.complex(v_real, v_imag)
    return vrf


def rate_func(
    h: torch.Tensor,
    v: torch.Tensor,
    snr_input: torch.Tensor,
    num_antennas: int = Nt,
) -> torch.Tensor:
    """Compute negative spectral efficiency (loss function).

    Calculates the achievable rate under the given beamforming vector and
    channel, then returns its NEGATIVE for use as a loss function to minimize.

    Args:
        h: True channel vectors of shape (batch, Nt), complex-valued.
        v: Phase values from CNN of shape (batch, Nt), real-valued in [-1, 1].
        snr_input: SNR values of shape (batch, 1), linear scale (not dB).
        num_antennas: Number of transmit antennas N_t.

    Returns:
        Negative spectral efficiency of shape (batch, 1).
        Minimizing this loss maximizes the achievable rate.

    Note:
        The spectral efficiency is computed as:
            R = log2(1 + (SNR/N_t) * |h^H v|^2)

        PRESERVED EXACTLY from original implementation.
    """
    v = trans_vrf(v)

    h = h.unsqueeze(1)      # (batch, 1, Nt)
    v = v.unsqueeze(2)      # (batch, Nt, 1)
    hv = torch.bmm(h.to(torch.complex64), v)  # (batch, 1, 1)
    hv = hv.squeeze(dim=-1)  # (batch, 1)
    rate = torch.log2(1 + snr_input / num_antennas * torch.pow(torch.abs(hv), 2))

    return -rate


def load_channel_data(
    data_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load channel data from .mat files.

    Loads perfect CSI (pcsi.mat) and estimated CSI (ecsi.mat) from the
    specified directory.

    Args:
        data_path: Path to directory containing pcsi.mat and ecsi.mat.

    Returns:
        Tuple of (H, H_est) where:
        - H: Perfect CSI matrix of shape (num_samples, Nt), complex.
        - H_est: Estimated CSI matrix of shape (num_samples, Nt), complex.
        Returns (None, None) if files are not found.

    Note:
        Gracefully handles missing .mat files by logging a warning
        and returning None values.
    """
    data_dir = Path(data_path)
    pcsi_path = data_dir / "pcsi.mat"
    ecsi_path = data_dir / "ecsi.mat"

    if not pcsi_path.exists() or not ecsi_path.exists():
        logger.warning(
            f"Channel data files not found in {data_path}. "
            f"Expected pcsi.mat and ecsi.mat. "
            f"Use generate_synthetic_data() for testing."
        )
        return None, None

    try:
        import scipy.io as sio

        logger.info(f"Loading channel data from {data_path}...")
        h = sio.loadmat(str(pcsi_path))["pcsi"]
        h_est = sio.loadmat(str(ecsi_path))["ecsi"]
        logger.info(
            f"Loaded CSI: perfect shape={h.shape}, estimated shape={h_est.shape}"
        )
        return h, h_est
    except Exception as e:
        logger.error(f"Error loading channel data: {e}")
        return None, None


def prepare_input_features(h_est: np.ndarray) -> np.ndarray:
    """Convert complex estimated CSI to CNN input format.

    Stacks real and imaginary parts along a new dimension to form
    the CNN input tensor.

    Args:
        h_est: Estimated CSI of shape (num_samples, Nt), complex.

    Returns:
        Input features of shape (num_samples, 1, 2, Nt), float32.
        Channel 0 = real part, Channel 1 = imaginary part.
    """
    real_part = np.real(h_est)
    imag_part = np.imag(h_est)
    # Stack along axis=1 (new axis after batch): (N, 2, Nt) -> add channel dim (N, 1, 2, Nt)
    features = np.stack([real_part, imag_part], axis=1)  # (N, 2, Nt)
    features = np.expand_dims(features, axis=1)  # (N, 1, 2, Nt)
    return features.astype(np.float32)


def generate_synthetic_data(
    num_samples: int = 5000,
    num_antennas: int = 256,
    noise_std: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic channel data for testing.

    Creates random near-field-like channel realizations with noisy estimates.

    Args:
        num_samples: Number of channel samples.
        num_antennas: Number of antennas N_t.
        noise_std: Standard deviation of estimation noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (H, H_est):
        - H: Perfect CSI of shape (num_samples, num_antennas), complex.
        - H_est: Noisy estimated CSI of shape (num_samples, num_antennas), complex.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random channels with near-field-like structure
    # Spherical wave: phase varies nonlinearly across array
    H = np.zeros((num_samples, num_antennas), dtype=np.complex128)
    positions = (
        np.arange(num_antennas) - (num_antennas - 1) / 2.0
    ) / num_antennas  # normalized positions

    for i in range(num_samples):
        # Random user parameters
        distance = np.random.uniform(10, 100)
        angle = np.random.uniform(-np.pi / 3, np.pi / 3)

        # Spherical wave model
        r_n = np.sqrt(
            distance**2 + positions**2 - 2 * distance * positions * np.sin(angle)
        )
        wavelength = 0.01
        h_i = np.exp(-1j * 2 * np.pi / wavelength * r_n)
        h_i /= np.linalg.norm(h_i)  # normalize
        H[i] = h_i

    # Add noise for channel estimate
    noise = (noise_std / np.sqrt(2)) * (
        np.random.randn(num_samples, num_antennas)
        + 1j * np.random.randn(num_samples, num_antennas)
    )
    H_est = H + noise

    return H, H_est


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath} (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: str = "cpu",
) -> Tuple[int, float]:
    """Load model checkpoint.

    Args:
        model: PyTorch model to load weights into.
        optimizer: Optimizer to load state into (optional).
        filepath: Path to the checkpoint file.
        device: Device to map the checkpoint to.

    Returns:
        Tuple of (epoch, loss) from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    logger.info(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.6f})")
    return epoch, loss
