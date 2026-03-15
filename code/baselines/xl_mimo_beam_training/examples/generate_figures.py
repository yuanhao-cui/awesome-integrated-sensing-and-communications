"""
Generate publication-quality figures for P0-C baseline (Near-Field Beam Training for XL-MIMO).

Figures:
  C1: CNN model architecture diagram
  C2: Training loss curve (20 epochs, synthetic data)
  C3: Beam pattern comparison — CNN vs DFT codebook
  C4: Achievable rate vs SNR for different methods

Usage:
    cd xl_mimo_beam_training
    source .venv/bin/activate
    python examples/generate_figures.py
"""

import sys
import os
import math
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import BeamTrainingNet
from src.channel import NearFieldChannel
from src.beamforming import BeamformingCodebook
from src.utils import (
    generate_synthetic_data,
    prepare_input_features,
    rate_func,
    trans_vrf,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_ANTENNAS = 256
NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_SYNTHETIC = 5000
LEARNING_RATE = 1e-3
SNR_DPI = 300

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": SNR_DPI,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "cnn": "#E63946",
    "dft": "#457B9D",
    "mrt": "#2A9D8F",
    "polar": "#E9C46A",
    "loss_train": "#E63946",
    "loss_val": "#457B9D",
    "encoder": "#A8DADC",
    "decoder": "#F1FAEE",
    "pool": "#457B9D",
    "upconv": "#E63946",
}


# ============================================================================
# Figure C1 — CNN Model Architecture Diagram
# ============================================================================
def generate_c1_architecture():
    """Draw a clean, publication-quality CNN architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Figure C1: BeamTrainingNet Architecture", fontweight="bold", pad=15)

    def draw_box(ax, x, y, w, h, label, color, sublabel=None, alpha=0.85):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#333333",
            linewidth=1.5, alpha=alpha, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.15, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#222222", zorder=4)
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                    fontsize=7.5, color="#555555", zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2, label="", color="#555555"):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
            zorder=2,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.25, my, label, fontsize=7, color=color, ha="left", va="center")

    # --- Input ---
    draw_box(ax, 0, 4, 1.2, 2.5, "Input", "#D3D3D3", "(batch, 1, 2, 256)")

    # --- Encoder 1 ---
    draw_box(ax, 2.5, 4, 1.6, 2.5, "Encoder 1", COLORS["encoder"],
             "[Conv-BN-ReLU]×2\nFeatures: 8")
    draw_arrow(ax, 0.6, 4, 1.7, 4)

    # --- AvgPool 1 ---
    draw_box(ax, 4.5, 4, 1.0, 1.2, "AvgPool", COLORS["pool"], "↓×2")
    draw_arrow(ax, 3.3, 4, 4.0, 4)

    # --- Encoder 2 ---
    draw_box(ax, 6.5, 4, 1.6, 2.5, "Encoder 2", COLORS["encoder"],
             "[Conv-BN-ReLU]×2\nFeatures: 16")
    draw_arrow(ax, 5.0, 4, 5.7, 4)

    # --- AvgPool 2 ---
    draw_box(ax, 8.5, 4, 1.0, 1.2, "AvgPool", COLORS["pool"], "↓×2")
    draw_arrow(ax, 7.3, 4, 8.0, 4)

    # --- Encoder 3 (bottleneck) ---
    draw_box(ax, 10.5, 4, 1.6, 2.5, "Encoder 3", COLORS["encoder"],
             "[Conv-BN-ReLU]×2\nFeatures: 16")
    draw_arrow(ax, 9.0, 4, 9.7, 4)

    # --- Decoder path (below) ---
    draw_box(ax, 8.5, 1.5, 1.0, 1.2, "ConvTranspose", COLORS["upconv"], "↑×2")
    draw_arrow(ax, 10.5, 2.75, 9.0, 2.1, "", "#E63946")

    draw_box(ax, 6.5, 1.5, 1.6, 2.5, "Decoder 2", COLORS["decoder"],
             "[Conv-BN-ReLU]×2\nFeatures: 8", alpha=0.9)
    draw_arrow(ax, 8.0, 1.5, 7.3, 1.5)

    draw_box(ax, 4.5, 1.5, 1.0, 1.2, "ConvTranspose", COLORS["upconv"], "↑×2")
    draw_arrow(ax, 5.7, 1.5, 5.0, 1.5)

    draw_box(ax, 2.5, 1.5, 1.6, 2.5, "Decoder 1", COLORS["decoder"],
             "[Conv-BN-ReLU]×2\nFeatures: 1", alpha=0.9)
    draw_arrow(ax, 4.0, 1.5, 3.3, 1.5)

    # --- Flatten + Linear + Tanh ---
    draw_box(ax, 0, 1.5, 1.2, 2.5, "Linear\n+ Tanh", "#FFE0B2",
             "Flatten → Linear(512, 256)")
    draw_arrow(ax, 1.7, 1.5, 0.6, 1.5)

    # --- Output ---
    draw_box(ax, -0.0, -1.0, 1.4, 1.5, "trans_vrf", "#C8E6C9",
             "Phase → complex v\n(batch, 256)")
    draw_arrow(ax, 0, 0.25, 0, -0.25)

    # Labels
    ax.text(5.5, 7.5, "Encoder Path", fontsize=10, ha="center",
            fontstyle="italic", color="#333333")
    ax.text(5.5, -0.3, "Decoder Path", fontsize=10, ha="center",
            fontstyle="italic", color="#333333")

    # Legend / annotation
    ax.text(12.5, 7.0, "Block: Conv2d → BN → ReLU", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.savefig(RESULTS_DIR / "c1_architecture.png")
    plt.close(fig)
    print(f"  ✓ C1 saved → {RESULTS_DIR / 'c1_architecture.png'}")


# ============================================================================
# Figure C2 — Training Loss Curve
# ============================================================================
def generate_c2_training_loss():
    """Train the model for NUM_EPOCHS on synthetic data and plot loss curves."""
    print(f"  Training BeamTrainingNet for {NUM_EPOCHS} epochs on synthetic data...")

    # Generate data
    H, H_est = generate_synthetic_data(
        num_samples=NUM_SYNTHETIC, num_antennas=NUM_ANTENNAS, seed=42
    )
    H_input = prepare_input_features(H_est)
    H_true = np.squeeze(H)
    num_samples = H_true.shape[0]
    snr_values = np.power(
        10.0,
        np.random.randint(-20, 20, size=(num_samples, 1)).astype(np.float32) / 10.0,
    )

    ds = TensorDataset(
        torch.tensor(H_input, dtype=torch.float32),
        torch.tensor(H_true, dtype=torch.complex64),
        torch.tensor(snr_values, dtype=torch.float32),
    )
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = "cpu"
    model = BeamTrainingNet(antenna_count=NUM_ANTENNAS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        running = 0.0
        for inputs, targets, snr_v in train_loader:
            inputs, targets, snr_v = inputs.to(device), targets.to(device), snr_v.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean(rate_func(targets, outputs, snr_v))
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_losses.append(running / len(train_loader))

        # Validate
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for inputs, targets, snr_v in val_loader:
                inputs, targets, snr_v = inputs.to(device), targets.to(device), snr_v.to(device)
                outputs = model(inputs)
                val_running += torch.mean(rate_func(targets, outputs, snr_v)).item()
        val_losses.append(val_running / len(val_loader))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{NUM_EPOCHS}  "
                  f"Train: {train_losses[-1]:.4f}  Val: {val_losses[-1]:.4f}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, NUM_EPOCHS + 1)
    ax.plot(epochs, train_losses, "-o", color=COLORS["loss_train"],
            label="Training Loss", markersize=4, linewidth=2)
    ax.plot(epochs, val_losses, "-s", color=COLORS["loss_val"],
            label="Validation Loss", markersize=4, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (−Rate)")
    ax.set_title("Figure C2: Training Loss Curve", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(epochs[::2])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "c2_training_loss.png")
    plt.close(fig)
    print(f"  ✓ C2 saved → {RESULTS_DIR / 'c2_training_loss.png'}")

    return model, device


# ============================================================================
# Figure C3 — Beam Pattern Comparison (CNN vs DFT)
# ============================================================================
def generate_c3_beam_pattern(model, device):
    """Compare beam patterns of CNN-predicted beamforming vs DFT codebook."""
    print("  Generating beam pattern comparison...")

    channel_model = NearFieldChannel(num_antennas=NUM_ANTENNAS)
    codebook_gen = BeamformingCodebook(num_antennas=NUM_ANTENNAS)
    dft_codebook = codebook_gen.generate_dft_codebook()  # (256, 256)

    # Generate a test channel at specific (distance, angle)
    distance = 30.0  # meters
    angle = 0.15     # radians (~8.6 degrees off broadside)
    h_true = channel_model.generate_channel(distance=distance, angle=angle)
    h_est = channel_model.estimate_channel(h_true, snr_dB=10.0)

    # CNN beamformer
    h_input = prepare_input_features(h_est.reshape(1, -1))  # (1, 1, 2, 256)
    h_tensor = torch.tensor(h_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        v_cnn_phase = model(h_tensor)  # (1, 256) in [-1, 1]
    v_cnn = trans_vrf(v_cnn_phase).squeeze().cpu().numpy()  # complex (256,)

    # DFT beamformer (select best beam)
    dft_gains = np.abs(np.conj(h_true) @ dft_codebook) ** 2  # gain for each beam
    best_dft_idx = np.argmax(dft_gains)
    v_dft = dft_codebook[:, best_dft_idx]

    # Evaluate gains across all DFT beams for both methods
    angles_deg = np.linspace(-60, 60, NUM_ANTENNAS)  # approximate beam angles

    gains_cnn = []
    gains_dft = []
    for k in range(NUM_ANTENNAS):
        v_k = dft_codebook[:, k]
        # For each DFT beam, evaluate |h^H v|^2
        gains_cnn.append(np.abs(np.vdot(h_true, v_cnn)) ** 2)
        gains_dft.append(np.abs(np.vdot(h_true, v_k)) ** 2)

    gains_cnn_arr = np.array(gains_cnn)
    gains_dft_arr = np.array(gains_dft)

    # Convert to dB
    gains_cnn_dB = 10 * np.log10(gains_cnn_arr + 1e-12)
    gains_dft_dB = 10 * np.log10(gains_dft_arr + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(angles_deg, gains_dft_dB, "-", color=COLORS["dft"],
            label="DFT Codebook (best beam index scan)", linewidth=1.8, alpha=0.8)
    # CNN is a constant value (one beamformer, same gain for all scan positions — show as marker)
    cnn_gain_dB = 10 * np.log10(np.abs(np.vdot(h_true, v_cnn)) ** 2 + 1e-12)
    dft_best_gain_dB = 10 * np.log10(np.abs(np.vdot(h_true, v_dft)) ** 2 + 1e-12)

    ax.axhline(y=cnn_gain_dB, color=COLORS["cnn"], linestyle="--",
               linewidth=2, label=f"CNN Beamformer ({cnn_gain_dB:.1f} dB)")
    ax.axhline(y=dft_best_gain_dB, color=COLORS["dft"], linestyle=":",
               linewidth=1.5, label=f"DFT Best Beam ({dft_best_gain_dB:.1f} dB)")

    ax.set_xlabel("Beam Index (≈ Angle [°])")
    ax.set_ylabel("Beamforming Gain |hᴴv|² [dB]")
    ax.set_title(f"Figure C3: Beam Pattern — CNN vs DFT Codebook\n"
                 f"(d={distance}m, θ={math.degrees(angle):.1f}°)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-60, 60)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "c3_beam_pattern.png")
    plt.close(fig)
    print(f"  ✓ C3 saved → {RESULTS_DIR / 'c3_beam_pattern.png'}")


# ============================================================================
# Figure C4 — Achievable Rate vs SNR
# ============================================================================
def generate_c4_rate_vs_snr(model, device):
    """Compare achievable rate vs SNR for CNN, DFT, MRT, and Polar codebook."""
    print("  Generating rate vs SNR comparison...")

    channel_model = NearFieldChannel(num_antennas=NUM_ANTENNAS)
    codebook_gen = BeamformingCodebook(num_antennas=NUM_ANTENNAS)
    dft_codebook = codebook_gen.generate_dft_codebook()

    # Polar codebook: distance grid + angle grid
    dist_grid = np.linspace(10, 80, 10)
    angle_grid = np.linspace(-np.pi / 4, np.pi / 4, 16)
    polar_codebook, _, _ = codebook_gen.generate_polar_codebook(
        num_beams=len(dist_grid) * len(angle_grid),
        distance_grid=dist_grid,
        angle_grid=angle_grid,
    )

    snr_dB_range = np.arange(-20, 25, 2)
    num_test = 200  # test channels for averaging

    # Generate test channels
    np.random.seed(123)
    test_channels = []
    test_estimates = []
    for _ in range(num_test):
        d = np.random.uniform(10, 80)
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        h = channel_model.generate_channel(distance=d, angle=theta)
        h_est = channel_model.estimate_channel(h, snr_dB=10.0)
        test_channels.append(h)
        test_estimates.append(h_est)

    methods = {
        "CNN (Proposed)": [],
        "DFT Codebook": [],
        "Polar Codebook": [],
        "MRT (Upper Bound)": [],
    }

    for snr_db in snr_dB_range:
        rho = 10 ** (snr_db / 10.0)
        rates_cnn, rates_dft, rates_polar, rates_mrt = [], [], [], []

        for i in range(num_test):
            h_true = test_channels[i]
            h_est = test_estimates[i]

            # CNN
            h_input = prepare_input_features(h_est.reshape(1, -1))
            h_tensor = torch.tensor(h_input, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                v_phase = model(h_tensor)
            v_cnn = trans_vrf(v_phase).squeeze().cpu().numpy()
            gain_cnn = np.abs(np.vdot(h_true, v_cnn)) ** 2
            rates_cnn.append(np.log2(1 + rho / NUM_ANTENNAS * gain_cnn))

            # DFT (best beam)
            dft_gains = np.abs(np.conj(h_true) @ dft_codebook) ** 2
            rates_dft.append(np.log2(1 + rho / NUM_ANTENNAS * np.max(dft_gains)))

            # Polar (best beam)
            polar_gains = np.abs(np.conj(h_true) @ polar_codebook) ** 2
            rates_polar.append(np.log2(1 + rho / NUM_ANTENNAS * np.max(polar_gains)))

            # MRT (upper bound)
            v_mrt = h_true / np.linalg.norm(h_true)
            gain_mrt = np.abs(np.vdot(h_true, v_mrt)) ** 2
            rates_mrt.append(np.log2(1 + rho / NUM_ANTENNAS * gain_mrt))

        methods["CNN (Proposed)"].append(np.mean(rates_cnn))
        methods["DFT Codebook"].append(np.mean(rates_dft))
        methods["Polar Codebook"].append(np.mean(rates_polar))
        methods["MRT (Upper Bound)"].append(np.mean(rates_mrt))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {
        "CNN (Proposed)":      ("o-", COLORS["cnn"], 2.2),
        "DFT Codebook":        ("s--", COLORS["dft"], 1.8),
        "Polar Codebook":      ("^:", COLORS["polar"], 1.8),
        "MRT (Upper Bound)":   ("D-", COLORS["mrt"], 1.5),
    }
    for name, rates in methods.items():
        marker, color, lw = style[name]
        ax.plot(snr_dB_range, rates, marker, color=color, linewidth=lw,
                markersize=5, label=name)

    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Achievable Rate [bps/Hz]")
    ax.set_title("Figure C4: Achievable Rate vs SNR — Method Comparison", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "c4_rate_vs_snr.png")
    plt.close(fig)
    print(f"  ✓ C4 saved → {RESULTS_DIR / 'c4_rate_vs_snr.png'}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  Generating P0-C Baseline Figures")
    print("=" * 60)

    print("\n[C1] Architecture Diagram")
    generate_c1_architecture()

    print("\n[C2] Training Loss Curve")
    model, device = generate_c2_training_loss()

    print("\n[C3] Beam Pattern Comparison")
    generate_c3_beam_pattern(model, device)

    print("\n[C4] Rate vs SNR")
    generate_c4_rate_vs_snr(model, device)

    print("\n" + "=" * 60)
    print("  All figures saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
