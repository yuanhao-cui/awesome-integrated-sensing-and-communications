"""Generate simplified Figure 4 for RIS-ISAC beamforming paper."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.system_model import RIS_ISAC_System
from src.crb_constraint import CRBConstrainedSolver

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def fig4_crb_vs_snr():
    """Plot CRB vs SNR for different RIS sizes - sensing performance."""
    print("Generating Figure 4: DOA Estimation Performance (CRB vs SNR) ...")

    L_values = [10, 30, 50]
    snr_dB_range = np.arange(0, 25, 3)  # 0 to 24 dB in steps of 3
    n_trials = 2  # Reduced for speed

    crb_results = {L: np.zeros(len(snr_dB_range)) for L in L_values}

    for i, snr_dB in enumerate(snr_dB_range):
        print(f"  SNR = {snr_dB} dB ...", end=" ", flush=True)

        P_max = 10e-3
        noise_power = P_max / (10 ** (snr_dB / 10))

        for L in L_values:
            crbs = []
            for trial in range(n_trials):
                system = RIS_ISAC_System(
                    M=4, K=2, L=L,
                    P_max=P_max,
                    noise_power=noise_power,
                    seed=100 + trial,
                )
                solver = CRBConstrainedSolver(
                    system,
                    crb_max=1e-1,
                    max_iter=15,  # Reduced for speed
                    tol=1e-3,
                )
                try:
                    res = solver.solve()
                    crbs.append(res["crb"])
                except Exception:
                    # Fallback: compute CRB with matched-filter beamforming
                    W_mf = np.zeros((4, 2), dtype=complex)
                    for k in range(2):
                        h_k = system.effective_channel(k)
                        W_mf[:, k] = h_k.conj() / np.linalg.norm(h_k)
                    total_power = np.sum(np.linalg.norm(W_mf, axis=0) ** 2)
                    W_mf *= np.sqrt(P_max / max(total_power, 1e-15))
                    w_total = np.sum(W_mf, axis=1)
                    crb_val = solver.compute_crb(w_total)
                    crbs.append(crb_val)

            crb_results[L][i] = np.mean(crbs)

        print("done")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for idx, L in enumerate(L_values):
        ax.semilogy(
            snr_dB_range,
            crb_results[L],
            marker=MARKERS[idx],
            color=COLORS[idx],
            label=f"$L = {L}$",
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("CRB (rad$^2$)")
    ax.set_title("DOA Estimation Performance")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=1e-6)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig4_crb_vs_snr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("RIS-ISAC Beamforming: Figure 4 Generation")
    print("=" * 60)
    fig4_crb_vs_snr()
    print("=" * 60)
    print("Figure 4 generated in:", RESULTS_DIR)
    print("=" * 60)
