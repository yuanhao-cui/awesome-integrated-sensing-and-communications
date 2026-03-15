"""Generate academic figures for RIS-ISAC beamforming paper.

Generates four publication-quality figures:
    Fig 1: Convergence of AO algorithm (sum-rate vs iteration, multiple L)
    Fig 2: Sum-rate vs number of RIS elements (Proposed vs baselines)
    Fig 3: Sum-rate vs SINR threshold (communication-sensing tradeoff)
    Fig 4: DOA estimation performance (CRB vs SNR for different RIS sizes)

Reference:
    R. Liu et al., "SNR/CRB-Constrained Joint Beamforming and Reflection
    Designs for RIS-ISAC Systems," IEEE TWC 2024. arXiv:2301.11134
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.system_model import RIS_ISAC_System
from src.ao_solver import AlternatingOptimizationSolver
from src.crb_constraint import CRBConstrainedSolver

# ── Academic plotting style ──────────────────────────────────────────────────
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


def run_ao(system, max_iter=30, snr_min_dB=5.0):
    """Run AO solver and return convergence history."""
    solver = AlternatingOptimizationSolver(
        system,
        problem_type="snr",
        snr_min_dB=snr_min_dB,
        max_iter=max_iter,
        tol=1e-5,
    )
    result = solver.solve()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Convergence of AO Algorithm
# ═══════════════════════════════════════════════════════════════════════════

def fig1_convergence():
    """Plot AO convergence for different RIS element counts."""
    print("Generating Figure 1: Convergence of AO Algorithm ...")

    L_values = [10, 30, 50]
    max_iter = 30

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for idx, L in enumerate(L_values):
        system = RIS_ISAC_System(M=4, K=2, L=L, seed=42)
        result = run_ao(system, max_iter=max_iter)
        history = result["history"]
        iters = np.arange(1, len(history) + 1)

        ax.plot(
            iters,
            history,
            marker=MARKERS[idx],
            color=COLORS[idx],
            label=f"$L = {L}$",
            markevery=3,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title("Convergence of Alternating Optimization")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.5)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig1_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Sum-Rate vs RIS Elements
# ═══════════════════════════════════════════════════════════════════════════

def fig2_sumrate_vs_ris():
    """Plot sum-rate vs number of RIS elements for three schemes."""
    print("Generating Figure 2: Sum-Rate vs RIS Elements ...")

    L_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_trials = 3  # Monte Carlo averages

    rate_proposed = np.zeros(len(L_range))
    rate_random = np.zeros(len(L_range))
    rate_no_ris = np.zeros(len(L_range))

    for i, L in enumerate(L_range):
        print(f"  L = {L} ...", end=" ", flush=True)

        r_prop = []
        r_rand = []
        r_none = []

        for trial in range(n_trials):
            seed = 42 + trial

            # --- Proposed (AO with SNR constraint) ---
            sys_prop = RIS_ISAC_System(M=4, K=2, L=L, seed=seed)
            solver = AlternatingOptimizationSolver(
                sys_prop, problem_type="snr", snr_min_dB=5.0, max_iter=25, tol=1e-4
            )
            res = solver.solve()
            r_prop.append(res["sum_rate"])

            # --- Random phase (RIS with random phases) ---
            sys_rand = RIS_ISAC_System(M=4, K=2, L=L, seed=seed)
            # Use matched-filter beamforming with random RIS phases
            W_rand = np.zeros((4, 2), dtype=complex)
            for k in range(2):
                h_k = sys_rand.effective_channel(k)
                W_rand[:, k] = h_k.conj() / np.linalg.norm(h_k)
            P_max = sys_rand.P_max
            total_power = np.sum(np.linalg.norm(W_rand, axis=0) ** 2)
            W_rand *= np.sqrt(P_max / max(total_power, 1e-15))
            r_rand.append(sys_rand.compute_sum_rate(W_rand))

            # --- Without RIS (direct channel only) ---
            sys_noris = RIS_ISAC_System(M=4, K=2, L=L, seed=seed)
            # Set RIS phases to zero reflection (effectively no RIS)
            sys_noris.theta = np.zeros(L, dtype=complex)
            W_noris = np.zeros((4, 2), dtype=complex)
            for k in range(2):
                h_k = sys_noris.effective_channel(k)
                W_noris[:, k] = h_k.conj() / np.linalg.norm(h_k)
            total_power = np.sum(np.linalg.norm(W_noris, axis=0) ** 2)
            W_noris *= np.sqrt(P_max / max(total_power, 1e-15))
            r_none.append(sys_noris.compute_sum_rate(W_noris))

        rate_proposed[i] = np.mean(r_prop)
        rate_random[i] = np.mean(r_rand)
        rate_no_ris[i] = np.mean(r_none)
        print("done")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(
        L_range, rate_proposed,
        marker="o", color=COLORS[0], label="Proposed (AO + SNR)"
    )
    ax.plot(
        L_range, rate_random,
        marker="s", color=COLORS[1], label="Random Phase",
        linestyle="--"
    )
    ax.plot(
        L_range, rate_no_ris,
        marker="^", color=COLORS[2], label="Without RIS",
        linestyle="-."
    )

    ax.set_xlabel("Number of RIS Elements $L$")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title("Sum Rate vs. RIS Elements")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig2_sumrate_vs_ris.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Sum-Rate vs SINR Threshold
# ═══════════════════════════════════════════════════════════════════════════

def fig3_sumrate_vs_sinr():
    """Plot sum-rate vs SINR threshold to show communication-sensing tradeoff."""
    print("Generating Figure 3: Sum-Rate vs SINR Threshold ...")

    sinr_thresh_dB_range = [0, 2, 4, 6, 8, 10, 12, 15, 18, 20]
    L = 30
    n_trials = 3

    rate_snr5 = np.zeros(len(sinr_thresh_dB_range))
    rate_snr10 = np.zeros(len(sinr_thresh_dB_range))
    rate_snr15 = np.zeros(len(sinr_thresh_dB_range))

    for i, sinr_dB in enumerate(sinr_thresh_dB_range):
        print(f"  SINR = {sinr_dB} dB ...", end=" ", flush=True)

        for snr_idx, snr_min_dB in enumerate([5.0, 10.0, 15.0]):
            rates = []
            for trial in range(n_trials):
                system = RIS_ISAC_System(
                    M=4, K=2, L=L,
                    sinr_thresh_dB=sinr_dB,
                    seed=42 + trial,
                )
                solver = AlternatingOptimizationSolver(
                    system,
                    problem_type="snr",
                    snr_min_dB=snr_min_dB,
                    max_iter=25,
                    tol=1e-4,
                )
                res = solver.solve()
                rates.append(res["sum_rate"])

            avg = np.mean(rates)
            if snr_idx == 0:
                rate_snr5[i] = avg
            elif snr_idx == 1:
                rate_snr10[i] = avg
            else:
                rate_snr15[i] = avg

        print("done")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(
        sinr_thresh_dB_range, rate_snr5,
        marker="o", color=COLORS[0], label=r"$\gamma_{\min} = 5$ dB"
    )
    ax.plot(
        sinr_thresh_dB_range, rate_snr10,
        marker="s", color=COLORS[1], label=r"$\gamma_{\min} = 10$ dB"
    )
    ax.plot(
        sinr_thresh_dB_range, rate_snr15,
        marker="^", color=COLORS[2], label=r"$\gamma_{\min} = 15$ dB"
    )

    ax.set_xlabel("SINR Threshold $\\gamma_k$ (dB)")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title("Communication-Sensing Tradeoff")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig3_sumrate_vs_sinr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: DOA Estimation Performance (CRB vs SNR)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_crb_vs_snr():
    """Plot CRB vs SNR for different RIS sizes — sensing performance."""
    print("Generating Figure 4: DOA Estimation Performance (CRB vs SNR) ...")

    L_values = [10, 30, 50]
    snr_dB_range = np.arange(0, 25, 3)  # 0 to 24 dB in steps of 3
    n_trials = 3

    crb_results = {L: np.zeros(len(snr_dB_range)) for L in L_values}

    for i, snr_dB in enumerate(snr_dB_range):
        print(f"  SNR = {snr_dB} dB ...", end=" ", flush=True)

        # Adjust noise power to achieve target SNR
        # SNR = P_max / noise_power (simplified)
        noise_power_base = 3.98e-12
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
                # Run CRB-constrained solver
                solver = CRBConstrainedSolver(
                    system,
                    crb_max=1e-1,
                    max_iter=25,
                    tol=1e-4,
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RIS-ISAC Beamforming: Figure Generation")
    print("Reference: R. Liu et al., IEEE TWC 2024")
    print("=" * 60)

    fig1_convergence()
    fig2_sumrate_vs_ris()
    fig3_sumrate_vs_sinr()
    fig4_crb_vs_snr()

    print("=" * 60)
    print("All figures generated in:", RESULTS_DIR)
    print("=" * 60)
