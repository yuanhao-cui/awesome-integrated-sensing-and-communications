#!/usr/bin/env python3
"""
Generate Simulation Figures for P0-D Baseline (ISAC Energy-Efficient Beamforming)
=================================================================================

Generates four publication-quality figures:
    - Figure D1: Comm-EE vs Dinkelbach Iterations (convergence)
    - Figure D2: Comm-EE vs SINR Requirement (γ_min)
    - Figure D3: Sensing-EE vs Maximum Transmit Power
    - Figure D4: Pareto Boundary (Comm-EE vs Sensing-EE) for different K

Saves to `results/` as PNG (300 DPI).

Reference: Zou et al., IEEE Trans. Commun., 2024
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.dinkelbach_solver import DinkelbachSolver
from src.ee_metrics import compute_ee_c, compute_ee_s, compute_total_power, compute_crb

# ── Style Configuration ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'blue': '#2166AC',
    'red': '#B2182B',
    'green': '#1B7837',
    'orange': '#E66101',
    'purple': '#762A83',
    'teal': '#00A5CF',
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']


# ── Figure D1: Comm-EE vs Dinkelbach Iterations ─────────────────────────────

def generate_fig_d1(output_dir):
    """
    Generate Figure D1: Communication EE convergence over Dinkelbach iterations.

    Shows how EE_C improves with each Dinkelbach iteration for different
    antenna configurations (M=8 and M=16).
    """
    print("\n[D1] Generating Comm-EE vs Iterations (convergence)...")

    configs = [
        {'M': 8,  'K': 2, 'label': 'M=8, K=2',  'color': COLORS['blue']},
        {'M': 16, 'K': 3, 'label': 'M=16, K=3', 'color': COLORS['red']},
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        model = ISACSystemModel(
            M=cfg['M'], K=cfg['K'], N=20,
            P_max_dbm=30, P0_dbm=33, epsilon=0.35, L=30,
            seed=42,
        )
        solver = DinkelbachSolver(model, max_dinkelbach_iter=30, verbose=False)
        result = solver.solve(target_angle_deg=90.0)

        iterations = range(1, len(result.obj_history) + 1)
        ax.plot(iterations, result.obj_history,
                color=cfg['color'], marker=MARKERS[configs.index(cfg)],
                markersize=8, linewidth=2.2, label=cfg['label'],
                markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Dinkelbach Iteration')
    ax.set_ylabel('$EE_C$ (bits/Hz/J)')
    ax.set_title('(a) Communication EE Convergence')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    output_path = os.path.join(output_dir, 'fig_d1_comm_ee_convergence.png')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")


# ── Figure D2: Comm-EE vs SINR Requirement ──────────────────────────────────

def generate_fig_d2(output_dir):
    """
    Generate Figure D2: Communication EE vs minimum SINR requirement (γ_min).

    Shows how EE_C decreases as SINR requirements become stricter.
    """
    print("\n[D2] Generating Comm-EE vs SINR Requirement...")

    gamma_db_values = np.arange(-5, 25, 2)  # dB
    gamma_linear = 10 ** (gamma_db_values / 10)

    configs = [
        {'M': 8,  'K': 2, 'label': 'M=8, K=2',  'color': COLORS['blue']},
        {'M': 16, 'K': 2, 'label': 'M=16, K=2', 'color': COLORS['red']},
        {'M': 16, 'K': 3, 'label': 'M=16, K=3', 'color': COLORS['green']},
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        model = ISACSystemModel(
            M=cfg['M'], K=cfg['K'], N=20,
            P_max_dbm=30, P0_dbm=33, epsilon=0.35, L=30,
            seed=42,
        )
        solver = DinkelbachSolver(model, max_dinkelbach_iter=30, verbose=False)

        ee_c_values = []
        for gamma_min in gamma_linear:
            try:
                result = solver.solve(
                    target_angle_deg=90.0,
                    gamma_min=gamma_min,
                )
                ee_c_values.append(result.ee_c if result.ee_c > 0 else np.nan)
            except Exception:
                ee_c_values.append(np.nan)

        ax.plot(gamma_db_values, ee_c_values,
                color=cfg['color'], marker=MARKERS[configs.index(cfg)],
                markersize=7, linewidth=2.2, label=cfg['label'],
                markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Minimum SINR Requirement $\\gamma_{\\min}$ (dB)')
    ax.set_ylabel('$EE_C$ (bits/Hz/J)')
    ax.set_title('(b) Comm-EE vs SINR Requirement')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')

    output_path = os.path.join(output_dir, 'fig_d2_comm_ee_vs_sinr.png')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")


# ── Figure D3: Sensing-EE vs Max Transmit Power ─────────────────────────────

def generate_fig_d3(output_dir):
    """
    Generate Figure D3: Sensing EE vs maximum transmit power (P_max).

    Shows how sensing EE (EE_S) varies with power budget.
    Uses beamforming toward target direction for sensing-centric operation.
    """
    print("\n[D3] Generating Sensing-EE vs Max Transmit Power...")

    p_max_dbm_values = np.arange(10, 45, 2)  # dBm

    configs = [
        {'M': 8,  'K': 2, 'N': 20, 'label': 'M=8, K=2, N=20',  'color': COLORS['blue']},
        {'M': 16, 'K': 2, 'N': 20, 'label': 'M=16, K=2, N=20', 'color': COLORS['red']},
        {'M': 16, 'K': 3, 'N': 20, 'label': 'M=16, K=3, N=20', 'color': COLORS['green']},
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        ee_s_values = []

        for p_max_dbm in p_max_dbm_values:
            model = ISACSystemModel(
                M=cfg['M'], K=cfg['K'], N=cfg['N'],
                P_max_dbm=p_max_dbm, P0_dbm=33, epsilon=0.35, L=30,
                seed=42,
            )

            theta_rad = np.radians(90.0)
            a_t = model.steering_vector_tx(theta_rad)
            a_r = model.steering_vector_rx(theta_rad)

            # Sensing-centric beamforming: beam toward target
            M_ant, K_users = model.M, model.K
            W = np.zeros((M_ant, K_users), dtype=complex)
            P_per_user = model.P_max / K_users

            for k in range(K_users):
                W[:, k] = a_t / np.linalg.norm(a_t) * np.sqrt(P_per_user)

            ee_s = compute_ee_s(
                W, a_t, a_r, model.sigma_s2, model.L,
                model.epsilon, model.P0,
            )
            ee_s_values.append(ee_s)

        ax.plot(p_max_dbm_values, ee_s_values,
                color=cfg['color'], marker=MARKERS[configs.index(cfg)],
                markersize=7, linewidth=2.2, label=cfg['label'],
                markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Maximum Transmit Power $P_{\\max}$ (dBm)')
    ax.set_ylabel('$EE_S$ (1/J)')
    ax.set_title('(c) Sensing-EE vs Transmit Power')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')

    output_path = os.path.join(output_dir, 'fig_d3_sensing_ee_vs_power.png')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")


# ── Figure D4: Pareto Boundary ──────────────────────────────────────────────

def generate_fig_d4(output_dir):
    """
    Generate Figure D4: Pareto boundary (Comm-EE vs Sensing-EE).

    Traces the optimal tradeoff between communication and sensing EE
    for different numbers of users K.
    """
    print("\n[D4] Generating Pareto Boundary...")

    configs = [
        {'K': 2, 'label': 'K=2', 'color': COLORS['blue']},
        {'K': 3, 'label': 'K=3', 'color': COLORS['red']},
        {'K': 4, 'label': 'K=4', 'color': COLORS['green']},
    ]

    M = 8
    n_pareto_points = 12

    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        K = cfg['K']
        print(f"    Computing Pareto boundary for K={K}...")

        model = ISACSystemModel(
            M=M, K=K, N=20,
            P_max_dbm=30, P0_dbm=33, epsilon=0.35, L=30,
            seed=42,
        )

        theta_rad = np.radians(90.0)
        a_t = model.steering_vector_tx(theta_rad)
        a_r = model.steering_vector_rx(theta_rad)
        H = model.get_csi()

        pareto_ee_c = []
        pareto_ee_s = []

        # Endpoint 1: Pure Comm-EE optimization (Dinkelbach)
        solver = DinkelbachSolver(model, max_dinkelbach_iter=30, verbose=False)
        result_comm = solver.solve(target_angle_deg=90.0)
        ee_c_max = result_comm.ee_c
        ee_s_at_comm = compute_ee_s(
            result_comm.W, a_t, a_r, model.sigma_s2, model.L,
            model.epsilon, model.P0,
        )

        # Endpoint 2: Pure Sensing-EE (beam toward target)
        W_sense = np.zeros((M, K), dtype=complex)
        for k in range(K):
            W_sense[:, k] = a_t / np.linalg.norm(a_t) * np.sqrt(model.P_max / K)

        ee_c_at_sense = compute_ee_c(H, W_sense, model.sigma_c2, model.epsilon, model.P0)
        ee_s_max = compute_ee_s(
            W_sense, a_t, a_r, model.sigma_s2, model.L,
            model.epsilon, model.P0,
        )

        # Generate Pareto boundary by varying power allocation fraction
        # α = fraction of power for communication, (1-α) for sensing
        alphas = np.linspace(0.05, 0.95, n_pareto_points)

        for alpha in alphas:
            W_pareto = np.zeros((M, K), dtype=complex)

            # Comm component: matched filter
            for k in range(K):
                h_k = H[k, :]
                w_comm = h_k / np.linalg.norm(h_k) * np.sqrt(alpha * model.P_max / K)
                w_sense = a_t / np.linalg.norm(a_t) * np.sqrt((1 - alpha) * model.P_max / K)
                W_pareto[:, k] = w_comm + w_sense

            ee_c = compute_ee_c(H, W_pareto, model.sigma_c2, model.epsilon, model.P0)
            ee_s = compute_ee_s(
                W_pareto, a_t, a_r, model.sigma_s2, model.L,
                model.epsilon, model.P0,
            )

            pareto_ee_c.append(ee_c)
            pareto_ee_s.append(ee_s)

        # Add endpoints
        all_ee_c = [ee_c_at_sense] + pareto_ee_c + [ee_c_max]
        all_ee_s = [ee_s_max] + pareto_ee_s + [ee_s_at_comm]

        # Sort by ee_c for clean boundary
        sorted_pairs = sorted(zip(all_ee_c, all_ee_s), key=lambda x: x[0])
        all_ee_c, all_ee_s = zip(*sorted_pairs)

        # Remove dominated points (keep Pareto optimal)
        pareto_c = [all_ee_c[0]]
        pareto_s = [all_ee_s[0]]
        for c, s in zip(all_ee_c[1:], all_ee_s[1:]):
            if s > pareto_s[-1]:
                pareto_c.append(c)
                pareto_s.append(s)

        ax.plot(pareto_c, pareto_s,
                color=cfg['color'], marker=MARKERS[configs.index(cfg)],
                markersize=7, linewidth=2.2, label=cfg['label'],
                markerfacecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('$EE_C$ (bits/Hz/J)')
    ax.set_ylabel('$EE_S$ (1/J)')
    ax.set_title('(d) Pareto Boundary: Comm-EE vs Sensing-EE')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')

    output_path = os.path.join(output_dir, 'fig_d4_pareto_boundary.png')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"    ✓ Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Generate all four figures."""
    print("=" * 70)
    print("  ISAC Energy-Efficient Beamforming — Simulation Figures")
    print("  P0-D Baseline (Dinkelbach Algorithm)")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    generate_fig_d1(output_dir)
    generate_fig_d2(output_dir)
    generate_fig_d3(output_dir)
    generate_fig_d4(output_dir)

    print("\n" + "=" * 70)
    print("  All figures generated successfully!")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
