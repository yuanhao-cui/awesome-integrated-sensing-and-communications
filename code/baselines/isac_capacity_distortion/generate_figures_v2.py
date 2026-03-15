#!/usr/bin/env python3
"""
Generate high-quality academic figures for ISAC capacity-distortion tradeoff.

Uses actual algorithm implementations from src/:
- covariance_shaping (optimization.py)
- compute_rate, compute_crb (system_model.py)
- outer_bound, gaussian_inner_bound, semi_unitary_inner_bound, pentagon_inner_bound (bounds.py)

Based on:
Xiong et al., "On the Fundamental Tradeoff of Integrated Sensing and
Communications Under Gaussian Channels," IEEE TIT, 2023.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.ticker as ticker

from optimization import covariance_shaping, optimize_sensing_rx, optimize_comm_rx
from system_model import compute_rate, compute_crb, compute_bfim
from bounds import (
    pentagon_inner_bound, gaussian_inner_bound,
    semi_unitary_inner_bound, outer_bound, compute_corner_points,
)

# ── Academic Plot Style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
})

os.makedirs('results', exist_ok=True)

np.random.seed(2023)

# ══════════════════════════════════════════════════════════════════════
# System Configuration
# ══════════════════════════════════════════════════════════════════════

sigma_c2 = 1.0   # Communication noise variance
sigma_s2 = 1.0   # Sensing noise variance
T = 1            # Coherent processing interval


def make_phi_func(M, d=0.5):
    """Create Phi(Rx) for angle estimation with ULA.

    For angle estimation, the BFIM element is:
    J = (T / sigma_s^2) * 4*pi^2 * cos^2(theta) * tr(D^2 Rx)

    where D = diag(positions), positions = (m - (M-1)/2) * d.
    We evaluate at theta = 0 (broadside, cos(theta)=1) and absorb
    constants into the Phi map.

    Phi(Rx) = D^2 @ Rx  (element-wise: sum_m d_m^2 * Rx_mm for diagonal Rx)

    Returns a 1x1 matrix containing tr(D^2 Rx).
    """
    positions = (np.arange(M) - (M - 1) / 2.0) * d
    D2 = positions ** 2  # squared positions

    def phi_func(Rx):
        # tr(D^2 Rx) = sum of D2[m] * Rx[m,m] for diagonal case
        # More generally: tr(diag(D2) @ Rx)
        val = np.real(np.trace(np.diag(D2) @ Rx))
        return np.array([[val]])

    return phi_func


def compute_sensing_crb_from_rx(Rx, M, T=1, sigma_s2=1.0, d=0.5):
    """Compute CRB for angle estimation from covariance Rx.

    Uses the BFIM formula for ULA angle estimation:
    J = (T / sigma_s^2) * 4*pi^2 * tr(D^2 Rx)
    CRB = 1 / J  (scalar parameter)
    """
    phi_func = make_phi_func(M, d)
    bfim = compute_bfim(Rx, T, sigma_s2, phi_func=phi_func)
    crb = np.real(np.trace(np.linalg.pinv(bfim + 1e-15 * np.eye(bfim.shape[0]))))
    return max(crb, 0.0)


def snr_to_pt(snr_db):
    """Convert SNR in dB to transmit power (with sigma_c2=1)."""
    return 10 ** (snr_db / 10.0)


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Rate-CRB Pareto Frontier
# ══════════════════════════════════════════════════════════════════════

def run_pareto_frontier(M=4, Nc=2, snr_db_list=(10, 15, 20), n_alpha=25):
    """Run covariance shaping for multiple alpha values and SNRs."""
    alpha_values = np.linspace(0.01, 0.99, n_alpha)
    results = {}

    for snr_db in snr_db_list:
        P_T = snr_to_pt(snr_db)
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

        rates = []
        crbs = []
        alphas_used = []

        for alpha in alpha_values:
            try:
                Rx_opt = covariance_shaping(
                    alpha=alpha, P_T=P_T, M=M, Hc=Hc,
                    sigma_c2=sigma_c2, sigma_s2=sigma_s2, T=T,
                )

                eigvals = np.linalg.eigvalsh(Rx_opt)
                if np.any(eigvals < -1e-6):
                    continue

                R = compute_rate(Rx_opt, Hc, sigma_c2)
                crb = compute_sensing_crb_from_rx(Rx_opt, M, T, sigma_s2)

                if R > 0 and crb > 0:
                    rates.append(R)
                    crbs.append(crb)
                    alphas_used.append(alpha)
            except Exception as e:
                print(f"  [skip] alpha={alpha:.2f}, SNR={snr_db}dB: {e}")
                continue

        results[snr_db] = {
            'rates': np.array(rates),
            'crbs': np.array(crbs),
            'alphas': np.array(alphas_used),
        }
        print(f"  SNR={snr_db}dB: {len(rates)} points computed")

    return results


def plot_figure1(results, M=4, Nc=2):
    """Plot Rate-CRB Pareto Frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#1976D2', '#388E3C', '#E64A19']
    markers = ['o', 's', '^']
    snr_list = sorted(results.keys())

    for idx, snr_db in enumerate(snr_list):
        data = results[snr_db]
        ax.plot(
            data['crbs'], data['rates'],
            color=colors[idx], marker=markers[idx],
            markersize=5, linewidth=2.0,
            label=f'SNR = {snr_db} dB',
            zorder=3,
        )

    ax.set_xlabel('CRB (rad²)')
    ax.set_ylabel('Rate (nats/channel use)')
    ax.set_title('Rate–CRB Pareto Frontier for ISAC')
    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.set_xlim(left=None, right=None)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=50))
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')

    # Annotations
    ax.annotate('Sensing-\noptimal', xy=(data['crbs'][0], data['rates'][0]),
                fontsize=8, color='gray', ha='right')
    ax.annotate('Comm-\noptimal', xy=(data['crbs'][-1], data['rates'][-1]),
                fontsize=8, color='gray', ha='left')

    fig.tight_layout()
    path = 'results/figure1_pareto_frontier.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 2: CRB-Rate Region with Bounds
# ══════════════════════════════════════════════════════════════════════

def run_bounds(M=4, Nc=2, snr_db=20, n_alpha=20):
    """Compute all bounds for the CRB-Rate region."""
    P_T = snr_to_pt(snr_db)
    Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)
    alpha_values = np.linspace(0.01, 0.99, n_alpha)

    phi_func = make_phi_func(M)

    print("  Computing outer bound...")
    e_outer, R_outer, _ = outer_bound(
        alpha_values, Hc, phi_func=phi_func,
        T=T, sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Nc=Nc,
    )

    print("  Computing Gaussian inner bound...")
    e_gauss, R_gauss, _ = gaussian_inner_bound(
        alpha_values, Hc, phi_func=phi_func,
        T=T, sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Nc=Nc, n_samples=1,
    )

    print("  Computing semi-unitary inner bound...")
    e_su, R_su, _ = semi_unitary_inner_bound(
        alpha_values, Hc, phi_func=phi_func,
        T=T, sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Nc=Nc, n_stiefel_samples=20,
    )

    print("  Computing corner points...")
    corners = compute_corner_points(
        Hc, phi_func=phi_func,
        T=T, sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Nc=Nc,
    )

    return {
        'outer': (e_outer, R_outer),
        'gaussian': (e_gauss, R_gauss),
        'semi_unitary': (e_su, R_su),
        'corners': corners,
    }


def plot_figure2(bounds_data, snr_db=20):
    """Plot CRB-Rate Region with Bounds."""
    fig, ax = plt.subplots(figsize=(7, 5))

    e_outer, R_outer = bounds_data['outer']
    e_gauss, R_gauss = bounds_data['gaussian']
    e_su, R_su = bounds_data['semi_unitary']
    corners = bounds_data['corners']

    # Outer bound
    if len(e_outer) > 0:
        sort_idx = np.argsort(e_outer)
        ax.plot(e_outer[sort_idx], R_outer[sort_idx],
                'k-', linewidth=2.5, label='Outer Bound', zorder=5)

    # Gaussian inner bound
    if len(e_gauss) > 0:
        sort_idx = np.argsort(e_gauss)
        ax.plot(e_gauss[sort_idx], R_gauss[sort_idx],
                'b--', linewidth=2.0, marker='o', markersize=4,
                label='Gaussian Inner Bound', zorder=4)

    # Semi-unitary inner bound
    if len(e_su) > 0:
        sort_idx = np.argsort(e_su)
        ax.plot(e_su[sort_idx], R_su[sort_idx],
                'r-.', linewidth=2.0, marker='s', markersize=4,
                label='Semi-Unitary Inner Bound', zorder=4)

    # Pentagon bound (time-sharing between corner points)
    P_sc = (corners['e_min'], corners['R_sc'])
    P_cs = (corners['e_cs'], corners['R_max'])
    e_pent, R_pent = pentagon_inner_bound(P_sc, P_cs, corners['e_min'], corners['R_max'])
    ax.plot(e_pent, R_pent,
            'g:', linewidth=2.0, label='Pentagon Bound (Time-Sharing)', zorder=3)

    # Mark corner points
    ax.plot(corners['e_min'], corners['R_sc'], 'ko', markersize=8, zorder=6)
    ax.plot(corners['e_cs'], corners['R_max'], 'kD', markersize=8, zorder=6)
    ax.annotate('$P_{sc}$', (corners['e_min'], corners['R_sc']),
                textcoords="offset points", xytext=(10, -10), fontsize=11)
    ax.annotate('$P_{cs}$', (corners['e_cs'], corners['R_max']),
                textcoords="offset points", xytext=(10, 5), fontsize=11)

    # Fill achievable region (between outer and pentagon)
    if len(e_outer) > 0 and len(e_pent) > 0:
        try:
            e_all = np.sort(np.unique(np.concatenate([e_outer, e_pent])))
            R_outer_interp = np.interp(e_all, e_outer[sort_idx], R_outer[sort_idx])
            R_pent_interp = np.interp(e_all, e_pent, R_pent, left=0, right=0)
            ax.fill_between(e_all, R_pent_interp, R_outer_interp,
                            alpha=0.08, color='steelblue', label='Achievable Region')
        except Exception:
            pass

    ax.set_xlabel('CRB (rad²)')
    ax.set_ylabel('Rate (nats/channel use)')
    ax.set_title(f'CRB–Rate Achievable Region (SNR = {snr_db} dB)')
    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray',
              fontsize=9)
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')

    fig.tight_layout()
    path = 'results/figure2_bounds_region.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Effect of Antenna Number
# ══════════════════════════════════════════════════════════════════════

def run_antenna_sweep(M_list=(2, 4, 8), Nc=2, snr_db=20, n_alpha=20):
    """Run Pareto frontier for different antenna numbers."""
    alpha_values = np.linspace(0.01, 0.99, n_alpha)
    results = {}

    for M in M_list:
        P_T = snr_to_pt(snr_db)
        # Generate random channel (Nc x M)
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

        rates = []
        crbs = []

        for alpha in alpha_values:
            try:
                Rx_opt = covariance_shaping(
                    alpha=alpha, P_T=P_T, M=M, Hc=Hc,
                    sigma_c2=sigma_c2, sigma_s2=sigma_s2, T=T,
                )

                eigvals = np.linalg.eigvalsh(Rx_opt)
                if np.any(eigvals < -1e-6):
                    continue

                R = compute_rate(Rx_opt, Hc, sigma_c2)
                crb = compute_sensing_crb_from_rx(Rx_opt, M, T, sigma_s2)

                if R > 0 and crb > 0:
                    rates.append(R)
                    crbs.append(crb)
            except Exception as e:
                print(f"  [skip] M={M}, alpha={alpha:.2f}: {e}")
                continue

        results[M] = {
            'rates': np.array(rates),
            'crbs': np.array(crbs),
        }
        print(f"  M={M}: {len(rates)} points computed")

    return results


def plot_figure3(results, snr_db=20):
    """Plot Effect of Antenna Number on Pareto Frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['#7B1FA2', '#1976D2', '#E64A19']
    markers = ['D', 'o', 's']
    M_list = sorted(results.keys())

    for idx, M in enumerate(M_list):
        data = results[M]
        ax.plot(
            data['crbs'], data['rates'],
            color=colors[idx], marker=markers[idx],
            markersize=5, linewidth=2.0,
            label=f'$M = {M}$',
            zorder=3,
        )

    ax.set_xlabel('CRB (rad²)')
    ax.set_ylabel('Rate (nats/channel use)')
    ax.set_title(f'Effect of Antenna Number on Rate–CRB Tradeoff (SNR = {snr_db} dB)')
    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')

    fig.tight_layout()
    path = 'results/figure3_antenna_effect.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Helper: compute CRB from Rx with proper angle-estimation FIM
# ══════════════════════════════════════════════════════════════════════

def compute_sensing_crb_from_rx(Rx, M, T, sigma_s2, d=0.5):
    """Compute sensing CRB for angle estimation with ULA.

    Uses the Fisher information:
        J = (T / sigma_s^2) * 4*pi^2 * tr(D^2 Rx)

    where D = diag(positions), positions = (m - (M-1)/2) * d.

    CRB = sigma_s^2 / (T * 4*pi^2 * tr(D^2 Rx))
    """
    positions = (np.arange(M) - (M - 1) / 2.0) * d
    D2 = positions ** 2

    # tr(D^2 Rx) = sum of D2[m] * Rx[m,m] for diagonal Rx
    # More precisely: tr(diag(D2) @ Rx)
    tr_D2_Rx = np.real(np.trace(np.diag(D2) @ Rx))

    if tr_D2_Rx < 1e-15:
        return np.inf

    crb = sigma_s2 / (T * 4 * np.pi**2 * tr_D2_Rx)
    return max(crb, 0.0)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    M = 4
    Nc = 2

    print("=" * 60)
    print("ISAC Capacity-Distortion Tradeoff — Figure Generation")
    print("=" * 60)

    # ── Figure 1: Pareto Frontier ──────────────────────────────
    print("\n[Figure 1] Rate-CRB Pareto Frontier...")
    pareto_results = run_pareto_frontier(M=M, Nc=Nc, snr_db_list=(10, 15, 20), n_alpha=20)
    path1 = plot_figure1(pareto_results, M=M, Nc=Nc)

    # ── Figure 2: Bounds Region ────────────────────────────────
    print("\n[Figure 2] CRB-Rate Achievable Region with Bounds...")
    bounds_data = run_bounds(M=M, Nc=Nc, snr_db=20, n_alpha=15)
    path2 = plot_figure2(bounds_data, snr_db=20)

    # ── Figure 3: Antenna Effect ───────────────────────────────
    print("\n[Figure 3] Effect of Antenna Number...")
    antenna_results = run_antenna_sweep(M_list=(2, 4, 8), Nc=2, snr_db=20, n_alpha=20)
    path3 = plot_figure3(antenna_results, snr_db=20)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
