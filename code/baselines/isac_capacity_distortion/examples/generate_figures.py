#!/usr/bin/env python3
"""
Generate simulation figures for P0-A baseline.

Creates:
- Figure A1: Capacity-Distortion Pareto curve (Rate vs CRB) for different SNR values
- Figure A2: CRB-Rate region with inner bounds (Pentagon, Gaussian, Semi-Unitary) and outer bound
- Figure A3: Tradeoff curves for different antenna numbers

Usage:
    python generate_figures.py [--output-dir results]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 100

from system_model import (
    compute_rate, compute_crb, compute_phi_angle,
    angle_to_channel, angle_to_hfunc, make_uniform_linear_array,
)
from bounds import (
    pentagon_inner_bound, gaussian_inner_bound,
    semi_unitary_inner_bound, outer_bound, compute_corner_points,
)
from optimization import optimize_sensing_rx, optimize_comm_rx


def setup_angle_estimation(
    M=10, Ns=10, Nc=1, theta_c_deg=42.0,
    sensing_snr_db=20.0, comm_snr_db=33.0, d=0.5
):
    theta_c = np.deg2rad(theta_c_deg)
    Hc = angle_to_channel(theta_c, M, Nc, d, d)
    Hc_norm = np.linalg.norm(Hc, 'fro')
    if Hc_norm > 0:
        Hc = Hc / Hc_norm * np.sqrt(M * Nc)
    Hs_func = angle_to_hfunc(M, Ns, d, d)
    P_T = 1.0
    sigma_s2 = P_T * 10 ** (-sensing_snr_db / 10)
    sigma_c2 = P_T * 10 ** (-comm_snr_db / 10)
    a_c = make_uniform_linear_array(M, d)(theta_c).flatten()
    a_s = make_uniform_linear_array(M, d)(np.deg2rad(30)).flatten()
    rho = np.abs(np.vdot(a_c, a_s)) / (np.linalg.norm(a_c) * np.linalg.norm(a_s))
    return {
        'M': M, 'Ns': Ns, 'Nc': Nc, 'Hc': Hc, 'Hs_func': Hs_func,
        'sigma_c2': sigma_c2, 'sigma_s2': sigma_s2, 'P_T': P_T,
        'theta_c': theta_c, 'theta_c_deg': theta_c_deg, 'rho': rho, 'd': d,
    }


def make_phi_angle_func(M, Ns, theta_target, d=0.5, Jp=0.0):
    def phi_func(Rx):
        return compute_phi_angle(Rx, 1, theta_target, M, Ns, d, d,
                                 Jp=Jp if Jp > 0 else None)
    return phi_func


def save_figure(fig, filepath):
    """Save figure safely without bbox_inches='tight' to avoid renderer issues."""
    fig.savefig(filepath, dpi=200, facecolor='white', pad_inches=0.3)
    print(f"  Saved to {filepath}")


def generate_figure_a1(output_dir="results", save=True):
    """Figure A1: Capacity-Distortion Pareto curve for different SNR values."""
    print("Generating Figure A1: Pareto curves for different SNR values...")
    T = 3
    snr_values = [5, 10, 15, 20]
    comm_snr_db = 33.0
    theta_target = np.deg2rad(30)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'd']
    alpha_vals = np.linspace(0, 0.99, 15)

    all_e_values = []
    all_R_values = []

    for idx, sensing_snr_db in enumerate(snr_values):
        print(f"  Processing sensing SNR = {sensing_snr_db} dB...")
        params = setup_angle_estimation(theta_c_deg=42.0, sensing_snr_db=sensing_snr_db,
                                        comm_snr_db=comm_snr_db)
        M, Ns = params['M'], params['Ns']
        Jp_scalar = (180 / (np.pi * 5))**2
        phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
        Jp = np.array([[Jp_scalar]])

        corners = compute_corner_points(
            params['Hc'], Hs_func=params['Hs_func'], phi_func=phi_func,
            T=T, sigma_c2=params['sigma_c2'], sigma_s2=params['sigma_s2'],
            P_T=params['P_T'], M=M, Jp=Jp, Nc=params['Nc'])

        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, params['Hc'], params['Hs_func'], phi_func,
            T, params['sigma_c2'], params['sigma_s2'], params['P_T'],
            M, Jp, params['Nc'])

        if len(e_gauss) > 0:
            valid = np.isfinite(e_gauss) & np.isfinite(R_gauss) & (e_gauss < 1e6)
            e_g = e_gauss[valid]; R_g = R_gauss[valid]
            if len(e_g) > 0:
                all_e_values.extend(e_g); all_R_values.extend(R_g)
                ax.plot(R_g, e_g, '-', color=colors[idx], linewidth=2.5,
                        label=f'SNR = {sensing_snr_db} dB', marker=markers[idx],
                        markersize=4, markevery=max(1, len(e_g)//6))
                ax.plot(corners['R_sc'], corners['e_min'], markers[idx],
                        color=colors[idx], markersize=8, markeredgecolor='black', markeredgewidth=0.5)
                ax.plot(corners['R_max'], corners['e_cs'], markers[idx],
                        color=colors[idx], markersize=8, markeredgecolor='black',
                        markeredgewidth=0.5, fillstyle='none')

    if all_e_values:
        ax.set_ylim(0, min(max(all_e_values) * 1.1, np.percentile(all_e_values, 95) * 2))

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)')
    ax.set_ylabel('Sensing CRB $e$')
    ax.set_title(f'Figure A1: Capacity-Distortion Pareto Curve for Different SNR Values\n'
                 f'($T={T}$, $\\theta_c=42°$, Comm SNR = {int(comm_snr_db)} dB)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_figure(fig, Path(output_dir) / 'figure_a1_pareto_snr.png')
    return fig, ax


def generate_figure_a2(output_dir="results", save=True):
    """Figure A2: CRB-Rate region with all bounds."""
    print("Generating Figure A2: CRB-Rate region with bounds...")
    T = 3
    theta_target = np.deg2rad(30)
    params = setup_angle_estimation(theta_c_deg=42.0)
    M, Ns = params['M'], params['Ns']
    Jp_scalar = (180 / (np.pi * 5))**2
    phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
    Jp = np.array([[Jp_scalar]])
    alpha_vals = np.linspace(0, 0.99, 15)

    corners = compute_corner_points(
        params['Hc'], Hs_func=params['Hs_func'], phi_func=phi_func,
        T=T, sigma_c2=params['sigma_c2'], sigma_s2=params['sigma_s2'],
        P_T=params['P_T'], M=M, Jp=Jp, Nc=params['Nc'])

    print(f"  P_sc = ({corners['e_min']:.6f}, {corners['R_sc']:.4f})")
    print(f"  P_cs = ({corners['e_cs']:.6f}, {corners['R_max']:.4f})")

    print("  Pentagon bound...")
    e_pent, R_pent = pentagon_inner_bound(
        (corners['e_min'], corners['R_sc']),
        (corners['e_cs'], corners['R_max']),
        corners['e_min'], corners['R_max'], n_points=60)

    print("  Gaussian bound...")
    e_gauss, R_gauss, _ = gaussian_inner_bound(
        alpha_vals, params['Hc'], params['Hs_func'], phi_func,
        T, params['sigma_c2'], params['sigma_s2'], params['P_T'],
        M, Jp, params['Nc'])

    print("  Semi-unitary bound...")
    e_su, R_su, _ = semi_unitary_inner_bound(
        alpha_vals, params['Hc'], params['Hs_func'], phi_func,
        T, params['sigma_c2'], params['sigma_s2'], params['P_T'],
        M, M_sc=min(M, T), Jp=Jp, Nc=params['Nc'], n_stiefel_samples=10)

    print("  Outer bound...")
    e_outer, R_outer, _ = outer_bound(
        alpha_vals, params['Hc'], params['Hs_func'], phi_func,
        T, params['sigma_c2'], params['sigma_s2'], params['P_T'],
        M, Jp, params['Nc'])

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Determine reasonable y-limit from valid data
    all_e = []
    for arr in [e_pent, e_gauss, e_su, e_outer]:
        valid = arr[np.isfinite(arr) & (arr < 1e6)]
        if len(valid) > 0: all_e.extend(valid)
    e_max = min(max(all_e) * 1.1, np.percentile(all_e, 95) * 3) if all_e else 1.0

    # Filter data for plotting
    def filter_data(e, R, emax=e_max):
        valid = np.isfinite(e) & np.isfinite(R) & (e < emax) & (R >= 0)
        return e[valid], R[valid]

    e_p, R_p = filter_data(e_pent, R_pent)
    if len(e_p) > 0:
        ax.fill_between(R_p, 0, e_p, alpha=0.15, color='#1f77b4', label='Pentagon Bound')
        ax.plot(R_p, e_p, 'b-', linewidth=1.5, alpha=0.7)

    e_g, R_g = filter_data(e_gauss, R_gauss)
    if len(e_g) > 0:
        ax.plot(R_g, e_g, 'g-', linewidth=2.5, label='Gaussian Inner Bound',
                marker='s', markersize=4, markevery=3)

    e_s, R_s = filter_data(e_su, R_su)
    if len(e_s) > 0:
        ax.plot(R_s, e_s, 'm--', linewidth=2.5, label='Semi-Unitary Inner Bound',
                marker='^', markersize=4, markevery=3)

    e_o, R_o = filter_data(e_outer, R_outer)
    if len(e_o) > 0:
        ax.plot(R_o, e_o, 'r:', linewidth=2.5, label='Outer Bound',
                marker='d', markersize=4, markevery=3)

    ax.plot(corners['R_sc'], corners['e_min'], 'ko', markersize=10, label='$P_{sc}$', zorder=5)
    ax.plot(corners['R_max'], corners['e_cs'], 'k^', markersize=10, label='$P_{cs}$', zorder=5)

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)')
    ax.set_ylabel('Sensing CRB $e$')
    ax.set_title(f'Figure A2: CRB-Rate Region with Inner and Outer Bounds\n'
                 f'($T={T}$, $\\theta_c=42°$, $\\rho \\approx {params["rho"]:.2f}$)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(0, e_max)

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_figure(fig, Path(output_dir) / 'figure_a2_bounds_comparison.png')
    return fig, ax


def generate_figure_a3(output_dir="results", save=True):
    """Figure A3: Tradeoff curves for different antenna numbers."""
    print("Generating Figure A3: Tradeoff curves for different antenna numbers...")
    T = 3
    theta_target = np.deg2rad(30)
    antenna_configs = [4, 6, 8, 10]
    sensing_snr_db, comm_snr_db, theta_c_deg = 20.0, 33.0, 42.0

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    markers = ['o', 's', '^', 'd']
    alpha_vals = np.linspace(0, 0.99, 15)

    all_e_values = []

    for idx, M in enumerate(antenna_configs):
        print(f"  Processing M = {M} antennas...")
        Ns, Nc = M, 1
        params = setup_angle_estimation(M=M, Ns=Ns, Nc=Nc, theta_c_deg=theta_c_deg,
                                        sensing_snr_db=sensing_snr_db, comm_snr_db=comm_snr_db)
        Jp_scalar = (180 / (np.pi * 5))**2
        phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
        Jp = np.array([[Jp_scalar]])

        corners = compute_corner_points(
            params['Hc'], Hs_func=params['Hs_func'], phi_func=phi_func,
            T=T, sigma_c2=params['sigma_c2'], sigma_s2=params['sigma_s2'],
            P_T=params['P_T'], M=M, Jp=Jp, Nc=Nc)

        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, params['Hc'], params['Hs_func'], phi_func,
            T, params['sigma_c2'], params['sigma_s2'], params['P_T'],
            M, Jp, Nc)

        if len(e_gauss) > 0:
            valid = np.isfinite(e_gauss) & np.isfinite(R_gauss) & (e_gauss < 1e6)
            e_g = e_gauss[valid]; R_g = R_gauss[valid]
            if len(e_g) > 0:
                all_e_values.extend(e_g)
                ax.plot(R_g, e_g, '-', color=colors[idx], linewidth=2.5,
                        label=f'$M = N_s = {M}$', marker=markers[idx],
                        markersize=4, markevery=max(1, len(e_g)//6))
                ax.plot(corners['R_sc'], corners['e_min'], markers[idx],
                        color=colors[idx], markersize=8, markeredgecolor='black', markeredgewidth=0.5)
                ax.plot(corners['R_max'], corners['e_cs'], markers[idx],
                        color=colors[idx], markersize=8, markeredgecolor='black',
                        markeredgewidth=0.5, fillstyle='none')

    if all_e_values:
        ax.set_ylim(0, min(max(all_e_values) * 1.1, np.percentile(all_e_values, 95) * 2))

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)')
    ax.set_ylabel('Sensing CRB $e$')
    ax.set_title(f'Figure A3: Capacity-Distortion Tradeoff for Different Antenna Numbers\n'
                 f'($T={T}$, $\\theta_c={int(theta_c_deg)}°$, $N_c=1$)',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, title='Configuration')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_figure(fig, Path(output_dir) / 'figure_a3_antenna_tradeoff.png')
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Generate P0-A ISAC Capacity-Distortion baseline figures")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figures", default="all",
                        help="all, a1, a2, a3, or comma-separated")
    args = parser.parse_args()

    output_dir = args.output_dir
    figures = args.figures.lower()

    print("=" * 70)
    print("P0-A Baseline: ISAC Capacity-Distortion - Figure Generation")
    print("=" * 70)
    print(f"Output: {output_dir} | Figures: {figures}")
    print()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if figures in ["all", "a1"]:
        print("-" * 70)
        fig, ax = generate_figure_a1(output_dir)
        plt.close()
        print()

    if figures in ["all", "a2"]:
        print("-" * 70)
        fig, ax = generate_figure_a2(output_dir)
        plt.close()
        print()

    if figures in ["all", "a3"]:
        print("-" * 70)
        fig, ax = generate_figure_a3(output_dir)
        plt.close()
        print()

    print("=" * 70)
    print(f"Done! Figures saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
