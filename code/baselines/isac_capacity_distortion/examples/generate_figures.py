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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set professional plotting style
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 100

from system_model import (
    compute_rate,
    compute_crb,
    compute_phi_angle,
    angle_to_channel,
    angle_to_hfunc,
    make_uniform_linear_array,
)
from bounds import (
    pentagon_inner_bound,
    gaussian_inner_bound,
    semi_unitary_inner_bound,
    outer_bound,
    compute_corner_points,
)
from optimization import (
    optimize_sensing_rx,
    optimize_comm_rx,
)


def setup_angle_estimation(
    M: int = 10,
    Ns: int = 10,
    Nc: int = 1,
    theta_c_deg: float = 42.0,
    sensing_snr_db: float = 20.0,
    comm_snr_db: float = 33.0,
    d: float = 0.5,
) -> dict:
    """Setup the angle estimation case study."""
    theta_c = np.deg2rad(theta_c_deg)

    # Communication channel: LoS from angle theta_c
    Hc = angle_to_channel(theta_c, M, Nc, d, d)

    # Normalize Hc to match SNR
    Hc_norm = np.linalg.norm(Hc, 'fro')
    if Hc_norm > 0:
        Hc = Hc / Hc_norm * np.sqrt(M * Nc)

    # Sensing channel function
    Hs_func = angle_to_hfunc(M, Ns, d, d)

    # Noise variances from SNR
    P_T = 1.0
    sigma_s2 = P_T * 10 ** (-sensing_snr_db / 10)
    sigma_c2 = P_T * 10 ** (-comm_snr_db / 10)

    # Compute correlation coefficient
    a_c = make_uniform_linear_array(M, d)(theta_c).flatten()
    a_s = make_uniform_linear_array(M, d)(np.deg2rad(30)).flatten()
    rho = np.abs(np.vdot(a_c, a_s)) / (np.linalg.norm(a_c) * np.linalg.norm(a_s))

    return {
        'M': M, 'Ns': Ns, 'Nc': Nc,
        'Hc': Hc,
        'Hs_func': Hs_func,
        'sigma_c2': sigma_c2,
        'sigma_s2': sigma_s2,
        'P_T': P_T,
        'theta_c': theta_c,
        'theta_c_deg': theta_c_deg,
        'rho': rho,
        'd': d,
    }


def make_phi_angle_func(M: int, Ns: int, theta_target: float, d: float = 0.5, Jp: float = 0.0):
    """Create Phi function for angle estimation."""
    def phi_func(Rx):
        phi_val = compute_phi_angle(
            Rx, 1, theta_target, M, Ns, d, d,
            Jp=Jp if Jp > 0 else None
        )
        return phi_val
    return phi_func


def generate_figure_a1(output_dir: str = "results", save: bool = True):
    """
    Figure A1: Capacity-Distortion Pareto curve (Rate vs CRB) for different SNR values.
    
    Shows how the tradeoff changes with SNR (5, 10, 15, 20 dB).
    """
    print("Generating Figure A1: Capacity-Distortion Pareto curve for different SNR values...")

    T = 3
    snr_values = [5, 10, 15, 20]  # dB
    theta_target = np.deg2rad(30)
    
    # Fixed comm SNR, varying sensing SNR
    comm_snr_db = 33.0
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use a professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'd']

    for idx, sensing_snr_db in enumerate(snr_values):
        print(f"  Processing sensing SNR = {sensing_snr_db} dB...")
        
        params = setup_angle_estimation(
            theta_c_deg=42.0,
            sensing_snr_db=sensing_snr_db,
            comm_snr_db=comm_snr_db
        )
        
        M = params['M']
        Ns = params['Ns']
        Hc = params['Hc']
        sigma_c2 = params['sigma_c2']
        sigma_s2 = params['sigma_s2']
        P_T = params['P_T']

        # Prior information
        Jp_scalar = (180 / (np.pi * 5))**2
        phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
        Jp = np.array([[Jp_scalar]])

        # Compute corner points
        corners = compute_corner_points(
            Hc, Hs_func=params['Hs_func'],
            phi_func=phi_func, T=T,
            sigma_c2=sigma_c2, sigma_s2=sigma_s2,
            P_T=P_T, M=M, Jp=Jp, Nc=params['Nc']
        )

        # Alpha values for tradeoff
        alpha_vals = np.linspace(0, 0.99, 20)

        # Gaussian inner bound (represents the Pareto frontier)
        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, Hc, params['Hs_func'], phi_func,
            T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
        )

        if len(e_gauss) > 0:
            label = f'SNR = {sensing_snr_db} dB'
            ax.plot(R_gauss, e_gauss, '-', color=colors[idx], 
                    linewidth=2.5, label=label, marker=markers[idx], 
                    markersize=4, markevery=5)
            
            # Mark corner points
            ax.plot(corners['R_sc'], corners['e_min'], markers[idx],
                    color=colors[idx], markersize=8, markeredgecolor='black',
                    markeredgewidth=0.5)
            ax.plot(corners['R_max'], corners['e_cs'], markers[idx],
                    color=colors[idx], markersize=8, markeredgecolor='black',
                    markeredgewidth=0.5, fillstyle='none')

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)', fontsize=13)
    ax.set_ylabel('Sensing CRB $e$', fontsize=13)
    ax.set_title('Figure A1: Capacity-Distortion Pareto Curve for Different SNR Values\n'
                 f'($T={T}$, $\theta_c=42°$, Comm SNR = {int(comm_snr_db)} dB)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / 'figure_a1_pareto_snr.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved to {filepath}")

    return fig, ax


def generate_figure_a2(output_dir: str = "results", save: bool = True):
    """
    Figure A2: CRB-Rate region with inner bounds (Pentagon, Gaussian, Semi-Unitary) 
    and outer bound.
    
    Shows the complete achievable region characterization.
    """
    print("Generating Figure A2: CRB-Rate region with inner and outer bounds...")

    T = 3
    theta_target = np.deg2rad(30)
    params = setup_angle_estimation(theta_c_deg=42.0)

    M = params['M']
    Ns = params['Ns']
    Hc = params['Hc']
    sigma_c2 = params['sigma_c2']
    sigma_s2 = params['sigma_s2']
    P_T = params['P_T']

    # Prior information
    Jp_scalar = (180 / (np.pi * 5))**2
    phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
    Jp = np.array([[Jp_scalar]])

    # Compute corner points
    corners = compute_corner_points(
        Hc, Hs_func=params['Hs_func'],
        phi_func=phi_func, T=T,
        sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Jp=Jp, Nc=params['Nc']
    )

    print(f"  Corner points:")
    print(f"    P_sc = (e_min={corners['e_min']:.4f}, R_sc={corners['R_sc']:.4f})")
    print(f"    P_cs = (e_cs={corners['e_cs']:.4f}, R_max={corners['R_max']:.4f})")
    print(f"    rho = {params['rho']:.4f}")

    # Alpha values for tradeoff
    alpha_vals = np.linspace(0, 0.99, 25)

    # Pentagon bound
    print("  Computing Pentagon inner bound...")
    e_pent, R_pent = pentagon_inner_bound(
        (corners['e_min'], corners['R_sc']),
        (corners['e_cs'], corners['R_max']),
        corners['e_min'], corners['R_max'],
        n_points=100
    )

    # Gaussian inner bound
    print("  Computing Gaussian inner bound...")
    e_gauss, R_gauss, _ = gaussian_inner_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
    )

    # Semi-unitary inner bound
    print("  Computing Semi-unitary inner bound...")
    e_su, R_su, _ = semi_unitary_inner_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, M_sc=min(M, T),
        Jp=Jp, Nc=params['Nc'], n_stiefel_samples=10,
    )

    # Outer bound
    print("  Computing Outer bound...")
    e_outer, R_outer, _ = outer_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pentagon bound (filled region)
    ax.fill_between(R_pent, 0, e_pent, alpha=0.15, color='#1f77b4', label='Pentagon Bound (Achievable)')
    ax.plot(R_pent, e_pent, 'b-', linewidth=1.5, alpha=0.7)

    # Gaussian inner bound
    if len(e_gauss) > 0:
        ax.plot(R_gauss, e_gauss, 'g-', linewidth=2.5, label='Gaussian Inner Bound', marker='s', 
                markersize=4, markevery=6)

    # Semi-unitary inner bound
    if len(e_su) > 0:
        ax.plot(R_su, e_su, 'm--', linewidth=2.5, label='Semi-Unitary Inner Bound', marker='^',
                markersize=4, markevery=6)

    # Outer bound
    if len(e_outer) > 0:
        ax.plot(R_outer, e_outer, 'r:', linewidth=2.5, label='Outer Bound', marker='d',
                markersize=4, markevery=6)

    # Corner points
    ax.plot(corners['R_sc'], corners['e_min'], 'ko', markersize=10, 
            label='$P_{sc}$ (Sensing-Optimal)', zorder=5)
    ax.plot(corners['R_max'], corners['e_cs'], 'k^', markersize=10, 
            label='$P_{cs}$ (Comm-Optimal)', zorder=5)
    
    # Add annotations for corner points
    ax.annotate('$P_{sc}$', 
                xy=(corners['R_sc'], corners['e_min']),
                xytext=(corners['R_sc'] - 0.3, corners['e_min'] + 0.5),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.annotate('$P_{cs}$', 
                xy=(corners['R_max'], corners['e_cs']),
                xytext=(corners['R_max'] - 0.5, corners['e_cs'] + 2),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)', fontsize=13)
    ax.set_ylabel('Sensing CRB $e$', fontsize=13)
    ax.set_title('Figure A2: CRB-Rate Region with Inner and Outer Bounds\n'
                 f'($T={T}$, $\theta_c=42°$, $\\rho \\approx {params["rho"]:.2f}$)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / 'figure_a2_bounds_comparison.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved to {filepath}")

    return fig, ax


def generate_figure_a3(output_dir: str = "results", save: bool = True):
    """
    Figure A3: Tradeoff curves for different antenna numbers.
    
    Shows how the CRB-Rate region changes with M (number of Tx antennas).
    """
    print("Generating Figure A3: Tradeoff curves for different antenna numbers...")

    T = 3
    theta_target = np.deg2rad(30)
    antenna_configs = [4, 6, 8, 10]  # Different M values
    
    # Fixed SNR values
    sensing_snr_db = 20.0
    comm_snr_db = 33.0
    theta_c_deg = 42.0

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Professional color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    markers = ['o', 's', '^', 'd']

    for idx, M in enumerate(antenna_configs):
        print(f"  Processing M = {M} antennas...")
        
        Ns = M  # Match sensing antennas to Tx antennas
        Nc = 1  # Single comm Rx antenna
        
        params = setup_angle_estimation(
            M=M, Ns=Ns, Nc=Nc,
            theta_c_deg=theta_c_deg,
            sensing_snr_db=sensing_snr_db,
            comm_snr_db=comm_snr_db
        )
        
        Hc = params['Hc']
        sigma_c2 = params['sigma_c2']
        sigma_s2 = params['sigma_s2']
        P_T = params['P_T']

        # Prior information
        Jp_scalar = (180 / (np.pi * 5))**2
        phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
        Jp = np.array([[Jp_scalar]])

        # Compute corner points
        corners = compute_corner_points(
            Hc, Hs_func=params['Hs_func'],
            phi_func=phi_func, T=T,
            sigma_c2=sigma_c2, sigma_s2=sigma_s2,
            P_T=P_T, M=M, Jp=Jp, Nc=Nc
        )

        # Alpha values for tradeoff
        alpha_vals = np.linspace(0, 0.99, 20)

        # Gaussian inner bound
        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, Hc, params['Hs_func'], phi_func,
            T, sigma_c2, sigma_s2, P_T, M, Jp, Nc,
        )

        if len(e_gauss) > 0:
            label = f'$M = N_s = {M}$'
            ax.plot(R_gauss, e_gauss, '-', color=colors[idx], 
                    linewidth=2.5, label=label, marker=markers[idx],
                    markersize=4, markevery=5)
            
            # Mark corner points
            ax.plot(corners['R_sc'], corners['e_min'], markers[idx],
                    color=colors[idx], markersize=8, markeredgecolor='black',
                    markeredgewidth=0.5)
            ax.plot(corners['R_max'], corners['e_cs'], markers[idx],
                    color=colors[idx], markersize=8, markeredgecolor='black',
                    markeredgewidth=0.5, fillstyle='none')

    ax.set_xlabel('Communication Rate $R$ (nats/channel use)', fontsize=13)
    ax.set_ylabel('Sensing CRB $e$', fontsize=13)
    ax.set_title('Figure A3: Capacity-Distortion Tradeoff for Different Antenna Numbers\n'
                 f'($T={T}$, $\theta_c={int(theta_c_deg)}°$, $N_c=1$)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, title='Configuration')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / 'figure_a3_antenna_tradeoff.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved to {filepath}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation figures for P0-A ISAC Capacity-Distortion baseline"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save figures (default: results)"
    )
    parser.add_argument(
        "--figures", type=str, default="all",
        help="Figures to generate: all, a1, a2, a3, or comma-separated (default: all)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure DPI (default: 300)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    figures = args.figures.lower()

    print("=" * 70)
    print("P0-A Baseline: ISAC Capacity-Distortion Tradeoff - Figure Generation")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Figures to generate: {figures}")
    print(f"DPI: {args.dpi}")
    print()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    if figures in ["all", "a1"]:
        print("-" * 70)
        fig_a1, ax_a1 = generate_figure_a1(output_dir)
        results['fig_a1'] = (fig_a1, ax_a1)
        plt.close()
        print()

    if figures in ["all", "a2"]:
        print("-" * 70)
        fig_a2, ax_a2 = generate_figure_a2(output_dir)
        results['fig_a2'] = (fig_a2, ax_a2)
        plt.close()
        print()

    if figures in ["all", "a3"]:
        print("-" * 70)
        fig_a3, ax_a3 = generate_figure_a3(output_dir)
        results['fig_a3'] = (fig_a3, ax_a3)
        plt.close()
        print()

    print("=" * 70)
    print(f"Done! Figures saved to {output_dir}/")
    print("=" * 70)
    print("\nGenerated figures:")
    if 'fig_a1' in results:
        print(f"  - figure_a1_pareto_snr.png: Pareto curves for different SNR values")
    if 'fig_a2' in results:
        print(f"  - figure_a2_bounds_comparison.png: CRB-Rate region with bounds")
    if 'fig_a3' in results:
        print(f"  - figure_a3_antenna_tradeoff.png: Tradeoff curves for different antenna numbers")


if __name__ == "__main__":
    main()
