#!/usr/bin/env python3
"""
Reproduce figures from the paper:
"On the Fundamental Tradeoff of Integrated Sensing and Communications
Under Gaussian Channels," IEEE TIT, 2023.

Usage:
    python reproduce_figures.py [--output-dir results] [--figures 5,8,10]

Figures:
    - Figure 5: Rate vs CRB for T=3, theta_c=42° (rho ≈ 0.61)
    - Figure 6: Rate vs CRB for different theta_c values
    - Figure 8: Rate vs CRB for different T values
    - Figure 10: Rate vs normalized CRB for matrix estimation
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

from case_study import (
    generate_figure5,
    generate_figure8,
    generate_figure10,
    target_angle_estimation,
    target_response_estimation,
    setup_angle_estimation,
)
from system_model import (
    compute_rate,
    compute_crb,
    compute_phi_angle,
    make_uniform_linear_array,
)
from bounds import (
    gaussian_inner_bound,
    outer_bound,
    compute_corner_points,
)


def generate_figure6(output_dir: str = "results", save: bool = True):
    """Generate Figure 6: Rate vs CRB for different theta_c values.

    Shows the effect of correlation rho between comm and sensing channels.
    theta_c = {90°, 50°, 40°, 30°} with rho ≈ {0.05, 0.22, 0.72, 0.99}.
    """
    print("Generating Figure 6: Rate vs CRB for different theta_c...")

    T = 3
    theta_c_values = [90.0, 50.0, 40.0, 30.0]
    alpha_vals = np.linspace(0, 0.99, 40)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(theta_c_values)))

    for idx, theta_c_deg in enumerate(theta_c_values):
        print(f"  Processing theta_c = {theta_c_deg}°...")

        params = setup_angle_estimation(theta_c_deg=theta_c_deg)
        M = params['M']
        Ns = params['Ns']
        Hc = params['Hc']
        sigma_c2 = params['sigma_c2']
        sigma_s2 = params['sigma_s2']
        P_T = params['P_T']

        theta_target = np.deg2rad(30)
        Jp_scalar = (180 / (np.pi * 5))**2

        def phi_func(Rx):
            return compute_phi_angle(Rx, T, theta_target, M, Ns, params['d'],
                                     params['d'], Jp=Jp_scalar)

        Jp = np.array([[Jp_scalar]])

        # Compute corner points
        corners = compute_corner_points(
            Hc, params['Hs_func'], phi_func, T,
            sigma_c2, sigma_s2, P_T, M, Jp, params['Nc']
        )

        # Gaussian inner bound
        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, Hc, params['Hs_func'], phi_func,
            T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
        )

        rho = params['rho']
        label = f'$\\theta_c = {int(theta_c_deg)}°$, $\\rho \\approx {rho:.2f}$'

        if len(e_gauss) > 0:
            ax.plot(R_gauss, e_gauss, '-', color=colors[idx],
                    linewidth=2, label=label)

        # Corner points
        ax.plot(corners['R_sc'], corners['e_min'], 'o',
                color=colors[idx], markersize=6)
        ax.plot(corners['R_max'], corners['e_cs'], '^',
                color=colors[idx], markersize=6)

    ax.set_xlabel('Communication Rate R (nats/channel use)', fontsize=12)
    ax.set_ylabel('Sensing CRB e', fontsize=12)
    ax.set_title(f'CRB-Rate Region: Effect of Channel Correlation ($T={T}$)',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / 'figure6_theta_c_effect.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to {out / 'figure6_theta_c_effect.png'}")

    plt.close()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper figures for ISAC Capacity-Distortion Tradeoff"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save figures (default: results)"
    )
    parser.add_argument(
        "--figures", type=str, default="5,6,8,10",
        help="Comma-separated list of figures to generate (default: 5,6,8,10)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: 150)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    figures = [int(f.strip()) for f in args.figures.split(",")]

    print("=" * 60)
    print("ISAC Capacity-Distortion Tradeoff - Figure Reproduction")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Figures to generate: {figures}")
    print()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    if 5 in figures:
        fig5, ax5 = generate_figure5(output_dir)
        results['fig5'] = (fig5, ax5)
        plt.close()

    if 6 in figures:
        fig6, ax6 = generate_figure6(output_dir)
        results['fig6'] = (fig6, ax6)
        plt.close()

    if 8 in figures:
        fig8, ax8 = generate_figure8(output_dir)
        results['fig8'] = (fig8, ax8)
        plt.close()

    if 10 in figures:
        fig10, ax10 = generate_figure10(output_dir)
        results['fig10'] = (fig10, ax10)
        plt.close()

    print()
    print("=" * 60)
    print(f"Done! Figures saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
