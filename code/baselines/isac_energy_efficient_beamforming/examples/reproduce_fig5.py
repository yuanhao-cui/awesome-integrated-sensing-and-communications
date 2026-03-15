#!/usr/bin/env python3
"""
Reproduce Figure 5: Pareto Boundary (EE_C vs EE_S)
===================================================

This script reproduces Figure 5 from the paper, showing the
tradeoff between communication-centric EE (EE_C) and
sensing-centric EE (EE_S).

Reference: Zou et al., IEEE Trans. Commun., 2024 (Fig. 5)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.pareto_optimizer import ParetoOptimizer
from src.baselines import EMaxBaseline, FixBeamBaseline, run_all_baselines


def plot_pareto_boundary(ax, model, n_points=15):
    """
    Plot Pareto boundary between EE_C and EE_S.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    model : ISACSystemModel
        System model
    n_points : int
        Number of Pareto boundary points
    """
    print("Computing Pareto boundary...")
    optimizer = ParetoOptimizer(model, n_pareto_points=n_points, verbose=True)

    try:
        pareto_points = optimizer.trace_pareto_boundary(
            target_angle_deg=90.0,
            n_points=n_points,
        )

        if pareto_points:
            ee_c_values = [pt.ee_c for pt in pareto_points]
            ee_s_values = [pt.ee_s for pt in pareto_points]

            ax.plot(ee_c_values, ee_s_values, 'b-o', markersize=8,
                    linewidth=2, label='Pareto Boundary', zorder=5)

            # Mark endpoints
            ax.plot(ee_c_values[0], ee_s_values[0], 'gs', markersize=12,
                    label='EE$_S$ Max', zorder=6)
            ax.plot(ee_c_values[-1], ee_s_values[-1], 'r^', markersize=12,
                    label='EE$_C$ Max', zorder=6)

            print(f"Pareto boundary: {len(pareto_points)} points")
            print(f"  EE_C range: [{min(ee_c_values):.4f}, {max(ee_c_values):.4f}]")
            print(f"  EE_S range: [{min(ee_s_values):.4f}, {max(ee_s_values):.4f}]")
        else:
            print("Warning: No Pareto points found")

    except Exception as e:
        print(f"Pareto optimization failed: {e}")

    ax.set_xlabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    ax.set_ylabel('$EE_S$ (1/J)', fontsize=12)
    ax.set_title('Pareto Boundary: EE$_C$ vs EE$_S$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)


def plot_baselines(ax, model):
    """
    Plot baseline schemes for comparison.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    model : ISACSystemModel
        System model
    """
    print("\nComputing baseline schemes...")
    results = run_all_baselines(model, target_angle_deg=90.0)

    colors = {'EMax': 'red', 'FixBeam': 'green', 'Random': 'orange'}
    markers = {'EMax': 's', 'FixBeam': 'D', 'Random': 'v'}

    for name, result in results.items():
        base_name = name.split('_')[0]
        color = colors.get(base_name, 'gray')
        marker = markers.get(base_name, 'o')

        ax.plot(result.ee_c, result.ee_s, marker=marker, color=color,
                markersize=10, label=name, zorder=4)
        print(f"  {name}: EE_C={result.ee_c:.4f}, EE_S={result.ee_s:.4f}")


def plot_pareto_vs_antennas(ax):
    """
    Plot Pareto boundary for different numbers of antennas.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    M_values = [14, 16, 18, 20]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for M, color in zip(M_values, colors):
        print(f"\nM = {M} antennas...")
        model = ISACSystemModel(M=M, K=4, N=20, seed=42)
        optimizer = ParetoOptimizer(model, n_pareto_points=10, verbose=False)

        try:
            pareto_points = optimizer.trace_pareto_boundary(n_points=10)
            if pareto_points:
                ee_c = [pt.ee_c for pt in pareto_points]
                ee_s = [pt.ee_s for pt in pareto_points]
                ax.plot(ee_c, ee_s, '-o', color=color, markersize=5,
                        linewidth=1.5, label=f'M = {M}')
        except Exception as e:
            print(f"  Failed for M={M}: {e}")

    ax.set_xlabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    ax.set_ylabel('$EE_S$ (1/J)', fontsize=12)
    ax.set_title('Pareto Boundary vs Antennas', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)


def plot_pareto_vs_users(ax):
    """
    Plot Pareto boundary for different numbers of users.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    K_values = [2, 4, 6, 8]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for K, color in zip(K_values, colors):
        print(f"\nK = {K} users...")
        model = ISACSystemModel(M=16, K=K, N=20, seed=42)
        optimizer = ParetoOptimizer(model, n_pareto_points=10, verbose=False)

        try:
            pareto_points = optimizer.trace_pareto_boundary(n_points=10)
            if pareto_points:
                ee_c = [pt.ee_c for pt in pareto_points]
                ee_s = [pt.ee_s for pt in pareto_points]
                ax.plot(ee_c, ee_s, '-o', color=color, markersize=5,
                        linewidth=1.5, label=f'K = {K}')
        except Exception as e:
            print(f"  Failed for K={K}: {e}")

    ax.set_xlabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    ax.set_ylabel('$EE_S$ (1/J)', fontsize=12)
    ax.set_title('Pareto Boundary vs Users', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)


def main():
    """Main function to generate Figure 5."""
    print("=" * 60)
    print("Reproducing Figure 5: Pareto Boundary")
    print("=" * 60)

    # Default parameters
    model = ISACSystemModel(
        M=16,
        K=4,
        N=20,
        P_max_dbm=30,
        P0_dbm=33,
        epsilon=0.35,
        L=30,
        seed=42,
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Pareto boundary with baselines
    print("\n(a) Pareto boundary with baselines...")
    plot_pareto_boundary(axes[0, 0], model, n_points=15)
    plot_baselines(axes[0, 0], model)

    # (b) Pareto boundary vs antennas
    print("\n(b) Pareto boundary vs antennas...")
    plot_pareto_vs_antennas(axes[0, 1])

    # (c) Pareto boundary vs users
    print("\n(c) Pareto boundary vs users...")
    plot_pareto_vs_users(axes[1, 0])

    # (d) Rate-power tradeoff
    print("\n(d) Rate vs Power tradeoff...")
    # Simple rate vs power plot
    P_max_values_dbm = np.arange(20, 41, 2)
    ee_c_values = []
    for P_dbm in P_max_values_dbm:
        model_p = ISACSystemModel(M=16, K=4, N=20, P_max_dbm=P_dbm, seed=42)
        from src.baselines import EMaxBaseline
        emax = EMaxBaseline(model_p)
        result = emax.solve()
        ee_c_values.append(result.ee_c)

    axes[1, 1].plot(P_max_values_dbm, ee_c_values, 'b-s', markersize=6, linewidth=2)
    axes[1, 1].set_xlabel('$P_{max}$ (dBm)', fontsize=12)
    axes[1, 1].set_ylabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    axes[1, 1].set_title('EE$_C$ vs Max Power', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig5_pareto_boundary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
