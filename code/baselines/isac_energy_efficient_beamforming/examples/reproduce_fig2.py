#!/usr/bin/env python3
"""
Reproduce Figure 2: EE_C Convergence + EE_C vs CRB Threshold
=============================================================

This script reproduces Figure 2 from the paper:
    - Left: EE_C convergence of Dinkelbach method
    - Right: EE_C vs CRB threshold

Reference: Zou et al., IEEE Trans. Commun., 2024 (Fig. 2)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system_model import ISACSystemModel
from src.dinkelbach_solver import DinkelbachSolver
from src.ee_metrics import compute_ee_c, compute_crb


def plot_ee_c_convergence(ax, model, target_angle_deg=90.0):
    """
    Plot EE_C convergence over Dinkelbach iterations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    model : ISACSystemModel
        System model
    target_angle_deg : float
        Target angle
    """
    solver = DinkelbachSolver(model, max_dinkelbach_iter=30, verbose=True)
    result = solver.solve(target_angle_deg=target_angle_deg)

    iterations = range(1, len(result.obj_history) + 1)

    ax.plot(iterations, result.obj_history, 'b-o', markersize=6, linewidth=2)
    ax.set_xlabel('Dinkelbach Iteration', fontsize=12)
    ax.set_ylabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    ax.set_title('(a) EE$_C$ Convergence', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(result.obj_history) + 0.5)


def plot_ee_c_vs_crb_threshold(ax, model, target_angle_deg=90.0):
    """
    Plot EE_C vs CRB threshold.

    Varies the CRB constraint and shows how EE_C decreases.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    model : ISACSystemModel
        System model
    target_angle_deg : float
        Target angle
    """
    # Generate CRB thresholds
    crb_thresholds = np.logspace(-4, -1, 10)  # From tight to loose
    ee_c_values = []

    solver = DinkelbachSolver(model, max_dinkelbach_iter=20, verbose=False)

    for crb_max in crb_thresholds:
        try:
            result = solver.solve(
                target_angle_deg=target_angle_deg,
                crb_max=crb_max,
            )
            ee_c_values.append(result.ee_c if result.converged else 0)
        except Exception:
            ee_c_values.append(0)

    ax.semilogx(crb_thresholds, ee_c_values, 'r-s', markersize=6, linewidth=2)
    ax.set_xlabel('CRB Threshold', fontsize=12)
    ax.set_ylabel('$EE_C$ (bits/Hz/J)', fontsize=12)
    ax.set_title('(b) EE$_C$ vs CRB Threshold', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')


def main():
    """Main function to generate Figure 2."""
    print("=" * 60)
    print("Reproducing Figure 2: EE_C Convergence + EE_C vs CRB Threshold")
    print("=" * 60)

    # Default parameters (Section VI)
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

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot (a): Convergence
    print("\nGenerating convergence plot...")
    plot_ee_c_convergence(ax1, model)

    # Plot (b): EE_C vs CRB threshold
    print("\nGenerating EE_C vs CRB threshold plot...")
    plot_ee_c_vs_crb_threshold(ax2, model)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig2_ee_c_convergence.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
