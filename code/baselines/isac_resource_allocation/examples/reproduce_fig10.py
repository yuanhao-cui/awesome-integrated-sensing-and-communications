"""
Reproduce Figure 10: Localization CRB vs Rate Threshold.

This script reproduces the Cramér-Rao Bound performance for localization
as a function of communication rate threshold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.system_model import ISACSystem
from src.ao_solver import AOSolver
from src.localization_qos import LocalizationQoS


def main():
    """Main function to reproduce Figure 10."""
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Create system with paper parameters
    system = ISACSystem(
        Nt=32,
        Nr=32,
        Q=3,  # 3 sensing targets
        K=3,  # 3 communication users
        L=1,  # 1 ISAC user
        fc=30e9,  # 30 GHz carrier
        P_total=40.0,  # 40W total power
        B_total=100e6,  # 100 MHz total bandwidth
        rng=rng
    )
    
    localization = LocalizationQoS(system)
    
    # Rate threshold values (Γc) from paper
    Gamma_c_values = np.linspace(0.1, 5.0, 12)
    
    # Store results
    crb_range_results = []  # Mean range CRB
    crb_angle_results = []  # Mean angle CRB
    crb_combined_results = []  # Mean combined metric
    
    print("Running simulations for Figure 10...")
    print("=" * 50)
    
    for Gamma_c in Gamma_c_values:
        print(f"Γc = {Gamma_c:.2f} bps/Hz")
        
        # Solve for localization QoS
        solver = AOSolver(
            system,
            qos_type='localization',
            fairness='maxmin',
            max_iter=20,
            tol=1e-4
        )
        
        result = solver.solve(Gamma_c=Gamma_c)
        
        # Compute CRB metrics with obtained allocation
        Q = system.params.Q
        p_sensing = result.p[:Q]
        b_sensing = result.b[:Q]
        
        crb_range = localization.compute_crb_range(p_sensing, b_sensing)
        crb_angle = localization.compute_crb_angle(p_sensing, b_sensing)
        crb_combined = localization.compute_crb_combined(p_sensing, b_sensing)
        
        crb_range_results.append(np.mean(crb_range))
        crb_angle_results.append(np.mean(crb_angle))
        crb_combined_results.append(np.mean(crb_combined))
    
    # Convert to numpy arrays
    crb_range_results = np.array(crb_range_results)
    crb_angle_results = np.array(crb_angle_results)
    crb_combined_results = np.array(crb_combined_results)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Range CRB vs Rate Threshold
    axes[0, 0].semilogy(Gamma_c_values, crb_range_results, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    axes[0, 0].set_ylabel('Mean Range CRB (m²)', fontsize=12)
    axes[0, 0].set_title('Range Estimation CRB', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Angle CRB vs Rate Threshold
    axes[0, 1].semilogy(Gamma_c_values, crb_angle_results, 's-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    axes[0, 1].set_ylabel('Mean Angle CRB (rad²)', fontsize=12)
    axes[0, 1].set_title('Angle Estimation CRB', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Combined ρ metric vs Rate Threshold
    axes[1, 0].plot(Gamma_c_values, crb_combined_results, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    axes[1, 0].set_ylabel('Mean ρ (Combined Metric)', fontsize=12)
    axes[1, 0].set_title('Combined Localization Metric', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: RMSE vs Rate Threshold
    rmse_range = np.sqrt(crb_range_results)
    rmse_angle = np.sqrt(crb_angle_results)
    
    axes[1, 1].plot(Gamma_c_values, rmse_range, 'o-', linewidth=2, markersize=8, color='blue', label='Range RMSE')
    axes[1, 1].plot(Gamma_c_values, rmse_angle, 's-', linewidth=2, markersize=8, color='red', label='Angle RMSE')
    axes[1, 1].set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    axes[1, 1].set_ylabel('RMSE', fontsize=12)
    axes[1, 1].set_title('Localization RMSE', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('fig10_localization_crb_vs_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFigure 10 saved as 'fig10_localization_crb_vs_rate.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Γc range: [{Gamma_c_values[0]:.2f}, {Gamma_c_values[-1]:.2f}] bps/Hz")
    print(f"Range CRB range: [{crb_range_results.min():.2e}, {crb_range_results.max():.2e}] m²")
    print(f"Angle CRB range: [{crb_angle_results.min():.2e}, {crb_angle_results.max():.2e}] rad²")
    print(f"Combined ρ range: [{crb_combined_results.min():.2e}, {crb_combined_results.max():.2e}]")


if __name__ == "__main__":
    main()
