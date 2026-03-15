"""
Reproduce Figure 6: Detection Probability vs Rate Threshold.

This script reproduces the tradeoff curve between detection probability
and communication rate threshold from the paper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.system_model import ISACSystem
from src.ao_solver import AOSolver
from src.fairness import FairnessType


def main():
    """Main function to reproduce Figure 6."""
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
    
    # Rate threshold values (Γc) from paper
    Gamma_c_values = np.linspace(0.1, 5.0, 15)
    
    # Store results for different fairness criteria
    results_maxmin = []
    results_sum = []
    
    print("Running simulations for Figure 6...")
    print("=" * 50)
    
    for Gamma_c in Gamma_c_values:
        print(f"Γc = {Gamma_c:.2f} bps/Hz")
        
        # Max-min fairness
        solver_maxmin = AOSolver(
            system, 
            qos_type='detection', 
            fairness='maxmin', 
            max_iter=20,
            tol=1e-4
        )
        result_maxmin = solver_maxmin.solve(Gamma_c=Gamma_c)
        
        if result_maxmin.detection_probs is not None:
            # Use minimum detection probability (max-min fairness)
            results_maxmin.append(np.min(result_maxmin.detection_probs))
        else:
            results_maxmin.append(0)
        
        # Sum fairness (comprehensiveness)
        solver_sum = AOSolver(
            system,
            qos_type='detection',
            fairness='sum',
            max_iter=20,
            tol=1e-4
        )
        result_sum = solver_sum.solve(Gamma_c=Gamma_c)
        
        if result_sum.detection_probs is not None:
            # Use sum of detection probabilities
            results_sum.append(np.sum(result_sum.detection_probs))
        else:
            results_sum.append(0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Min detection probability vs rate threshold (max-min fairness)
    ax1.plot(Gamma_c_values, results_maxmin, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    ax1.set_ylabel('Min Detection Probability', fontsize=12)
    ax1.set_title('Max-Min Fairness: Detection vs Rate Tradeoff', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Sum detection probability vs rate threshold
    ax2.plot(Gamma_c_values, results_sum, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Rate Threshold Γc (bps/Hz)', fontsize=12)
    ax2.set_ylabel('Sum Detection Probability', fontsize=12)
    ax2.set_title('Comprehensiveness: Detection vs Rate Tradeoff', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig6_detection_vs_rate_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFigure 6 saved as 'fig6_detection_vs_rate_tradeoff.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Γc range: [{Gamma_c_values[0]:.2f}, {Gamma_c_values[-1]:.2f}] bps/Hz")
    print(f"Min P_D range (max-min): [{min(results_maxmin):.4f}, {max(results_maxmin):.4f}]")
    print(f"Sum P_D range: [{min(results_sum):.4f}, {max(results_sum):.4f}]")


if __name__ == "__main__":
    main()
