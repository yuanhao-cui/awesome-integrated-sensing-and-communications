"""
Reproduce Figure 12: Tracking PCRB over Time.

This script reproduces the Posterior Cramér-Rao Bound performance for tracking
as a function of time steps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.system_model import ISACSystem
from src.ao_solver import AOSolver
from src.tracking_qos import TrackingQoS


def main():
    """Main function to reproduce Figure 12."""
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
    
    # Simulation parameters
    num_steps = 50
    time_steps = np.arange(num_steps)
    
    # Test different resource allocation schemes
    schemes = {
        'Optimal (AO)': None,  # Will be computed by solver
        'Equal Allocation': np.ones(system.total_objects) * system.params.P_total / system.total_objects,
        'Power-focused': np.concatenate([
            np.ones(system.params.Q) * 20.0,  # More power to sensing
            np.ones(system.params.K + system.params.L) * 5.0  # Less to comm/ISAC
        ]),
    }
    
    # Normalize power allocations to satisfy budget
    for name, p in schemes.items():
        if p is not None:
            schemes[name] = p * system.params.P_total / np.sum(p)
    
    # Get optimal allocation from AO solver
    solver = AOSolver(
        system,
        qos_type='tracking',
        fairness='maxmin',
        max_iter=20,
        tol=1e-4
    )
    
    result = solver.solve(Gamma_c=1.0)
    schemes['Optimal (AO)'] = result.p
    
    print("Running simulations for Figure 12...")
    print("=" * 50)
    
    # Store results
    results = {}
    
    for scheme_name, p_alloc in schemes.items():
        print(f"\nSimulating: {scheme_name}")
        
        # Create tracking QoS for this scheme
        tracking = TrackingQoS(system, dt=0.1, process_noise_std=0.5, measurement_noise_std=0.1)
        
        # Use bandwidth allocation from AO result or equal
        if scheme_name == 'Optimal (AO)':
            b_alloc = result.b
        else:
            b_alloc = np.ones(system.total_objects) * system.params.B_total / system.total_objects
        
        Q = system.params.Q
        p_sensing = p_alloc[:Q]
        b_sensing = b_alloc[:Q]
        
        # Simulate tracking
        pcrb_history, trace_history = tracking.simulate_tracking(
            p_sensing, b_sensing, num_steps=num_steps
        )
        
        results[scheme_name] = {
            'pcrb_history': pcrb_history,
            'trace_history': trace_history,
            'p_alloc': p_alloc,
            'b_alloc': b_alloc
        }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sum of PCRB traces over time
    for scheme_name, data in results.items():
        axes[0, 0].plot(time_steps, data['trace_history'], 'o-', linewidth=2, 
                        markersize=4, label=scheme_name)
    
    axes[0, 0].set_xlabel('Time Step', fontsize=12)
    axes[0, 0].set_ylabel('Sum of PCRB Traces', fontsize=12)
    axes[0, 0].set_title('Tracking Performance Over Time', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Position PCRB for each target (optimal scheme)
    optimal_data = results['Optimal (AO)']
    pcrb_history = optimal_data['pcrb_history']
    
    for q in range(system.params.Q):
        pos_trace = [np.trace(pcrb_history[t][q, :2, :2]) for t in range(num_steps)]
        axes[0, 1].plot(time_steps, pos_trace, 'o-', linewidth=2, 
                        markersize=4, label=f'Target {q}')
    
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].set_ylabel('Position PCRB Trace', fontsize=12)
    axes[0, 1].set_title('Position Tracking PCRB (Optimal)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Velocity PCRB for each target (optimal scheme)
    for q in range(system.params.Q):
        vel_trace = [np.trace(pcrb_history[t][q, 2:, 2:]) for t in range(num_steps)]
        axes[1, 0].plot(time_steps, vel_trace, 'o-', linewidth=2, 
                        markersize=4, label=f'Target {q}')
    
    axes[1, 0].set_xlabel('Time Step', fontsize=12)
    axes[1, 0].set_ylabel('Velocity PCRB Trace', fontsize=12)
    axes[1, 0].set_title('Velocity Tracking PCRB (Optimal)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Final PCRB comparison
    final_traces = {name: data['trace_history'][-1] for name, data in results.items()}
    
    scheme_names = list(final_traces.keys())
    final_values = [final_traces[name] for name in scheme_names]
    
    bars = axes[1, 1].bar(scheme_names, final_values, color=['blue', 'orange', 'green', 'red'])
    axes[1, 1].set_ylabel('Final Sum of PCRB Traces', fontsize=12)
    axes[1, 1].set_title('Final Tracking Performance Comparison', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fig12_tracking_pcrb_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFigure 12 saved as 'fig12_tracking_pcrb_over_time.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of time steps: {num_steps}")
    print(f"Number of sensing targets: {system.params.Q}")
    
    for scheme_name, data in results.items():
        initial_trace = data['trace_history'][0]
        final_trace = data['trace_history'][-1]
        improvement = (initial_trace - final_trace) / initial_trace * 100
        
        print(f"\n{scheme_name}:")
        print(f"  Initial trace: {initial_trace:.2e}")
        print(f"  Final trace: {final_trace:.2e}")
        print(f"  Improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
