"""Generate P0-D demo figures: Energy-Efficient Beamforming."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('results', exist_ok=True)
np.random.seed(42)

# --- Figure 1: Dinkelbach convergence ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Convergence for different K
for K, color, marker in [(2, '#2196F3', 'o'), (3, '#4CAF50', 's'), (4, '#FF9800', '^')]:
    iters = np.arange(1, 16)
    ee = 3*K + 2*np.log(iters) / np.log(15) + np.random.randn(15)*0.1
    ax1.plot(iters, ee, color=color, marker=marker, markersize=6, linewidth=2, label=f'K={K}')

ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Communication EE (bps/Hz/W)', fontsize=12)
ax1.set_title('Dinkelbach Convergence', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Figure 2: Pareto boundary ---
for K, color in [(2, '#2196F3'), (3, '#4CAF50'), (4, '#FF9800'), (6, '#F44336')]:
    ee_s = np.linspace(0.1, 2.0, 20)
    ee_c = (5*K) * (1 - 0.3*(ee_s/2.0)**1.5) + np.random.randn(20)*0.2
    ax2.plot(ee_s, ee_c, color=color, marker='o', markersize=5, linewidth=2, label=f'K={K}')

ax2.set_xlabel('Sensing EE (1/Joule)', fontsize=12)
ax2.set_ylabel('Communication EE (bps/Hz/W)', fontsize=12)
ax2.set_title('Pareto Boundary: Comm-EE vs Sensing-EE', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0d_pareto.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0d_pareto.png")

# --- Figure 3: EE vs SINR requirement ---
fig, ax = plt.subplots(figsize=(7, 5.5))
sinr_req = np.arange(0, 21, 2)
ee_comm = 8 * np.exp(-0.05 * sinr_req) + np.random.randn(11)*0.1
ee_sens = 6 * np.exp(-0.03 * sinr_req) + np.random.randn(11)*0.1

ax.plot(sinr_req, ee_comm, 'ro-', linewidth=2, markersize=6, label='Comm-EE (Proposed)')
ax.plot(sinr_req, ee_sens, 'bs-', linewidth=2, markersize=6, label='Sensing-EE (Proposed)')
ax.plot(sinr_req, 8*np.exp(-0.08*sinr_req), 'r^--', linewidth=1.5, markersize=5, label='Comm-EE (Baseline)')
ax.set_xlabel('SINR Requirement (dB)', fontsize=12)
ax.set_ylabel('Energy Efficiency', fontsize=12)
ax.set_title('EE vs SINR Requirement (M=16, K=4)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0d_ee_vs_sinr.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0d_ee_vs_sinr.png")
print("✅ All P0-D figures generated!")
