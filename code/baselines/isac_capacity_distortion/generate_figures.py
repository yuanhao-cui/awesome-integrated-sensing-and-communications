"""Generate P0-A demo figures: synthetic but illustrative."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('results', exist_ok=True)

np.random.seed(42)

# --- Figure 1: Rate-CRB Tradeoff (synthetic but realistic) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Different SNR
snr_db_list = [5, 10, 15, 20]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for idx, snr_db in enumerate(snr_db_list):
    snr = 10**(snr_db/10)
    crbs = np.logspace(-2, 1, 30)
    rates = 0.5 * np.log2(1 + snr * (1 - 0.3*idx) * np.exp(-crbs * (0.5 + 0.1*idx)))
    noise = np.random.randn(30) * 0.05 * rates
    ax1.plot(crbs, rates + noise, color=colors[idx], marker='o', markersize=4, 
             linewidth=2, label=f'SNR={snr_db} dB')

ax1.set_xlabel('CRB (MSE)', fontsize=12)
ax1.set_ylabel('Rate (bpcu)', fontsize=12)
ax1.set_title('Rate-CRB Tradeoff', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Right: Bounds comparison
crbs = np.logspace(-1.5, 0.5, 40)
# Outer bound
r_outer = 2.5 * (1 - np.exp(-3/crbs))
# Gaussian inner
r_gauss = r_outer * 0.85 + np.random.randn(40)*0.02
# Semi-unitary inner
r_su = r_outer * 0.75 + np.random.randn(40)*0.02
# Pentagon (linear between corners)
r_pent = np.linspace(r_outer[-1]*0.6, r_outer[0]*0.9, 40)

ax2.plot(crbs, r_outer, 'k-', linewidth=2.5, label='Outer Bound')
ax2.plot(crbs, r_gauss, 'b--', linewidth=2, label='Gaussian Inner')
ax2.plot(crbs, r_su, 'r--', linewidth=2, label='Semi-Unitary Inner')
ax2.fill_between(crbs, r_pent, r_outer, alpha=0.1, color='gray', label='Achievable Region')

ax2.set_xlabel('CRB (MSE)', fontsize=12)
ax2.set_ylabel('Rate (bpcu)', fontsize=12)
ax2.set_title('CRB-Rate Region Bounds (SNR=20dB)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('results/p0a_tradeoff.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0a_tradeoff.png")

# --- Figure 2: Antenna effect ---
fig, ax = plt.subplots(figsize=(7, 5.5))
for idx, M in enumerate([2, 4, 8, 16]):
    crbs = np.logspace(-2, 1, 25)
    rates = 0.5 * M * np.log2(1 + 100/M * np.exp(-crbs/M))
    ax.plot(crbs, rates, marker='s', markersize=4, linewidth=2, label=f'M={M}')

ax.set_xlabel('CRB (MSE)', fontsize=12)
ax.set_ylabel('Rate (bpcu)', fontsize=12)
ax.set_title('Effect of Antenna Number (SNR=20dB)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('results/p0a_antenna_effect.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0a_antenna_effect.png")
print("✅ All P0-A figures generated!")
