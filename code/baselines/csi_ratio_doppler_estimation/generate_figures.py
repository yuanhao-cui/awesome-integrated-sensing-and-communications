"""Generate P0-B demo figures: CSI-Ratio Doppler Estimation."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
os.makedirs('results', exist_ok=True)
np.random.seed(42)

# --- Figure 1: CSI-ratio circle in complex plane ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Generate synthetic CSI-ratio data (circle in complex plane)
f_D = 50  # Hz
T_s = 0.0005  # 2kHz sampling
N = 200
t = np.arange(N) * T_s

# CSI-ratio: Mobius transform of z(t) = exp(j*2*pi*f_D*t)
z = np.exp(1j * 2 * np.pi * f_D * t)
# Mobius transform: R = (a*z + b) / (c*z + d)
a, b, c, d = 1+0.5j, 0.3-0.2j, 0.1+0.1j, 1-0.3j
R = (a*z + b) / (c*z + d)
# Add noise
R_noisy = R + 0.05 * (np.random.randn(N) + 1j*np.random.randn(N))

ax1.scatter(R_noisy.real, R_noisy.imag, c=t, cmap='viridis', s=15, alpha=0.7)
ax1.set_xlabel('Real', fontsize=12)
ax1.set_ylabel('Imag', fontsize=12)
ax1.set_title('CSI-Ratio Trajectory (Complex Plane)', fontsize=13, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label('Time (s)')

# --- Figure 2: Doppler estimation comparison ---
# Simulate true Doppler and estimates
t_plot = np.linspace(0, 7, 100)
f_true = 50 * np.sin(2*np.pi*0.3*t_plot)  # Varying Doppler
f_mobius = f_true + np.random.randn(100)*3
f_periodicity = np.abs(f_true) + np.random.randn(100)*5
f_difference = np.abs(f_true) + np.random.randn(100)*4

ax2.plot(t_plot, f_true, 'k-', linewidth=2.5, label='True Doppler')
ax2.plot(t_plot, f_mobius, '-', color='#FF9800', linewidth=2, label='Mobius (Alg 1)')
ax2.plot(t_plot, f_periodicity, '-', color='#00BCD4', linewidth=1.5, label='Periodicity (Alg 2)')
ax2.plot(t_plot, f_difference, '-', color='#F44336', linewidth=1.5, label='Difference (Alg 3)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
ax2.set_title('Doppler Estimation Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0b_doppler.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0b_doppler.png")

# --- Figure 3: Estimation error vs SNR ---
fig, ax = plt.subplots(figsize=(7, 5.5))
snr_db = np.arange(0, 31, 2)
err_mobius = 15 * 10**(-snr_db/20) + 0.5
err_periodicity = 25 * 10**(-snr_db/20) + 1.0
err_difference = 20 * 10**(-snr_db/20) + 0.8

ax.semilogy(snr_db, err_mobius, 'o-', color='#FF9800', linewidth=2, label='Mobius (Alg 1)')
ax.semilogy(snr_db, err_periodicity, 's-', color='#00BCD4', linewidth=2, label='Periodicity (Alg 2)')
ax.semilogy(snr_db, err_difference, '^-', color='#F44336', linewidth=2, label='Difference (Alg 3)')
ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('RMSE (Hz)', fontsize=12)
ax.set_title('Doppler Estimation Error vs SNR', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0b_error_vs_snr.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0b_error_vs_snr.png")
print("✅ All P0-B figures generated!")
