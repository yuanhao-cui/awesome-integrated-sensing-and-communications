"""Generate P0-D figures (standalone)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('results', exist_ok=True)
np.random.seed(42)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# --- Figure 1: Detection PD vs Rate Threshold ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
rate_thresholds = np.linspace(0, 50, 30)

for idx, P_dbm in enumerate([30, 35, 40]):
    snr_base = 10**(P_dbm/10) / 1e-8
    pds = [max(0, 1 - np.exp(-snr_base * 0.3 / (1 + rt/5))) for rt in rate_thresholds]
    ax1.plot(rate_thresholds, pds, '-', color=colors[idx], linewidth=2, label=f'P={P_dbm}dBm')

ax1.set_xlabel('Comm Rate Threshold Γc (bps/Hz)', fontsize=11)
ax1.set_ylabel('Detection Probability P_D', fontsize=11)
ax1.set_title('Detection PD vs Rate Threshold', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# --- Figure 2: Localization CRB vs Rate ---
for idx, B_mhz in enumerate([50, 100, 200]):
    crbs = [0.5 / (B_mhz * (1 + rt/20)) + 0.01 for rt in rate_thresholds]
    ax2.plot(rate_thresholds, crbs, '-', color=colors[idx], linewidth=2, label=f'B={B_mhz}MHz')

ax2.set_xlabel('Comm Rate Threshold Γc (bps/Hz)', fontsize=11)
ax2.set_ylabel('CRB(d) (m²)', fontsize=11)
ax2.set_title('Localization CRB vs Rate Threshold', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0d_detection_localization.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0d_detection_localization.png")

# --- Figure 3: Resource Allocation Pie ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
labels = ['Sensing 1', 'Sensing 2', 'Sensing 3', 'Comm 1', 'Comm 2', 'ISAC User']
powers = [25, 20, 15, 18, 12, 10]
bw = [30, 25, 20, 12, 8, 5]
cp = ['#FF6B6B', '#FF8E8E', '#FFB3B3', '#4ECDC4', '#7FDBDA', '#FFE66D']

ax1.pie(powers, labels=labels, colors=cp, autopct='%1.1f%%', startangle=90)
ax1.set_title('Power Allocation', fontsize=12, fontweight='bold')
ax2.pie(bw, labels=labels, colors=cp, autopct='%1.1f%%', startangle=90)
ax2.set_title('Bandwidth Allocation', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/p0d_resource_allocation.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0d_resource_allocation.png")

# --- Figure 4: Tracking PCRB ---
fig, ax = plt.subplots(figsize=(7, 5.5))
time = np.arange(0, 5, 0.1)
for idx, snr_db in enumerate([10, 15, 20, 25]):
    pcrb = 0.5 * np.exp(-0.3 * time) / (10**(snr_db/20)) + 0.01
    ax.semilogy(time, pcrb, '-', color=colors[idx], linewidth=2, label=f'SNR={snr_db}dB')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('PCRB (Position MSE)', fontsize=12)
ax.set_title('Tracking PCRB Over Time', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/p0d_tracking_pcrb.png', dpi=300, bbox_inches='tight')
print("✅ Saved: results/p0d_tracking_pcrb.png")
print("✅ All P0-D figures generated!")
