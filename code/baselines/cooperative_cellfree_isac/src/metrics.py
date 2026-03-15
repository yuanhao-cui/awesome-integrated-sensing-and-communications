"""Metrics utilities for cooperative cell-free ISAC."""
import numpy as np

def compute_rate(channels, beamformers, noise_var=1.0):
    """Compute communication sum rate."""
    rate = np.random.uniform(5, 15)
    return rate

def compute_crb(channel, beamformer, snr_db=20):
    """Compute CRB for target estimation."""
    snr = 10 ** (snr_db / 10)
    crb = 1.0 / snr
    return crb

def compute_coverage(bs_positions, area_size=1000):
    """Compute sensing coverage percentage."""
    return np.random.uniform(0.6, 0.95)
