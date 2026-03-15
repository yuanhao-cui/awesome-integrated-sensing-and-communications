#!/usr/bin/env python3
"""Simple demo of ISAC Capacity-Distortion Tradeoff."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from system_model import (
    GaussianISACChannel,
    compute_rate,
    compute_crb,
    angle_to_hfunc,
    compute_phi_angle,
)
from optimization import (
    optimize_sensing_rx,
    optimize_comm_rx,
)
from bounds import compute_corner_points

def main():
    print("=" * 60)
    print("ISAC Capacity-Distortion Tradeoff - Simple Demo")
    print("=" * 60)

    # Setup parameters
    M, Nc, Ns, T = 4, 2, 4, 5
    P_T = 1.0
    sigma_c2 = 0.1
    sigma_s2 = 0.1

    # Random channel
    np.random.seed(42)
    Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

    print(f"\nSystem Parameters:")
    print(f"  M = {M} (Tx antennas)")
    print(f"  Nc = {Nc} (Comm Rx antennas)")
    print(f"  Ns = {Ns} (Sensing Rx antennas)")
    print(f"  T = {T} (Coherent interval)")
    print(f"  P_T = {P_T}")

    # Compute corner points
    print("\nComputing corner points...")
    corners = compute_corner_points(
        Hc, None, None, T, sigma_c2, sigma_s2, P_T, M, None, Nc
    )

    print(f"\nCorner Points:")
    print(f"  P_sc = (e_min={corners['e_min']:.6f}, R_sc={corners['R_sc']:.4f})")
    print(f"  P_cs = (e_cs={corners['e_cs']:.6f}, R_max={corners['R_max']:.4f})")

    # Test sensing-optimal
    Rx_sense = optimize_sensing_rx(P_T, M)
    R_sc = compute_rate(Rx_sense, Hc, sigma_c2)
    print(f"\nSensing-optimal covariance:")
    print(f"  Rate: {R_sc:.4f}")

    # Test comm-optimal
    Rx_comm = optimize_comm_rx(P_T, M, Hc)
    R_max = compute_rate(Rx_comm, Hc, sigma_c2)
    print(f"\nComm-optimal covariance:")
    print(f"  Rate: {R_max:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
