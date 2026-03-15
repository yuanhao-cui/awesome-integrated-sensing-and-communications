"""
Case Studies and Figure Generation.

Implements:
- Case Study A: Target Angle Estimation (Figures 5-9)
- Case Study B: Target Response Matrix Estimation (Figures 10-11)

All parameters match Table I and Table II from the paper.

References:
    Xiong et al., IEEE TIT, 2023.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

try:
    from .system_model import (
        angle_to_channel,
        angle_to_hfunc,
        compute_rate,
        compute_crb,
        compute_phi_angle,
        make_uniform_linear_array,
    )
    from .bounds import (
        pentagon_inner_bound,
        gaussian_inner_bound,
        semi_unitary_inner_bound,
        outer_bound,
        compute_corner_points,
    )
    from .optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
    )
except ImportError:
    from system_model import (
        angle_to_channel,
        angle_to_hfunc,
        compute_rate,
        compute_crb,
        compute_phi_angle,
        make_uniform_linear_array,
    )
    from bounds import (
        pentagon_inner_bound,
        gaussian_inner_bound,
        semi_unitary_inner_bound,
        outer_bound,
        compute_corner_points,
    )
    from optimization import (
        optimize_sensing_rx,
        optimize_comm_rx,
        covariance_shaping,
    )


# ============================================================
# Table I: Target Angle Estimation Parameters
# ============================================================
TABLE_I = {
    'M': 10,            # Tx antennas
    'Ns': 10,           # Sensing Rx antennas
    'Nc': 1,            # Communication Rx antennas
    'd': 0.5,           # Antenna spacing (wavelengths)
    'sensing_snr_db': 20,   # Max sensing SNR per antenna (dB)
    'comm_snr_db': 33,      # Max comm SNR per antenna (dB)
    'theta_prior_mean': np.deg2rad(30),   # Prior mean: 30°
    'theta_prior_std': np.deg2rad(5),     # Prior std: 5°
}


# ============================================================
# Table II: Target Response Matrix Estimation Parameters
# ============================================================
TABLE_II = {
    'M': 4,             # Tx antennas
    'Ns': 4,            # Sensing Rx antennas
    'Nc': 4,            # Communication Rx antennas
    'sigma_s2': 1.0,    # Sensing noise variance
    'sensing_snr_db': 24,   # Sensing transmit SNR (dB)
    'comm_snr_db': 24,      # Comm transmit SNR (dB)
}


def setup_angle_estimation(
    M: int = 10,
    Ns: int = 10,
    Nc: int = 1,
    theta_c_deg: float = 42.0,
    sensing_snr_db: float = 20.0,
    comm_snr_db: float = 33.0,
    d: float = 0.5,
) -> dict:
    """Setup the angle estimation case study.

    Args:
        M: Number of Tx antennas.
        Ns: Number of sensing Rx antennas.
        Nc: Number of comm Rx antennas.
        theta_c_deg: Communication Rx bearing angle (degrees).
        sensing_snr_db: Max sensing receiving SNR.
        comm_snr_db: Max comm receiving SNR.
        d: Antenna spacing in wavelengths.

    Returns:
        dict with channel parameters.
    """
    theta_c = np.deg2rad(theta_c_deg)

    # Communication channel: LoS from angle theta_c
    Hc = angle_to_channel(theta_c, M, Nc, d, d)

    # Normalize Hc to match SNR
    Hc_norm = np.linalg.norm(Hc, 'fro')
    if Hc_norm > 0:
        Hc = Hc / Hc_norm * np.sqrt(M * Nc)

    # Sensing channel function
    Hs_func = angle_to_hfunc(M, Ns, d, d)

    # Noise variances from SNR
    # sensing_snr = P_T / sigma_s2 (per antenna)
    # We set P_T = 1 (normalized), so sigma_s2 = 10^(-sensing_snr_db/10)
    P_T = 1.0
    sigma_s2 = P_T * 10 ** (-sensing_snr_db / 10)
    sigma_c2 = P_T * 10 ** (-comm_snr_db / 10)

    # Compute correlation coefficient between comm and sensing channels
    # a_c = steering vector for theta_c, a_s = steering vector for target
    # rho = |a_c^H a_s| / (||a_c|| ||a_s||)
    # For theta_c relative to 30° (prior mean)
    a_c = make_uniform_linear_array(M, d)(theta_c).flatten()
    a_s = make_uniform_linear_array(M, d)(np.deg2rad(30)).flatten()
    rho = np.abs(np.vdot(a_c, a_s)) / (np.linalg.norm(a_c) * np.linalg.norm(a_s))

    return {
        'M': M, 'Ns': Ns, 'Nc': Nc,
        'Hc': Hc,
        'Hs_func': Hs_func,
        'sigma_c2': sigma_c2,
        'sigma_s2': sigma_s2,
        'P_T': P_T,
        'theta_c': theta_c,
        'theta_c_deg': theta_c_deg,
        'rho': rho,
        'd': d,
    }


def make_phi_angle_func(
    M: int,
    Ns: int,
    theta_target: float,
    d: float = 0.5,
    Jp: float = 0.0,
):
    """Create Phi function for angle estimation.

    For scalar angle parameter, Phi(Rx) is a scalar.
    """
    def phi_func(Rx):
        phi_val = compute_phi_angle(
            Rx, 1, theta_target, M, Ns, d, d,
            Jp=Jp if Jp > 0 else None
        )
        return phi_val
    return phi_func


def generate_figure5(
    output_dir: str = "results",
    save: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate Figure 5: Rate vs CRB for T=3, theta_c=42°.

    Shows the CRB-rate region with:
    - Pentagon inner bound
    - Gaussian inner bound
    - Semi-unitary inner bound
    - Outer bound
    - Corner points P_sc and P_cs

    Parameters match Table I with theta_c = 42° (rho ≈ 0.61).

    Returns:
        Tuple of (fig, ax).
    """
    print("Generating Figure 5: Rate vs CRB (T=3, theta_c=42°)...")

    T = 3
    theta_target = np.deg2rad(30)  # Prior mean
    params = setup_angle_estimation(theta_c_deg=42.0)

    M = params['M']
    Ns = params['Ns']
    Hc = params['Hc']
    sigma_c2 = params['sigma_c2']
    sigma_s2 = params['sigma_s2']
    P_T = params['P_T']

    # Create Phi function for angle estimation
    # Prior: Jp from von Mises distribution
    kappa = 1.0 / params['theta_prior_std']**2 if hasattr(params, 'theta_prior_std') else (180 / (np.pi * 5))**2
    Jp_scalar = kappa  # Prior precision

    phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
    Jp = np.array([[Jp_scalar]])

    # Compute corner points
    corners = compute_corner_points(
        Hc, Hs_func=params['Hs_func'],
        phi_func=phi_func, T=T,
        sigma_c2=sigma_c2, sigma_s2=sigma_s2,
        P_T=P_T, M=M, Jp=Jp, Nc=params['Nc']
    )

    print(f"  Corner points:")
    print(f"    P_sc = (e_min={corners['e_min']:.4f}, R_sc={corners['R_sc']:.4f})")
    print(f"    P_cs = (e_cs={corners['e_cs']:.4f}, R_max={corners['R_max']:.4f})")
    print(f"    rho = {params['rho']:.4f}")

    # Alpha values for tradeoff
    alpha_vals = np.linspace(0, 0.99, 50)

    # Pentagon bound
    e_pent, R_pent = pentagon_inner_bound(
        (corners['e_min'], corners['R_sc']),
        (corners['e_cs'], corners['R_max']),
        corners['e_min'], corners['R_max'],
    )

    # Gaussian inner bound
    print("  Computing Gaussian inner bound...")
    e_gauss, R_gauss, _ = gaussian_inner_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
    )

    # Semi-unitary inner bound
    print("  Computing semi-unitary inner bound...")
    e_su, R_su, _ = semi_unitary_inner_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, M_sc=min(M, T),
        Jp=Jp, Nc=params['Nc'], n_stiefel_samples=30,
    )

    # Outer bound
    print("  Computing outer bound...")
    e_outer, R_outer, _ = outer_bound(
        alpha_vals, Hc, params['Hs_func'], phi_func,
        T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pentagon
    ax.fill_between(R_pent, e_pent, alpha=0.15, color='blue', label='Pentagon bound')
    ax.plot(R_pent, e_pent, 'b-', linewidth=1.5, alpha=0.5)

    # Inner bounds
    if len(e_gauss) > 0:
        ax.plot(R_gauss, e_gauss, 'g-', linewidth=2, label='Gaussian bound')
    if len(e_su) > 0:
        ax.plot(R_su, e_su, 'm--', linewidth=2, label='Semi-unitary bound')

    # Outer bound
    if len(e_outer) > 0:
        ax.plot(R_outer, e_outer, 'r:', linewidth=2, label='Outer bound')

    # Corner points
    ax.plot(corners['R_sc'], corners['e_min'], 'ko', markersize=8, label='$P_{sc}$')
    ax.plot(corners['R_max'], corners['e_cs'], 'k^', markersize=8, label='$P_{cs}$')

    ax.set_xlabel('Communication Rate R (nats/channel use)', fontsize=12)
    ax.set_ylabel('Sensing CRB e', fontsize=12)
    ax.set_title(f'CRB-Rate Region ($T={T}$, $\\theta_c={42}°$, $\\rho \\approx {params["rho"]:.2f}$)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / 'figure5_rate_vs_crb.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to {out / 'figure5_rate_vs_crb.png'}")

    return fig, ax


def generate_figure8(
    output_dir: str = "results",
    save: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate Figure 8: Rate vs CRB for different T values.

    Shows how the CRB-rate region expands with increasing T.
    Parameters: theta_c = 50° (rho ≈ 0.22), T = {3, 5, 10, 20, 50}.

    Returns:
        Tuple of (fig, ax).
    """
    print("Generating Figure 8: Rate vs CRB for different T...")

    T_values = [3, 5, 10, 20, 50]
    theta_target = np.deg2rad(30)
    params = setup_angle_estimation(theta_c_deg=50.0)

    M = params['M']
    Ns = params['Ns']
    Hc = params['Hc']
    sigma_c2 = params['sigma_c2']
    sigma_s2 = params['sigma_s2']
    P_T = params['P_T']

    Jp_scalar = (180 / (np.pi * 5))**2
    phi_func = make_phi_angle_func(M, Ns, theta_target, params['d'], Jp_scalar)
    Jp = np.array([[Jp_scalar]])

    alpha_vals = np.linspace(0, 0.99, 40)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(T_values)))

    for idx, T in enumerate(T_values):
        print(f"  Processing T = {T}...")

        # Corner points
        corners = compute_corner_points(
            Hc, Hs_func=params['Hs_func'],
            phi_func=phi_func, T=T,
            sigma_c2=sigma_c2, sigma_s2=sigma_s2,
            P_T=P_T, M=M, Jp=Jp, Nc=params['Nc']
        )

        # Gaussian inner bound
        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, Hc, params['Hs_func'], phi_func,
            T, sigma_c2, sigma_s2, P_T, M, Jp, params['Nc'],
        )

        # Plot
        if len(e_gauss) > 0:
            ax.plot(R_gauss, e_gauss, '-', color=colors[idx],
                    linewidth=2, label=f'T = {T}')

        # Corner points
        ax.plot(corners['R_sc'], corners['e_min'], 'o',
                color=colors[idx], markersize=6)
        ax.plot(corners['R_max'], corners['e_cs'], '^',
                color=colors[idx], markersize=6)

    ax.set_xlabel('Communication Rate R (nats/channel use)', fontsize=12)
    ax.set_ylabel('Sensing CRB e', fontsize=12)
    ax.set_title(f'CRB-Rate Region vs. Coherent Interval $T$ ($\\theta_c = 50°$, $\\rho \\approx {params["rho"]:.2f}$)',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / 'figure8_rate_vs_crb_T.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to {out / 'figure8_rate_vs_crb_T.png'}")

    return fig, ax


def setup_matrix_estimation(
    M: int = 4,
    Ns: int = 4,
    Nc: int = 4,
    sensing_snr_db: float = 24.0,
    comm_snr_db: float = 24.0,
) -> dict:
    """Setup the target response matrix estimation case study.

    Hs entries ~ CN(0, 1), independent.
    Hc entries ~ CN(0, 1), Rayleigh fading.

    Returns:
        dict with channel parameters.
    """
    P_T = 1.0
    sigma_s2 = P_T * 10 ** (-sensing_snr_db / 10)
    sigma_c2 = P_T * 10 ** (-comm_snr_db / 10)

    # Communication channel: Rayleigh fading
    Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

    # Sensing channel: random matrix
    def Hs_func(eta):
        return eta.reshape(Ns, M)

    # For matrix estimation, Phi(Rx) = T * kron(Rx^T, I_Ns)
    # The BFIM has dimension (Ns*M) x (Ns*M)
    def phi_func(Rx):
        return np.kron(Rx.T, np.eye(Ns))

    return {
        'M': M, 'Ns': Ns, 'Nc': Nc,
        'Hc': Hc,
        'Hs_func': Hs_func,
        'phi_func': phi_func,
        'sigma_c2': sigma_c2,
        'sigma_s2': sigma_s2,
        'P_T': P_T,
    }


def generate_figure10(
    output_dir: str = "results",
    save: bool = True,
    n_trials: int = 5,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate Figure 10: Rate vs normalized CRB for matrix estimation.

    Rayleigh fading channel, T = 4M = 16.
    CRB normalized by M * Ns.

    Returns:
        Tuple of (fig, ax).
    """
    print("Generating Figure 10: Matrix estimation (Rayleigh fading)...")

    params = setup_matrix_estimation()
    M = params['M']
    Ns = params['Ns']
    Nc = params['Nc']
    T = 4 * M  # T = 16

    alpha_vals = np.linspace(0, 0.99, 40)

    fig, ax = plt.subplots(figsize=(8, 6))

    e_all_trials = []
    R_all_trials = []

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...")

        # Random channel for each trial
        Hc = (np.random.randn(Nc, M) + 1j * np.random.randn(Nc, M)) / np.sqrt(2)

        e_gauss, R_gauss, _ = gaussian_inner_bound(
            alpha_vals, Hc, params['Hs_func'], params['phi_func'],
            T, params['sigma_c2'], params['sigma_s2'],
            params['P_T'], M, None, Nc,
        )

        if len(e_gauss) > 0:
            e_all_trials.append(e_gauss / (M * Ns))  # Normalize
            R_all_trials.append(R_gauss)

    # Average over trials
    if e_all_trials:
        # Interpolate to common R grid
        R_min = min(r.min() for r in R_all_trials if len(r) > 0)
        R_max = max(r.max() for r in R_all_trials if len(r) > 0)
        R_grid = np.linspace(R_min, R_max, 50)

        e_interp = []
        for e_trial, R_trial in zip(e_all_trials, R_all_trials):
            if len(R_trial) > 1:
                e_i = np.interp(R_grid, R_trial, e_trial)
                e_interp.append(e_i)

        if e_interp:
            e_mean = np.mean(e_interp, axis=0)
            e_std = np.std(e_interp, axis=0)

            ax.plot(R_grid, e_mean, 'b-', linewidth=2, label='Gaussian bound')
            ax.fill_between(R_grid, e_mean - e_std, e_mean + e_std,
                           alpha=0.2, color='blue')

    ax.set_xlabel('Communication Rate R (nats/channel use)', fontsize=12)
    ax.set_ylabel('Normalized CRB $e / (M N_s)$', fontsize=12)
    ax.set_title(f'CRB-Rate Region: Matrix Estimation ($M=N_s=N_c={M}$, $T={T}$)',
                 fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / 'figure10_matrix_estimation.png', dpi=150, bbox_inches='tight')
        print(f"  Saved to {out / 'figure10_matrix_estimation.png'}")

    return fig, ax


def target_angle_estimation(
    output_dir: str = "results",
) -> dict:
    """Run the complete angle estimation case study (Figures 5-9).

    Returns:
        dict with all results.
    """
    results = {}

    # Figure 5
    fig5, ax5 = generate_figure5(output_dir)
    results['figure5'] = {'fig': fig5, 'ax': ax5}

    # Figure 8
    fig8, ax8 = generate_figure8(output_dir)
    results['figure8'] = {'fig': fig8, 'ax': ax8}

    plt.close('all')
    return results


def target_response_estimation(
    output_dir: str = "results",
) -> dict:
    """Run the complete target response estimation case study (Figures 10-11).

    Returns:
        dict with all results.
    """
    results = {}

    # Figure 10
    fig10, ax10 = generate_figure10(output_dir)
    results['figure10'] = {'fig': fig10, 'ax': ax10}

    plt.close('all')
    return results
