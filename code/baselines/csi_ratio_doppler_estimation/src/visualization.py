"""
Visualization Utilities for CSI-Ratio Doppler Estimation.

Provides plotting functions for:
- CSI-ratio samples in complex plane (with fitted circle)
- Phase evolution and linear regression
- Difference function Δ_Σ(n)
- Comparison of three algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from typing import Dict, Optional, Tuple


def plot_csi_ratio_complex(
    R: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    title: str = "CSI-Ratio in Complex Plane",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot CSI-ratio samples in the complex plane with optional fitted circle.

    Parameters
    ----------
    R : np.ndarray
        Complex CSI-ratio samples, shape (N,).
    center : tuple, optional
        (A, B) center of fitted circle.
    radius : float, optional
        Radius of fitted circle.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates new figure if None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot samples
    ax.scatter(np.real(R), np.imag(R), c=np.arange(len(R)),
               cmap='viridis', s=20, alpha=0.7, zorder=5)

    # Plot fitted circle
    if center is not None and radius is not None:
        circle = CirclePatch(center, radius, fill=False,
                             edgecolor='red', linewidth=2,
                             linestyle='--', label='Fitted circle')
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'rx', markersize=12,
                markeredgewidth=3, label=f'Center ({center[0]:.3f}, {center[1]:.3f})')

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_phase_evolution(
    t: np.ndarray,
    theta: np.ndarray,
    beta_0: float,
    beta_1: float,
    weights: Optional[np.ndarray] = None,
    title: str = "Phase Evolution and Linear Fit",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot phase evolution with weighted linear regression fit.

    Parameters
    ----------
    t : np.ndarray
        Time samples (seconds).
    theta : np.ndarray
        Phase angles (radians), unwrapped.
    beta_0 : float
        Intercept of linear fit.
    beta_1 : float
        Slope of linear fit (rad/s).
    weights : np.ndarray, optional
        Weights for each sample (shown as marker sizes).
    title : str
        Plot title.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Marker sizes proportional to weights
    if weights is not None:
        sizes = 20 + 100 * weights / np.max(weights)
    else:
        sizes = 20

    # Plot samples
    ax.scatter(t, theta, s=sizes, c='blue', alpha=0.5, label='Phase samples')

    # Plot linear fit
    t_fit = np.linspace(t[0], t[-1], 100)
    theta_fit = beta_0 + beta_1 * t_fit
    ax.plot(t_fit, theta_fit, 'r-', linewidth=2,
            label=f'Fit: θ = {beta_0:.3f} + {beta_1:.1f}·t')

    f_D = beta_1 / (2 * np.pi)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title(f'{title}\n(f_D = {f_D:.2f} Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_difference_function(
    delta_sigma: np.ndarray,
    n_star: int,
    T_s: float,
    title: str = "Difference Function Δ_Σ(n)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the difference function Δ_Σ(n) with the detected minimum.

    Parameters
    ----------
    delta_sigma : np.ndarray
        Difference values for each lag.
    n_star : int
        Optimal lag (1-indexed).
    T_s : float
        Sampling interval (seconds).
    title : str
        Plot title.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    lags = np.arange(1, len(delta_sigma) + 1)
    times = lags * T_s * 1000  # Convert to ms

    ax.plot(times, delta_sigma, 'b-', linewidth=1)
    ax.axvline(x=n_star * T_s * 1000, color='r', linestyle='--',
               label=f'n* = {n_star} ({n_star * T_s * 1000:.2f} ms)')
    ax.plot(n_star * T_s * 1000, delta_sigma[n_star - 1], 'ro',
            markersize=10, zorder=5)

    f_D = 1.0 / (n_star * T_s)
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Δ_Σ(n)')
    ax.set_title(f'{title}\n(f_D = {f_D:.2f} Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_comparison(
    results: Dict[str, Dict],
    true_f_D: float,
    title: str = "Algorithm Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare Doppler estimates from all three algorithms.

    Parameters
    ----------
    results : dict
        Dictionary with keys 'mobius', 'periodicity', 'difference',
        each containing a result dict from the respective estimator.
    true_f_D : float
        True Doppler frequency for reference.
    title : str
        Plot title.
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    algorithms = ['Mobius (Alg 1)', 'Periodicity (Alg 2)', 'Difference (Alg 3)']
    keys = ['mobius', 'periodicity', 'difference']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    estimates = []
    for key in keys:
        if key in results:
            estimates.append(abs(results[key]['f_D']))
        else:
            estimates.append(0)

    x = np.arange(len(algorithms))
    bars = ax.bar(x, estimates, color=colors, alpha=0.7, width=0.6)

    # Add true value line
    ax.axhline(y=abs(true_f_D), color='red', linestyle='--', linewidth=2,
               label=f'True |f_D| = {abs(true_f_D):.1f} Hz')

    # Add value labels on bars
    for bar, est in zip(bars, estimates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{est:.1f} Hz', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylabel('|f_D| (Hz)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    return ax


def plot_all_results(
    R: np.ndarray,
    t: np.ndarray,
    mobius_result: Dict,
    periodicity_result: Dict,
    difference_result: Dict,
    true_f_D: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate comprehensive visualization of all algorithm results.

    Parameters
    ----------
    R : np.ndarray
        CSI-ratio samples.
    t : np.ndarray
        Time samples.
    mobius_result, periodicity_result, difference_result : dict
        Results from each algorithm.
    true_f_D : float
        True Doppler frequency.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Complex plane with circle
    ax1 = fig.add_subplot(2, 2, 1)
    plot_csi_ratio_complex(
        R,
        center=(mobius_result['center_A'], mobius_result['center_B']),
        radius=mobius_result['radius'],
        title='CSI-Ratio in Complex Plane',
        ax=ax1,
    )

    # 2. Phase evolution
    ax2 = fig.add_subplot(2, 2, 2)
    R_s = R - (mobius_result['center_A'] + 1j * mobius_result['center_B'])
    theta = np.unwrap(np.angle(R_s))
    weights = np.abs(R_s)
    plot_phase_evolution(
        t, theta,
        mobius_result['beta_0'], mobius_result['beta_1'],
        weights=weights,
        title='Phase Evolution (Mobius)',
        ax=ax2,
    )

    # 3. Difference function
    ax3 = fig.add_subplot(2, 2, 3)
    plot_difference_function(
        difference_result['delta_sigma'],
        difference_result['n_star'],
        t[1] - t[0] if len(t) > 1 else 0.0005,
        title='Difference Function',
        ax=ax3,
    )

    # 4. Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    plot_comparison(
        {'mobius': mobius_result,
         'periodicity': periodicity_result,
         'difference': difference_result},
        true_f_D,
        title='Algorithm Comparison',
        ax=ax4,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
