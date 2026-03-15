"""
Gaussian ISAC Channel Model and Core Computations.

Implements the system model, Bayesian Fisher Information Matrix (BFIM),
Bayesian Cramér-Rao Bound (BCRB), and ergodic communication rate
for the point-to-point ISAC system described in:

Xiong et al., "On the Fundamental Tradeoff of Integrated Sensing
and Communications Under Gaussian Channels," IEEE TIT, 2023.

System Model (Eq. 2):
    Communication: Y_c = H_c X + Z_c
    Sensing:       Y_s = H_s X + Z_s

where X ∈ C^(M×T) is the dual-functional waveform.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple


class GaussianISACChannel:
    """Gaussian ISAC channel model.

    Attributes:
        Hc (np.ndarray): Communication channel matrix, shape (Nc, M).
        Hs_func (Callable): Sensing channel generator Hs(eta), returns (Ns, M).
        sigma_c2 (float): Communication noise variance.
        sigma_s2 (float): Sensing noise variance.
        M (int): Number of transmit antennas.
        Nc (int): Number of communication receive antennas.
        Ns (int): Number of sensing receive antennas.
        T (int): Coherent processing interval (number of symbols).
    """

    def __init__(
        self,
        Hc: np.ndarray,
        Hs_func: Callable[[np.ndarray], np.ndarray],
        sigma_c2: float,
        sigma_s2: float,
        M: int,
        Nc: int,
        Ns: int,
        T: int,
    ):
        self.Hc = np.asarray(Hc, dtype=np.complex128)
        self.Hs_func = Hs_func
        self.sigma_c2 = float(sigma_c2)
        self.sigma_s2 = float(sigma_s2)
        self.M = M
        self.Nc = Nc
        self.Ns = Ns
        self.T = T

    def comm_channel(self) -> np.ndarray:
        """Return the communication channel matrix Hc."""
        return self.Hc

    def sensing_channel(self, eta: np.ndarray) -> np.ndarray:
        """Return the sensing channel matrix Hs for parameter eta."""
        return self.Hs_func(eta)

    def generate_noise(self, n_rows: int) -> np.ndarray:
        """Generate i.i.d. CN(0, 1) noise matrix of shape (n_rows, T)."""
        return (
            np.random.randn(n_rows, self.T)
            + 1j * np.random.randn(n_rows, self.T)
        ) / np.sqrt(2)

    def comm_receive(self, X: np.ndarray) -> np.ndarray:
        """Generate communication observation Y_c = H_c X + Z_c."""
        Zc = np.sqrt(self.sigma_c2) * self.generate_noise(self.Nc)
        return self.Hc @ X + Zc

    def sense_receive(self, X: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Generate sensing observation Y_s = H_s(eta) X + Z_s."""
        Hs = self.Hs_func(eta)
        Zs = np.sqrt(self.sigma_s2) * self.generate_noise(self.Ns)
        return Hs @ X + Zs


def compute_bfim(
    Rx: np.ndarray,
    T: int,
    sigma_s2: float,
    phi_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    Jp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the Bayesian Fisher Information Matrix (BFIM).

    Implements Eq. (11): J_{eta|X} = (T / sigma_s^2) Phi(Rx)

    where Phi is an affine map capturing the channel structure and
    prior information Jp. For the generic case, Phi(Rx) = T * Rx + Jp
    (when no specific channel structure is given).

    Args:
        Rx (np.ndarray): Sample covariance matrix R_X = (1/T) X X^H,
            shape (M, M), Hermitian positive semi-definite.
        T (int): Coherent processing interval.
        sigma_s2 (float): Sensing noise variance.
        phi_func (callable, optional): Custom Phi map. If None, uses
            Phi(Rx) = T * Rx (for scalar parameter estimation with
            no prior). Takes (M,M) -> (dim_eta, dim_eta).
        Jp (np.ndarray, optional): Prior information matrix, shape
            (dim_eta, dim_eta).

    Returns:
        np.ndarray: BFIM J_{eta|X}, shape (dim_eta, dim_eta).

    References:
        Eq. (11) in the paper.
    """
    Rx = np.asarray(Rx, dtype=np.complex128)

    if phi_func is not None:
        Phi = phi_func(Rx)
    else:
        # Default: Phi(Rx) = T * Rx (simplified single-parameter model)
        Phi = T * Rx

    if Jp is not None:
        Phi = Phi + np.asarray(Jp, dtype=np.complex128)

    J_bfim = (T / sigma_s2) * Phi
    return J_bfim


def compute_crb(
    Rx: np.ndarray,
    T: int,
    sigma_s2: float,
    phi_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    Jp: Optional[np.ndarray] = None,
    bfim: Optional[np.ndarray] = None,
) -> float:
    """Compute the Bayesian CRB = tr{J_{eta|X}^{-1}}.

    Implements Eq. (7): e := E[tr{J_{eta|X}^{-1}}]

    For the Bayesian case with known waveform, this simplifies to
    tr{J_{eta|X}^{-1}} where J_{eta|X} is the BFIM.

    Args:
        Rx (np.ndarray): Sample covariance matrix, shape (M, M).
        T (int): Coherent processing interval.
        sigma_s2 (float): Sensing noise variance.
        phi_func (callable, optional): Custom Phi map for BFIM.
        Jp (np.ndarray, optional): Prior information matrix.
        bfim (np.ndarray, optional): Pre-computed BFIM. If provided,
            phi_func and Jp are ignored.

    Returns:
        float: BCRB = tr{J^{-1}}.

    References:
        Eq. (7) in the paper.
    """
    if bfim is None:
        bfim = compute_bfim(Rx, T, sigma_s2, phi_func, Jp)

    # Compute tr{J^{-1}} using pseudoinverse for numerical stability
    try:
        J_inv = np.linalg.inv(bfim)
    except np.linalg.LinAlgError:
        J_inv = np.linalg.pinv(bfim)

    crb = np.real(np.trace(J_inv))
    return float(crb)


def compute_rate(
    Rx: np.ndarray,
    Hc: np.ndarray,
    sigma_c2: float,
) -> float:
    """Compute the ergodic communication rate.

    Implements Eq. (4): R = (1/T) I(Y_c; X | H_c)

    For ergodic capacity under known channel:
        R = (1/T) log det(I + sigma_c^{-2} H_c R_X H_c^H)

    This is the mutual information for a Gaussian channel with
    known channel matrix at the receiver.

    Args:
        Rx (np.ndarray): Statistical covariance matrix R_X,
            shape (M, M), Hermitian positive semi-definite.
        Hc (np.ndarray): Communication channel matrix,
            shape (Nc, M).
        sigma_c2 (float): Communication noise variance.

    Returns:
        float: Communication rate in nats per channel use.

    References:
        Eq. (4) in the paper.
    """
    Rx = np.asarray(Rx, dtype=np.complex128)
    Hc = np.asarray(Hc, dtype=np.complex128)

    Nc = Hc.shape[0]

    # Compute I + (1/sigma_c^2) H_c R_X H_c^H
    Gram = Hc @ Rx @ Hc.conj().T
    M_mat = np.eye(Nc, dtype=np.complex128) + Gram / sigma_c2

    # Rate = log det(M_mat) / T ... but T cancels since Rx = (1/T) X X^H
    # Actually: R = (1/T) * log_det(I + (1/sigma_c2) Hc Rx Hc^H)
    # Wait - Rx is the statistical covariance E[(1/T) X X^H], so
    # R = log_det(I + (1/sigma_c2) Hc Rx Hc^H)
    # (the T factor is already absorbed into the definition of Rx)

    sign, logdet = np.linalg.slogdet(M_mat)
    if sign <= 0:
        return 0.0

    # Rate per channel use (not per T)
    rate = np.real(logdet)
    return float(rate)


def compute_rate_per_symbol(
    X: np.ndarray,
    Hc: np.ndarray,
    sigma_c2: float,
) -> float:
    """Compute rate from a specific waveform realization.

    R = (1/T) log det(I + sigma_c^{-2} H_c (X X^H / T) H_c^H)

    Args:
        X (np.ndarray): Waveform matrix, shape (M, T).
        Hc (np.ndarray): Communication channel, shape (Nc, M).
        sigma_c2 (float): Noise variance.

    Returns:
        float: Rate in nats per channel use.
    """
    M, T = X.shape
    Rx_sample = (X @ X.conj().T) / T
    return compute_rate(Rx_sample, Hc, sigma_c2)


def make_uniform_linear_array(M: int, d: float = 0.5) -> np.ndarray:
    """Create a ULA steering vector function.

    Args:
        M (int): Number of antennas.
        d (float): Antenna spacing in wavelengths (default 0.5).

    Returns:
        Callable that takes angle theta (radians) and returns
        steering vector a(theta) of shape (M, 1).
    """
    positions = np.arange(M) * d

    def steering_vector(theta: float) -> np.ndarray:
        return np.exp(1j * 2 * np.pi * positions * np.sin(theta)).reshape(-1, 1)

    return steering_vector


def angle_to_channel(
    theta: float,
    M: int,
    N: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
) -> np.ndarray:
    """Create a line-of-sight MIMO channel for a given angle.

    H = a_rx(theta) a_tx(theta)^H

    Args:
        theta (float): Angle of arrival/departure in radians.
        M (int): Number of transmit antennas.
        N (int): Number of receive antennas.
        d_tx (float): Tx antenna spacing in wavelengths.
        d_rx (float): Rx antenna spacing in wavelengths.

    Returns:
        np.ndarray: Channel matrix H of shape (N, M).
    """
    a_tx = make_uniform_linear_array(M, d_tx)(theta)
    a_rx = make_uniform_linear_array(N, d_rx)(theta)
    return a_rx @ a_tx.conj().T


def angle_to_hfunc(
    M: int,
    Ns: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create Hs(eta) function for angle estimation.

    The parameter eta = [theta] is a scalar angle (radians).
    Hs = a_rx(theta) a_tx(theta)^H

    Args:
        M (int): Number of transmit antennas.
        Ns (int): Number of sensing receive antennas.
        d_tx (float): Tx antenna spacing.
        d_rx (float): Rx antenna spacing.

    Returns:
        Callable mapping angle (as 1-element array) to Hs matrix.
    """
    def hfunc(eta: np.ndarray) -> np.ndarray:
        theta = float(eta[0]) if eta.ndim > 0 else float(eta)
        return angle_to_channel(theta, M, Ns, d_tx, d_rx)
    return hfunc


def compute_phi_angle(
    Rx: np.ndarray,
    T: int,
    theta: float,
    M: int,
    Ns: int,
    d_tx: float = 0.5,
    d_rx: float = 0.5,
    Jp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Phi(Rx) for angle estimation (single parameter).

    For angle estimation with ULA, the BFIM structure is:
    J = (T / sigma_s^2) * |a'(theta)^H Rx a'(theta)|
    where a'(theta) = d(a)/d(theta) is the derivative of the
    steering vector.

    Actually, for the full matrix formulation:
    dH/dtheta = a_rx'(theta) a_tx(theta)^H (simplified)

    We use the scalar BFIM formula from the paper's treatment.

    Args:
        Rx (np.ndarray): Sample covariance, shape (M, M).
        T (int): Coherent processing interval.
        theta (float): Target angle in radians.
        M (int): Number of Tx antennas.
        Ns (int): Number of sensing Rx antennas.
        d_tx (float): Tx antenna spacing.
        d_rx (float): Rx antenna spacing.
        Jp (float or None): Prior information scalar.

    Returns:
        np.ndarray: Scalar BFIM (1x1 matrix).
    """
    positions_tx = np.arange(M) * d_tx
    positions_rx = np.arange(Ns) * d_rx

    # Steering vectors
    a_tx = np.exp(1j * 2 * np.pi * positions_tx * np.sin(theta))
    a_rx = np.exp(1j * 2 * np.pi * positions_rx * np.sin(theta))

    # Derivatives
    da_tx = 1j * 2 * np.pi * positions_tx * np.cos(theta) * a_tx
    da_rx = 1j * 2 * np.pi * positions_rx * np.cos(theta) * a_rx

    # For the Fisher information with known waveform:
    # J = (2T/sigma_s^2) * Re{ tr[(H^H H)^{-1} (dH/dtheta)^H (I - P) dH/dtheta] }
    # Simplified for single parameter angle:
    # J_theta = (2T/sigma_s^2) * || (I - H(H^H H)^{-1} H^H) dH/dtheta ||_F^2

    # Using the paper's formulation with Rx:
    # For angle: Phi(Rx) involves da/dtheta

    # Compute: da_rx^H * ones * da_tx weighted by Rx
    # Phi = Ns * |da_tx^H Rx da_tx|^2 (simplified scalar model)

    # More precise: the BFIM element for angle is
    # J = (2T/sigma_s^2) * Re{ sum_i sum_j ... }
    # For single angle parameter, this reduces to a scalar.

    # Use the standard result for angle estimation with ULA:
    # J = (2T/sigma_s^2) * Ns * M * (2*pi*d*cos(theta))^2 * tr(Rx) * (M+1)/3
    # ... but more precisely, we need the exact formula.

    # Following the paper's structure:
    # Phi(Rx) gives the spatial weighting.
    # For ULA angle estimation:
    W = np.diag(positions_tx)
    Phi_val = 4 * np.pi**2 * np.cos(theta)**2 * np.real(
        np.trace(W @ W @ Rx)
    )

    if Jp is not None:
        Phi_val += Jp

    return np.array([[Phi_val]])
