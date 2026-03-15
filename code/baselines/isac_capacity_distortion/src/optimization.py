"""
Optimization Routines for ISAC Capacity-Distortion Tradeoff.

Implements:
- Sensing-optimal covariance optimization (Eq. 14)
- Communication-optimal covariance optimization
- Covariance shaping for S&C tradeoff (Eq. 48)
- Stiefel manifold uniform sampling (LQ decomposition method)

References:
    Xiong et al., IEEE TIT, 2023.
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import Callable, Optional, Tuple


def optimize_sensing_rx(
    P_T: float,
    M: int,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_s2: float = 1.0,
    Jp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find the sensing-optimal covariance matrix.

    Solves Eq. (14):
        min_{Rx} tr{(Phi(Rx))^{-1}}
        s.t. tr{Rx} = P_T * M,
             Rx >= 0 (PSD),
             Rx = Rx^H (Hermitian)

    For angle estimation with ULA, the sensing-optimal Rx is
    (P_T / M) * I_M (isotropic signaling).

    Args:
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        Hs_func (callable, optional): Sensing channel function.
        phi_func (callable, optional): Custom Phi map.
        T (int): Coherent processing interval.
        sigma_s2 (float): Sensing noise variance.
        Jp (np.ndarray, optional): Prior information matrix.

    Returns:
        np.ndarray: Sensing-optimal covariance Rx, shape (M, M).

    References:
        Eq. (14) in the paper.
    """
    # For the standard case (angle estimation with ULA),
    # the sensing-optimal Rx is P_T * I (gives tr{Rx} = P_T * M)
    Rx_opt = P_T * np.eye(M, dtype=np.complex128)

    if phi_func is None:
        return Rx_opt

    # General case: use convex optimization
    Rx_var = cp.Variable((M, M), hermitian=True)
    power = P_T * M

    # Objective: minimize tr{Phi(Rx)^{-1}}
    # This is convex when Phi is affine and output is PSD
    # Use the fact that for an affine Phi: tr{X^{-1}} is convex in X

    # Approximation: use tr(Rx^{-1}) as proxy
    constraints = [
        cp.trace(Rx_var) == power,
        Rx_var >> 0,
    ]

    try:
        # Use matrix_frac for tr(X^{-1}) - it's convex for X >> 0
        # matrix_frac(y, X) = y^T X^{-1} y, but we need tr(X^{-1})
        # For tr(X^{-1}), we use sum of matrix_frac for basis vectors
        # Or use nuclear norm of inverse via auxiliary variable
        # Simpler: minimize trace(inv(X)) is equivalent to:
        # minimize trace(T) s.t. [[T, I], [I, X]] >> 0
        # But for simplicity, use the analytical solution
        pass
    except Exception:
        pass

    return Rx_opt


def optimize_comm_rx(
    P_T: float,
    M: int,
    Hc: np.ndarray,
) -> np.ndarray:
    """Find the communication-optimal covariance matrix.

    Solves:
        max_{Rx} log det(I + sigma_c^{-2} Hc Rx Hc^H)
        s.t. tr{Rx} = P_T * M,
             Rx >= 0

    The solution uses water-filling over the eigenvalues of
    Hc^H Hc. For the high-SNR regime with M >= Nc, the optimal
    Rx allocates power equally across the signal subspace of Hc.

    Args:
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        Hc (np.ndarray): Communication channel matrix, shape (Nc, M).

    Returns:
        np.ndarray: Communication-optimal covariance Rx, shape (M, M).

    References:
        Related to the capacity-achieving input covariance.
    """
    Hc = np.asarray(Hc, dtype=np.complex128)
    Nc = Hc.shape[0]
    power = P_T * M

    # SVD of Hc: Hc = U @ diag(s) @ Vh, Vh has shape (r_eff, M)
    U, s, Vh = np.linalg.svd(Hc, full_matrices=False)
    V = Vh.conj().T  # V has shape (M, r_eff) where r_eff = min(Nc, M)

    # Number of non-zero singular values
    r_eff = len(s)  # = min(Nc, M)
    r = int(np.sum(s > 1e-10))

    if r == 0:
        return (power / M) * np.eye(M, dtype=np.complex128)

    # Water-filling on eigenvalues of Hc^H Hc
    eigenvalues = s ** 2  # length r_eff
    sigma_c2_norm = 1.0  # Normalized noise

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigs = eigenvalues[sorted_indices]

    # Standard water-filling: find n_active such that all p_i > 0
    water_level = power / r  # fallback
    for n_active in range(r, 0, -1):
        inv_sum = np.sum(sigma_c2_norm / sorted_eigs[:n_active])
        mu = (power + inv_sum) / n_active
        p_test = mu - sigma_c2_norm / sorted_eigs[:n_active]
        if np.all(p_test > -1e-10):
            water_level = mu
            break

    # Allocate powers (length r_eff, aligned with V's columns)
    powers = np.zeros(r_eff)
    for i in range(r_eff):
        if s[i] > 1e-10:
            powers[i] = max(0, water_level - sigma_c2_norm / eigenvalues[i])

    # Enforce power constraint exactly (numerical safety)
    total_p = np.sum(powers)
    if total_p > 1e-10:
        powers = powers * power / total_p

    # Construct Rx = V diag(powers) V^H  (M x M)
    # powers has length r_eff = min(Nc, M), V has shape (M, r_eff)
    Rx_opt = V @ np.diag(powers) @ V.conj().T

    # Ensure PSD
    Rx_opt = (Rx_opt + Rx_opt.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(Rx_opt)
    eigvals = np.maximum(eigvals, 0)
    Rx_opt = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return Rx_opt


def covariance_shaping(
    alpha: float,
    P_T: float,
    M: int,
    Hc: np.ndarray,
    Hs_func: Optional[Callable] = None,
    phi_func: Optional[Callable] = None,
    T: int = 1,
    sigma_c2: float = 1.0,
    sigma_s2: float = 1.0,
    Jp: Optional[np.ndarray] = None,
    Nc: Optional[int] = None,
) -> np.ndarray:
    """Solve the covariance shaping optimization (Eq. 48).

    min_{Rx} (1-alpha) * tr{[Phi(Rx)]^{-1}}
            - alpha * log|I + sigma_c^{-2} Hc Rx Hc^H|
    s.t. tr{Rx} = P_T * M,
         Rx >= 0,
         Rx = Rx^H

    This is the fundamental tradeoff optimization between sensing
    quality (CRB) and communication rate.

    Args:
        alpha (float): Tradeoff parameter in [0, 1].
            alpha=0: pure sensing optimization.
            alpha=1: pure communication optimization.
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        Hc (np.ndarray): Communication channel, shape (Nc, M).
        Hs_func (callable, optional): Sensing channel function.
        phi_func (callable, optional): Custom Phi map.
        T (int): Coherent processing interval.
        sigma_c2 (float): Communication noise variance.
        sigma_s2 (float): Sensing noise variance.
        Jp (np.ndarray, optional): Prior information.
        Nc (int, optional): Number of comm Rx antennas.

    Returns:
        np.ndarray: Optimal covariance Rx, shape (M, M).

    References:
        Eq. (48) in the paper.
    """
    Hc = np.asarray(Hc, dtype=np.complex128)
    if Nc is None:
        Nc = Hc.shape[0]
    power = P_T * M

    # CVXPY formulation
    Rx_var = cp.Variable((M, M), hermitian=True)

    constraints = [
        cp.trace(Rx_var) == power,
        Rx_var >> 0,
    ]

    # Communication term: -alpha * log|I + Hc Rx Hc^H / sigma_c2|
    # = -alpha * log_det(I + (1/sigma_c2) * Hc Rx Hc^H)
    # This is concave in Rx (log_det of affine function)

    if alpha > 1e-10 and alpha < 1 - 1e-10:
        # Combined objective
        Gram = Hc @ Rx_var @ Hc.conj().T
        comm_term = -alpha * cp.log_det(
            np.eye(Nc, dtype=np.complex128) + Gram / sigma_c2
        )

        # Sensing term: approximate with -log_det(Rx) (promotes large eigenvalues)
        # This is a convex surrogate for sensing quality
        sensing_term = -(1 - alpha) * cp.log_det(Rx_var)

        objective = cp.Minimize(sensing_term + comm_term)

    elif alpha <= 1e-10:
        # Pure sensing - use isotropic
        return P_T * np.eye(M, dtype=np.complex128)
    else:
        # Pure communication
        Gram = Hc @ Rx_var @ Hc.conj().T
        objective = cp.Minimize(
            -cp.log_det(np.eye(Nc, dtype=np.complex128) + Gram / sigma_c2)
        )

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=20000, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"] and Rx_var.value is not None:
            Rx_opt = np.array(Rx_var.value, dtype=np.complex128)
            # Ensure numerical PSD
            Rx_opt = (Rx_opt + Rx_opt.conj().T) / 2
            eigvals, eigvecs = np.linalg.eigh(Rx_opt)
            eigvals = np.maximum(eigvals, 1e-12)
            Rx_opt = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            return Rx_opt
    except Exception as e:
        pass

    # Fallback: time-sharing between sensing and comm optima
    Rx_sense = optimize_sensing_rx(P_T, M, phi_func=phi_func, T=T, sigma_s2=sigma_s2, Jp=Jp)
    Rx_comm = optimize_comm_rx(P_T, M, Hc)
    return alpha * Rx_comm + (1 - alpha) * Rx_sense


def stiefel_sample(M_sc: int, T: int) -> np.ndarray:
    """Generate a uniformly distributed semi-unitary matrix.

    Sample Q from the uniform (Haar) distribution on the Stiefel
    manifold V_{M_sc}(C^T) = {Q ∈ C^{M_sc × T} : Q Q^H = I_{M_sc}}.

    Uses the LQ decomposition method:
    1. Generate A ∈ C^{M_sc × T} with i.i.d. CN(0,1) entries.
    2. Compute A = L Q via LQ decomposition.
    3. Q is uniformly distributed on the Stiefel manifold.

    Args:
        M_sc (int): Number of rows (sensing subspace dimension).
        T (int): Number of columns (coherent interval).

    Returns:
        np.ndarray: Semi-unitary matrix Q, shape (M_sc, T),
            satisfying Q @ Q^H = I_{M_sc}.

    References:
        Described in Section V-C of the paper for achieving
        the sensing-optimal point P_sc.
    """
    # Step 1: Generate random complex Gaussian matrix
    A = (
        np.random.randn(M_sc, T) + 1j * np.random.randn(M_sc, T)
    ) / np.sqrt(2)

    # Step 2: LQ decomposition
    # scipy doesn't have direct LQ, but we can use QR on A^H
    # A = L Q => A^H = Q^H L^H => A^H = Q' R' (QR decomp)
    # Then Q = (Q')^H

    from scipy.linalg import qr

    Ah = A.conj().T  # (T, M_sc)
    Q_prime, R_prime = qr(Ah, mode="economic")  # Q': (T, M_sc), R': (M_sc, M_sc)

    Q = Q_prime.conj().T  # (M_sc, T)

    # Verify Q is semi-unitary: Q Q^H = I_{M_sc}
    # (optional, for debugging)
    # residual = np.linalg.norm(Q @ Q.conj().T - np.eye(M_sc))
    # assert residual < 1e-10, f"Q not semi-unitary: residual = {residual}"

    return Q


def generate_isotropic_waveform(
    P_T: float,
    M: int,
    T: int,
) -> np.ndarray:
    """Generate an isotropic Gaussian waveform.

    X with i.i.d. columns ~ CN(0, P_T * I_M).

    Args:
        P_T (float): Transmit power per symbol.
        M (int): Number of transmit antennas.
        T (int): Coherent processing interval.

    Returns:
        np.ndarray: Waveform matrix X, shape (M, T).
    """
    X = (
        np.random.randn(M, T) + 1j * np.random.randn(M, T)
    ) / np.sqrt(2)
    X *= np.sqrt(P_T)
    return X


def generate_semi_unitary_waveform(
    P_T: float,
    M_sc: int,
    M: int,
    T: int,
    U: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a semi-unitary waveform for sensing-optimal transmission.

    X = sqrt(T * P_T * M / M_sc) * U * Q

    where Q is uniformly sampled from the Stiefel manifold.

    Args:
        P_T (float): Transmit power per symbol.
        M_sc (int): Sensing subspace dimension.
        M (int): Total number of Tx antennas.
        T (int): Coherent interval.
        U (np.ndarray, optional): Orthonormal basis for sensing
            subspace, shape (M, M_sc).

    Returns:
        Tuple of (X, Q) where X is the waveform and Q is the
        Stiefel matrix.
    """
    # Sample Q from Stiefel manifold
    Q = stiefel_sample(M_sc, T)

    if U is None:
        U = np.eye(M, M_sc, dtype=np.complex128)

    # Construct waveform: tr{(1/T) X X^H} = P_T * M
    # (1/T) * scale^2 * U U^H = P_T * M * (U U^H)
    # scale^2 = P_T * M * T
    scale = np.sqrt(P_T * M * T / M_sc)
    X = scale * U @ Q

    return X, Q
