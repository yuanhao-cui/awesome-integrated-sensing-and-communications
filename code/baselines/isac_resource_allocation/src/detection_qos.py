"""
Detection QoS for ISAC Resource Allocation.

Implements detection probability metrics from "Sensing as a Service in 6G Perceptive Networks"
(Eq. 18-21).
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple
from dataclasses import dataclass

from .system_model import ISACSystem


class DetectionQoS:
    """
    Detection Quality of Service (Eq. 18-21).
    
    Implements detection probability computation under Neyman-Pearson criterion
    with both max-min fairness and sum (comprehensiveness) objectives.
    """
    
    def __init__(self, system: ISACSystem, Pfa: float = 0.01):
        """
        Initialize Detection QoS.
        
        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        Pfa : float
            Probability of false alarm (default: 0.01)
        """
        self.system = system
        self.Pfa = Pfa
        self._chi2 = stats.chi2(df=2)  # Central chi-squared with 2 DOF
        self._chi2_nc = None  # Will be set for non-central
        
    def _compute_threshold(self, N: int = 1000) -> float:
        """
        Compute detection threshold for given Pfa.
        
        Parameters
        ----------
        N : int
            Number of samples for threshold computation
            
        Returns
        -------
        threshold : float
            Detection threshold
        """
        # For Neyman-Pearson: threshold determined by Pfa
        return self._chi2.ppf(1 - self.Pfa)
    
    def compute_detection_probability(self, p: np.ndarray, b: np.ndarray,
                                       sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute detection probability for each sensing target.
        
        From Eq. 18:
        P_D,q = 1 - F_χ²( 2δ / (N0 σ²) / (1 + p_q S_q) )
        
        where:
        - δ is the detection threshold (determined by Pfa)
        - N0 is the noise power spectral density
        - σ² is the target RCS
        - p_q is the transmit power for target q
        - S_q is the signal-to-noise ratio factor
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation for sensing targets (Q,)
        b : np.ndarray
            Bandwidth allocation for sensing targets (Q,)
        sigma : np.ndarray, optional
            Target radar cross sections. If None, uses system defaults.
            
        Returns
        -------
        P_D : np.ndarray
            Detection probabilities for each target (Q,)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        if sigma is None:
            sigma = self.system.rcs
        
        Q = self.system.params.Q
        P_D = np.zeros(Q)
        
        # Detection threshold from Pfa
        delta = self._compute_threshold()
        
        for q in range(Q):
            # SNR factor (Eq. 18)
            snr_q = (p[q] * self.system.beta_sensing[q] * sigma[q]) / \
                    (self.system.N0 * b[q])
            
            # Argument for chi-squared CDF
            # Scale factor: 2δ / (N0 σ²) / (1 + SNR)
            arg = 2 * delta / (1 + snr_q)
            
            # Non-central chi-squared CDF (2 DOF, non-centrality parameter = SNR)
            ncp = 2 * snr_q  # Non-centrality parameter
            P_D[q] = 1 - stats.ncx2.cdf(arg, df=2, nc=ncp)
        
        return P_D
    
    def compute_detection_prob_simplified(self, p: np.ndarray, b: np.ndarray,
                                          sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simplified detection probability computation (closed-form approximation).
        
        From Eq. 18:
        P_D,q = Q_1(√(2 * SNR_q), √(2δ))
        
        where Q_1 is the Marcum Q-function.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        sigma : np.ndarray, optional
            Target RCS (Q,)
            
        Returns
        -------
        P_D : np.ndarray
            Detection probabilities (Q,)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        if sigma is None:
            sigma = self.system.rcs
        
        Q = self.system.params.Q
        P_D = np.zeros(Q)
        
        delta = self._compute_threshold()
        
        for q in range(Q):
            snr_q = (p[q] * self.system.beta_sensing[q] * sigma[q]) / \
                    (self.system.N0 * b[q])
            
            # Using chi-squared approximation
            # P_D = 1 - F_χ²(2δ / (1 + SNR_q), 2)
            arg = 2 * delta / (1 + snr_q)
            P_D[q] = 1 - stats.chi2.cdf(arg, df=2)
        
        return P_D
    
    def compute_objective_maxmin(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute max-min fairness objective (Eq. 20).
        
        max min_q P_D,q
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        objective : float
            Minimum detection probability across all targets
        """
        P_D = self.compute_detection_probability(p, b)
        return np.min(P_D)
    
    def compute_objective_sum(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute sum (comprehensiveness) objective (Eq. 21).
        
        max Σ_q P_D,q
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        objective : float
            Sum of detection probabilities
        """
        P_D = self.compute_detection_probability(p, b)
        return np.sum(P_D)
    
    def detection_probability_gradient(self, p: np.ndarray, b: np.ndarray,
                                        sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient of detection probability w.r.t. power.
        
        ∂P_D,q/∂p_q for gradient-based optimization.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        sigma : np.ndarray, optional
            Target RCS (Q,)
            
        Returns
        -------
        grad : np.ndarray
            Gradient w.r.t. power (Q,)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        if sigma is None:
            sigma = self.system.rcs
        
        Q = self.system.params.Q
        grad = np.zeros(Q)
        
        delta = self._compute_threshold()
        eps = 1e-6
        
        for q in range(Q):
            # Finite difference approximation for gradient
            p_plus = p.copy()
            p_plus[q] += eps
            P_D_plus = self.compute_detection_probability(p_plus, b, sigma)[q]
            
            p_minus = p.copy()
            p_minus[q] = max(eps, p[q] - eps)
            P_D_minus = self.compute_detection_probability(p_minus, b, sigma)[q]
            
            grad[q] = (P_D_plus - P_D_minus) / (2 * eps)
        
        return grad
    
    def is_detectable(self, p: np.ndarray, b: np.ndarray, 
                      threshold: float = 0.9) -> np.ndarray:
        """
        Check if targets are detectable with given probability.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        threshold : float
            Required detection probability
            
        Returns
        -------
        detectable : np.ndarray
            Boolean array indicating detectability (Q,)
        """
        P_D = self.compute_detection_probability(p, b)
        return P_D >= threshold
