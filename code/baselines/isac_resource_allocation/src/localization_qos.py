"""
Localization QoS for ISAC Resource Allocation.

Implements Cramér-Rao Bound based localization metrics from "Sensing as a Service in 6G Perceptive Networks"
(Eq. 22-31).
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .system_model import ISACSystem


@dataclass
class CRBResults:
    """Cramér-Rao Bound results for a target."""
    crb_range: float      # CRB for range estimation
    crb_angle: float      # CRB for angle estimation
    crb_combined: float   # Combined CRB (weighted sum)


class LocalizationQoS:
    """
    Localization Quality of Service (Eq. 22-31).
    
    Implements CRB-based localization metrics with range and angle estimation
    performance bounds.
    """
    
    def __init__(self, system: ISACSystem, w_d: float = 1.0, w_theta: float = 1.0):
        """
        Initialize Localization QoS.
        
        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        w_d : float
            Weight for range estimation (default: 1.0)
        w_theta : float
            Weight for angle estimation (default: 1.0)
        """
        self.system = system
        self.w_d = w_d
        self.w_theta = w_theta
        self.c = 3e8  # Speed of light
        
    def compute_crb_range(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute CRB for range estimation (Eq. 22).
        
        CRB(d_q) ∝ 1 / (p_q * |s_q|² * b_q)
        
        More precisely:
        CRB(d_q) = c² / (8π² * SNR_q * b_q²)
        
        where SNR_q = (p_q * β_q * σ_q) / (N0 * b_q)
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation for sensing targets (Q,)
        b : np.ndarray
            Bandwidth allocation for sensing targets (Q,)
            
        Returns
        -------
        crb_range : np.ndarray
            CRB for range estimation (Q,)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        snr = self.system.compute_sensing_snr(p, b)
        
        # CRB(d) = c² / (8π² * SNR * B²)
        crb = (self.c**2) / (8 * np.pi**2 * snr * b**2)
        
        return crb
    
    def compute_crb_angle(self, p: np.ndarray, b: np.ndarray,
                          d_lambda: float = 0.5) -> np.ndarray:
        """
        Compute CRB for angle estimation (Eq. 29-31).
        
        CRB(θ_q) ∝ 1 / (p_q * |s_q|²)
        
        More precisely:
        CRB(θ_q) = 6 / (SNR_q * Nt * (Nt² - 1) * π² * cos²(θ_q))
        
        Note: Angle CRB depends on SNR which includes bandwidth, but the
        fundamental limit is set by the power and antenna array configuration.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        d_lambda : float
            Antenna spacing in wavelengths (default: 0.5)
            
        Returns
        -------
        crb_angle : np.ndarray
            CRB for angle estimation (Q,)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        # SNR for sensing: SNR = (p * β * σ) / (N0 * b)
        snr = self.system.compute_sensing_snr(p, b)
        Nt = self.system.params.Nt
        angles = self.system.target_angles
        
        # CRB(θ) = 6 / (SNR * Nt * (Nt² - 1) * π² * cos²(θ) * d_λ²)
        crb = 6 / (snr * Nt * (Nt**2 - 1) * np.pi**2 * 
                   np.cos(angles)**2 * d_lambda**2)
        
        return crb
    
    def compute_crb_combined(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute combined CRB for localization (Eq. 22, 29-31).
        
        ρ_q = w_d / CRB(d_q) + w_θ / CRB(θ_q)
        
        This is the inverse-weighted sum of individual CRBs.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        crb_combined : np.ndarray
            Combined CRB metric (Q,)
        """
        crb_d = self.compute_crb_range(p, b)
        crb_theta = self.compute_crb_angle(p, b)
        
        # ρ_q = w_d / CRB(d_q) + w_θ / CRB(θ_q)
        rho = self.w_d / crb_d + self.w_theta / crb_theta
        
        return rho
    
    def compute_localization_rmse(self, p: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute root mean square error (RMSE) bounds for localization.
        
        RMSE(d_q) = √CRB(d_q)
        RMSE(θ_q) = √CRB(θ_q)
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        rmse_range : np.ndarray
            RMSE for range estimation (Q,)
        rmse_angle : np.ndarray
            RMSE for angle estimation (Q,)
        """
        crb_d = self.compute_crb_range(p, b)
        crb_theta = self.compute_crb_angle(p, b)
        
        return np.sqrt(crb_d), np.sqrt(crb_theta)
    
    def compute_objective_sum(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute sum objective for localization QoS (Eq. 31).
        
        max Σ_q ρ_q
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        objective : float
            Sum of combined CRB metrics
        """
        rho = self.compute_crb_combined(p, b)
        return np.sum(rho)
    
    def compute_objective_proportional_fairness(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute proportional fairness objective for localization (Eq. 31).
        
        max Σ_q log(ρ_q)
        
        This ensures proportional fairness among targets.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        objective : float
            Proportional fairness metric
        """
        rho = self.compute_crb_combined(p, b)
        # Proportional fairness: sum of log
        return np.sum(np.log(rho + 1e-10))
    
    def compute_objective_maxmin(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute max-min fairness objective for localization.
        
        max min_q ρ_q
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        objective : float
            Minimum combined CRB metric
        """
        rho = self.compute_crb_combined(p, b)
        return np.min(rho)
    
    def compute_fim(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Fisher Information Matrix (FIM) for target parameters.
        
        FIM(q) = [F_dd, F_dθ; F_θd, F_θθ]
        
        where:
        - F_dd = ∂²lnL/∂d²
        - F_θθ = ∂²lnL/∂θ²
        - F_dθ = ∂²lnL/∂d∂θ
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        fim : np.ndarray
            Fisher Information Matrices (Q, 2, 2)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        Q = self.system.params.Q
        Nt = self.system.params.Nt
        snr = self.system.compute_sensing_snr(p, b)
        angles = self.system.target_angles
        
        fim = np.zeros((Q, 2, 2))
        
        for q in range(Q):
            # F_dd = 8π² B² SNR / c²
            F_dd = 8 * np.pi**2 * b[q]**2 * snr[q] / self.c**2
            
            # F_θθ = SNR * Nt * (Nt² - 1) * π² * cos²(θ) * d_λ² / 3
            F_theta_theta = snr[q] * Nt * (Nt**2 - 1) * np.pi**2 * \
                           np.cos(angles[q])**2 / 3
            
            # Off-diagonal term (coupling between range and angle)
            F_d_theta = 0  # Simplified: assume decoupled
            
            fim[q] = [[F_dd, F_d_theta], [F_d_theta, F_theta_theta]]
        
        return fim
    
    def validate_localization_performance(self, p: np.ndarray, b: np.ndarray,
                                          max_range_error: float = 1.0,
                                          max_angle_error: float = 0.1) -> bool:
        """
        Validate that localization performance meets requirements.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        max_range_error : float
            Maximum allowed range RMSE (meters)
        max_angle_error : float
            Maximum allowed angle RMSE (radians)
            
        Returns
        -------
        valid : bool
            True if performance requirements are met
        """
        rmse_range, rmse_angle = self.compute_localization_rmse(p, b)
        
        return (np.all(rmse_range <= max_range_error) and 
                np.all(rmse_angle <= max_angle_error))
