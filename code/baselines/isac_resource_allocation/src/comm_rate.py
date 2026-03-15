"""
Communication Rate for ISAC Resource Allocation.

Implements communication rate computation from "Sensing as a Service in 6G Perceptive Networks"
(Eq. 9).
"""

import numpy as np
from typing import Optional, Tuple

from .system_model import ISACSystem


class CommunicationRate:
    """
    Communication Rate Computation (Eq. 9).
    
    Implements rate computation for communication users in ISAC system.
    """
    
    def __init__(self, system: ISACSystem):
        """
        Initialize Communication Rate.
        
        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        """
        self.system = system
    
    def compute_rate(self, p: np.ndarray, b: np.ndarray,
                     user_type: str = 'comm') -> np.ndarray:
        """
        Compute communication rate (Eq. 9).
        
        R_k = b_k * log2(1 + SNR_k)
        
        where:
        - SNR_k = (p_k * β_k) / (N0 * b_k)
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation
        user_type : str
            'comm' for communication users, 'isac' for ISAC users
            
        Returns
        -------
        rates : np.ndarray
            Communication rates in bps/Hz
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        if user_type == 'comm':
            snr = self.system.get_comm_snr(p, b)
        elif user_type == 'isac':
            # ISAC user also gets sensing SNR benefit
            snr = self.system.get_comm_snr(p, b)
        else:
            raise ValueError(f"Unknown user_type: {user_type}")
        
        # R = B * log2(1 + SNR)
        rates = b * np.log2(1 + snr)
        
        return rates
    
    def compute_sum_rate(self, p_comm: np.ndarray, b_comm: np.ndarray,
                         p_isac: Optional[np.ndarray] = None,
                         b_isac: Optional[np.ndarray] = None) -> float:
        """
        Compute sum communication rate.
        
        R_total = Σ_k R_k + Σ_l R_l (ISAC users)
        
        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)
            
        Returns
        -------
        sum_rate : float
            Total communication rate
        """
        sum_rate = np.sum(self.compute_rate(p_comm, b_comm, 'comm'))
        
        if p_isac is not None and b_isac is not None:
            sum_rate += np.sum(self.compute_rate(p_isac, b_isac, 'isac'))
        
        return sum_rate
    
    def compute_min_rate(self, p_comm: np.ndarray, b_comm: np.ndarray,
                         p_isac: Optional[np.ndarray] = None,
                         b_isac: Optional[np.ndarray] = None) -> float:
        """
        Compute minimum communication rate (for fairness).
        
        min(R_k, R_l)
        
        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)
            
        Returns
        -------
        min_rate : float
            Minimum communication rate
        """
        rates_comm = self.compute_rate(p_comm, b_comm, 'comm')
        
        if p_isac is not None and b_isac is not None:
            rates_isac = self.compute_rate(p_isac, b_isac, 'isac')
            all_rates = np.concatenate([rates_comm, rates_isac])
        else:
            all_rates = rates_comm
        
        return np.min(all_rates)
    
    def check_rate_constraints(self, p_comm: np.ndarray, b_comm: np.ndarray,
                               Gamma_c: float,
                               p_isac: Optional[np.ndarray] = None,
                               b_isac: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """
        Check if communication rate constraints are satisfied.
        
        R_k ≥ Γc for all k
        
        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        Gamma_c : float
            Rate threshold (bps/Hz)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)
            
        Returns
        -------
        satisfied : bool
            True if all rate constraints are met
        rates : np.ndarray
            Actual rates for all users
        """
        rates_comm = self.compute_rate(p_comm, b_comm, 'comm')
        
        if p_isac is not None and b_isac is not None:
            rates_isac = self.compute_rate(p_isac, b_isac, 'isac')
            all_rates = np.concatenate([rates_comm, rates_isac])
        else:
            all_rates = rates_comm
        
        satisfied = np.all(all_rates >= Gamma_c)
        
        return satisfied, all_rates
    
    def compute_spectral_efficiency(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute spectral efficiency (rate per unit bandwidth).
        
        η_k = R_k / b_k = log2(1 + SNR_k) [bps/Hz]
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation
            
        Returns
        -------
        eta : np.ndarray
            Spectral efficiency (bps/Hz)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        snr = self.system.get_comm_snr(p, b)
        return np.log2(1 + snr)
    
    def compute_energy_efficiency(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute energy efficiency (rate per unit power).
        
        EE_k = R_k / p_k [bps/Hz/W]
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation
            
        Returns
        -------
        ee : np.ndarray
            Energy efficiency
        """
        rates = self.compute_rate(p, b, 'comm')
        # Avoid division by zero
        p_safe = np.maximum(p, 1e-10)
        return rates / p_safe
