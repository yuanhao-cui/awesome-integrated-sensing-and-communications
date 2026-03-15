"""
ISAC System Model for Resource Allocation.

Implements the system model from "Sensing as a Service in 6G Perceptive Networks" (Eq. 1-9).
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SystemParameters:
    """ISAC system parameters."""
    Nt: int = 32          # Number of transmit antennas
    Nr: int = 32          # Number of receive antennas  
    Q: int = 3            # Number of sensing targets
    K: int = 3            # Number of communication users
    L: int = 1            # Number of ISAC users
    fc: float = 30e9      # Carrier frequency (Hz)
    P_total: float = 40.0 # Total transmit power (W)
    B_total: float = 100e6 # Total bandwidth (Hz)
    N0_dBm: float = -174.0 # Noise PSD (dBm/Hz)
    NF_dB: float = 10.0   # Noise figure (dB)
    
    @property
    def M(self) -> int:
        """Total number of objects."""
        return self.Q + self.K + self.L


class ISACSystem:
    """
    ISAC System Model (Eq. 1-9).
    
    Models an ISAC system with Q sensing targets, K communication users,
    and L ISAC users performing joint sensing and communication.
    """
    
    def __init__(self, Nt: int = 32, Nr: int = 32, Q: int = 3, K: int = 3, L: int = 1,
                 fc: float = 30e9, P_total: float = 40.0, B_total: float = 100e6,
                 N0_dBm: float = -174.0, NF_dB: float = 10.0,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize ISAC system.
        
        Parameters
        ----------
        Nt : int
            Number of transmit antennas (default: 32)
        Nr : int
            Number of receive antennas (default: 32)
        Q : int
            Number of sensing targets (default: 3)
        K : int
            Number of communication users (default: 3)
        L : int
            Number of ISAC users (default: 1)
        fc : float
            Carrier frequency in Hz (default: 30 GHz)
        P_total : float
            Total transmit power in Watts (default: 40W)
        B_total : float
            Total bandwidth in Hz (default: 100 MHz)
        N0_dBm : float
            Noise power spectral density in dBm/Hz (default: -174 dBm/Hz)
        NF_dB : float
            Noise figure in dB (default: 10 dB)
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.params = SystemParameters(Nt, Nr, Q, K, L, fc, P_total, B_total, N0_dBm, NF_dB)
        self.rng = rng or np.random.default_rng(42)
        
        # Noise power
        self.N0 = 10**((N0_dBm + NF_dB) / 10) / 1000  # Convert dBm to Watts
        
        # Initialize channels
        self._init_channels()
        
    def _init_channels(self):
        """Initialize synthetic channels for testing."""
        p = self.params
        
        # Sensing target positions (randomized)
        self.target_positions = self.rng.uniform(10, 100, p.Q)  # Range 10-100m
        self.target_angles = self.rng.uniform(-np.pi/3, np.pi/3, p.Q)  # Azimuth angles
        
        # Communication user positions
        self.user_positions = self.rng.uniform(50, 200, p.K)  # Range 50-200m
        self.user_angles = self.rng.uniform(-np.pi/2, np.pi/2, p.K)
        
        # ISAC user positions (communication + sensing)
        self.isac_position = self.rng.uniform(30, 150, 1)[0]
        self.isac_angle = self.rng.uniform(-np.pi/4, np.pi/4, 1)[0]
        
        # Path loss factors (Eq. 1)
        self.alpha_sensing = self._compute_path_loss(self.target_positions)
        self.alpha_comm = self._compute_path_loss(self.user_positions)
        self.alpha_isac = np.array([self._compute_path_loss(np.array([self.isac_position]))[0]])
        
        # Radar cross sections for sensing targets
        self.rcs = self.rng.uniform(1.0, 10.0, p.Q)  # m^2
        
        # Channel gains (normalized)
        self.beta_sensing = self._compute_channel_gain(self.target_positions, self.target_angles)
        self.beta_comm = self._compute_channel_gain(self.user_positions, self.user_angles)
        self.beta_isac = np.array([self._compute_channel_gain(
            np.array([self.isac_position]), np.array([self.isac_angle]))[0]])
        
    def _compute_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute path loss in linear scale.
        
        Path loss model: PL = 32.4 + 20*log10(d) + 20*log10(fc) dB (Eq. 1)
        """
        fc_GHz = self.params.fc / 1e9
        # Convert to linear scale
        pl_dB = 32.4 + 20 * np.log10(distances) + 20 * np.log10(fc_GHz)
        return 10 ** (-pl_dB / 10)
    
    def _compute_channel_gain(self, distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Compute channel gain including path loss and small-scale fading.
        
        β = α * |h|^2 where h ~ CN(0, I)
        """
        # Small-scale fading: Rayleigh
        h = (self.rng.normal(0, 1/np.sqrt(2), (len(distances), self.params.Nt)) + 
             1j * self.rng.normal(0, 1/np.sqrt(2), (len(distances), self.params.Nt)))
        
        # Array response
        d_lambda = 0.5  # Half-wavelength spacing
        array_response = np.exp(1j * 2 * np.pi * d_lambda * np.outer(np.sin(angles), 
                                                                       np.arange(self.params.Nt)))
        
        # Combined channel
        h_combined = h * array_response
        
        # Path loss
        alpha = self._compute_path_loss(distances)
        
        # Channel gain: β = α * ||h||^2 / Nt
        return alpha * np.linalg.norm(h_combined, axis=1)**2 / self.params.Nt
    
    def get_channel_matrix(self, idx: int, target_type: str = 'sensing') -> np.ndarray:
        """
        Get channel matrix for a specific target/user.
        
        Parameters
        ----------
        idx : int
            Index of target or user
        target_type : str
            'sensing', 'comm', or 'isac'
            
        Returns
        -------
        h : np.ndarray
            Channel vector of shape (Nt,)
        """
        if target_type == 'sensing':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) + 
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_sensing[idx])
        elif target_type == 'comm':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) + 
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_comm[idx])
        elif target_type == 'isac':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) + 
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_isac[idx])
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
    
    def get_snr(self, power: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """
        Compute SNR for each sensing target.
        
        SNR_q = (p_q * β_q * σ_q) / (N0 * b_q)
        
        Parameters
        ----------
        power : np.ndarray
            Power allocation (Q,)
        bandwidth : np.ndarray
            Bandwidth allocation (Q,)
            
        Returns
        -------
        snr : np.ndarray
            SNR for each target (Q,)
        """
        return (power * self.beta_sensing * self.rcs) / (self.N0 * bandwidth)
    
    def get_comm_snr(self, power: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """
        Compute SNR for communication users.
        
        SNR_k = (p_k * β_k) / (N0 * b_k)
        
        Parameters
        ----------
        power : np.ndarray
            Power allocation (K,)
        bandwidth : np.ndarray
            Bandwidth allocation (K,)
            
        Returns
        -------
        snr : np.ndarray
            SNR for each user (K,)
        """
        return (power * self.beta_comm) / (self.N0 * bandwidth)
    
    def validate_allocations(self, p: np.ndarray, b: np.ndarray) -> bool:
        """
        Validate power and bandwidth allocations.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation vector (M,) = [p_sensing, p_comm, p_isac]
        b : np.ndarray
            Bandwidth allocation vector (M,) = [b_sensing, b_comm, b_isac]
            
        Returns
        -------
        valid : bool
            True if allocations satisfy constraints
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        # Non-negativity
        if np.any(p < 0) or np.any(b < 0):
            return False
        
        # Budget constraints
        if not np.isclose(np.sum(p), self.params.P_total, rtol=1e-3):
            return False
        if not np.isclose(np.sum(b), self.params.B_total, rtol=1e-3):
            return False
            
        return True
    
    def compute_communication_rate(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute communication rate for each user.
        
        R_k = b_k * log2(1 + SNR_k) [bps/Hz when b_k=1]
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (K + L,)
        b : np.ndarray
            Bandwidth allocation (K + L,)
            
        Returns
        -------
        rates : np.ndarray
            Communication rates (K + L,)
        """
        snr = self.get_comm_snr(p, b)
        return b * np.log2(1 + snr)
    
    def compute_sensing_snr(self, p_sensing: np.ndarray, b_sensing: np.ndarray) -> np.ndarray:
        """
        Compute sensing SNR for each target.
        
        SNR_q = (p_q * β_q * σ_q) / (N0 * b_q)
        
        Parameters
        ----------
        p_sensing : np.ndarray
            Power allocation for sensing targets (Q,)
        b_sensing : np.ndarray
            Bandwidth allocation for sensing targets (Q,)
            
        Returns
        -------
        snr : np.ndarray
            Sensing SNR for each target (Q,)
        """
        return self.get_snr(p_sensing, b_sensing)
    
    @property
    def total_objects(self) -> int:
        """Total number of objects (M = Q + K + L)."""
        return self.params.M
