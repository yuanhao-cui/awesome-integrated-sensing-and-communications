"""
Tracking QoS for ISAC Resource Allocation.

Implements Posterior Cramér-Rao Bound (PCRB) based tracking metrics from
"Sensing as a Service in 6G Perceptive Networks" (Eq. 44-47).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .system_model import ISACSystem


@dataclass
class TargetState:
    """State of a tracking target."""
    position: np.ndarray  # [x, y] or [x, y, z]
    velocity: np.ndarray  # [vx, vy] or [vx, vy, vz]
    acceleration: Optional[np.ndarray] = None


class TrackingQoS:
    """
    Tracking Quality of Service (Eq. 44-47).
    
    Implements Posterior Cramér-Rao Bound (PCRB) for sequential tracking
    using Extended Kalman Filter (EKF) framework.
    """
    
    def __init__(self, system: ISACSystem, dt: float = 0.1,
                 process_noise_std: float = 0.5,
                 measurement_noise_std: float = 0.1):
        """
        Initialize Tracking QoS.
        
        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        dt : float
            Time step for tracking (seconds)
        process_noise_std : float
            Standard deviation of process noise
        measurement_noise_std : float
            Standard deviation of measurement noise
        """
        self.system = system
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        
        # Initialize target states
        self.target_states = self._initialize_target_states()
        
    def _initialize_target_states(self) -> List[TargetState]:
        """Initialize target states from system model."""
        states = []
        Q = self.system.params.Q
        
        for q in range(Q):
            # Convert polar to Cartesian
            d = self.system.target_positions[q]
            theta = self.system.target_angles[q]
            
            x = d * np.cos(theta)
            y = d * np.sin(theta)
            
            # Random initial velocity
            vx = np.random.uniform(-10, 10)
            vy = np.random.uniform(-10, 10)
            
            states.append(TargetState(
                position=np.array([x, y]),
                velocity=np.array([vx, vy])
            ))
        
        return states
    
    def _get_transition_matrix(self) -> np.ndarray:
        """
        Get state transition matrix for constant velocity model.
        
        F = [1, 0, dt, 0;
             0, 1, 0, dt;
             0, 0, 1,  0;
             0, 0, 0,  1]
        """
        dt = self.dt
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
    
    def _get_process_noise_cov(self) -> np.ndarray:
        """
        Get process noise covariance matrix.
        
        Q = σ_p² * [dt⁴/4, 0,     dt³/2, 0;
                    0,     dt⁴/4, 0,     dt³/2;
                    dt³/2, 0,     dt²,   0;
                    0,     dt³/2, 0,     dt²]
        """
        dt = self.dt
        sigma = self.process_noise_std
        
        return sigma**2 * np.array([
            [dt**4/4, 0,       dt**3/2, 0],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0],
            [0,       dt**3/2, 0,       dt**2]
        ])
    
    def _compute_measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute measurement Jacobian H.
        
        For polar measurements [range, angle]:
        H = [x/r, y/r, 0, 0;
             -y/r², x/r², 0, 0]
        """
        x, y = state[0], state[1]
        r = np.sqrt(x**2 + y**2)
        
        if r < 1e-10:
            r = 1e-10
        
        return np.array([
            [x/r,   y/r,   0, 0],
            [-y/r**2, x/r**2, 0, 0]
        ])
    
    def compute_fim(self, p: np.ndarray, b: np.ndarray,
                    state_idx: int = 0) -> np.ndarray:
        """
        Compute Fisher Information Matrix for tracking.
        
        J(x) = J_prior + J_likelihood
        
        where:
        - J_prior = (F * J_{k-1}^{-1} * F^T + Q)^{-1}
        - J_likelihood = H^T * R^{-1} * H
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        state_idx : int
            Target index for tracking
            
        Returns
        -------
        fim : np.ndarray
            Fisher Information Matrix (4, 4)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        F = self._get_transition_matrix()
        Q_proc = self._get_process_noise_cov()
        
        # Get target state
        state = self.target_states[state_idx]
        state_vec = np.array([state.position[0], state.position[1],
                             state.velocity[0], state.velocity[1]])
        
        # Measurement Jacobian
        H = self._compute_measurement_jacobian(state_vec)
        
        # Measurement noise covariance (depends on SNR)
        snr = self.system.compute_sensing_snr(p[state_idx], b[state_idx])
        R = self._compute_measurement_covariance(snr)
        
        # FIM for likelihood
        J_likelihood = H.T @ np.linalg.inv(R) @ H
        
        return J_likelihood
    
    def _compute_measurement_covariance(self, snr: float) -> np.ndarray:
        """
        Compute measurement noise covariance based on SNR.
        
        R = diag(σ_r², σ_θ²)
        where:
        - σ_r² = c² / (8π² SNR B²)
        - σ_θ² = 6 / (SNR Nt (Nt² - 1) π² cos²θ)
        """
        c = 3e8
        Nt = self.system.params.Nt
        
        # Range variance
        sigma_r_sq = c**2 / (8 * np.pi**2 * snr * self.system.params.B_total**2)
        
        # Angle variance (simplified)
        sigma_theta_sq = 6 / (snr * Nt * (Nt**2 - 1) * np.pi**2)
        
        return np.diag([sigma_r_sq, sigma_theta_sq])
    
    def compute_pcrb(self, p: np.ndarray, b: np.ndarray,
                     prior_pcrb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Posterior Cramér-Rao Bound (Eq. 44).
        
        PCRB_k = (F * PCRB_{k-1}^{-1} * F^T + Q)^{-1} + H^T * R^{-1} * H
        
        This is the recursive PCRB computation.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        prior_pcrb : np.ndarray, optional
            PCRB from previous time step (4, 4). If None, uses identity.
            
        Returns
        -------
        pcrb : np.ndarray
            Posterior CRB for each target (Q, 4, 4)
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        Q = self.system.params.Q
        F = self._get_transition_matrix()
        Q_proc = self._get_process_noise_cov()
        
        pcrb = np.zeros((Q, 4, 4))
        
        for q in range(Q):
            # Prior PCRB (or large initial uncertainty)
            if prior_pcrb is None or q >= len(prior_pcrb):
                J_prior_inv = np.eye(4) * 1e6  # Large initial uncertainty
            else:
                J_prior_inv = prior_pcrb[q]
            
            # Predict: J_pred^{-1} = F * J_prior^{-1} * F^T + Q
            J_pred_inv = F @ J_prior_inv @ F.T + Q_proc
            
            # Measurement information
            state = self.target_states[q]
            state_vec = np.array([state.position[0], state.position[1],
                                 state.velocity[0], state.velocity[1]])
            
            H = self._compute_measurement_jacobian(state_vec)
            snr = self.system.compute_sensing_snr(p[q:q+1], b[q:q+1])[0]
            R = self._compute_measurement_covariance(snr)
            
            # Update: J_post = J_pred + H^T * R^{-1} * H
            J_post = np.linalg.inv(J_pred_inv) + H.T @ np.linalg.inv(R) @ H
            
            # PCRB is inverse of posterior FIM
            pcrb[q] = np.linalg.inv(J_post)
        
        return pcrb
    
    def compute_pcrb_trace(self, p: np.ndarray, b: np.ndarray,
                           prior_pcrb: Optional[np.ndarray] = None) -> float:
        """
        Compute trace of PCRB for all targets (Eq. 47).
        
        min Σ_q trace(PCRB_q)
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        prior_pcrb : np.ndarray, optional
            Prior PCRB (Q, 4, 4)
            
        Returns
        -------
        trace_sum : float
            Sum of PCRB traces across all targets
        """
        pcrb = self.compute_pcrb(p, b, prior_pcrb)
        return np.sum([np.trace(pcrb[q]) for q in range(pcrb.shape[0])])
    
    def compute_pcrb_position_trace(self, p: np.ndarray, b: np.ndarray,
                                     prior_pcrb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute PCRB trace for position only (first 2x2 block).
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        prior_pcrb : np.ndarray, optional
            Prior PCRB (Q, 4, 4)
            
        Returns
        -------
        position_trace : np.ndarray
            Position PCRB trace for each target (Q,)
        """
        pcrb = self.compute_pcrb(p, b, prior_pcrb)
        return np.array([np.trace(pcrb[q, :2, :2]) for q in range(pcrb.shape[0])])
    
    def update_target_states(self, measurements: Optional[np.ndarray] = None):
        """
        Update target states using EKF prediction and update.
        
        Parameters
        ----------
        measurements : np.ndarray, optional
            Range and angle measurements (Q, 2). If None, uses predicted states.
        """
        F = self._get_transition_matrix()
        
        for q in range(len(self.target_states)):
            state = self.target_states[q]
            state_vec = np.array([state.position[0], state.position[1],
                                 state.velocity[0], state.velocity[1]])
            
            # Predict
            state_vec_pred = F @ state_vec
            
            if measurements is not None:
                # Update with measurement
                H = self._compute_measurement_jacobian(state_vec_pred)
                meas = measurements[q]
                
                # Predicted measurement
                r_pred = np.sqrt(state_vec_pred[0]**2 + state_vec_pred[1]**2)
                theta_pred = np.arctan2(state_vec_pred[1], state_vec_pred[0])
                meas_pred = np.array([r_pred, theta_pred])
                
                # Innovation
                innovation = meas - meas_pred
                
                # Simplified Kalman gain
                K = np.eye(4)[:, :2] @ np.linalg.inv(H @ np.eye(4)[:, :2].T + 
                                                      np.eye(2) * self.measurement_noise_std**2)
                
                # Update
                state_vec = state_vec_pred + K @ innovation
            else:
                state_vec = state_vec_pred
            
            # Add process noise
            state_vec += np.random.normal(0, self.process_noise_std, 4)
            
            # Update state
            self.target_states[q].position = state_vec[:2]
            self.target_states[q].velocity = state_vec[2:4]
    
    def simulate_tracking(self, p: np.ndarray, b: np.ndarray,
                          num_steps: int = 50) -> Tuple[List[np.ndarray], List[float]]:
        """
        Simulate tracking over multiple time steps.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        num_steps : int
            Number of time steps to simulate
            
        Returns
        -------
        pcrb_history : list
            PCRB at each time step
        trace_history : list
            Trace of PCRB at each time step
        """
        p = np.asarray(p)
        b = np.asarray(b)
        
        pcrb_history = []
        trace_history = []
        prior_pcrb = None
        
        for t in range(num_steps):
            # Compute PCRB
            pcrb = self.compute_pcrb(p, b, prior_pcrb)
            pcrb_history.append(pcrb.copy())
            trace_history.append(np.sum([np.trace(pcrb[q]) for q in range(pcrb.shape[0])]))
            
            # Update prior for next step
            prior_pcrb = pcrb
            
            # Update target states
            self.update_target_states()
        
        return pcrb_history, trace_history
    
    def compute_tracking_error_bound(self, p: np.ndarray, b: np.ndarray,
                                     prior_pcrb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute tracking error bound (RMSE) from PCRB.
        
        TEB(q) = √trace(PCRB_q)
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        prior_pcrb : np.ndarray, optional
            Prior PCRB (Q, 4, 4)
            
        Returns
        -------
        teb : np.ndarray
            Tracking error bounds for each target (Q,)
        """
        pcrb = self.compute_pcrb(p, b, prior_pcrb)
        return np.sqrt(np.array([np.trace(pcrb[q]) for q in range(pcrb.shape[0])]))
