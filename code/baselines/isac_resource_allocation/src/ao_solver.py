"""
Alternating Optimization Solver for ISAC Resource Allocation.

Implements Algorithm 1 from "Sensing as a Service in 6G Perceptive Networks".
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings

from .system_model import ISACSystem
from .detection_qos import DetectionQoS
from .localization_qos import LocalizationQoS
from .tracking_qos import TrackingQoS
from .comm_rate import CommunicationRate
from .fairness import FairnessMetrics, FairnessType


@dataclass
class AOResult:
    """Result from Alternating Optimization."""
    p: np.ndarray          # Power allocation
    b: np.ndarray          # Bandwidth allocation
    objective: float       # Final objective value
    iterations: int        # Number of iterations
    converged: bool        # Whether converged
    detection_probs: Optional[np.ndarray] = None
    localization_rho: Optional[np.ndarray] = None
    tracking_pcrb: Optional[np.ndarray] = None
    comm_rates: Optional[np.ndarray] = None


class AOSolver:
    """
    Alternating Optimization Solver (Algorithm 1).
    
    Solves the unified ISAC resource allocation problem:
    
    maximize   Sensing QoS
      p, b
    subject to
      R(p, b) ≥ Γc  (communication rate threshold)
      1^T * p = P_total  (power budget)
      1^T * b = B_total  (bandwidth budget)
      
    Algorithm:
    1. Initialize bandwidth b uniformly
    2. Solve for power p given b (convex subproblem)
    3. Solve for bandwidth b given p (convex subproblem)
    4. Iterate until convergence
    """
    
    def __init__(self, system: ISACSystem, qos_type: str = 'detection',
                 fairness: str = 'maxmin', max_iter: int = 50,
                 tol: float = 1e-4, solver: str = 'default'):
        """
        Initialize AO Solver.
        
        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        qos_type : str
            Type of sensing QoS: 'detection', 'localization', 'tracking'
        fairness : str
            Fairness criterion: 'maxmin', 'sum', 'proportional'
        max_iter : int
            Maximum iterations (default: 50)
        tol : float
            Convergence tolerance (default: 1e-4)
        solver : str
            CVXPY solver: 'default', 'MOSEK', 'SCS', 'ECOS'
        """
        self.system = system
        self.qos_type = qos_type
        self.fairness_type = FairnessType(fairness)
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        
        # Initialize QoS modules
        self.detection_qos = DetectionQoS(system)
        self.localization_qos = LocalizationQoS(system)
        self.tracking_qos = TrackingQoS(system)
        self.comm_rate = CommunicationRate(system)
        self.fairness = FairnessMetrics()
        
    def _get_solver(self):
        """Get CVXPY solver with fallback."""
        if self.solver == 'default':
            try:
                return cp.MOSEK
            except:
                return cp.SCS
        elif self.solver == 'MOSEK':
            try:
                return cp.MOSEK
            except:
                warnings.warn("MOSEK not available, falling back to SCS")
                return cp.SCS
        else:
            return getattr(cp, self.solver)
    
    def _initialize_bandwidth(self) -> np.ndarray:
        """
        Initialize bandwidth uniformly.
        
        Step 1 of Algorithm 1.
        """
        M = self.system.total_objects
        return np.ones(M) * self.system.params.B_total / M
    
    def _solve_power_subproblem(self, b: np.ndarray, Gamma_c: float) -> np.ndarray:
        """
        Solve power allocation subproblem given bandwidth.
        
        Step 2 of Algorithm 1.
        
        Parameters
        ----------
        b : np.ndarray
            Bandwidth allocation (M,)
        Gamma_c : float
            Communication rate threshold
            
        Returns
        -------
        p : np.ndarray
            Power allocation (M,)
        """
        Q = self.system.params.Q
        K = self.system.params.K
        L = self.system.params.L
        M = self.system.total_objects
        
        # Decision variables
        p = cp.Variable(M, nonneg=True)
        
        # Split allocations
        p_sensing = p[:Q]
        p_comm = p[Q:Q+K]
        p_isac = p[Q+K:]
        b_sensing = b[:Q]
        b_comm = b[Q:Q+K]
        b_isac = b[Q+K:]
        
        # Build objective based on QoS type
        if self.qos_type == 'detection':
            objective = self._build_detection_objective(p_sensing, b_sensing)
        elif self.qos_type == 'localization':
            objective = self._build_localization_objective(p_sensing, b_sensing)
        elif self.qos_type == 'tracking':
            objective = self._build_tracking_objective(p_sensing, b_sensing)
        else:
            raise ValueError(f"Unknown QoS type: {self.qos_type}")
        
        # Constraints
        constraints = [
            cp.sum(p) == self.system.params.P_total,  # Power budget
        ]
        
        # Communication rate constraints
        for k in range(K):
            snr_k = (p_comm[k] * self.system.beta_comm[k]) / \
                    (self.system.N0 * b_comm[k])
            constraints.append(
                b_comm[k] * cp.log(1 + snr_k) / np.log(2) >= Gamma_c
            )
        
        # ISAC user rate constraint
        for l in range(L):
            snr_l = (p_isac[l] * self.system.beta_isac[l]) / \
                    (self.system.N0 * b_isac[l])
            constraints.append(
                b_isac[l] * cp.log(1 + snr_l) / np.log(2) >= Gamma_c
            )
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            solver = self._get_solver()
            problem.solve(solver=solver, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                return np.maximum(p.value, 0)
            else:
                warnings.warn(f"Power subproblem status: {problem.status}")
                return np.ones(M) * self.system.params.P_total / M
        except Exception as e:
            warnings.warn(f"Power subproblem failed: {e}")
            return np.ones(M) * self.system.params.P_total / M
    
    def _solve_bandwidth_subproblem(self, p: np.ndarray, Gamma_c: float) -> np.ndarray:
        """
        Solve bandwidth allocation subproblem given power.
        
        Step 3 of Algorithm 1.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation (M,)
        Gamma_c : float
            Communication rate threshold
            
        Returns
        -------
        b : np.ndarray
            Bandwidth allocation (M,)
        """
        Q = self.system.params.Q
        K = self.system.params.K
        L = self.system.params.L
        M = self.system.total_objects
        
        # Decision variables
        b = cp.Variable(M, nonneg=True)
        
        # Split allocations
        p_sensing = p[:Q]
        p_comm = p[Q:Q+K]
        p_isac = p[Q+K:]
        b_sensing = b[:Q]
        b_comm = b[Q:Q+K]
        b_isac = b[Q+K:]
        
        # Build objective
        if self.qos_type == 'detection':
            objective = self._build_detection_objective_bw(p_sensing, b_sensing)
        elif self.qos_type == 'localization':
            objective = self._build_localization_objective_bw(p_sensing, b_sensing)
        elif self.qos_type == 'tracking':
            objective = self._build_tracking_objective_bw(p_sensing, b_sensing)
        else:
            raise ValueError(f"Unknown QoS type: {self.qos_type}")
        
        # Constraints
        constraints = [
            cp.sum(b) == self.system.params.B_total,  # Bandwidth budget
        ]
        
        # Communication rate constraints
        for k in range(K):
            snr_k = (p_comm[k] * self.system.beta_comm[k]) / \
                    (self.system.N0 * b_comm[k])
            constraints.append(
                b_comm[k] * cp.log(1 + snr_k) / np.log(2) >= Gamma_c
            )
        
        # ISAC user rate constraint
        for l in range(L):
            snr_l = (p_isac[l] * self.system.beta_isac[l]) / \
                    (self.system.N0 * b_isac[l])
            constraints.append(
                b_isac[l] * cp.log(1 + snr_l) / np.log(2) >= Gamma_c
            )
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            solver = self._get_solver()
            problem.solve(solver=solver, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                return np.maximum(b.value, 0)
            else:
                warnings.warn(f"Bandwidth subproblem status: {problem.status}")
                return np.ones(M) * self.system.params.B_total / M
        except Exception as e:
            warnings.warn(f"Bandwidth subproblem failed: {e}")
            return np.ones(M) * self.system.params.B_total / M
    
    def _build_detection_objective(self, p_sensing: cp.Variable,
                                    b_sensing: np.ndarray) -> cp.Expression:
        """
        Build detection QoS objective (for power optimization).
        
        Uses approximation: P_D ≈ 1 - exp(-SNR)
        """
        Q = self.system.params.Q
        
        # SNR approximation for convex optimization
        snr_terms = []
        for q in range(Q):
            snr_q = (p_sensing[q] * self.system.beta_sensing[q] * self.system.rcs[q]) / \
                    (self.system.N0 * b_sensing[q])
            snr_terms.append(snr_q)
        
        if self.fairness_type == FairnessType.MAXMIN:
            # Max-min: maximize minimum SNR
            return cp.minimum(*snr_terms)
        else:
            # Sum: maximize total SNR
            return cp.sum(snr_terms)
    
    def _build_detection_objective_bw(self, p_sensing: np.ndarray,
                                       b_sensing: cp.Variable) -> cp.Expression:
        """Build detection QoS objective for bandwidth optimization."""
        Q = self.system.params.Q
        
        snr_terms = []
        for q in range(Q):
            snr_q = (p_sensing[q] * self.system.beta_sensing[q] * self.system.rcs[q]) / \
                    (self.system.N0 * b_sensing[q])
            snr_terms.append(snr_q)
        
        if self.fairness_type == FairnessType.MAXMIN:
            return cp.minimum(*snr_terms)
        else:
            return cp.sum(snr_terms)
    
    def _build_localization_objective(self, p_sensing: cp.Variable,
                                       b_sensing: np.ndarray) -> cp.Expression:
        """
        Build localization QoS objective (for power optimization).
        
        Maximize ρ = w_d / CRB_d + w_θ / CRB_θ
        CRB_d ∝ 1/(p * b²), CRB_θ ∝ 1/p
        """
        Q = self.system.params.Q
        
        rho_terms = []
        for q in range(Q):
            snr_q = (p_sensing[q] * self.system.beta_sensing[q] * self.system.rcs[q]) / \
                    (self.system.N0 * b_sensing[q])
            # CRB inverse proportional to SNR * B² (range) and SNR (angle)
            crb_d_inv = 8 * np.pi**2 * b_sensing[q]**2 * snr_q / (3e8)**2
            crb_theta_inv = snr_q * self.system.params.Nt * (self.system.params.Nt**2 - 1) * np.pi**2 / 6
            rho_q = self.localization_qos.w_d * crb_d_inv + self.localization_qos.w_theta * crb_theta_inv
            rho_terms.append(rho_q)
        
        if self.fairness_type == FairnessType.MAXMIN:
            return cp.minimum(*rho_terms)
        else:
            return cp.sum(rho_terms)
    
    def _build_localization_objective_bw(self, p_sensing: np.ndarray,
                                          b_sensing: cp.Variable) -> cp.Expression:
        """Build localization QoS objective for bandwidth optimization."""
        Q = self.system.params.Q
        
        rho_terms = []
        for q in range(Q):
            snr_q = (p_sensing[q] * self.system.beta_sensing[q] * self.system.rcs[q]) / \
                    (self.system.N0 * b_sensing[q])
            crb_d_inv = 8 * np.pi**2 * b_sensing[q]**2 * snr_q / (3e8)**2
            crb_theta_inv = snr_q * self.system.params.Nt * (self.system.params.Nt**2 - 1) * np.pi**2 / 6
            rho_q = self.localization_qos.w_d * crb_d_inv + self.localization_qos.w_theta * crb_theta_inv
            rho_terms.append(rho_q)
        
        if self.fairness_type == FairnessType.MAXMIN:
            return cp.minimum(*rho_terms)
        else:
            return cp.sum(rho_terms)
    
    def _build_tracking_objective(self, p_sensing: cp.Variable,
                                   b_sensing: np.ndarray) -> cp.Expression:
        """
        Build tracking QoS objective (for power optimization).
        
        Minimize Σ trace(PCRB) ≈ maximize Σ (p * b²)
        """
        Q = self.system.params.Q
        
        trace_terms = []
        for q in range(Q):
            # Approximation: trace(PCRB) ∝ 1/(p * b²)
            trace_inv = p_sensing[q] * b_sensing[q]**2
            trace_terms.append(trace_inv)
        
        if self.fairness_type == FairnessType.MAXMIN:
            return cp.minimum(*trace_terms)
        else:
            return cp.sum(trace_terms)
    
    def _build_tracking_objective_bw(self, p_sensing: np.ndarray,
                                      b_sensing: cp.Variable) -> cp.Expression:
        """Build tracking QoS objective for bandwidth optimization."""
        Q = self.system.params.Q
        
        trace_terms = []
        for q in range(Q):
            trace_inv = p_sensing[q] * b_sensing[q]**2
            trace_terms.append(trace_inv)
        
        if self.fairness_type == FairnessType.MAXMIN:
            return cp.minimum(*trace_terms)
        else:
            return cp.sum(trace_terms)
    
    def solve(self, Gamma_c: float = 1.0,
              initial_p: Optional[np.ndarray] = None,
              initial_b: Optional[np.ndarray] = None) -> AOResult:
        """
        Solve the ISAC resource allocation problem using AO (Algorithm 1).
        
        Parameters
        ----------
        Gamma_c : float
            Communication rate threshold (bps/Hz)
        initial_p : np.ndarray, optional
            Initial power allocation
        initial_b : np.ndarray, optional
            Initial bandwidth allocation
            
        Returns
        -------
        result : AOResult
            Optimization result
        """
        M = self.system.total_objects
        
        # Initialize (Step 1)
        if initial_b is None:
            b = self._initialize_bandwidth()
        else:
            b = initial_b.copy()
        
        if initial_p is None:
            p = np.ones(M) * self.system.params.P_total / M
        else:
            p = initial_p.copy()
        
        # Track convergence
        prev_objective = -np.inf
        converged = False
        
        for iteration in range(self.max_iter):
            # Step 2: Solve for power given bandwidth
            p = self._solve_power_subproblem(b, Gamma_c)
            
            # Step 3: Solve for bandwidth given power
            b = self._solve_bandwidth_subproblem(p, Gamma_c)
            
            # Check convergence
            current_objective = self._compute_current_objective(p, b)
            
            if abs(current_objective - prev_objective) < self.tol:
                converged = True
                break
            
            prev_objective = current_objective
        
        # Compute final metrics
        p_sensing = p[:self.system.params.Q]
        b_sensing = b[:self.system.params.Q]
        p_comm = p[self.system.params.Q:self.system.params.Q+self.system.params.K]
        b_comm = b[self.system.params.Q:self.system.params.Q+self.system.params.K]
        
        detection_probs = None
        localization_rho = None
        tracking_pcrb = None
        comm_rates = None
        
        if self.qos_type == 'detection':
            detection_probs = self.detection_qos.compute_detection_probability(p_sensing, b_sensing)
        elif self.qos_type == 'localization':
            localization_rho = self.localization_qos.compute_crb_combined(p_sensing, b_sensing)
        elif self.qos_type == 'tracking':
            tracking_pcrb = self.tracking_qos.compute_pcrb(p_sensing, b_sensing)
        
        comm_rates = self.comm_rate.compute_rate(p_comm, b_comm, 'comm')
        
        return AOResult(
            p=p,
            b=b,
            objective=prev_objective,
            iterations=iteration + 1,
            converged=converged,
            detection_probs=detection_probs,
            localization_rho=localization_rho,
            tracking_pcrb=tracking_pcrb,
            comm_rates=comm_rates
        )
    
    def _compute_current_objective(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute current objective value for convergence check.
        
        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation
            
        Returns
        -------
        objective : float
            Current objective value
        """
        Q = self.system.params.Q
        p_sensing = p[:Q]
        b_sensing = b[:Q]
        
        if self.qos_type == 'detection':
            P_D = self.detection_qos.compute_detection_probability(p_sensing, b_sensing)
            if self.fairness_type == FairnessType.MAXMIN:
                return np.min(P_D)
            else:
                return np.sum(P_D)
        elif self.qos_type == 'localization':
            rho = self.localization_qos.compute_crb_combined(p_sensing, b_sensing)
            if self.fairness_type == FairnessType.MAXMIN:
                return np.min(rho)
            else:
                return np.sum(rho)
        elif self.qos_type == 'tracking':
            return -self.tracking_qos.compute_pcrb_trace(p_sensing, b_sensing)
        
        return 0.0
    
    def solve_multiple_qos(self, Gamma_c: float = 1.0) -> Dict[str, AOResult]:
        """
        Solve for all three QoS types.
        
        Parameters
        ----------
        Gamma_c : float
            Communication rate threshold
            
        Returns
        -------
        results : dict
            Results for each QoS type
        """
        results = {}
        
        for qos_type in ['detection', 'localization', 'tracking']:
            self.qos_type = qos_type
            results[qos_type] = self.solve(Gamma_c)
        
        return results
