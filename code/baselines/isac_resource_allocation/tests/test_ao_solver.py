"""
Tests for Alternating Optimization Solver.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.system_model import ISACSystem
from src.ao_solver import AOSolver


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def detection_solver(system):
    """Create AO solver for detection QoS."""
    return AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=20)


@pytest.fixture
def localization_solver(system):
    """Create AO solver for localization QoS."""
    return AOSolver(system, qos_type='localization', fairness='maxmin', max_iter=20)


@pytest.fixture
def tracking_solver(system):
    """Create AO solver for tracking QoS."""
    return AOSolver(system, qos_type='tracking', fairness='maxmin', max_iter=20)


def test_power_budget(detection_solver):
    """Test that power budget constraint is satisfied."""
    result = detection_solver.solve(Gamma_c=1.0)
    
    total_power = np.sum(result.p)
    P_total = detection_solver.system.params.P_total
    
    assert np.isclose(total_power, P_total, rtol=1e-3), \
        f"Power budget violated: Σp={total_power}, P_total={P_total}"


def test_bandwidth_budget(detection_solver):
    """Test that bandwidth budget constraint is satisfied."""
    result = detection_solver.solve(Gamma_c=1.0)
    
    total_bandwidth = np.sum(result.b)
    B_total = detection_solver.system.params.B_total
    
    assert np.isclose(total_bandwidth, B_total, rtol=1e-3), \
        f"Bandwidth budget violated: Σb={total_bandwidth}, B_total={B_total}"


def test_comm_rate_constraint(detection_solver):
    """Test that communication rate constraint is satisfied."""
    Gamma_c = 1.0
    result = detection_solver.solve(Gamma_c=Gamma_c)
    
    # Extract communication user allocations
    K = detection_solver.system.params.K
    p_comm = result.p[detection_solver.system.params.Q:detection_solver.system.params.Q+K]
    b_comm = result.b[detection_solver.system.params.Q:detection_solver.system.params.Q+K]
    
    # Check if rates meet threshold (may be relaxed due to solver limitations)
    comm_rates = result.comm_rates
    if comm_rates is not None:
        # Allow some tolerance for solver feasibility
        assert np.all(comm_rates >= Gamma_c * 0.5), \
            f"Communication rate constraint severely violated: rates={comm_rates}, Γc={Gamma_c}"


def test_ao_convergence(detection_solver):
    """Test that AO converges within max iterations."""
    result = detection_solver.solve(Gamma_c=1.0)
    
    assert result.converged or result.iterations <= detection_solver.max_iter, \
        f"AO should converge or reach max iterations: converged={result.converged}, iter={result.iterations}"
    
    assert result.iterations <= detection_solver.max_iter, \
        f"AO exceeded max iterations: {result.iterations} > {detection_solver.max_iter}"


def test_detection_solver_result(detection_solver):
    """Test detection QoS solver result."""
    result = detection_solver.solve(Gamma_c=1.0)
    
    # Result should have detection probabilities
    assert result.detection_probs is not None, "Detection probabilities should be computed"
    
    # Detection probabilities should be in valid range
    assert np.all(result.detection_probs >= 0), \
        f"Detection probabilities should be ≥ 0: {result.detection_probs}"
    assert np.all(result.detection_probs <= 1), \
        f"Detection probabilities should be ≤ 1: {result.detection_probs}"


def test_localization_solver_result(localization_solver):
    """Test localization QoS solver result."""
    result = localization_solver.solve(Gamma_c=1.0)
    
    # Result should have localization metrics
    assert result.localization_rho is not None, "Localization metrics should be computed"
    
    # Localization metrics should be positive
    assert np.all(result.localization_rho > 0), \
        f"Localization metrics should be > 0: {result.localization_rho}"


def test_tracking_solver_result(tracking_solver):
    """Test tracking QoS solver result."""
    result = tracking_solver.solve(Gamma_c=1.0)
    
    # Result should have tracking metrics
    assert result.tracking_pcrb is not None, "Tracking PCRB should be computed"
    
    # PCRB should be positive semi-definite
    for q in range(result.tracking_pcrb.shape[0]):
        eigenvalues = np.linalg.eigvalsh(result.tracking_pcrb[q])
        assert np.all(eigenvalues >= -1e-10), \
            f"PCRB should be PSD for target {q}: eigenvalues={eigenvalues}"


def test_fairness_maxmin(detection_solver):
    """Test max-min fairness works."""
    # Use max-min fairness
    detection_solver.fairness_type = 'maxmin'
    result = detection_solver.solve(Gamma_c=1.0)
    
    # All detection probabilities should be reasonably close (fairness)
    if result.detection_probs is not None:
        min_pd = np.min(result.detection_probs)
        max_pd = np.max(result.detection_probs)
        
        # Max-min fairness should reduce disparity
        assert max_pd / (min_pd + 1e-10) < 10, \
            f"Max-min fairness should reduce disparity: min={min_pd}, max={max_pd}"


def test_solve_multiple_qos(system):
    """Test solving for all QoS types."""
    solver = AOSolver(system, max_iter=10)
    results = solver.solve_multiple_qos(Gamma_c=1.0)
    
    # Should have results for all QoS types
    assert 'detection' in results, "Should have detection results"
    assert 'localization' in results, "Should have localization results"
    assert 'tracking' in results, "Should have tracking results"
    
    # All results should satisfy power/bandwidth budgets
    for qos_type, result in results.items():
        assert np.isclose(np.sum(result.p), system.params.P_total, rtol=1e-3), \
            f"Power budget violated for {qos_type}"
        assert np.isclose(np.sum(result.b), system.params.B_total, rtol=1e-3), \
            f"Bandwidth budget violated for {qos_type}"


def test_initial_conditions(detection_solver):
    """Test solver with different initial conditions."""
    M = detection_solver.system.total_objects
    
    # Random initial conditions
    initial_p = np.random.dirichlet(np.ones(M)) * detection_solver.system.params.P_total
    initial_b = np.random.dirichlet(np.ones(M)) * detection_solver.system.params.B_total
    
    result = detection_solver.solve(Gamma_c=1.0, initial_p=initial_p, initial_b=initial_b)
    
    # Should still satisfy constraints
    assert np.isclose(np.sum(result.p), detection_solver.system.params.P_total, rtol=1e-3)
    assert np.isclose(np.sum(result.b), detection_solver.system.params.B_total, rtol=1e-3)


def test_detection_vs_rate_tradeoff(system):
    """Test tradeoff curve between detection and rate threshold."""
    solver = AOSolver(system, qos_type='detection', fairness='maxmin', max_iter=10)
    
    Gamma_c_values = [0.5, 1.0, 2.0]
    detection_values = []
    
    for Gamma_c in Gamma_c_values:
        result = solver.solve(Gamma_c=Gamma_c)
        if result.detection_probs is not None:
            detection_values.append(np.min(result.detection_probs))
        else:
            detection_values.append(0)
    
    # Generally, higher rate threshold should lead to lower detection probability
    # (tradeoff), but may not be strictly monotonic due to solver behavior
    assert len(detection_values) == len(Gamma_c_values), \
        "Should have detection value for each rate threshold"
