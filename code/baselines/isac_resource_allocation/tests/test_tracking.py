"""
Tests for Tracking QoS module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.system_model import ISACSystem
from src.tracking_qos import TrackingQoS


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def tracking_qos(system):
    """Create Tracking QoS module."""
    return TrackingQoS(system, dt=0.1, process_noise_std=0.5, measurement_noise_std=0.1)


def test_pcrb_recursive(tracking_qos):
    """Test that PCRB recursion is correct."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    # Compute PCRB without prior (large initial uncertainty)
    pcrb1 = tracking_qos.compute_pcrb(p, b, prior_pcrb=None)
    
    # Use first PCRB as prior for second computation
    pcrb2 = tracking_qos.compute_pcrb(p, b, prior_pcrb=pcrb1)
    
    # With prior information, PCRB should decrease (better estimation)
    trace1 = np.sum([np.trace(pcrb1[q]) for q in range(pcrb1.shape[0])])
    trace2 = np.sum([np.trace(pcrb2[q]) for q in range(pcrb2.shape[0])])
    
    assert trace2 <= trace1, \
        f"PCRB should decrease with prior information: trace1={trace1}, trace2={trace2}"


def test_pcrb_trace(tracking_qos):
    """Test PCRB trace computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    pcrb = tracking_qos.compute_pcrb(p, b)
    trace = tracking_qos.compute_pcrb_trace(p, b)
    
    # Manual trace computation
    trace_manual = np.sum([np.trace(pcrb[q]) for q in range(pcrb.shape[0])])
    
    assert np.isclose(trace, trace_manual), \
        f"PCRB trace should match manual computation: {trace} vs {trace_manual}"


def test_pcrb_position_trace(tracking_qos):
    """Test position-only PCRB trace."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    pos_trace = tracking_qos.compute_pcrb_position_trace(p, b)
    
    # Position trace should be positive
    assert np.all(pos_trace > 0), f"Position trace should be positive: {pos_trace}"
    
    # Should be less than full trace
    full_trace = tracking_qos.compute_pcrb_trace(p, b)
    assert np.sum(pos_trace) <= full_trace, \
        f"Position trace should be ≤ full trace: {np.sum(pos_trace)} vs {full_trace}"


def test_tracking_error_bound(tracking_qos):
    """Test tracking error bound computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    teb = tracking_qos.compute_tracking_error_bound(p, b)
    
    # TEB should be positive
    assert np.all(teb > 0), f"Tracking error bound should be positive: {teb}"
    
    # TEB should equal sqrt(trace(PCRB))
    pcrb = tracking_qos.compute_pcrb(p, b)
    trace_pcrb = np.array([np.trace(pcrb[q]) for q in range(pcrb.shape[0])])
    
    np.testing.assert_allclose(teb, np.sqrt(trace_pcrb), rtol=1e-5)


def test_update_target_states(tracking_qos):
    """Test target state update."""
    initial_positions = [state.position.copy() for state in tracking_qos.target_states]
    
    # Update states
    tracking_qos.update_target_states()
    
    # States should have changed (due to dynamics + noise)
    for q in range(len(tracking_qos.target_states)):
        assert not np.allclose(tracking_qos.target_states[q].position, 
                              initial_positions[q]), \
            f"Target {q} position should have changed after update"


def test_simulate_tracking(tracking_qos):
    """Test tracking simulation over multiple steps."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    pcrb_history, trace_history = tracking_qos.simulate_tracking(p, b, num_steps=10)
    
    # Should have 10 time steps
    assert len(pcrb_history) == 10, f"Expected 10 PCRB snapshots, got {len(pcrb_history)}"
    assert len(trace_history) == 10, f"Expected 10 trace values, got {len(trace_history)}"
    
    # Traces should generally decrease over time (learning)
    # But may not be strictly monotonic due to target motion
    assert trace_history[-1] <= trace_history[0] * 2, \
        f"Trace should not increase too much: initial={trace_history[0]}, final={trace_history[-1]}"


def test_transition_matrix(tracking_qos):
    """Test state transition matrix."""
    F = tracking_qos._get_transition_matrix()
    
    # Should be 4x4 for constant velocity model
    assert F.shape == (4, 4), f"Transition matrix should be 4x4, got {F.shape}"
    
    # Should be upper triangular
    assert np.allclose(F, np.triu(F)), "Transition matrix should be upper triangular"
    
    # Diagonal should be 1
    assert np.allclose(np.diag(F), 1), "Diagonal elements should be 1"


def test_process_noise_covariance(tracking_qos):
    """Test process noise covariance."""
    Q_proc = tracking_qos._get_process_noise_cov()
    
    # Should be 4x4
    assert Q_proc.shape == (4, 4), f"Process noise covariance should be 4x4, got {Q_proc.shape}"
    
    # Should be symmetric positive semi-definite
    assert np.allclose(Q_proc, Q_proc.T), "Process noise covariance should be symmetric"
    eigenvalues = np.linalg.eigvalsh(Q_proc)
    assert np.all(eigenvalues >= -1e-10), \
        f"Process noise covariance should be PSD: eigenvalues={eigenvalues}"


def test_measurement_jacobian(tracking_qos):
    """Test measurement Jacobian."""
    state = np.array([10.0, 20.0, 1.0, 0.5])
    H = tracking_qos._compute_measurement_jacobian(state)
    
    # Should be 2x4 (range, angle measurements)
    assert H.shape == (2, 4), f"Measurement Jacobian should be 2x4, got {H.shape}"
    
    # Range measurement should depend on position
    assert H[0, 0] != 0 or H[0, 1] != 0, "Range measurement should depend on position"
    
    # Angle measurement should depend on position
    assert H[1, 0] != 0 or H[1, 1] != 0, "Angle measurement should depend on position"
