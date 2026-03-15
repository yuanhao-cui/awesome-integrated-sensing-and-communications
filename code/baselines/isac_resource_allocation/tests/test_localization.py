"""
Tests for Localization QoS module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.system_model import ISACSystem
from src.localization_qos import LocalizationQoS


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def localization_qos(system):
    """Create Localization QoS module."""
    return LocalizationQoS(system, w_d=1.0, w_theta=1.0)


def test_crb_decreases_with_power(localization_qos):
    """Test that CRB decreases with more power."""
    b = np.array([30e6, 30e6, 25e6])
    
    p_low = np.array([5.0, 5.0, 5.0])
    p_high = np.array([20.0, 20.0, 20.0])
    
    crb_low = localization_qos.compute_crb_range(p_low, b)
    crb_high = localization_qos.compute_crb_range(p_high, b)
    
    # Higher power should give lower CRB (better estimation)
    assert np.all(crb_low >= crb_high), \
        f"CRB should decrease with power: low={crb_low}, high={crb_high}"


def test_crb_decreases_with_bandwidth(localization_qos):
    """Test that range CRB decreases with more bandwidth."""
    p = np.array([10.0, 10.0, 10.0])
    
    b_low = np.array([10e6, 10e6, 10e6])
    b_high = np.array([50e6, 50e6, 50e6])
    
    crb_low = localization_qos.compute_crb_range(p, b_low)
    crb_high = localization_qos.compute_crb_range(p, b_high)
    
    # Higher bandwidth should give lower CRB (better range estimation)
    assert np.all(crb_low >= crb_high), \
        f"Range CRB should decrease with bandwidth: low={crb_low}, high={crb_high}"


def test_crb_angle_with_bandwidth(localization_qos):
    """Test that angle CRB changes with bandwidth (through SNR)."""
    # With fixed power, higher bandwidth means lower power spectral density
    # So SNR decreases with bandwidth, and CRB increases
    p = np.array([10.0, 10.0, 10.0])
    
    b_low = np.array([10e6, 10e6, 10e6])
    b_high = np.array([50e6, 50e6, 50e6])
    
    crb_theta_low = localization_qos.compute_crb_angle(p, b_low)
    crb_theta_high = localization_qos.compute_crb_angle(p, b_high)
    
    # With fixed power, higher bandwidth → lower SNR → higher CRB
    assert np.all(crb_theta_high >= crb_theta_low), \
        f"With fixed power, higher bandwidth should give higher CRB: low={crb_theta_low}, high={crb_theta_high}"


def test_crb_combined(localization_qos):
    """Test combined CRB metric."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    rho = localization_qos.compute_crb_combined(p, b)
    
    # ρ = w_d/CRB_d + w_θ/CRB_θ should be positive
    assert np.all(rho > 0), f"Combined CRB should be positive: {rho}"
    
    # Manual verification
    crb_d = localization_qos.compute_crb_range(p, b)
    crb_theta = localization_qos.compute_crb_angle(p, b)
    rho_manual = 1.0/crb_d + 1.0/crb_theta
    
    np.testing.assert_allclose(rho, rho_manual, rtol=1e-5)


def test_localization_rmse(localization_qos):
    """Test RMSE computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    rmse_range, rmse_angle = localization_qos.compute_localization_rmse(p, b)
    
    # RMSE should be positive
    assert np.all(rmse_range > 0), f"Range RMSE should be positive: {rmse_range}"
    assert np.all(rmse_angle > 0), f"Angle RMSE should be positive: {rmse_angle}"
    
    # RMSE should equal sqrt(CRB)
    crb_d = localization_qos.compute_crb_range(p, b)
    crb_theta = localization_qos.compute_crb_angle(p, b)
    
    np.testing.assert_allclose(rmse_range, np.sqrt(crb_d), rtol=1e-5)
    np.testing.assert_allclose(rmse_angle, np.sqrt(crb_theta), rtol=1e-5)


def test_objective_sum(localization_qos):
    """Test sum objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    obj = localization_qos.compute_objective_sum(p, b)
    rho = localization_qos.compute_crb_combined(p, b)
    
    assert np.isclose(obj, np.sum(rho)), \
        f"Sum objective should equal sum(rho): {obj} vs {np.sum(rho)}"


def test_objective_proportional_fairness(localization_qos):
    """Test proportional fairness objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    obj = localization_qos.compute_objective_proportional_fairness(p, b)
    rho = localization_qos.compute_crb_combined(p, b)
    
    assert np.isclose(obj, np.sum(np.log(rho + 1e-10))), \
        f"Proportional fairness should equal sum(log(rho)): {obj}"


def test_fisher_information_matrix(localization_qos):
    """Test Fisher Information Matrix computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])
    
    fim = localization_qos.compute_fim(p, b)
    
    # FIM should be positive semi-definite
    for q in range(fim.shape[0]):
        eigenvalues = np.linalg.eigvalsh(fim[q])
        assert np.all(eigenvalues >= -1e-10), \
            f"FIM should be PSD for target {q}: eigenvalues={eigenvalues}"
    
    # Diagonal elements should be positive
    for q in range(fim.shape[0]):
        assert fim[q, 0, 0] > 0, f"FIM[0,0] should be positive for target {q}"
        assert fim[q, 1, 1] > 0, f"FIM[1,1] should be positive for target {q}"


def test_validate_localization_performance(localization_qos):
    """Test localization performance validation."""
    # High resources should meet performance requirements
    p_high = np.array([30.0, 30.0, 30.0])
    b_high = np.array([90e6, 90e6, 90e6])
    
    valid = localization_qos.validate_localization_performance(
        p_high, b_high, max_range_error=100.0, max_angle_error=1.0)
    assert valid, "High resources should meet relaxed performance requirements"
    
    # Low resources should not meet strict requirements
    p_low = np.array([0.1, 0.1, 0.1])
    b_low = np.array([1e6, 1e6, 1e6])
    
    valid = localization_qos.validate_localization_performance(
        p_low, b_low, max_range_error=0.01, max_angle_error=0.001)
    assert not valid, "Low resources should not meet strict performance requirements"
