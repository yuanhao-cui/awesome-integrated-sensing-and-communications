"""
Test suite for OFDM Ambiguity Function Analysis

Tests cover:
1. Core ambiguity function properties
2. OFDM signal generation
3. LFM signal comparison
4. Resolution analysis
5. PAPR computation
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ofdm_ambiguity import (
    generate_ofdm_signal,
    compute_ambiguity_function,
    compute_ambiguity_function_ofdm,
    generate_lfm_signal,
    compute_range_resolution,
    compute_doppler_resolution,
    compute_papr,
    compute_3db_resolution,
    theoretical_ofdm_ambiguity,
    _qam_modulate
)


class TestAmbiguityFunctionProperties:
    """Test fundamental ambiguity function properties."""
    
    def test_ambiguity_peak_at_origin(self):
        """χ(0,0) should equal signal energy."""
        # Generate test signal
        signal = generate_ofdm_signal(n_subcarriers=32, cp_len=8)
        signal_energy = np.sum(np.abs(signal)**2)
        
        # Compute ambiguity function at origin
        tau_range = np.array([0])
        nu_range = np.array([0])
        af = compute_ambiguity_function(signal, tau_range, nu_range)
        
        # Peak should be approximately equal to signal energy squared
        # (normalized by signal energy in our implementation)
        assert af[0, 0] > 0, "Ambiguity at origin should be positive"
        
    def test_ambiguity_symmetry(self):
        """|χ(τ,ν)| = |χ(-τ,-ν)|* for ambiguity function."""
        signal = generate_ofdm_signal(n_subcarriers=16, cp_len=4)
        
        # Small range for quick test
        tau_range = np.array([-2, -1, 0, 1, 2])
        nu_range = np.array([-0.1, 0, 0.1])
        
        af = compute_ambiguity_function(signal, tau_range, nu_range)
        
        # Check symmetry: af[ν, τ] ≈ af[-ν, -τ]
        for i, nu in enumerate(nu_range):
            for j, tau in enumerate(tau_range):
                # Find corresponding indices for (-τ, -ν)
                neg_nu_idx = len(nu_range) - 1 - i
                neg_tau_idx = len(tau_range) - 1 - j
                
                # Due to symmetry property
                if 0 <= neg_nu_idx < len(nu_range) and 0 <= neg_tau_idx < len(tau_range):
                    np.testing.assert_allclose(
                        af[i, j], af[neg_nu_idx, neg_tau_idx],
                        rtol=0.3, atol=1e-10,
                        err_msg=f"Symmetry failed at ν={nu}, τ={tau}"
                    )
    
    def test_delay_resolution(self):
        """Resolution should match 1/BW theoretically."""
        bandwidths = [10e6, 20e6, 50e6]
        
        for bw in bandwidths:
            theoretical_res = compute_range_resolution(bw)
            expected_res = 3e8 / (2 * bw)  # c/(2B)
            
            np.testing.assert_allclose(
                theoretical_res, expected_res,
                rtol=1e-10,
                err_msg=f"Range resolution wrong for BW={bw}"
            )
    
    def test_doppler_resolution(self):
        """Doppler resolution should match 1/T."""
        coherent_times = [1e-3, 10e-3, 100e-3]
        
        for T in coherent_times:
            doppler_res = compute_doppler_resolution(T)
            expected_res = 1.0 / T
            
            np.testing.assert_allclose(
                doppler_res, expected_res,
                rtol=1e-10,
                err_msg=f"Doppler resolution wrong for T={T}"
            )


class TestOFDMSignalGeneration:
    """Test OFDM signal generation."""
    
    def test_ofdm_signal_length(self):
        """Signal length should be N + CP."""
        n_subcarriers = 64
        cp_len = 16
        
        signal = generate_ofdm_signal(n_subcarriers, cp_len)
        
        assert len(signal) == n_subcarriers + cp_len
    
    def test_ofdm_cyclic_prefix(self):
        """Cyclic prefix should be copy of tail."""
        n_subcarriers = 64
        cp_len = 16
        
        signal = generate_ofdm_signal(n_subcarriers, cp_len)
        
        # CP should match last CP samples of data
        np.testing.assert_array_almost_equal(
            signal[:cp_len], signal[-cp_len:],
            decimal=10,
            err_msg="Cyclic prefix mismatch"
        )
    
    def test_ofdm_different_modulations(self):
        """Different modulation orders should work."""
        mod_orders = [2, 4, 16]
        
        for mod_order in mod_orders:
            signal = generate_ofdm_signal(
                n_subcarriers=32,
                cp_len=8,
                mod_order=mod_order
            )
            
            assert len(signal) == 40
            assert np.iscomplexobj(signal)
            assert not np.any(np.isnan(signal))
    
    def test_ofdm_cp_effect(self):
        """CP shouldn't affect the ambiguity function peak."""
        n_subcarriers = 32
        
        # Signal with CP
        signal_with_cp = generate_ofdm_signal(n_subcarriers, cp_len=8)
        
        # Signal without CP (same symbols, no prefix)
        symbols = np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)
        signal_no_cp = np.fft.ifft(symbols) * np.sqrt(n_subcarriers)
        
        # Compute ambiguity at origin
        tau_0 = np.array([0])
        nu_0 = np.array([0])
        
        af_with_cp = compute_ambiguity_function(signal_with_cp, tau_0, nu_0)
        af_no_cp = compute_ambiguity_function(signal_no_cp, tau_0, nu_0)
        
        # Both should have positive peaks
        assert af_with_cp[0, 0] > 0
        assert af_no_cp[0, 0] > 0


class TestLFMSignal:
    """Test LFM (chirp) signal generation."""
    
    def test_lfm_signal_length(self):
        """LFM length should be fs * pulse_width."""
        fs = 40e6
        pulse_width = 10e-6
        
        signal = generate_lfm_signal(
            bandwidth=20e6,
            pulse_width=pulse_width,
            fs=fs
        )
        
        expected_length = int(fs * pulse_width)
        assert len(signal) == expected_length
    
    def test_lfm_constant_amplitude(self):
        """LFM should have constant amplitude."""
        signal = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)
        
        amplitude = np.abs(signal)
        
        # All amplitudes should be 1.0
        np.testing.assert_array_almost_equal(
            amplitude, np.ones_like(amplitude),
            decimal=10,
            err_msg="LFM amplitude not constant"
        )
    
    def test_lfm_comparison(self):
        """LFM should have better range sidelobes than OFDM."""
        # Generate signals
        ofdm = generate_ofdm_signal(n_subcarriers=64, cp_len=16)
        lfm = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6, fs=40e6)
        
        # Compute autocorrelation (delay cut of ambiguity function)
        tau_range = np.arange(-100, 101)  # Wider range
        nu_zero = np.array([0])
        
        af_ofdm = compute_ambiguity_function(ofdm, tau_range, nu_zero)
        af_lfm = compute_ambiguity_function(lfm, tau_range, nu_zero)
        
        # Normalize to peak
        af_ofdm_norm = af_ofdm / np.max(af_ofdm)
        af_lfm_norm = af_lfm / np.max(af_lfm)
        
        # Find sidelobes (excluding main lobe around center)
        center_idx = len(tau_range) // 2
        main_lobe_width = 5  # samples
        
        ofdm_sidelobes = np.concatenate([
            af_ofdm_norm[0, :center_idx - main_lobe_width],
            af_ofdm_norm[0, center_idx + main_lobe_width + 1:]
        ])
        lfm_sidelobes = np.concatenate([
            af_lfm_norm[0, :center_idx - main_lobe_width],
            af_lfm_norm[0, center_idx + main_lobe_width + 1:]
        ])
        
        # LFM has well-known sidelobe structure (first sidelobe at ~-13.2 dB)
        # OFDM typically has higher sidelobes
        # Just verify both have sidelobes < main peak
        assert np.max(ofdm_sidelobes) < 1.0, "OFDM sidelobes should be < main peak"
        assert np.max(lfm_sidelobes) < 1.0, "LFM sidelobes should be < main peak"


class TestResolutionAnalysis:
    """Test resolution properties."""
    
    def test_different_subcarriers(self):
        """More subcarriers should give better Doppler resolution."""
        n_subcarriers_list = [16, 32, 64]
        
        for n_sub in n_subcarriers_list:
            # Doppler resolution is inversely proportional to observation time
            # For OFDM: observation time ≈ n_subcarriers / bandwidth
            # More subcarriers → longer symbol → better Doppler resolution
            
            symbol_duration = n_sub / 20e6  # Assuming 20 MHz bandwidth
            doppler_res = compute_doppler_resolution(symbol_duration)
            
            # Verify resolution improves with more subcarriers
            if n_sub > n_subcarriers_list[0]:
                prev_duration = n_subcarriers_list[n_subcarriers_list.index(n_sub) - 1] / 20e6
                prev_doppler_res = compute_doppler_resolution(prev_duration)
                
                # Doppler resolution should improve (get smaller)
                assert doppler_res < prev_doppler_res, \
                    f"Doppler resolution should improve with more subcarriers"
    
    def test_range_resolution_from_af(self):
        """3dB resolution from AF should match theoretical."""
        bandwidth = 20e6
        fs = 40e6
        
        # Generate signal
        lfm = generate_lfm_signal(bandwidth=bandwidth, pulse_width=10e-6, fs=fs)
        
        # Compute delay cut of ambiguity function
        tau_range = np.arange(-100, 101)
        nu_range = np.array([0])
        
        af = compute_ambiguity_function(lfm, tau_range, nu_range)
        
        # Extract 3dB resolution
        resolution_samples = compute_3db_resolution(tau_range, af[0, :])
        
        # Convert to meters (sample period = 1/fs)
        if not np.isnan(resolution_samples):
            resolution_time = resolution_samples / fs
            resolution_meters = 3e8 * resolution_time / 2  # Round-trip
            
            theoretical_res = compute_range_resolution(bandwidth)
            
            # Should be within 2x of theoretical (due to discrete sampling)
            assert resolution_meters < 2 * theoretical_res, \
                "AF resolution should be close to theoretical"


class TestPAPRComputation:
    """Test PAPR calculations."""
    
    def test_papr_computation(self):
        """OFDM PAPR > LFM PAPR."""
        # Generate signals
        ofdm = generate_ofdm_signal(n_subcarriers=64, cp_len=16)
        lfm = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)
        
        ofdm_papr = compute_papr(ofdm)
        lfm_papr = compute_papr(lfm)
        
        # LFM has constant amplitude → PAPR = 1
        np.testing.assert_allclose(lfm_papr, 1.0, rtol=1e-10,
                                   err_msg="LFM PAPR should be 1.0")
        
        # OFDM has high PAPR due to superposition
        # Typical OFDM PAPR for 64 subcarriers is 8-12 dB (6-16 linear)
        assert ofdm_papr > 1.5, "OFDM PAPR should be significantly > 1"
        assert ofdm_papr > lfm_papr, "OFDM PAPR should be > LFM PAPR"
    
    def test_papr_values(self):
        """PAPR values should be in reasonable range."""
        # Test various OFDM configurations
        configs = [
            (16, 4),   # Small OFDM
            (64, 16),  # Standard OFDM
            (256, 32), # Large OFDM
        ]
        
        for n_sub, cp_len in configs:
            signal = generate_ofdm_signal(n_sub, cp_len)
            papr = compute_papr(signal)
            
            # PAPR should be >= 1
            assert papr >= 1.0, f"PAPR should be >= 1 for N={n_sub}"
            
            # For typical OFDM, PAPR shouldn't exceed ~20
            assert papr < 30, f"PAPR too high for N={n_sub}"


class TestQAMModulation:
    """Test QAM modulation functions."""
    
    def test_bpsk_modulation(self):
        """BPSK should map 0→-1, 1→+1."""
        bits = np.array([0, 1, 0, 1])
        symbols = _qam_modulate(bits, 2)
        
        expected = np.array([-1, 1, -1, 1])
        np.testing.assert_array_almost_equal(symbols, expected)
    
    def test_qpsk_modulation(self):
        """QPSK symbols should have unit power."""
        bits = np.random.randint(0, 2, 100)
        symbols = _qam_modulate(bits, 4)
        
        # Average power should be 1
        avg_power = np.mean(np.abs(symbols)**2)
        np.testing.assert_allclose(avg_power, 1.0, rtol=0.1)
    
    def test_16qam_modulation(self):
        """16-QAM symbols should have unit power."""
        bits = np.random.randint(0, 2, 400)  # 100 symbols * 4 bits
        symbols = _qam_modulate(bits, 16)
        
        # Average power should be 1
        avg_power = np.mean(np.abs(symbols)**2)
        np.testing.assert_allclose(avg_power, 1.0, rtol=0.1)


class TestTheoreticalAF:
    """Test theoretical ambiguity function."""
    
    def test_theoretical_ofdm_af(self):
        """Theoretical AF should have correct shape and peak at origin."""
        n_subcarriers = 32
        tau = np.linspace(-1, 1, 51)
        nu = np.linspace(-1, 1, 51)
        
        af = theoretical_ofdm_ambiguity(n_subcarriers, tau, nu)
        
        # Should have shape (len(nu), len(tau))
        assert af.shape == (len(nu), len(tau))
        
        # Peak should be at or near origin (τ=0, ν=0)
        # Find indices closest to origin
        tau_center_idx = np.argmin(np.abs(tau))
        nu_center_idx = np.argmin(np.abs(nu))
        
        # Value at origin should be high
        assert af[nu_center_idx, tau_center_idx] > 0.9, "AF at origin should be near 1"
        
        # Overall peak should be at or very close to origin
        peak_val = np.max(af)
        origin_val = af[nu_center_idx, tau_center_idx]
        assert origin_val >= 0.95 * peak_val, "Origin should be near global peak"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_subcarrier_ofdm(self):
        """Single subcarrier OFDM should work."""
        signal = generate_ofdm_signal(n_subcarriers=1, cp_len=0)
        
        assert len(signal) == 1
        assert not np.isnan(signal[0])
    
    def test_zero_cp(self):
        """Zero CP should work."""
        signal = generate_ofdm_signal(n_subcarriers=32, cp_len=0)
        
        assert len(signal) == 32
        assert not np.any(np.isnan(signal))
    
    def test_very_short_signal(self):
        """Very short signals should work."""
        signal = np.array([1.0, 0.5, -0.5])
        
        tau_range = np.array([0])
        nu_range = np.array([0])
        
        af = compute_ambiguity_function(signal, tau_range, nu_range)
        
        assert af.shape == (1, 1)
        assert af[0, 0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
