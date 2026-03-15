"""Basic tests for DOA array partitioning - module import verification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_system_model_import():
    import system_model
    assert hasattr(system_model, '__file__')

def test_doa_crb_import():
    import doa_crb
    assert hasattr(doa_crb, 'compute_crb')

def test_array_partition_import():
    import array_partition
    assert hasattr(array_partition, '__file__')

def test_beamforming_import():
    import beamforming
    assert hasattr(beamforming, '__file__')

def test_admm_solver_import():
    import admm_solver
    assert hasattr(admm_solver, '__file__')

def test_mm_solver_import():
    import mm_solver
    assert hasattr(mm_solver, '__file__')
