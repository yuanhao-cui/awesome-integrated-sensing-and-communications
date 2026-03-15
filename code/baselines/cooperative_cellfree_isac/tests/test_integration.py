"""Basic tests for cooperative cell-free ISAC - module import verification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_system_model_import():
    import system_model
    assert hasattr(system_model, '__file__')

def test_mode_selection_import():
    import mode_selection
    assert hasattr(mode_selection, '__file__')

def test_beamforming_import():
    import beamforming
    assert hasattr(beamforming, '__file__')

def test_cooperative_import():
    import cooperative
    assert hasattr(cooperative, '__file__')

def test_ao_solver_import():
    import ao_solver
    assert hasattr(ao_solver, '__file__')
