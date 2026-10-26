import numpy as np
import pytest
from src.monitoring.drift import calculate_psi

def test_same_distribution_has_zero_psi():
    a = np.linspace(.1, .9, 100)
    assert calculate_psi(a,a) == pytest.approx(0)

def test_outside_scores_are_counted():
    a = np.linspace(.1,.9,100)
    assert calculate_psi(a,np.ones(100)) > 0.25

def test_constant_baseline_and_empty_inputs():
    assert np.isfinite(calculate_psi(np.ones(100)*.5,np.ones(100)*.8))
    with pytest.raises(ValueError):calculate_psi([],[])
