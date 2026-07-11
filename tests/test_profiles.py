import numpy as np
import pytest

from cosmociety.grid import radial_grid
from cosmociety.profiles import core_source, density_profile, initial_temperature


def test_core_source_peaks_at_center_and_decays_outward():
    r = radial_grid(100)
    source = core_source(r)
    assert source[0] == pytest.approx(1.0)
    assert np.all(np.diff(source) <= 1e-12)


def test_initial_temperature_hot_center_cool_surface():
    r = radial_grid(100)
    T = initial_temperature(r, surface_temperature=0.1)
    assert T[0] == pytest.approx(1.0)
    assert T[-1] == pytest.approx(0.1)
    assert np.all(np.diff(T) <= 1e-12)


def test_density_profile_dense_core_light_envelope():
    r = radial_grid(100)
    density = density_profile(r, surface_density=0.05, central_density=1.0, concentration=2.0)
    assert density[0] == pytest.approx(1.0)
    assert density[-1] == pytest.approx(0.05)
    assert np.all(np.diff(density) <= 1e-12)


def test_density_profile_rejects_nonpositive_surface_density():
    r = radial_grid(10)
    with pytest.raises(ValueError):
        density_profile(r, surface_density=0.0)


def test_density_profile_rejects_central_below_surface():
    r = radial_grid(10)
    with pytest.raises(ValueError):
        density_profile(r, surface_density=0.5, central_density=0.1)
