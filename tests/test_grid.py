import numpy as np
import pytest

from cosmociety.grid import radial_grid


def test_radial_grid_spans_center_to_surface():
    r = radial_grid(50)
    assert r[0] == 0.0
    assert r[-1] == 1.0
    assert len(r) == 50


def test_radial_grid_is_uniformly_spaced():
    r = radial_grid(21)
    dr = np.diff(r)
    assert np.allclose(dr, dr[0])


def test_radial_grid_rejects_too_few_points():
    with pytest.raises(ValueError):
        radial_grid(2)
