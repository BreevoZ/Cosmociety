import numpy as np
import pytest

from cosmociety.opacity import opacity_from_state, radiative_diffusivity_from_state


def test_opacity_from_state_matches_analytic_power_law():
    # kappa = (rho / rho_surface)^q * T^-p, with rho_surface = density[-1].
    density = np.array([2.0, 1.0])
    temperature = np.array([2.0, 1.0])
    kappa = opacity_from_state(
        temperature=temperature,
        density=density,
        opacity_temperature_power=2.0,
        opacity_density_power=1.0,
    )
    expected = np.array([2.0**1 * 2.0**-2, 1.0**1 * 1.0**-2])
    assert kappa == pytest.approx(expected)


def test_radiative_diffusivity_is_scale_over_opacity():
    density = np.array([1.0, 1.0])
    temperature = np.array([2.0, 4.0])
    D = radiative_diffusivity_from_state(
        temperature=temperature,
        density=density,
        diffusivity_scale=10.0,
        opacity_temperature_power=1.0,
        opacity_density_power=0.0,
        diffusivity_floor=1e-8,
    )
    kappa = opacity_from_state(
        temperature=temperature,
        density=density,
        opacity_temperature_power=1.0,
        opacity_density_power=0.0,
    )
    assert D == pytest.approx(10.0 / kappa)


def test_radiative_diffusivity_is_floored():
    # T is tiny and opacity_temperature_power is large, so kappa = T^-p is
    # huge and D0/kappa would underflow toward zero without the floor.
    density = np.array([1.0])
    temperature = np.array([1e-6])
    D = radiative_diffusivity_from_state(
        temperature=temperature,
        density=density,
        diffusivity_scale=1.0,
        opacity_temperature_power=10.0,
        opacity_density_power=0.0,
        diffusivity_floor=1e-3,
    )
    assert D[0] == pytest.approx(1e-3)


def test_radiative_diffusivity_above_floor_ignores_floor():
    density = np.array([1.0])
    temperature = np.array([1.0])
    D = radiative_diffusivity_from_state(
        temperature=temperature,
        density=density,
        diffusivity_scale=1.0,
        opacity_temperature_power=1.0,
        opacity_density_power=0.0,
        diffusivity_floor=1e-8,
    )
    assert D[0] == pytest.approx(1.0)
