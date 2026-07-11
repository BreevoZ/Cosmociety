import numpy as np
import pytest

from cosmociety.transport import diffusion_step, interface_flux


def test_interface_flux_zero_for_uniform_temperature():
    T = np.full(6, 0.7)
    flux = interface_flux(T, diffusivity=0.05, dr=0.2)
    assert np.allclose(flux, 0.0)


def test_interface_flux_positive_when_temperature_decreases_outward():
    # Positive flux means heat flows outward (toward larger index/radius).
    T = np.linspace(1.0, 0.2, 6)
    flux = interface_flux(T, diffusivity=0.1, dr=0.2)
    assert np.all(flux > 0)
    # dT/dr per interface is -0.16/0.2, so flux = -D * dT/dr = 0.08 everywhere.
    assert flux == pytest.approx(0.08 * np.ones(5))


def test_interface_flux_rejects_mismatched_diffusivity_shape():
    T = np.zeros(5)
    with pytest.raises(ValueError):
        interface_flux(T, diffusivity=np.zeros(4), dr=0.1)


def _symmetric_profile(n, seed=0):
    rng = np.random.default_rng(seed)
    r = np.linspace(0.0, 1.0, n)
    T = 1.0 + 0.5 * np.cos(np.pi * r) + 0.05 * rng.standard_normal(n)
    T[0] = T[1]  # enforce center symmetry, as relax_to_equilibrium always has
    return r, T


@pytest.mark.parametrize("geometry", ["cartesian", "spherical"])
def test_diffusion_step_conserves_energy_with_no_source_or_cooling(geometry):
    n = 12
    r, T = _symmetric_profile(n)
    dr = r[1] - r[0]
    C = 0.3 + 0.7 * (1.0 - r)
    D = 0.05 + 0.02 * np.sin(3 * r) ** 2
    weight = r**2 if geometry == "spherical" else np.ones(n)

    before = np.sum(C[1:] * T[1:] * weight[1:])
    T_after = diffusion_step(
        temperature=T,
        source=np.zeros(n),
        dt=1e-3,
        diffusivity=D,
        heat_capacity=C,
        space_temperature=0.01,
        surface_cooling=0.0,
        dr=dr,
        radius=r,
        geometry=geometry,
    )
    after = np.sum(C[1:] * T_after[1:] * weight[1:])

    # Cell 0 is a mirror of cell 1, not an independently evolving control
    # volume (see "Center symmetry" in transport.diffusion_step), and with no
    # source/cooling the only way energy could leave cells [1:] is through
    # the center interface -- which carries zero flux here because T[0]==T[1].
    assert after == pytest.approx(before, abs=1e-9)


def test_diffusion_step_enforces_center_symmetry():
    n = 8
    r, T = _symmetric_profile(n)
    T_after = diffusion_step(
        temperature=T,
        source=np.zeros(n),
        dt=1e-3,
        diffusivity=0.05,
        heat_capacity=1.0,
        space_temperature=0.01,
        surface_cooling=1.0,
        dr=r[1] - r[0],
        radius=r,
        geometry="spherical",
    )
    assert T_after[0] == T_after[1]


def test_diffusion_step_surface_cooling_reduces_surface_temperature():
    n = 8
    r, T = _symmetric_profile(n)
    kwargs = dict(
        temperature=T,
        source=np.zeros(n),
        dt=1e-3,
        diffusivity=0.05,
        heat_capacity=1.0,
        space_temperature=0.01,
        dr=r[1] - r[0],
        radius=r,
        geometry="spherical",
    )
    no_cooling = diffusion_step(surface_cooling=0.0, **kwargs)
    with_cooling = diffusion_step(surface_cooling=10.0, **kwargs)
    assert with_cooling[-1] < no_cooling[-1]


def test_diffusion_step_clamps_to_space_temperature():
    n = 6
    r = np.linspace(0.0, 1.0, n)
    T = np.full(n, 0.5)
    T[0] = T[1]
    T_after = diffusion_step(
        temperature=T,
        source=np.full(n, -100.0),  # unphysical, but exercises the safety clamp
        dt=1e-2,
        diffusivity=np.zeros(n),
        heat_capacity=1.0,
        space_temperature=0.2,
        surface_cooling=0.0,
        dr=r[1] - r[0],
        radius=r,
        geometry="cartesian",
    )
    assert np.all(T_after >= 0.2)


def test_diffusion_step_rejects_nonpositive_heat_capacity():
    n = 6
    r = np.linspace(0.0, 1.0, n)
    T = np.full(n, 0.5)
    C = np.ones(n)
    C[2] = -1.0
    with pytest.raises(ValueError):
        diffusion_step(
            temperature=T,
            source=np.zeros(n),
            dt=1e-3,
            diffusivity=0.05,
            heat_capacity=C,
            space_temperature=0.0,
            surface_cooling=0.0,
            dr=r[1] - r[0],
            radius=r,
            geometry="cartesian",
        )


def test_diffusion_step_requires_radius_for_spherical_geometry():
    n = 6
    T = np.full(n, 0.5)
    with pytest.raises(ValueError):
        diffusion_step(
            temperature=T,
            source=np.zeros(n),
            dt=1e-3,
            diffusivity=0.05,
            heat_capacity=1.0,
            space_temperature=0.0,
            surface_cooling=0.0,
            dr=0.2,
            radius=None,
            geometry="spherical",
        )
