import numpy as np
import pytest

from cosmociety.convection import (
    convective_diffusivity_from_gradient,
    convective_diffusivity_from_radiative_demand,
    convective_diffusivity_from_schwarzschild,
    convective_flux_from_adiabatic_excess,
)


def test_gradient_criterion_inactive_below_threshold():
    n = 8
    r = np.linspace(0.0, 1.0, n)
    dr = r[1] - r[0]
    T = 1.0 - 0.1 * r  # shallow slope
    D, excess = convective_diffusivity_from_gradient(
        T, dr=dr, threshold=5.0, strength=0.1, max_diffusivity=1e-2
    )
    assert np.all(excess == 0.0)
    assert np.all(D == 0.0)


def test_gradient_criterion_caps_at_max_diffusivity_when_far_above_threshold():
    n = 8
    r = np.linspace(0.0, 1.0, n)
    dr = r[1] - r[0]
    T = 1.0 - 50.0 * r  # steep slope, far above threshold
    D, excess = convective_diffusivity_from_gradient(
        T, dr=dr, threshold=5.0, strength=0.1, max_diffusivity=1e-2
    )
    assert np.all(excess > 0.0)
    assert np.allclose(D[1:-1], 1e-2)


def test_radiative_demand_criterion_inactive_when_diffusivity_is_generous():
    n = 8
    r = np.linspace(0.0, 1.0, n)
    dr = r[1] - r[0]
    source = np.exp(-((r / 0.12) ** 2))
    D_rad = np.full(n, 0.02)  # plenty of radiative transport capacity
    D, excess = convective_diffusivity_from_radiative_demand(
        source, r, D_rad, dr=dr, threshold=1.0, strength=0.05, max_diffusivity=1e-3
    )
    assert np.all(excess == 0.0)
    assert np.all(D == 0.0)


def test_radiative_demand_criterion_active_when_diffusivity_is_starved():
    n = 8
    r = np.linspace(0.0, 1.0, n)
    dr = r[1] - r[0]
    source = np.exp(-((r / 0.12) ** 2))
    D_rad = np.full(n, 1e-6)  # radiation alone can't carry the luminosity
    D, excess = convective_diffusivity_from_radiative_demand(
        source, r, D_rad, dr=dr, threshold=1.0, strength=0.05, max_diffusivity=1e-3
    )
    assert np.any(excess > 0.0)
    assert np.any(D > 0.0)


def test_schwarzschild_nabla_rad_decreases_outward_for_centrally_concentrated_source():
    # Enclosed luminosity growth slows with radius once past the source's
    # width, so nabla_rad should trend downward away from the core.
    n = 10
    r = np.linspace(0.0, 1.0, n)
    dr = r[1] - r[0]
    source = np.exp(-((r / 0.12) ** 2))
    T = 1.0 - 0.5 * r
    pressure = (1.0 - 0.5 * r) ** 3 * T
    D_rad = np.full(n, 0.02)
    _, _, nabla_rad = convective_diffusivity_from_schwarzschild(
        source, r, T, pressure, D_rad, dr=dr, nabla_ad=0.4, strength=0.05, max_diffusivity=1e-3
    )
    assert nabla_rad[1] > nabla_rad[-1]


def test_convective_flux_from_adiabatic_excess_never_exceeds_ordinary_diffusive_flux():
    # This is the invariant the CFL bound in equilibrium.relax_to_equilibrium
    # relies on to stay conservative in "excess" transport mode (see the
    # comment above `stable_dt`): the superadiabatic drop this flux
    # transports can never exceed the raw outward temperature drop, so the
    # excess flux can never exceed what the same diffusivity would carry as
    # an ordinary diffusive flux on the same gradient.
    rng = np.random.default_rng(0)
    dr = 1.0 / 9
    for _ in range(200):
        temperature = np.sort(rng.uniform(0.5, 2.0, size=10))[::-1]
        density = np.sort(rng.uniform(0.1, 1.0, size=10))[::-1]
        pressure = density**3 * temperature
        convective_diffusivity = rng.uniform(0.0, 1e-3, size=10)
        nabla_ad = rng.uniform(0.05, 0.6)

        flux = convective_flux_from_adiabatic_excess(
            temperature, pressure, convective_diffusivity, dr, nabla_ad
        )
        diffusivity_interface = 0.5 * (convective_diffusivity[:-1] + convective_diffusivity[1:])
        ordinary_diffusive_flux = diffusivity_interface * (-np.diff(temperature) / dr)

        assert np.all(flux >= -1e-12)
        assert np.all(flux <= ordinary_diffusive_flux + 1e-9)


def test_convective_flux_from_adiabatic_excess_zero_on_exact_adiabat():
    # If the profile sits exactly on the adiabatic gradient, there is no
    # superadiabatic drop left to transport. Build T interface-by-interface
    # by directly solving the function's own discrete adiabat equation
    #   -(T[i+1]-T[i])/dr == nabla_ad * T_interface * outward_pressure_drop / P_interface
    # (rather than the continuum ODE d(lnT) = nabla_ad d(lnP), which only
    # agrees with this discretization in the dr -> 0 limit) so the result is
    # exactly zero rather than merely small.
    n = 8
    density = np.linspace(1.0, 0.2, n)
    nabla_ad = 0.4
    pressure = density**3.0
    dr = 1.0 / (n - 1)

    T = np.empty(n)
    T[0] = 1.0
    for i in range(n - 1):
        P0, P1 = pressure[i], pressure[i + 1]
        outward_pressure_drop = -(P1 - P0) / dr
        pressure_interface = 0.5 * (P0 + P1)
        k = nabla_ad * outward_pressure_drop / (2 * pressure_interface)
        T[i + 1] = T[i] * (1 - dr * k) / (1 + dr * k)

    D_conv = np.full(n, 5e-4)
    flux = convective_flux_from_adiabatic_excess(T, pressure, D_conv, dr, nabla_ad)
    assert np.allclose(flux, 0.0, atol=1e-12)
