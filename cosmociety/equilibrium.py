import numpy as np

from .convection import (
    convective_diffusivity_from_gradient,
    convective_diffusivity_from_radiative_demand,
    convective_diffusivity_from_schwarzschild,
    convective_flux_from_adiabatic_excess,
)
from .grid import radial_grid
from .opacity import opacity_from_state, radiative_diffusivity_from_state
from .profiles import core_source, density_profile, initial_temperature
from .transport import diffusion_step, interface_flux


def convective_threshold_for(
    convective_criterion: str,
    convective_gradient_threshold: float,
    convective_radiative_threshold: float,
    convective_nabla_ad: float,
) -> float:
    """Return the active threshold for a named convection criterion."""
    if convective_criterion == "gradient":
        return convective_gradient_threshold
    if convective_criterion == "radiative":
        return convective_radiative_threshold
    if convective_criterion == "schwarzschild":
        return convective_nabla_ad
    raise ValueError(
        "convective_criterion must be 'gradient', 'radiative', or 'schwarzschild'"
    )


def evaluate_convection(
    convective_criterion: str,
    temperature: np.ndarray,
    pressure: np.ndarray,
    source: np.ndarray,
    radius: np.ndarray,
    radiative_diffusivity: np.ndarray,
    dr: float,
    convective_gradient_threshold: float,
    convective_radiative_threshold: float,
    convective_nabla_ad: float,
    convective_strength: float,
    convective_max_diffusivity: float,
    enable_convection: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate the current convective state for one temperature profile."""
    convective_threshold = convective_threshold_for(
        convective_criterion=convective_criterion,
        convective_gradient_threshold=convective_gradient_threshold,
        convective_radiative_threshold=convective_radiative_threshold,
        convective_nabla_ad=convective_nabla_ad,
    )
    convective_diffusivity = np.zeros_like(temperature)
    convective_excess = np.zeros(len(temperature) - 1)
    nabla_rad = np.zeros(len(temperature) - 1)

    if not enable_convection:
        return (
            convective_diffusivity,
            convective_excess,
            nabla_rad,
            convective_threshold,
        )

    if convective_criterion == "gradient":
        convective_diffusivity, convective_excess = convective_diffusivity_from_gradient(
            temperature=temperature,
            dr=dr,
            threshold=convective_threshold,
            strength=convective_strength,
            max_diffusivity=convective_max_diffusivity,
        )
    elif convective_criterion == "radiative":
        convective_diffusivity, convective_excess = (
            convective_diffusivity_from_radiative_demand(
                source=source,
                radius=radius,
                diffusivity=radiative_diffusivity,
                dr=dr,
                threshold=convective_threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        )
    elif convective_criterion == "schwarzschild":
        convective_diffusivity, convective_excess, nabla_rad = (
            convective_diffusivity_from_schwarzschild(
                source=source,
                radius=radius,
                temperature=temperature,
                pressure=pressure,
                diffusivity=radiative_diffusivity,
                dr=dr,
                nabla_ad=convective_threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        )

    return convective_diffusivity, convective_excess, nabla_rad, convective_threshold


def relax_to_equilibrium(
    n: int = 200,
    dt: float = 1e-5,
    diffusivity: float = 0.02,
    space_temperature: float = 0.01,
    surface_cooling: float = 10.0,
    max_steps: int = 7_000_000,
    tolerance: float = 3e-8,
    save_every: int = 5000,
    stability_safety: float = 0.2,
    opacity_temperature_power: float = 10.0,
    opacity_density_power: float = 0.1,
    opacity_power: float | None = None,
    radiative_density_power: float | None = None,
    radiative_diffusivity_floor: float = 1e-8,
    surface_density: float = 0.1,
    central_density: float = 1.0,
    density_concentration: float = 2.0,
    pressure_density_power: float = 3.0,
    geometry: str = "spherical",
    enable_convection: bool = True,
    convective_gradient_threshold: float = 2.0,
    convective_radiative_threshold: float = 75_000.0,
    convective_nabla_ad: float = 0.4,
    convective_strength: float = 0.05,
    convective_max_diffusivity: float = 1e-3,
    convective_criterion: str = "schwarzschild",
    convective_transport: str = "excess",
) -> dict:
    """
    Relax the toy stellar temperature profile toward equilibrium.

    Equilibrium means max |T_new - T_old| < tolerance.
    """
    if opacity_power is not None:
        opacity_temperature_power = opacity_power
    if radiative_density_power is not None:
        opacity_density_power = radiative_density_power

    r = radial_grid(n)
    source = core_source(r)
    density = density_profile(
        r,
        surface_density=surface_density,
        central_density=central_density,
        concentration=density_concentration,
    )
    heat_capacity = density
    T = initial_temperature(r, surface_temperature = 0.1)

    history = [T.copy()]
    deltas = []
    timesteps = []

    converged_step = None
    dr = r[1] - r[0]
    convective_threshold = convective_threshold_for(
        convective_criterion=convective_criterion,
        convective_gradient_threshold=convective_gradient_threshold,
        convective_radiative_threshold=convective_radiative_threshold,
        convective_nabla_ad=convective_nabla_ad,
    )
    if convective_transport not in {"diffusive", "excess"}:
        raise ValueError("convective_transport must be 'diffusive' or 'excess'")

    for step in range(max_steps):
        old_T = T.copy()
        pressure = density**pressure_density_power * T
        radiative_diffusivity = radiative_diffusivity_from_state(
            temperature=T,
            density=density,
            diffusivity_scale=diffusivity,
            opacity_temperature_power=opacity_temperature_power,
            opacity_density_power=opacity_density_power,
            diffusivity_floor=radiative_diffusivity_floor,
        )
        convective_diffusivity, _, _, convective_threshold = evaluate_convection(
            convective_criterion=convective_criterion,
            temperature=T,
            pressure=pressure,
            source=source,
            radius=r,
            radiative_diffusivity=radiative_diffusivity,
            dr=dr,
            convective_gradient_threshold=convective_gradient_threshold,
            convective_radiative_threshold=convective_radiative_threshold,
            convective_nabla_ad=convective_nabla_ad,
            convective_strength=convective_strength,
            convective_max_diffusivity=convective_max_diffusivity,
            enable_convection=enable_convection,
        )

        effective_diffusivity = radiative_diffusivity + convective_diffusivity
        if convective_transport == "diffusive":
            step_diffusivity = effective_diffusivity
            extra_interface_flux = None
        else:
            step_diffusivity = radiative_diffusivity
            extra_interface_flux = convective_flux_from_adiabatic_excess(
                temperature=T,
                pressure=pressure,
                convective_diffusivity=convective_diffusivity,
                dr=dr,
                nabla_ad=convective_nabla_ad,
            )

        max_effective_diffusivity = float(np.max(effective_diffusivity / heat_capacity))
        stable_dt = stability_safety * dr**2 / max_effective_diffusivity
        step_dt = min(dt, stable_dt)

        T = diffusion_step(
            temperature=T,
            source=source,
            dt=step_dt,
            diffusivity=step_diffusivity,
            heat_capacity=heat_capacity,
            space_temperature=space_temperature,
            surface_cooling=surface_cooling,
            dr=dr,
            radius=r,
            geometry=geometry,
            extra_interface_flux=extra_interface_flux,
        )

        delta = float(np.max(np.abs(T - old_T)))
        deltas.append(delta)
        timesteps.append(step_dt)

        if not np.all(np.isfinite(T)):
            raise FloatingPointError(
                f"Temperature became non-finite at step {step}. "
                f"Try lowering dt or stability_safety."
            )

        if step % save_every == 0:
            history.append(T.copy())

        if delta < tolerance:
            converged_step = step
            history.append(T.copy())
            break

    kappa = opacity_from_state(
        temperature=T,
        density=density,
        opacity_temperature_power=opacity_temperature_power,
        opacity_density_power=opacity_density_power,
    )
    pressure = density**pressure_density_power * T
    radiative_diffusivity = radiative_diffusivity_from_state(
        temperature=T,
        density=density,
        diffusivity_scale=diffusivity,
        opacity_temperature_power=opacity_temperature_power,
        opacity_density_power=opacity_density_power,
        diffusivity_floor=radiative_diffusivity_floor,
    )
    convective_diffusivity, convective_excess, nabla_rad, convective_threshold = (
        evaluate_convection(
            convective_criterion=convective_criterion,
            temperature=T,
            pressure=pressure,
            source=source,
            radius=r,
            radiative_diffusivity=radiative_diffusivity,
            dr=dr,
            convective_gradient_threshold=convective_gradient_threshold,
            convective_radiative_threshold=convective_radiative_threshold,
            convective_nabla_ad=convective_nabla_ad,
            convective_strength=convective_strength,
            convective_max_diffusivity=convective_max_diffusivity,
            enable_convection=enable_convection,
        )
    )
    total_diffusivity = radiative_diffusivity + convective_diffusivity
    radiative_flux = interface_flux(T, radiative_diffusivity, dr)
    if convective_transport == "diffusive":
        convective_flux = interface_flux(T, convective_diffusivity, dr)
    else:
        convective_flux = convective_flux_from_adiabatic_excess(
            temperature=T,
            pressure=pressure,
            convective_diffusivity=convective_diffusivity,
            dr=dr,
            nabla_ad=convective_nabla_ad,
        )
    flux = radiative_flux + convective_flux
    r_interface = 0.5 * (r[:-1] + r[1:])
    luminosity = r_interface**2 * flux

    return {
        "r": r,
        "temperature": T,
        "source": source,
        "density": density,
        "pressure": pressure,
        "pressure_density_power": pressure_density_power,
        "heat_capacity": heat_capacity,
        "kappa": kappa,
        "opacity_temperature_power": opacity_temperature_power,
        "opacity_density_power": opacity_density_power,
        "opacity_power": opacity_temperature_power,
        "radiative_density_power": opacity_density_power,
        "radiative_diffusivity_scale": diffusivity,
        "radiative_diffusivity_floor": radiative_diffusivity_floor,
        "radiative_diffusivity": radiative_diffusivity,
        "convective_diffusivity": convective_diffusivity,
        "diffusivity": total_diffusivity,
        "temperature_gradient": np.diff(T) / dr,
        "convective_excess": convective_excess,
        "convective_threshold": convective_threshold,
        "convective_gradient_threshold": convective_gradient_threshold,
        "convective_radiative_threshold": convective_radiative_threshold,
        "convective_nabla_ad": convective_nabla_ad,
        "convective_strength": convective_strength,
        "nabla_rad": nabla_rad,
        "convective_max_diffusivity": convective_max_diffusivity,
        "convective_criterion": convective_criterion,
        "convective_transport": convective_transport,
        "enable_convection": enable_convection,
        "surface_cooling": surface_cooling,
        "radiative_flux": radiative_flux,
        "convective_flux": convective_flux,
        "flux": flux,
        "luminosity": luminosity,
        "r_interface": r_interface,
        "geometry": geometry,
        "history": np.array(history),
        "deltas": np.array(deltas),
        "timesteps": np.array(timesteps),
        "converged_step": converged_step,
    }
