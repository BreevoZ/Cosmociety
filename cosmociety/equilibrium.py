import numpy as np

from .grid import radial_grid
from .profiles import core_source, density_profile, initial_temperature
from .transport import diffusion_step, interface_flux


def convective_diffusivity_from_gradient(
    temperature: np.ndarray,
    dr: float,
    threshold: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a toy convective diffusivity triggered by steep temperature gradients.

    Convection appears where the outward temperature drop is steeper than
    threshold. The returned cell-centered diffusivity can be added to the
    radiative diffusivity.
    """
    outward_drop = -np.diff(temperature) / dr
    excess = np.maximum(outward_drop - threshold, 0.0)
    threshold_scale = max(threshold, 1e-30)
    interface_diffusivity = np.minimum(
        strength * excess / threshold_scale,
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(temperature)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess


def convective_diffusivity_from_radiative_demand(
    source: np.ndarray,
    radius: np.ndarray,
    diffusivity: np.ndarray,
    dr: float,
    threshold: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return convective diffusivity where radiation alone needs a steep gradient.

    This is a toy Schwarzschild-like trigger. It estimates the luminosity that
    must pass through each interface from the enclosed source, then asks what
    temperature gradient radiation alone would need to carry that luminosity.
    """
    r_interface = 0.5 * (radius[:-1] + radius[1:])
    source_shells = source * radius**2 * dr
    enclosed_luminosity = np.cumsum(source_shells)[:-1]

    diffusivity_interface = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    radiative_gradient = enclosed_luminosity / (
        np.maximum(r_interface**2 * diffusivity_interface, 1e-30)
    )
    excess = np.maximum(radiative_gradient - threshold, 0.0)
    threshold_scale = max(threshold, 1e-30)
    interface_diffusivity = np.minimum(
        strength * excess / threshold_scale,
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(source)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess


def convective_diffusivity_from_schwarzschild(
    source: np.ndarray,
    radius: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    diffusivity: np.ndarray,
    dr: float,
    nabla_ad: float,
    strength: float,
    max_diffusivity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return convective diffusivity from a toy Schwarzschild criterion.

    It estimates the temperature gradient radiation alone would need to carry
    the enclosed luminosity, converts that to nabla_rad = d ln T / d ln P, and
    triggers convection where nabla_rad > nabla_ad.
    """
    r_interface = 0.5 * (radius[:-1] + radius[1:])
    source_shells = source * radius**2 * dr
    enclosed_luminosity = np.cumsum(source_shells)[:-1]

    diffusivity_interface = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    required_temperature_drop = enclosed_luminosity / (
        np.maximum(r_interface**2 * diffusivity_interface, 1e-30)
    )

    temperature_interface = 0.5 * (temperature[:-1] + temperature[1:])
    pressure_interface = 0.5 * (pressure[:-1] + pressure[1:])
    pressure_drop = -np.diff(pressure) / dr

    nabla_rad = (
        pressure_interface
        * required_temperature_drop
        / np.maximum(temperature_interface * pressure_drop, 1e-30)
    )
    excess = np.maximum(nabla_rad - nabla_ad, 0.0)
    interface_diffusivity = np.minimum(
        strength * excess / max(nabla_ad, 1e-30),
        max_diffusivity,
    )

    cell_diffusivity = np.zeros_like(source)
    cell_diffusivity[0] = interface_diffusivity[0]
    cell_diffusivity[-1] = interface_diffusivity[-1]
    cell_diffusivity[1:-1] = 0.5 * (
        interface_diffusivity[:-1] + interface_diffusivity[1:]
    )

    return cell_diffusivity, excess, nabla_rad


def convective_flux_from_adiabatic_excess(
    temperature: np.ndarray,
    pressure: np.ndarray,
    convective_diffusivity: np.ndarray,
    dr: float,
    nabla_ad: float,
) -> np.ndarray:
    """
    Return convective flux that only transports superadiabatic temperature drop.

    The target retained gradient is the toy adiabatic gradient:

        dT/dr_ad = nabla_ad * T/P * dP/dr

    Since outward heat flux is positive, this uses outward temperature drops.
    """
    outward_temperature_drop = -np.diff(temperature) / dr
    outward_pressure_drop = np.maximum(-np.diff(pressure) / dr, 0.0)
    temperature_interface = 0.5 * (temperature[:-1] + temperature[1:])
    pressure_interface = 0.5 * (pressure[:-1] + pressure[1:])
    diffusivity_interface = 0.5 * (
        convective_diffusivity[:-1] + convective_diffusivity[1:]
    )

    adiabatic_temperature_drop = (
        nabla_ad
        * temperature_interface
        * outward_pressure_drop
        / np.maximum(pressure_interface, 1e-30)
    )
    superadiabatic_drop = np.maximum(
        outward_temperature_drop - adiabatic_temperature_drop,
        0.0,
    )
    return diffusivity_interface * superadiabatic_drop


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
    opacity_power: float = 10.0,
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
    if convective_criterion == "gradient":
        convective_threshold = convective_gradient_threshold
    elif convective_criterion == "radiative":
        convective_threshold = convective_radiative_threshold
    elif convective_criterion == "schwarzschild":
        convective_threshold = convective_nabla_ad
    else:
        raise ValueError(
            "convective_criterion must be 'gradient', 'radiative', or 'schwarzschild'"
        )
    if convective_transport not in {"diffusive", "excess"}:
        raise ValueError("convective_transport must be 'diffusive' or 'excess'")

    for step in range(max_steps):
        old_T = T.copy()
        pressure = density**pressure_density_power * T
        kappa = T ** (-opacity_power)
        radiative_diffusivity = np.maximum(diffusivity / kappa, radiative_diffusivity_floor)
        convective_diffusivity = np.zeros_like(T)
        if enable_convection:
            if convective_criterion == "gradient":
                convective_threshold = convective_gradient_threshold
                convective_diffusivity, _ = convective_diffusivity_from_gradient(
                    temperature=T,
                    dr=dr,
                    threshold=convective_threshold,
                    strength=convective_strength,
                    max_diffusivity=convective_max_diffusivity,
                )
            elif convective_criterion == "radiative":
                convective_threshold = convective_radiative_threshold
                convective_diffusivity, _ = convective_diffusivity_from_radiative_demand(
                    source=source,
                    radius=r,
                    diffusivity=radiative_diffusivity,
                    dr=dr,
                    threshold=convective_threshold,
                    strength=convective_strength,
                    max_diffusivity=convective_max_diffusivity,
                )
            elif convective_criterion == "schwarzschild":
                convective_threshold = convective_nabla_ad
                convective_diffusivity, _, _ = convective_diffusivity_from_schwarzschild(
                    source=source,
                    radius=r,
                    temperature=T,
                    pressure=pressure,
                    diffusivity=radiative_diffusivity,
                    dr=dr,
                    nabla_ad=convective_threshold,
                    strength=convective_strength,
                    max_diffusivity=convective_max_diffusivity,
                )
            else:
                raise ValueError(
                    "convective_criterion must be 'gradient', 'radiative', or 'schwarzschild'"
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

    kappa = T ** (-opacity_power)
    pressure = density**pressure_density_power * T
    radiative_diffusivity = np.maximum(
        diffusivity * T ** opacity_power,
        radiative_diffusivity_floor,
    )
    convective_diffusivity = np.zeros_like(T)
    convective_excess = np.zeros(len(T) - 1)
    nabla_rad = np.zeros(len(T) - 1)
    if enable_convection:
        if convective_criterion == "gradient":
            convective_threshold = convective_gradient_threshold
            convective_diffusivity, convective_excess = convective_diffusivity_from_gradient(
                temperature=T,
                dr=dr,
                threshold=convective_threshold,
                strength=convective_strength,
                max_diffusivity=convective_max_diffusivity,
            )
        elif convective_criterion == "radiative":
            convective_threshold = convective_radiative_threshold
            convective_diffusivity, convective_excess = (
                convective_diffusivity_from_radiative_demand(
                    source=source,
                    radius=r,
                    diffusivity=radiative_diffusivity,
                    dr=dr,
                    threshold=convective_threshold,
                    strength=convective_strength,
                    max_diffusivity=convective_max_diffusivity,
                )
            )
        elif convective_criterion == "schwarzschild":
            convective_threshold = convective_nabla_ad
            convective_diffusivity, convective_excess, nabla_rad = (
                convective_diffusivity_from_schwarzschild(
                    source=source,
                    radius=r,
                    temperature=T,
                    pressure=pressure,
                    diffusivity=radiative_diffusivity,
                    dr=dr,
                    nabla_ad=convective_threshold,
                    strength=convective_strength,
                    max_diffusivity=convective_max_diffusivity,
                )
            )
        else:
            raise ValueError(
                "convective_criterion must be 'gradient', 'radiative', or 'schwarzschild'"
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
        "opacity_power": opacity_power,
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
