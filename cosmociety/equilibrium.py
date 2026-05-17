import numpy as np

from .grid import radial_grid
from .profiles import core_source, initial_temperature
from .transport import diffusion_step, interface_flux


def relax_to_equilibrium(
    n: int = 200,
    dt: float = 1e-5,
    diffusivity: float = 0.02,
    space_temperature: float = 0.01,
    surface_cooling: float = 10.0,
    max_steps: int = 2_000_000,
    tolerance: float = 1e-7,
    save_every: int = 5000,
    stability_safety: float = 0.2,
    opacity_power: float = 3.0,
    geometry: str = "spherical",
) -> dict:
    """
    Relax the toy stellar temperature profile toward equilibrium.

    Equilibrium means max |T_new - T_old| < tolerance.
    """
    r = radial_grid(n)
    source = core_source(r)
    T = initial_temperature(r, surface_temperature = 0.1)

    history = [T.copy()]
    deltas = []
    timesteps = []

    converged_step = None
    dr = r[1] - r[0]

    for step in range(max_steps):
        old_T = T.copy()
        kappa = T ** (-opacity_power)
        diffusivity_profile = diffusivity / kappa
        max_diffusivity = float(np.max(diffusivity_profile))
        stable_dt = stability_safety * dr**2 / max_diffusivity
        step_dt = min(dt, stable_dt)

        T = diffusion_step(
            temperature=T,
            source=source,
            dt=step_dt,
            diffusivity=diffusivity_profile,
            space_temperature=space_temperature,
            surface_cooling=surface_cooling,
            dr=dr,
            radius=r,
            geometry=geometry,
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
    diffusivity_profile = diffusivity * T ** opacity_power
    flux = interface_flux(T, diffusivity_profile, dr)
    r_interface = 0.5 * (r[:-1] + r[1:])
    luminosity = r_interface**2 * flux

    return {
        "r": r,
        "temperature": T,
        "source": source,
        "kappa": kappa,
        "diffusivity": diffusivity_profile,
        "temperature_gradient": np.diff(T) / dr,
        "flux": flux,
        "luminosity": luminosity,
        "r_interface": r_interface,
        "geometry": geometry,
        "history": np.array(history),
        "deltas": np.array(deltas),
        "timesteps": np.array(timesteps),
        "converged_step": converged_step,
    }
