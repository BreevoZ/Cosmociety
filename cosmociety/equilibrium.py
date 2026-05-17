import numpy as np

from .grid import radial_grid
from .profiles import core_source, initial_temperature
from .transport import diffusion_step


def relax_to_equilibrium(
    n: int = 200,
    dt: float = 1e-5,
    diffusivity: float = 0.5,
    surface_temperature: float = 0.1,
    max_steps: int = 500_000,
    tolerance: float = 1e-7,
    save_every: int = 1000,
) -> dict:
    """
    Relax the toy stellar temperature profile toward equilibrium.

    Equilibrium means max |T_new - T_old| < tolerance.
    """
    r = radial_grid(n)
    source = core_source(r)
    T = initial_temperature(r, surface_temperature)

    history = [T.copy()]
    deltas = []

    converged_step = None
    dr = r[1] - r[0]

    for step in range(max_steps):
        old_T = T.copy()

        T = diffusion_step(
            temperature=T,
            source=source,
            dt=dt,
            diffusivity=diffusivity,
            surface_temperature=surface_temperature,
            dr=dr,
        )

        delta = float(np.max(np.abs(T - old_T)))
        deltas.append(delta)

        if step % save_every == 0:
            history.append(T.copy())

        if delta < tolerance:
            converged_step = step
            history.append(T.copy())
            break

    return {
        "r": r,
        "temperature": T,
        "source": source,
        "history": np.array(history),
        "deltas": np.array(deltas),
        "converged_step": converged_step,
    }
