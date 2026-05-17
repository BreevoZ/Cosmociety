import numpy as np


def diffusion_step(
    temperature: np.ndarray,
    source: np.ndarray,
    dt: float,
    diffusivity: float,
    surface_temperature: float,
) -> np.ndarray:
    """
    One explicit diffusion step with core heating and fixed surface temperature.

    This is the simplest toy model:

        dT/dt = diffusivity * d²T/dr² + source

    Boundary conditions:
        center: dT/dr = 0
        surface: T = surface_temperature
    """
    T = temperature.copy()

    laplacian = np.zeros_like(T)
    laplacian[1:-1] = T[:-2] - 2.0 * T[1:-1] + T[2:]

    T += dt * (diffusivity * laplacian + source)

    # Boundary conditions
    T[0] = T[1]
    T[-1] = surface_temperature

    return T
